import torch
import copy
import inspect
import warnings
import mdtex2html
import gradio as gr
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

logger = logging.get_logger("transformers")
logging.set_verbosity_error()


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id) -> None:
        super().__init__()
        self.eos_token_id = eos_token_id
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., self.eos_token_id] = 5e4
        return scores


class Chatter():
    """
    Copied from https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py
    """
    def __init__(self, model:PreTrainedModel, tokenizer:PreTrainedTokenizer, add_chat_prompt:bool = True, use_cache:bool = True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.add_chat_prompt = add_chat_prompt
        self.use_cache = use_cache

    def process_response(self, response):
        response = response.strip()
        return response
    
    def remove_unused_columns(self, inputs):
        signature = inspect.signature(self.model.forward)
        signature_columns = list(signature.parameters.keys())
        ignored_columns = list(set(inputs.keys()) - set(signature_columns))
        if len(ignored_columns):
            for ignored_column in ignored_columns:
                inputs.pop(ignored_column)
        return inputs

    def build_inputs(self, query: str, history: List[Tuple[str, str]] = None):
        if history is None:
            history = []
        prompt = ""
        if self.add_chat_prompt:
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n\nUser: {}\n\nAssistant: {}\n\n".format(i + 1, old_query, response)
            prompt += "[Round {}]\n\nUser: {}\n\nAssistant: ".format(len(history) + 1, query)
        else:
            for i, (old_query, response) in enumerate(history):
                prompt += old_query + response
            prompt += query

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = self.remove_unused_columns(inputs)
        inputs = inputs.to(self.model.device)
        return inputs

    def build_stream_inputs(self, query: str, history: List[Tuple[str, str]] = None):
        if history is None:
            history = []
        if self.add_chat_prompt:
            prompt = "[Round {}]\n\nUser: {}\n\nAssistant: ".format(len(history) + 1, query)
            if len(history):
                prompt = "\n\n" + prompt
        else:
            prompt = query
        
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=len(history) == 0)
        inputs = self.remove_unused_columns(inputs)
        inputs = inputs.to(self.model.device)
        return inputs

    @torch.no_grad()
    def chat(self, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor(eos_token_id=tokenizer.eos_token_id))
        gen_kwargs = {
            "max_length": max_length, 
            "num_beams": num_beams, 
            "do_sample": do_sample, 
            "top_p": top_p,
            "temperature": temperature, 
            "logits_processor": logits_processor, 
            **kwargs
        }
        inputs = self.build_inputs(query, history=history)
        print(f"Inputs: {self.tokenizer.batch_decode(inputs['input_ids'])}")

        outputs = self.model.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def stream_chat(
        self, 
        query: str, 
        history: List[Tuple[str, str]] = None, 
        past_key_values=None,
        max_length: int = 8192, 
        max_new_tokens: int = 100, 
        do_sample=True, 
        top_p=0.8, 
        temperature=0.8, 
        logits_processor=None,
        **kwargs
    ):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor(eos_token_id=self.tokenizer.eos_token_id))
        gen_kwargs = {
            "max_length": max_length,
            "max_new_tokens": max_new_tokens, 
            "do_sample": do_sample, 
            "top_p": top_p,
            "temperature": temperature, 
            "logits_processor": logits_processor, 
            **kwargs
        }
        # in case we disable caching
        if past_key_values is None and not self.use_cache:
            inputs = self.build_inputs(query, history=history)
        else:
            inputs = self.build_stream_inputs(query, history=history)

        print(f"Inputs: {self.tokenizer.batch_decode(inputs['input_ids'])}")

        if past_key_values is not None:
            # NOTE: split inputs into two parts: all previous tokens and the last token
            previous_inputs = {}
            current_inputs = {}
            for k, v in inputs.items():
                previous_inputs[k] = v[:, :-1]
                current_inputs[k] = v[:, -1:]
            past_length = past_key_values[0][0].shape[-2]

            if previous_inputs["input_ids"].shape[1] > 0:
                # encode all previous tokens to get past_key_values
                attention_mask = previous_inputs["attention_mask"]
                # NOTE: we must extend attention mask according to https://github.com/huggingface/transformers/issues/24741
                previous_inputs["attention_mask"] = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=-1)

                previous_outputs = self.model(**previous_inputs, past_key_values=past_key_values)
                past_key_values = previous_outputs.past_key_values
                # update past_length because there are newly encoded tokens
                past_length = previous_outputs.past_key_values[0][0].size(-2)

            attention_mask = current_inputs["attention_mask"]
            # update attention mask by 1 because when generating, the model automatically truncate input_ids to the last token 
            current_inputs["attention_mask"] = torch.cat((attention_mask.new_ones(1, past_length), attention_mask[:, :1]), dim=-1)
            inputs = current_inputs

        for outputs in self.stream_generate(**inputs, past_key_values=past_key_values, **gen_kwargs):
            if self.use_cache:
                outputs, past_key_values = outputs
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = self.tokenizer.decode(outputs)
            if response and response[-1] != "�":
                response = self.process_response(response)
                new_history = history + [(query, response)]
                if self.use_cache:
                    yield response, new_history, past_key_values
                else:
                    yield response, new_history, None

    @torch.no_grad()
    def stream_generate(
        self,
        input_ids,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        **kwargs,
    ):
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.model.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.model.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self.model._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self.model._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None
        while True:
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())
            if self.use_cache:
                yield input_ids, outputs.past_key_values
            else:
                yield input_ids
            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

def parse_text(text):
    # lines = text.split("\n")
    # lines = [line for line in lines if line != ""]
    # text = "".join(lines)
    # count = 0
    # for i, line in enumerate(lines):
    #     if "```" in line:
    #         count += 1
    #         items = line.split('`')
    #         if count % 2 == 1:
    #             lines[i] = f'<pre><code class="language-{items[-1]}">'
    #         else:
    #             lines[i] = f'<br></code></pre>'
    #     else:
    #         if i > 0:
    #             if count % 2 == 1:
    #                 line = line.replace("`", "\`")
    #                 line = line.replace("<", "&lt;")
    #                 line = line.replace(">", "&gt;")
    #                 line = line.replace(" ", "&nbsp;")
    #                 line = line.replace("*", "&ast;")
    #                 line = line.replace("_", "&lowbar;")
    #                 line = line.replace("-", "&#45;")
    #                 line = line.replace(".", "&#46;")
    #                 line = line.replace("!", "&#33;")
    #                 line = line.replace("(", "&#40;")
    #                 line = line.replace(")", "&#41;")
    #                 line = line.replace("$", "&#36;")
    #             lines[i] = "<br>"+line
    # text = text.strip()
    return text


def predict(_input, chatbot, max_length, max_new_tokens, top_p, temperature, history, past_key_values):
    chatbot.append((parse_text(_input), ""))
    if temperature == 0:
        temperature = 1e-8
    for response, history, past_key_values in chatter.stream_chat(
        _input, 
        history, 
        past_key_values=past_key_values,
        max_length=max_length,
        max_new_tokens=max_new_tokens, 
        top_p=top_p,
        temperature=temperature,
    ):
        chatbot[-1] = (parse_text(_input), parse_text(response))

        yield chatbot, history, past_key_values


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None


@dataclass
class DemoArgs:
    model_name_or_path: str = field(
        default='EleutherAI/gpt-neo-125m',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    model_save_dir: str = field(
        default='/share/LMs',
        metadata={'help': 'Default path to save language models'}
    )
    padding_side: str = field(
        default="left",
        metadata={'help': 'Tokenizer padding side'}
    )
    device: str = field(
        default="cpu",
        metadata={'help': 'What device to put the model?'}
    )
    no_cache: bool = field(
        default=False,
        metadata={'help': 'Cache past key values for faster decoding?'}
    )
    no_prompt: bool = field(
        default=True,
        metadata={'help': 'Disable chatting prompts? ([Round K] User: xxx\n\n Assistant: xxx\n\n)'}
    )
    no_html: bool = field(
        default=True,
        metadata={'help': 'Parse html?'}
    )
    use_fast: bool = field(
        default=True,
        metadata={'help': 'Use fast tokenizer?'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 model?'}
    )
    bf16: bool = field(
        default=False,
        metadata={'help': 'Use bf16 model?'}
    )
    def __post_init__(self):
        if self.fp16:
            torch_dtype = torch.float16
        elif self.bf16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = "auto"
        self.torch_dtype = torch_dtype
        try:
            self.device = int(self.device)
        except:
            pass


if __name__ == "__main__":
    parser = HfArgumentParser([DemoArgs])
    demo_args, = parser.parse_args_into_dataclasses()

    print(f"Loading model and tokenizer from {demo_args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(demo_args.model_name_or_path, cache_dir=demo_args.model_save_dir, padding_side=demo_args.padding_side, trust_remote_code=True, use_fast=demo_args.use_fast)
    model = AutoModelForCausalLM.from_pretrained(demo_args.model_name_or_path, cache_dir=demo_args.model_save_dir, trust_remote_code=True, torch_dtype=demo_args.torch_dtype)
    model = model.to(demo_args.device).eval()
    chatter = Chatter(model, tokenizer, add_chat_prompt=not demo_args.no_prompt, use_cache=not demo_args.no_cache)

    gr.Chatbot.postprocess = postprocess
    with gr.Blocks() as demo:
        gr.HTML(f"""<h1 align="center">{demo_args.model_name_or_path}</h1>""")
        
        with gr.Row():
            with gr.Column(scale=5):
                chatbot = gr.Chatbot()
                user_input = gr.Textbox(show_label=False, placeholder="Input...")
            with gr.Column(scale=1):         
                max_length = gr.Slider(0, 8192, value=2048, step=1.0, label="Max Length", interactive=True)
                max_new_tokens = gr.Slider(0, 1000, value=100, step=1.0, label="Max new tokens", interactive=True)
                top_p = gr.Slider(0, 1, value=1, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=1, step=0.01, label="Temperature", interactive=True)
                emptyBtn = gr.Button("Clear History")

        history = gr.State([])
        past_key_values = gr.State(None)
        user_input.submit(predict, [user_input, chatbot, max_length, max_new_tokens, top_p, temperature, history, past_key_values],
                        [chatbot, history, past_key_values], show_progress=True)
        user_input.submit(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

    demo.queue().launch()
