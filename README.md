# SimpleChat

This is a simple script for deploying local ChatBot with any available decoder-only language models on Huggingface.

```bash
git clone https://github.com/namespace-Pt/SimpleChat.git
cd SimpleChat
pip install -r requirements

python demo.py --model_name_or_path=EleutherAI/gpt-neo-125m --no_prompt
```
- This leverages [EleutherAI/gpt-neo-125m](https://huggingface.co/EleutherAI/gpt-neo-125m) as the decoder
- With `--no_prompt` argument, all chatting-related prompts are disabled, which makes the ChatBot do text completions. The input to the model is: 
  ```
  <1-th Round Input><1-th Round Output>...<Current Input>
  ```
  Without specifying `--no_prompt`, the default input to the model is: 
  ```
  ...

  [Round K]
  
  User: <K-th Round Input>
  
  Assistant: <K-th Round Output>

  ...

  User: <Current Input>
  Assistant: 
  ```
  
- By default, the hidden states of conversation histories are cached for faster decoding, you can turn off caching by specifying `--no_cache` when running the script

## TODOs
- [ ] user-specified chat prompts
- [ ] add support for encoder-decoder models
