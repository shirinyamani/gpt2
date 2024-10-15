# GPT2

Reproduction of GPT2 124M with PyTorch on Fine Web edu dataset by 🤗.  

<img src="./img/image.png" alt="img" width="600"/>

# Device 
The model is trained on a single A100 GPU with 40GB memory. Note that if the model does not fit in your GPU memory, you can reduce the batch size or sequence length. Also note that for the purpose of effitient training, try to use "NICE POWERS OF 2" numbers, becuase at core of the GPU design everything is in Tensor Cores manner, which are optimized for 2^n operations.

# Reference 
The official [GPT2](https://github.com/openai/gpt-2) code release from OpenAI, supposed to be our reference, however the code is in TensorFlow and the dataset is not available. Therefore, instead of using the official code, we use the 🤗 [implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt2) of it as our reference to reproduce the GPT2 model to have confidence that we are implementing correctly!

# Reference Papers and Dataset 
- [GPT2](https://arxiv.org/abs/2005.14165)
- [FineWeb edu🍷 dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

# Requirements
run `pip install -r requirements.txt` to install the required packages.

# Sample Output for max_length=30
```txt
> Hello, I'm a language model, and I'm a guy.
But the idea on all the lines is different that we have a background to be found


> Hello, I'm a language model, and I still can't remember what I want....
The words used to make the words "The B-man"


> Hello, I'm a language model, and I have a look, one thing, just a picture, and some, some of the other things I've done

> Hello, I'm a language model, and I'm currently planning for a few weeks now, yet, I'm getting some ideas about something, but just really
```

# Usage
```bash
python src/gpt2.py
```