# GPT2

Reproduction of GPT2 with PyTorch on Fine Web edu dataset by 🤗. 

![img](./img/image.png)

# Reference 
The official GPT2 code release from OpenAI, supposed to be our reference, however the code is in TensorFlow and the dataset is not available. Therefore, instead of using the official code, we use the 🤗 Transformers library to reproduce the GPT2 model to have confidence that we are implementing correctly!

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