# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface GPTJ model

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    print("downloading model...")
    GPTNeoForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-125M",
        low_cpu_mem_usage=True,
    )
    print("done")

    print("downloading tokenizer...")
    GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    print("done")


if __name__ == "__main__":
    download_model()
