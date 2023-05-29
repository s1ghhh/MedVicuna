# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
# from fastchat.train.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )


# import sys
# import os
# sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__).replace('/fastchat', ''))))
# print('------------------')
# print (sys.path)


# replace_llama_attn_with_flash_attn()

from fastchat.train.train import train

if __name__ == "__main__":
    train()
