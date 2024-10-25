# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math

import numpy as np
import spacy

import torchtext  
import tqdm
import evaluate  #MJ: for bleu = evaluate.load("bleu")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
from os.path import exists

from torch.nn.functional import log_softmax, pad

import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt

from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
#MJ: import torchtext.datasets as datasets
import datasets  #  for Huginface

import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")


# %%
print(f'torch: version={torch.__version__}')

#$ pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 torchtext==0.17.0 --index-url https://download.pytorch.org/whl/cu121


# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
#MJ: model.cuda(0) will be equivalent to model.cuda(3) without the restricting the visible GPUs
# as long as the code does not use the other gpus

# %%

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")


# %%
print(torch.cuda.current_device())  # Should print 0, because GPU 3 is now set as the only visible GPU
print(torch.cuda.get_device_name(0))  # Check the name of the GPU being used

# %%
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# %%
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# print(f'device={device}')

# %%


# %%
dataset = datasets.load_dataset("bentrevett/multi30k") #MJ from Hugginface
#MJ:A dataset is a directory that contains:

    # - some data files in generic formats (JSON, CSV, Parquet, text, etc.).
    # - and optionally a dataset script, if it requires some code to read the data files. This is used to load any kind of formats or structures.
    
# The "30K" in Multi30k refers to the fact that the dataset originally contained around 30,000 image-caption pairs. 
# The Multi30k dataset was developed for multilingual captioning and translation tasks 
# and is built upon the Flickr30k dataset, which has 30,000 images, each with multiple captions.

# Key Points:
# Flickr30k: The Multi30k dataset is based on the Flickr30k dataset, which contains 30,000 images 
# with captions in English.
# Multi30k: The "multi" part of Multi30k refers to the addition of multilingual captions
# and translations in German, French, Czech, and other languages for these 30,000 images.

# %%
train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)

# %%
#MJ: for en_nlp.tokenzier
spacy_en = spacy.load("en_core_web_sm")  #en_core_web_sm is a small-sized pretrained language model provided by spaCy for processing English text.
spacy_de = spacy.load("de_core_news_sm")  #he model is installed as a Python package in your environment. cf. path  = spacy.util.get_package_path("de_core_news_sm")

# %%
def tokenize_example(example, spacy_en, spacy_de, max_length, lower, sos_token, eos_token): #MJ: max_length = 1_000
    all_en_tokens = [ token.text for token in spacy_en.tokenizer(example["en"]) ]
    all_de_tokens = [ token.text for token in spacy_de.tokenizer(example["de"]) ]
    en_tokens = all_en_tokens[:max_length]
    de_tokens = all_de_tokens[:max_length]
    
    # If max_length is greater than the actual length of all_en_tokens (or all_de_tokens), the slicing operation:
    #  en_tokens = all_en_tokens[:max_length]
    #  de_tokens = all_de_tokens[:max_length]
    # will not cause any error and will simply return the entire list of tokens without truncation
    #Note that train_data = 
    # # Dataset({
    #     features: ['en', 'de'],
    #     num_rows: 29000
    # })
    # en_tokens = [ token.text for token in en_nlp.tokenizer(example["en"]) ][:max_length]
    # de_tokens = [ token.text for token in de_nlp.tokenizer(example["de"]) ][:max_length]
    
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]
    #MJ: add <sos> and <eos> to to the token sequence for each sentence    
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]
    return {"en_tokens": en_tokens, "de_tokens": de_tokens}

# %% [markdown]
# 

# %%
#MJ: tokenize_example adds <sos> and <eos>
max_length = 1_000
lower = True
sos_token = "<sos>"
eos_token = "<eos>"


# Index 0: <unk>
# Index 1: <pad>
# Index 2: <sos>
# Index 3: <eos>

fn_kwargs = {
    "spacy_en": spacy_en,
    "spacy_de": spacy_de,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}

#The returned values {"en_tokens": en_tokens, "de_tokens": de_tokens} are added to each example as new columns.
# The .map() function in the Hugging Face datasets library is designed to apply a function to each example in a dataset, and the result of that function can replace, modify, or add new columns.

# If the function you provide to .map() returns new columns, those columns will be added to the dataset.
# If the function returns modified values for existing columns, those columns will be updated.
# If the function returns fewer columns, you can remove columns by specifying them in the return.
# the .map() function in the Hugging Face datasets library can automatically detect whether the function returns an existing column or introduces new columns. It does this by analyzing the keys of the dictionary returned by the function you pass to .map().

#train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs, remove_columns=["en", "de"])

#MJ: tokenize_example adds <sos> and <eos>
train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

# %%
min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

from torchtext.vocab import build_vocab_from_iterator
# en_vocab = torchtext.vocab.build_vocab_from_iterator(
#     train_data["en_tokens"],
#     min_freq=min_freq,
#     specials=special_tokens,
# )

# de_vocab = torchtext.vocab.build_vocab_from_iterator(
#     train_data["de_tokens"],
#     min_freq=min_freq,
#     specials=special_tokens,
# )

#MJ: the vocab functionality is no longer included under the main torchtext namespace;
# To use torchtext.vocab.build_vocab_from_iterator as torchtext.vocab.build_vocab_from_iterator,
# the torchtext library would need to be structured such that the vocab module is exposed as part of the top-level namespace. This means that the vocab module would need to be accessible directly from the torchtext package without the need for additional imports.

# For example, in torchtext/__init__.py, there should be a line like:
# from .vocab import build_vocab_from_iterator
# Instead, you must explicitly import it from the specific submodule torchtext.vocab.


#MJ:
# When you define a custom Dataset in PyTorch by inheriting from torch.utils.data.Dataset,
# you typically implement two essential methods:

# __len__: Returns the total number of samples in the dataset.
# __getitem__: Returns a specific sample based on an index.
# This setup makes the Dataset class indexable and iterable, allowing it to act as a data source
# that provides one data sample at a time when used in a DataLoader.

#MJ:
# dataset = load_dataset("my_dataset")  # returns a Dataset (which is iterable)

# # You can loop over the dataset, because it is iterable
# for item in dataset:
#     print(item)

# # Under the hood, Python converts the iterable (dataset) into an iterator using iter()
# iterator = iter(dataset)  # You get an iterator from the iterable

#MJ: By calling iter(train_data["en_tokens"]), you are turning the iterable into an iterator, which can be passed to build_vocab_from_iterator

train_data_en_iterator = iter(train_data["en_tokens"])
en_vocab = build_vocab_from_iterator(
    train_data_en_iterator,
    min_freq=min_freq,
    specials=special_tokens,
)

train_data_de_iterator = iter(train_data["de_tokens"])
de_vocab = build_vocab_from_iterator(
    train_data_de_iterator,
    min_freq=min_freq,
    specials=special_tokens,
)

# %%
type(en_vocab)

# %%
# en_vocab = build_vocab_from_iterator(
#     train_data["en_tokens"],
#     min_freq=min_freq,
#     specials=special_tokens,
# )

#MJ: Iterables: In Python, objects like lists, tuples, and other collections are iterable. This means they can be used in a for loop or passed to functions expecting an iterator, as Python can implicitly convert an iterable into an iterator using the iter() function behind the scenes.
# build_vocab_from_iterator: This function expects an iterator (or iterable) that yields lists or sequences of tokens. If train_data["en_tokens"] is a list of tokenized sequences (e.g., each element is a list of tokens), it is an iterable object. When you pass it to build_vocab_from_iterator, Python treats it as an iterable and internally uses it like an iterator.


# %%
type(en_vocab)

# %%
assert en_vocab[unk_token] == de_vocab[unk_token]
assert en_vocab[pad_token] == de_vocab[pad_token]

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

# %%
# for idx, token in enumerate(en_vocab.get_itos()):
#     print(f"Index {idx}: {token}")

# %%
en_vocab["two"]  # Replace "word" with the token you want to check


# %%
print(en_vocab.lookup_token(16))

# %%
en_vocab.set_default_index(unk_index) 
de_vocab.set_default_index(unk_index)
# calling en_vocab.set_default_index(unk_index) changes the behavior of the en_vocab object
# by modifying how it handles out-of-vocabulary (OOV) tokens. 
# Specifically, it ensures that when you attempt to access a token that is not in the vocabulary, 
# it will return the unk_index instead of raising an error or returning None.

# %%


# %%
input_dim = len(de_vocab)
output_dim = len(en_vocab)
print('vocab size of DE=', input_dim)
print('vocab size of EN=', output_dim)

# %%
#MJ: numericalize tokens of the datasets
def numericalize_example(example, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    de_ids = de_vocab.lookup_indices(example["de_tokens"])
    return {"en_ids": en_ids, "de_ids": de_ids}

# %%
#MJ: numericalize tokens of the datasets using the vocabulary
fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}

train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)

# %%
#MJ: make numbers of the datasets into **tensors**
data_type = "torch"
format_columns = ["en_ids", "de_ids"]
#MJ: This means that both en_ids and de_ids will be returned as PyTorch TENSORS. 
# If there are any other columns in the dataset, they will still be included in the output without conversion, 
# as the output_all_columns=True flag ensures all columns are retained.   

train_data = train_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

valid_data = valid_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

test_data = test_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

# %%
#MJ add paddings to each batch
def get_collate_fn(device, pad_index):
    
    def collate_fn(batch):
        batch_en_ids = [example["en_ids"] for example in batch]  #MJ: len(batch_en_ids)=128; len(batch_en_ids[0]) = 17
        batch_de_ids = [example["de_ids"] for example in batch]
        #MJ: batch_en_ids is a tensor
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        #MJ: https://github.com/pfnet/pfrl/issues/154: 
#         File "/home/aurelien/.local/lib/python3.8/site-packages/torch/nn/utils/rnn.py", line 363, in pad_sequence
# return torch._C._nn.pad_sequence(sequences, batch_first, padding_value) TypeError: pad_sequence(): argument 'sequences' (position 1) must be tuple of Tensors, not Tensor

        batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
        
         # Move the padded tensors to the specified device
        batch_en_ids = batch_en_ids.to(device)
        batch_de_ids = batch_de_ids.to(device)
        
        batch = {
            "en_ids": batch_en_ids,
            "de_ids": batch_de_ids,
        }
        return batch

    return collate_fn

# %%
# def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    
#     collate_fn = get_collate_fn(pad_index) #MJ: add padding indices
    
#     data_loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         collate_fn=collate_fn,  #MJ: collate_fn takes batch as the input arg
#         shuffle=shuffle,
#     )
#     return data_loader

# %%
# batch_size = 128
# #pad_index = en_vocab[pad_token] == the pad token index
# train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
# valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
# test_data_loader = get_data_loader(test_data, batch_size, pad_index)

# %%
def create_dataloader(
    device,  #MJ: device = gpu no
    dataset,    
    batch_size=12000,
    max_padding=128,  #MJ: max_padding = 128 => the maximum num of tokens in a sequence ??
    is_distributed=True,
):
        
   
    sampler = (
            DistributedSampler(dataset) if is_distributed else None
        )
    pad_token = "<pad>"
    pad_index = en_vocab[pad_token]  #MJ  the pad token index
    
    collate_fn = get_collate_fn(device, pad_index) #MJ: add padding indices to the token sequence and move it to device
     
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
    )
    
    
    return dataloader

# %%
# def collate_batch(
#     batch,
#     tokenize_de, #MJ =  tokenize_de,           
#     tokenize_en,  #MJ: =  tokenize_en,
#     src_vocab,
#     tgt_vocab,
#     device,
#     max_padding=128,
#     pad_id=2,
# ):
#     bs_id = torch.tensor([0], device=device)  # <s> token id
#     eos_id = torch.tensor([1], device=device)  # </s> token id
#     src_list, tgt_list = [], []
#     for (_src, _tgt) in batch:
#         processed_src = torch.cat(
#             [
#                 bs_id,
#                 torch.tensor(
#                     src_vocab( tokenize_de(_src)),
#                     dtype=torch.int64,
#                     device=device,
#                 ),
#                 eos_id,
#             ],
#             0,
#         )
#         processed_tgt = torch.cat(
#             [
#                 bs_id,
#                 torch.tensor(
#                     tgt_vocab( tokenize_en(_tgt)),
#                     dtype=torch.int64,
#                     device=device,
#                 ),
#                 eos_id,
#             ],
#             0,
#         )
#         src_list.append(
#             # warning - overwrites values for negative values of "padding - len", e.g.,  max_padding - len(processed_src)
#             # =>  if the calculated padding length is negative, it will overwrite existing values in the tensor.
#             # pads the input sequence with a specified value (pad_id) to ensure that each sequence has a consistent length (max_padding).
#             pad(
#                 processed_src,
#                 (
#                     0, #0 means no padding at the beginning.
#                     max_padding - len(processed_src), #max_padding - len(processed_src) means how much padding is needed at the end to reach the desired length (max_padding).
#                 ),
#                 value=pad_id,
#             )
#         )
#         tgt_list.append(
#             pad(
#                 processed_tgt,
#                 (0, max_padding - len(processed_tgt)),
#                 value=pad_id,
#             )
#         )

#     src = torch.stack(src_list)  # stacks them along a new dimension to create a single tensor. 
    
# #     # Suppose src_list contains:
# # src_list = [
# #     torch.tensor([1, 2, 3, 0, 0]),  # Shape: (5,)
# #     torch.tensor([4, 5, 6, 0, 0]),  # Shape: (5,)
# #     torch.tensor([7, 8, 9, 0, 0]),  # Shape: (5,)
# # ]

# # # Stacking these tensors:
# # src = torch.stack(src_list)

# # # Resulting tensor `src` will have shape: (3, 5)
# # # Where 3 is the number of tensors in src_list and 5 is the length of each tensor
# # print(src)
# # # Output:
# # # tensor([[1, 2, 3, 0, 0],
# # #         [4, 5, 6, 0, 0],
# # #         [7, 8, 9, 0, 0]])


#     tgt = torch.stack(tgt_list)
#     return (src, tgt)

# %%
# def tokenize(text, tokenizer):
#     return [tok.text for tok in tokenizer.tokenizer(text)]


# def create_dataloaders(
#     device,  #MJ: device = gpu no
#     vocab_src,
#     vocab_tgt,
#     spacy_de,
#     spacy_en,
#     batch_size=12000,
#     max_padding=128,  #MJ: max_padding = 128 => the maximum num of tokens in a sequence ??
#     is_distributed=True,
# ):
#     # def create_dataloaders(batch_size=12000):
#     def tokenize_de(text):
#         return tokenize(text, spacy_de)

#     def tokenize_en(text):
#         return tokenize(text, spacy_en)

#     def collate_fn(batch):
        
#         return collate_batch(
#             batch,
#             tokenize_de,
#             tokenize_en,
#             vocab_src,
#             vocab_tgt,
#             device,
#             max_padding=max_padding,
#             pad_id=vocab_src.get_stoi()["<blank>"],
#         )

    
#     # MJ: Number of lines per split:   - train: 29000 -valid: 1014 - test: 1000
    
#     train_sampler = (
#         DistributedSampler(train_data) if is_distributed else None
#     )
   
#     valid_sampler = (
#         DistributedSampler(valid_data) if is_distributed else None
#     )
    
#     test_sampler = (
#         DistributedSampler(test_data) if is_distributed else None
#     )
    

#     train_dataloader = DataLoader(
#         train_data,
#         batch_size=batch_size,
#         shuffle=(train_sampler is None),
#         sampler=train_sampler,
#         collate_fn=collate_fn,
#     )
#     valid_dataloader = DataLoader(
#         valid_data,
#         batch_size=batch_size,
#         shuffle=(valid_sampler is None),
#         sampler=valid_sampler,
#         collate_fn=collate_fn,
#     )
    
#     test_dataloader = DataLoader(
#         test_data,
#         batch_size=batch_size,
#         shuffle=(valid_sampler is None),
#         sampler=test_sampler,
#         collate_fn=collate_fn,
#     )
    
#     return train_dataloader, valid_dataloader, test_dataloader

# %%

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model, bias=False) # Query transformation
        self.W_k = nn.Linear(d_model, d_model, bias=False) # Key transformation
        self.W_v = nn.Linear(d_model, d_model, bias=False) # Value transformation
        
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
        self.attn_matrix = None
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) #mask: [80,1,1,15] => [80,1,15,15]
            #MJ: If mask[i, j] == 0, it means the position j in sequence i should be masked out (ignored).
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)  #MJ: (batch_size, num_heads, seq_length, seq_length)).
        #MJ: The purpose of using -1e9 is to effectively "mask out" those positions in the attention scores 
        #  by setting them to a very large negative number. When you apply the softmax function later on these scores, the large negative values will result in near-zero probabilities
        #MJ:  attn_probs is called the "attention matrix" of shape [B, num_head, seq_length, d_k], wehre d_model = hum_head * d_k ;
        # The attention matrix is a key component in the scaled dot-product attention mechanism,
        # which computes the relationship between different positions in the input sequence.
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output, attn_probs  #MJ: attn_prob is added by MJ for visualizing attention matrix
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q_proj = self.split_heads(self.W_q(Q))  #MJ: Q : [B, seq_length, d_model] => [B, num_head, seq_length, d_k],
        # b wehre d_model = hum_head * d_k: 512 = 8 * 64
        K_proj = self.split_heads(self.W_k(K))  #K^{T} = [B, d_model, seq_length]
        V_proj = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output, attn_probs = self.scaled_dot_product_attention(Q_proj, K_proj, V_proj, mask)
        self.attn_matrix =  attn_probs
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))  #attn_output: [B, num_head, seq_length, d_k] => [B, seq_length, num_head*d_k ]
        return output

# %% [markdown]
# Learned Positional Embeddings: The Generative Pretrained Transformer (GPT) models, starting from GPT-1 and extending through GPT-2 and GPT-3, all use learned positional embeddings.
# Description: These models learn the positional encodings as part of the training process and apply them to the input sequence to capture word-order dependencies dynamically.

# %%
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()  #M: M1 * M2
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)  #MJ: W =[1, 1, d_model, d_model] @ Seq_vec = [1, 1 d_model]
        self.relu = nn.ReLU()

    def forward(self, x): #x:[B, seq_length, d_model]
        return self.fc2(self.relu(self.fc1(x)))

# %%
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        #MJ: register_buffer is a method provided by nn.Module that is used to register a tensor as a buffer in the module.
        # Buffers are tensors that are not considered parameters (i.e., they are not learnable or updated during training via backpropagation) but are part of the model's state.
        
        #MJ: The buffer is saved when you call model.state_dict() and loaded with model.load_state_dict().
        # Buffers are automatically moved to the correct device (CPU/GPU) when you call model.to(device) or model.cuda().
        # Buffers are typically used for things like fixed positional encodings, running statistics (in batch normalization), or any other non-learnable data that is part of the model.

        self.register_buffer('pe', pe.unsqueeze(0)) #MJ:  you can access the buffer using self.pe 
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# %%
class PositionalEmbedding(nn.Module):
    
    def __init__(self, d_model, max_seq_length):
        super(PositionalEmbedding, self).__init__()
        
        self.pe = nn.Embedding(max_seq_length, d_model) 
        #MH: self.pe is not a tensor, but an object inherited from nn.Module; it is a lookup table
               
        
        
    def forward(self, x): #MJ: x: [B,seq_length, d_model]
        # Generate position indices [0, 1, 2, ..., seq_length - 1]
        seq_length = x.size(1)
        pos = torch.arange(seq_length).unsqueeze(0)  # shape (1, seq_length)
        pos_emb = self.pe(pos)
        return x + pos_emb #MJ: x: word-embedding vector: [B,seq_length, d_mdoel]

# %%
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)  #MJ: x:[B, seq_length, d_model] src_mask: [B, 1, 1,  seq_length]
        x = self.norm1(x + self.dropout(attn_output))  #MJ: apply droput to the encoder self attenttion output
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x #atten_output = Q^K^{T} * V

# %%
import torch

def simulated_dropout(input, p=0.5, inplace=False):
    if p < 0.0 or p > 1.0:
        raise ValueError("Dropout probability must be between 0 and 1, but got {}".format(p))

    # If not in training mode or p == 0, return the input unchanged
    if p == 0 or not input.requires_grad:
        return input

    # Generate a mask with the same shape as the input
    # Each element in the mask is 1 with probability (1 - p), and 0 with probability p
    mask = torch.bernoulli(torch.ones_like(input) * (1 - p))

    if inplace:
        # Modify the input tensor in-place
        input.mul_(mask)  # Zero out elements where mask is 0
        input.div_(1 - p) # Scale the remaining elements by (1 / (1 - p))
        return input
    else:
        # Return a new tensor with the mask applied
        return input * mask / (1 - p) 
    #MJ: * = an element wise multiplication
    # n the typical case where the mask consists of 0s and 1s, input * mask selectively sets elements of the input to zero.

# Example usage
input_tensor = torch.randn(5, 5, requires_grad=True)
output_tensor = simulated_dropout(input_tensor, p=0.5, inplace=False)

print("Input Tensor:\n", input_tensor)
print("Output Tensor (with Dropout Applied):\n", output_tensor)


# %%
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# %%
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,pad=2):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model) #MJ: A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
         
        #MJ: Use a fixed positional embedding
        #self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        #MJ: use a learnable positional embedding
        self.positional_encoding = PositionalEmbedding(d_model,max_seq_length )
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pad = pad
        
         # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
           if p.dim() > 1:
               nn.init.xavier_uniform_(p)

    def generate_mask(self, src, tgt_x):
        src_mask = (src != self.pad).unsqueeze(1).unsqueeze(2)  #MJ: crc: [1, L] => src_mask: [B, 1, 1, L]
        tgt_mask = (tgt_x != self.pad).unsqueeze(1).unsqueeze(3)   #MJ:tgt: [1, 1] => tgt_mask: [B, 1, 1, 1] in inference
        # (batch_size, seq_length) to (batch_size, 1, seq_length). 
        # => (batch_size, 1, seq_length) to (batch_size, 1, seq_length, 1).
        #MJ: the attention mechanism (which often works with 4D tensors in the form of
        # (batch_size, num_heads, seq_length, seq_length)).
        #That is, his step further prepares the mask to be broadcast properly over the attention scores
        # when used in multi-head attention. In particular, the resulting mask will be able to match the shape of 
        # the attention scores, which are often of shape (batch_size, num_heads, seq_length, seq_length).
        
        seq_length = tgt_x.size(1) #MJ= 8; to ensure that the model does not "peek" at future tokens when making predictions for the current token.
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask  #MJ: peek: to look quickly or secretly at something, often without permission or in a way that is not meant to be seen

    def forward(self, src, tgt_x): #MJ: src, tgt:  (batch_size, seq_length); value = index to word
        src_mask, tgt_mask = self.generate_mask(src, tgt_x)
        
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt_x)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask) #MJ: torch.Size([80, 10, 512])

        dec_output = tgt_embedded  #MJ: torch.Size([80, 8, 512])
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output) #MJ: The same as Generator in Harvard tutorial
        #MJ: This tutorial does not use log_softmax(self.fc(x), dim=-1), because it is handled by the CrossEntropy function self
        #return output  #MJ: torch.Size([80, 8, 11])
    
        return  log_softmax( output, dim=-1)  #MJ: added to use the KLDiv loss
    #MJ: Added for the inference
    # self.encoder( self.src_embed(src), src_mask)
    
    #MJ: memory = self.encode(src, src_mask): src=tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    # def decode(self, memory, src_mask, tgt, tgt_mask): #MJ: memory: [1,10,512];  src_mask:[1,1,10]; tgt.shape=[1,1]
    #     return self.decoder( self.tgt_embed(tgt), memory, src_mask, tgt_mask) #tgt_mask: [1,1,1]
    
    
    def encoder(self, src):
        src_mask = (src != self.pad).unsqueeze(1).unsqueeze(2)  #MJ: crc: [1, L] => src_mask: [B, 1, 1, L]
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
                    
        for enc_layer in self.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask) 
        return enc_output 
         
    def decoder(self, enc_output, src_mask, tgt_x):
        
        tgt_mask = (tgt_x != self.pad).unsqueeze(1).unsqueeze(3)   #MJ:tgt: [1, 1] => tgt_mask: [B, 1, 1, 1] in inference
        # (batch_size, seq_length) to (batch_size, 1, seq_length). 
        # => (batch_size, 1, seq_length) to (batch_size, 1, seq_length, 1).
        #MJ: the attention mechanism (which often works with 4D tensors in the form of
        # (batch_size, num_heads, seq_length, seq_length)).
        #That is, his step further prepares the mask to be broadcast properly over the attention scores
        # when used in multi-head attention. In particular, the resulting mask will be able to match the shape of 
        # the attention scores, which are often of shape (batch_size, num_heads, seq_length, seq_length).
        
        seq_length = tgt_x.size(1) #MJ= 8; to ensure that the model does not "peek" at future tokens when making predictions for the current token.
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt_x)))
        
        
        dec_output = tgt_embedded  #MJ: torch.Size([80, 8, 512])
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output) #MJ: The same as Generator in Harvard tutorial
        #MJ: This tutorial does not use log_softmax(self.fc(x), dim=-1), because it is handled by the CrossEntropy function self
        return output  #MJ: torch.Size([80, 8, 11])
              

# %%
class Batch:
    """Object for holding a batch of data **with source and target masks**"""

# Index 0: <unk>
# Index 1: <pad>
# Index 2: <sos>
# Index 3: <eos> => src, tgt already contains <sos> and <eos>
    def __init__(self, src, tgt=None, pad=1): 
        
        self.src = src
        
        self.src_mask = (src != pad).unsqueeze(-2)
        #MJ: If src had shape (batch_size, seq_length), after unsqueeze(-2), the shape becomes (batch_size, 1, seq_length).
        # this src_mask is used to prevent the model from attending to padding tokens in the source sequence. 
        # self.src_mask = (src != pad).unsqueeze(-2) creates a mask that identifies non-padding tokens in the source sequence.
        if tgt is not None:
            self.tgt = tgt[:, :-1]  #MJ: tgt.shape=(batch_size,seq_length);  tgt[:, :-1] : (batch_size, seq_length - 1) = the right shifted output.
                                    #MJ: the decoder input seq= [<sos<,1,2,3...n], excluding the last token; tgt[:,0]= <sos>, tgt[]0,-1]=<eos>
            self.tgt_y = tgt[:, 1:]  #MJ:the decoder target seq = tgt_y = tgt[:, 1:] =<1,2,3...,<eos>]:  excluding the first token
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
            
# Example Recap:
# Target sequence: [A, B, C, D]
# Step 1:
# Input: The decoder receives the <SOS> token.
# Prediction: The model predicts A (or some other token).
# Comparison: The model's prediction is compared with the actual token at time step 1, which is A in the target sequence.
# So, at step 1, the model's prediction is compared against the actual token A.


    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        ) #MJ: tgt = self.tgt = tgt[:, :-1];  tgt.size(-1) = seq_length - 1
        return tgt_mask

# %%
class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

# %%
def rate(step, model_size, factor, warmup):
    #MJ: 
    # opts = [ model_size, factor, warmup]=[
    #     [512, 1, 4000],  # example 1
    #     [512, 1, 8000],  # example 2
    #     [256, 1, 4000],  # example 3
    # ]
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power. warmup =4000 ?
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

# %%
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    # The model less confident in its predictions by assigning a small portion of the probability mass to other classes,
    # instead of placing the entire probability mass on the ground truth class. 
    # This prevents the model from being too confident about a single class and has been found to improve generalization.
    

    def __init__(self, size, padding_idx, smoothing=0.0): #MJ: criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        #MJ: nn.KLDivLoss in PyTorch requires the input to be in the form of log-probabilities and the target to be in the form of probabilities.
        
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size  # size=Vocab-size
        #MJ: Dimension 0 corresponds to each example in the batch.
        #    Dimension 1 corresponds to the possible classes for each example.
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        #MJ: The denominator (self.size - 2) indicates the number of non-true and non-padding classes.
        # The true class will receive a different probability (higher confidence value).
        # The padding class should receive a probability of zero.
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  #MJ: target = [2, 0, 3, 1]  # For batch size B = 4
        true_dist[:, self.padding_idx] = 0
        
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            #0 = dim => Here, dim=0 refers to the batch dimension, so the line sets all values in the specified rows (those where the target was equal to the padding index) to 0.0.
            
            #MJ: mask.squeeze() removes any singleton dimensions from mask, converting it from a shape like [N, 1] to [N] (where N is the number of found indices).
            #mask.squeeze(): This removes all dimensions of size 1 from the tensor mask. It removes singleton dimensions along all axes.
            
    #     target = [3, 0, 2, 0]
    #     self.padding_idx = 0
    # ==>
    #     target.data == self.padding_idx -> [False, True, False, True]
    #     mask -> [[1], [3]]

        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

# %%
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
         
        self.generator = generator  #MJ: generator is the transformer model
        self.criterion = criterion

    def __call__(self, x, y, norm):
        out = self.generator(x)
       
        sloss = (
            self.criterion(
                out.contiguous().view(-1, out.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss

# %%


# %%
#  run_epoch(
#             epoch,
#             data_iter_gen(V, batch_size, 20),  #MJ: Generate nbatches=20 batches for each epoch
#             model,..
def run_epoch(  #MJ: 
    epoch, is_main_process,
    batch_generator,
    model, #module
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    
    # accum_iter: This is the number of iterations (batches) over which gradients are accumulated 
    # before an optimizer step is taken. This is typically used when you want to simulate a larger batch size 
    # than what can fit into memory by accumulating gradients over multiple smaller batches.
    
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens_since_logging = 0
    n_accum = 0
    
    #MJ: Train the net until the end of batch iterator, data_iter:
    #MJ: for batch in gen:
    #    print(batch)
    #  In place of gen, we can put an iterable, iterator, and generator
    
    for i, batch in enumerate(batch_generator):
        
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        ) #MJ: batch.tgt = tgt[:,:-1] = the right shifted output = the decoder input
        # batch.tgt = the decoder input
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens) #MJ: batch.tgt_y = tgt[:,1:] = the expected decoder output = the decodeer answers
        # batch.tgt_y = the decoder target
        
        # loss_node = loss_node / accum_iter
        #MJ: Train the network
        if mode == "train" or mode == "train+log":
            
            loss_node.backward()  #MJ: The gradient of the loss with respect to the params are 
                                  # accumulated automatically
            
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            
            #MJ: When i % accum_iter == 0, the accumulated gradients are used to perform an update (via optimizer.step()),
            # and then the gradients are cleared (via optimizer.zero_grad(set_to_none=True)).
            #MJ: computing the gradient of the loss with respect to a large batch and accumulating 
            # the gradient with respect to their subbatches are roughly equivalent?
            #=> Thus, accumulating gradients over smaller sub-batches is equivalent to computing the gradient of the loss
            # with respect to a large batch, 
            # provided that the gradients are accumulated correctly and applied in the same manner.
            
            if i % accum_iter == 0:
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
                
            scheduler.step()
            
        #if mode == "train" or mode == "train+log"
        else: #When mode =="eval"
            pass
                
        #MJ: mode =="eval" or "train"
        total_loss += loss
        total_tokens += batch.ntokens
        tokens_since_logging += batch.ntokens
        
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
        #if i % 2 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            if is_main_process:
                print(
                    (
                        "epoch: %d, mode: %s, Step: %6d | Accum Step: %3d | Loss: %6.2f " + "| Tokens/Sec: %7.1f | L-Rate: %6.1e"
                    )
                    % (epoch, mode, i, n_accum, loss / batch.ntokens, tokens_since_logging / elapsed, lr)
                )
            start = time.time()
            tokens_since_logging = 0
            
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

# %%
#mj: args=(ngpus, vocab_src, vocab_tgt, config, True),

def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,    
    config,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    print(f"Train worker process using mapped GPU: {gpu} (actual GPU: {torch.cuda.current_device()})", flush=True)
    torch.cuda.set_device(gpu) #MJ: torch.cuda.set_device(gpu) is used to set the default GPU device for PyTorch operations. 
    
    #MJ: After setting CUDA_VISIBLE_DEVICES=1,2,3, only GPUs 1, 2, and 3 are visible to the script.
    # PyTorch treats these visible devices as device 0, 1, and 2.
    # So when you print the gpu value, it will display 0, 1, and 2, but they actually correspond to GPUs 1, 2, and 3 on your machine.


    pad_idx = vocab_tgt["<blank>"]
    d_model=512
        
    #def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,pad=0):
    model = Transformer(len(vocab_src), len(vocab_tgt), d_model, num_heads=8, num_layers=6, 
                        d_ff=2048,  max_seq_length=72,dropout=0.1, pad=pad_idx )
        
     
    model.cuda(gpu)  
    #MJ: (1)device = torch.device(f'cuda:{gpu}') ; model.to(device)
    # (2) torch.cuda.set_device(gpu)  # Set the current device
    #     model.cuda()  # Move the model to the current default GPU
    # (3) for buffer in model.buffers():
    #   buffer.data = buffer.data.cuda(gpu)

    module = model
    is_main_process = True
    
    if is_distributed:
        
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        
        is_main_process = gpu == 0
    #MJ: backend="nccl": This is a highly optimized backend specifically designed for multi-GPU communication on NVIDIA GPUs. It handles tasks like gradient synchronization efficiently.
    # init_method="env://":: "env://" means that the environment variables (like MASTER_ADDR, MASTER_PORT, etc.) will be used to set up communication. These variables are usually set in distributed environments like SLURM or manually in scripts.
    # rank=gpu: The rank is a unique identifier for each process in the distributed system. For example, if you have 4 GPUs (4 processes), the rank will be 0, 1, 2, and 3 for each process.

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    # #MJ: creeate the dataloaders for each gpu
    # train_dataloader, valid_dataloader,_ = create_dataloaders(
    #     gpu,
    #     vocab_src,
    #     vocab_tgt,
    #     spacy_de,
    #     spacy_en,
    #     batch_size=config["batch_size"] // ngpus_per_node,
    #     max_padding=config["max_padding"],
    #     is_distributed=is_distributed,
    # )

# def create_dataloader(
#     device,  #MJ: device = gpu no
#     dataset,    
#     batch_size=12000,
#     max_padding=128,  #MJ: max_padding = 128 => the maximum num of tokens in a sequence ??
#     is_distributed=True,
# ):
    
    train_data_loader = create_dataloader( gpu, train_data,
          batch_size= config["batch_size"] // ngpus_per_node,
          max_padding=config["max_padding"],
          is_distributed=True)
    
    valid_data_loader = create_dataloader( gpu, valid_data,
        batch_size= config["batch_size"] // ngpus_per_node,
          max_padding=config["max_padding"],
          is_distributed=True)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    #MJ: The training loop
    for epoch in range(config["num_epochs"]):
        if is_distributed: 
            #MJ: 1) For the model to train properly, all devices need to shuffle their subsets of data in the same way, but within the subset assigned to each device. This is where set_epoch(epoch) comes in. 
            #2) Different shuffling in each epoch: The set_epoch(epoch) function makes sure that the data is shuffled differently in each epoch by modifying the random seed with the epoch number.
            
            train_data_loader.sampler.set_epoch(epoch)
            valid_data_loader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        
        _, train_state = run_epoch( epoch, is_main_process,
            
            #MJ: add pad_idx to each batch from train_data_loader (iterator)
           

# gen = (x * x for x in range(5))
# print(type(gen))  # Output: <class 'generator'>
# ==> () can create either a tuple or a generator depending on the context.
# Comma-separated values in () create a tuple.
# Expressions in () like (x for x in iterable) create a generator expression.

#MJ: produce a generator that yields Batch objects, one by one, as it iterates over train_data_loader.
# An iterator is any object that implements the iterator protocol (__iter__ and __next__), and it may or may not be lazily evaluated.
# A generator is a simpler and more powerful tool for creating iterators in Python, using a function and yield. It is inherently lazy, meaning it produces values only when needed, saving memory.
         
            (Batch(b[0], b[1], pad_idx) for b in train_data_loader),
            
            model, #MJ: ddp_wraped model if distributed training is on
            SimpleLossCompute(module, criterion), #MJ: module = transformer model itself
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        
        sloss = run_epoch(epoch, is_main_process,
                          
            (Batch(b[0], b[1], pad_idx) for b in valid_data_loader),
            model,
            SimpleLossCompute(module, criterion), #MJ: module refers to the transformer model
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()
    #for epoch in range(config["num_epochs"])
    
    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        
        torch.save(module.state_dict(), file_path)

# %%

def train_distributed_model(vocab_src, vocab_tgt, config):
    #from real_transformer import train_worker 
    #MJ: We need to import train_worker from a separate py file in order to use multiprocessing in Notebook
    #MJ: https://bobswinkels.com/posts/multiprocessing-python-windows-jupyter/
    
#     The first solution is to define the worker function in a separate python file 
#     and then import the worker function as a separate module. This will work because the worker function is not part of the main module and therefore can be pickled and sent to the new process. Let’s see how this works:

# # worker.py
# def square(x):
#     return x**2
# Copy
# # Cell in Jupyter Notebook
# from multiprocessing import Pool
# from worker import square

# numbers = [1, 2, 3, 4, 5]

# with Pool() as pool:
#     squares = pool.map(square, numbers)

# print(squares)

    
    # Count GPUs based on what's visible after setting CUDA_VISIBLE_DEVICES
    ngpus = torch.cuda.device_count()
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    
    mp.spawn(
        train_worker,
        nprocs=ngpus, # Now it will spawn processes for the 3 GPUs (1, 2, 3)
        args=(ngpus, vocab_src, vocab_tgt, config, True), #True = is_distributed
    ) #MJ:  assigns each gpu to train_worker


def train_model(vocab_src, vocab_tgt, config):
    
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt,  config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt,  config, False #MJ: #False = is_distributed
        )  #MJ: 0,1: 0 refers to the gpu; 1 =   ngpus_per_node,
        #MJ: args=(ngpus, vocab_src, vocab_tgt,  config, True) 
        #        == 0, 1, vocab_src, vocab_tgt, config, False


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        #"distributed": True,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72, #MJ: max-seq-length
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    #model_path = "multi30k_model_final.pt"
    
    # if not exists(model_path):
        
    #    train_model(de_vocab, en_vocab, config)

    pad_idx = en_vocab["<blank>"]
    model = Transformer(len(de_vocab), len(en_vocab), d_model=512, num_heads=8, num_layers=6, d_ff=2048, 
                        max_seq_length=72, dropout=0.1, pad=pad_idx)
   
    model.load_state_dict(
        torch.load( "multi30k_model_final.pt", map_location=torch.device("cpu") )
    )

    # model.load_state_dict( torch.load("multi30k_model_final.pt") )
    return model


#Use only GPUs 1, 2, and 3
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    

# model = load_trained_model()

# %%
#MJ: https://bobswinkels.com/posts/multiprocessing-python-windows-jupyter/

#MJ:  train the model:
config = {
        "batch_size": 32,
        #"distributed": False,
        "distributed": True,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }


#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# %%
# Run the trained model using sentence-pairs from test_dataloader which contains batches with batch-size =1
def check_outputs(
    test_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="<eos>",
):
    results = [()] * n_examples
    
    for idx in range(n_examples):
        
        print("\nExample %d ========\n" % idx)
        
        b = next(iter(test_dataloader)) #MJ:  What iter() Does: When you pass an iterable like valid_dataloader to iter(), 
                                         #it calls valid_dataloader.__iter__(), which returns an iterator over the dataset.
                                         
# for batch in valid_dataloader:
#     # process the batch
# Python automatically does something like this under the hood:

# valid_iterator = iter(valid_dataloader)
# while True:
#     try:
#         batch = next(valid_iterator)
#         # process the batch
#     except StopIteration:
#         break
    
        
        rb = Batch(b[0], b[1], pad_idx)
        
        #MJ: greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0] #MJ:  ?? greedy_decode(model, src, src_mask, max_len, start_symbol): returns    return ys

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ] #MJ: get the tokens from only the first element in the batch

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        
        eos_token = vocab_tgt.get_stoi()[eos_string] #MJ: == 1
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0, end_symbol=eos_token)[0]  #MJ: max_len = 72, 0 = <sos>; rb.src:[2,128]; rb.src_maskL[2,1,128]
        
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        ) 
        #MJ: x.split(eos_string, 1) splits x at the first occurrence of the eos_string, which is "</s>"
        #  This prevents any tokens appearing after the first "</s>" from being included in the output.
        
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(n_examples=5):
    
    #global vocab_src, vocab_tgt, spacy_de, spacy_en
    global de_vocab, en_vocab 
    print("Preparing Data ...") #MJ: <== 
                                    # train_iter, valid_iter, test_iter = datasets.Multi30k(
                                    #     language_pair=("de", "en")
                                    # )
                                    
#  def create_dataloader(
#     device,  #MJ: device = gpu no
#     dataset,    
#     batch_size=12000,
#     max_padding=128,  #MJ: max_padding = 128 => the maximum num of tokens in a sequence ??
#     is_distributed=True,
# ):
                                        
    test_dataloader = create_dataloader(
        torch.device("cpu"),
        test_data,
        batch_size=1, #MJ: the batch size for src and tgt should be 1 in this experiment
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    pad_idx = en_vocab["<blank>"]
    
    model = Transformer(len(de_vocab), len(en_vocab), d_model=512, num_heads=8, num_layers=6, d_ff=2048, 
                        max_seq_length=72, dropout=0.1, pad=pad_idx)
   
    model.load_state_dict(
        torch.load( "multi30k_model_final.pt", map_location=torch.device("cpu") )
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        test_dataloader, model, de_vocab, en_vocab, n_examples=n_examples
    ) #MJ: example_data = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    # check_outputs(    valid_dataloader,    model,    vocab_src,    vocab_tgt,    n_examples=15,    pad_idx=2,    eos_string="</s>",):
    # check_outputs calls  model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]  #MJ: max_len = 72, 0 = <sos>
    return model, example_data


# run_model_example(n_examples=1) #n_examples = 1



# %%


# %%


# %%
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)  #MJ: size = seq_len  = 20: 
    #MJ: the upper triangular part of the tensor X; The diagonal=1 specifies that the diagonal starts at the first superdiagonal (i.e., one position above the main diagonal).
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0  #MJ: broadcasting is applied ==> Upper triagnular boolean matrix, upper triangular = False

# %% [markdown]
#  def encoder(self, src):
#         src_mask = (src != self.pad).unsqueeze(1).unsqueeze(2)  #MJ: crc: [1, L] => src_mask: [B, 1, 1, L]
#         src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
#         enc_output = src_embedded
#                     
#         for enc_layer in self.encoder_layers:
#                 enc_output = enc_layer(enc_output, src_mask) 
#         return enc_output 
#          
#     def decoder(self, enc_output, src_mask, tgt):

# %%


# %%
def greedy_decode(transformer_model, src, max_len, start_token, end_token):
#greedy_decode(transformer, src_data,  max_gen_seq_length, sos, eos)    
    enc_output = transformer_model.encoder(src)    
    ys = torch.zeros(1, 1).fill_(start_token).type_as(src)  #MJ: Do not use src.data but use .detach() and/or with torch.no_grad() 
    #  torch.zeros(1, 1).fill_(start_token) = tensor([[0.]])
    src_mask = (src != transformer_model.pad).unsqueeze(1).unsqueeze(2)  #MJ: crc: [1, L] => src_mask: [B, 1, 1, L]
    
    for i in range(max_len - 1):
        out = transformer_model.decoder(enc_output, src_mask, ys)  #MJ: src: [1,15], ys: [1,1]; out_prob: [B, 1,11] =[B,location, seq_length]
        #print(f"out={out}") #out_prob=torch.Size([1, 1, 11]) => out_prob=torch.Size([1, 2, 11])
        #out_prob=tensor([[[-5.4479, -1.6750,  0.6865,  1.1233,  0.3584,  0.3773,  0.7420,
        #   1.0585, -0.2761,  1.0520,  1.0914]]], grad_fn=<ViewBackward0>)
        #out_prob=tensor([[[-5.4479, -1.6750,  0.6865,  1.1233,  0.3584,  0.3773,  0.7420,
        #    1.0585, -0.2761,  1.0520,  1.0914],
        #  [-5.4578, -1.2135,  0.7173,  0.9492,  0.3095,  0.3843,  0.7443,
        #    1.0394, -0.3098,  0.9794,  1.0174]]], grad_fn=<ViewBackward0>)
        last_logit = out[:, -1]
        #MJ: out[:, -1] selects the last time step along the sequence length dimension L
        # meaning you are extracting the features (of size 𝐷) at the last time step for each batch.
        # This operation slices the second dimension (sequence length), 
        #  reducing the tensor shape from (B, L, D) to (B, D).
        # (B, L, D) = (32, 100, 512), where: 
        # 32 is the batch size,
        # 100 is the sequence length (L),
        # 512 is the model dimension (D).
        # Then, out[:, -1] will give you a tensor with shape:

        # (B, D) = (32, 512), meaning you have selected the last token's representation (along the sequence dimension) for each batch.

        #print(f"last logit={last_logit}")
        _, next_word = torch.max(last_logit, dim=1)  
        
        #print(f"next_word={next_word}; shape={next_word.shape}")
                    
    
        # calling y = x.data will be a Tensor that shares the same data with x, is unrelated with the computation history of x, and has requires_grad=False.
        if next_word == end_token: #MJ next_word =tensor([7]) >
        #if (next_word == end_token).all():  # All values must be the end token
        #    print(f"reached the <eos> token") 
        #    print(f'ys={ys}')
           return ys 
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src).fill_(next_word.item())], dim=1
        ) 
    # print(f"reached the max_seq_length: {max_len}")   
    # print(f'ys={ys}')  
    return ys

# %%

 


# %%
# %% tags=[]
def get_encoder(model, layer):
    return model.encoder_layers[layer].self_attn.attn_matrix

def get_decoder_self(model, layer):
    return model.decoder_layers[layer].self_attn.attn_matrix


def get_decoder_src(model, layer):
    return model.decoder_layers[layer].cross_attn.attn_matrix



# %%
import altair as alt
import pandas as pd

# %%


# %%
def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    "convert a dense matrix to a data frame with row and column indices"
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s"
                % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


# %%
def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(  #MJ: matrix to DataFrame
        attn[0, head].data, #MJ: attn[0, head] refers to the attention map 
                             #for a particular attention head in the first batch.
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        #The function creates an Altair chart, specifically a heatmap, where each cell represents the attention value between two tokens
        # (one from the row and one from the column). 
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        #.properties(height=400, width=400)
        .properties(height=200, width=200)
        .interactive()
    )


# %%
def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    # ntokens = last_example[0].ntokens
    attn = getter_fn(model, layer)
    #MJ=> Get Attention: attn = getter_fn(model, layer) extracts the attention matrix for the given layer using get_encoder()
    n_heads = attn.shape[1]
    
    #MJ:
    # Loop through Attention Heads: The attention maps for each head in the layer are 
    # visualized. The loop generates attention maps for all attention heads (n_heads):
        
    charts = [
        attn_map(
            attn,
            0,  # ==layer, not used in attn_map; but attn is from particular layer as
                #   attn = getter_fn(model, layer)
            h,  # For each attention head
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    #assert n_heads == 8
    assert n_heads == 4
    return alt.vconcat(
        charts[0]
        | charts[1]
        | charts[2]
        | charts[3]
        # | charts[4]
        # # | charts[5]
        # | charts[6]
        # # | charts[7]
        # layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))



# %%
#MJ: Inference for Transformer (multi30K dataset)

# %%
# %% tags=[]
def viz_encoder_self():
    


    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(src_content), 
            src_content, src_content
        )
    #MJ: =>
    #      return alt.vconcat(
    #     charts[0]
    #     # | charts[1]
    #     | charts[2]
    #     # | charts[3]
    #     | charts[4]
    #     # | charts[5]
    #     | charts[6]
    #     # | charts[7]
    #     # layer + 1 due to 0-indexing
    # )
         
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )



# %%


# %%
# %% tags=[]
def viz_decoder_self():
        
    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self, #self-attention
            len(decoded_seq),
            decoded_seq,  #MJ: tokens
            decoded_seq
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )



# %%
# %% tags=[]
def viz_decoder_src():
     
    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src, #cross-attention
            max(len(src_content), len(decoded_seq)),
            src_content, #MJ: src_tokens,
            decoded_seq,  #MJ:  tgt_tokens
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )

if __name__ == '__main__':
    train_model(de_vocab, en_vocab, config)
   
   
# %%
# Run the model using batches with batch-size =1
    run_model_example(n_examples=1) #n_examples = 1  
    
    pad_idx = en_vocab["<blank>"]
    model = Transformer(len(de_vocab), len(en_vocab), d_model=512, num_heads=8, num_layers=6, d_ff=2048, 
                            max_seq_length=72, dropout=0.1, pad=pad_idx)
    model.load_state_dict(torch.load("multi30k-transformer.pt"))
    pad = 0
    sos = 1
    eos = 2 

                                        
    test_dataloader = create_dataloader(
            torch.device("cpu"),
            test_data,
            batch_size=1, #MJ: the batch size for src and tgt should be 1 in this experiment
            is_distributed=False,
        )
    
    # pad_idx = en_vocab["<blank>"]
    # model = Transformer(len(de_vocab), len(en_vocab), d_model=512, num_heads=8, num_layers=6, d_ff=2048, 
    #                         max_seq_length=72, dropout=0.1, pad=pad_idx)
    # model.load_state_dict(torch.load("multi30k-transformer.pt"))

    model.eval()  #eval mode: droput layer를 실행안해요.
    total_loss =0
    max_seq_length = 72
    
    with torch.no_grad():
    
        for i, batch in enumerate(test_dataloader): #MJ: use an iterator of batches 
          for j in range( len(batch.src) ): #MJ: = 80
                
            src_data = batch.src[j][None]  #MJ: [80,15] <==> [1,15], 15 = max_seq_length
            
            tgt_data_y = batch.tgt_y[j][None]
        
            # src_data = batch.src
            # tgt_data_y = batch.tgt_y
            decoded_seq  = greedy_decode(model, src_data,  max_seq_length, sos, eos)
            
            
            
            src_content =  src_data[0][: len(decoded_seq[0]) ]
            decoded_seq  = decoded_seq[0]
            
            print(f"i,j={i,j}: source  seq={src_content}")
            
            #print(f"target_y  seq={tgt_data_y[:,:]}")
            print(f"i,j={i,j}: decoded seq={ decoded_seq}")
            
            
            diff = (decoded_seq == src_content)
            loss = (diff == False).float().mean()     
        
            if loss > 0:
                print(f'***************************loss nonzero: i,j={i,j}:  loss={loss}') 
                total_loss += loss
            #for j in range( len(batch.src) )
        #for i, batch in enumerate(test_data_iter)      
    #with torch.no_grad()
    avg_loss = total_loss / (  len(test_dataloader) * len(batch.src) ) #MJ: / 10*80
    print(f'tut-transformer:total_loss={total_loss},len(test_data_iter) * len(batch.src)={len(test_dataloader) * len(batch.src)}, avg_loss={avg_loss}')  
       

    # %%
    viz_encoder_self()


    # %%
    viz_decoder_self()



    # %%
    viz_decoder_src()  #MJ: decoder cross attention
#if __name__ == '__main__':

