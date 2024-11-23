"""
Prepare Bhagavad Gita Dataset for Character-level modelling.
"""

import pickle
import numpy as np
import os

# Load the dataset
with open('Data/input.txt', 'r', encoding='utf-8') as f:
    data = f.read()

print(f"The length of the dataset is {len(data):,}")

# Character set and vocabulary size
chars = sorted(set(data))
vocab_size = len(chars)

print("".join(chars))
print(f"\nVocab Size: {vocab_size}")

# Create mappings
stoi = {s: i for i, s in enumerate(chars)}  # character to integer
itos = {i: s for s, i in stoi.items()}      # integer to character

# Encoder and decoder functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Split data into training and validation sets
n = len(data)
train = data[:int(n * 0.9)]
val = data[int(n * 0.9):]

# Encode both sets to integers
train_ids = np.array(encode(train), dtype=np.uint16)
val_ids = np.array(encode(val), dtype=np.uint16)

print(f"The length of training set is {len(train_ids)}")
print(f"The length of validation set is {len(val_ids)}")

# Ensure the directory exists
output_dir = "Data/CharLevelEncoded"
os.makedirs(output_dir, exist_ok=True)

# Save the training and validation data
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

# Save encoding/decoding information
meta = {
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos
}

with open(os.path.join(output_dir, "meta.pkl"), 'wb') as f:
    pickle.dump(meta, f)


# The length of the dataset is 297,518

#  !"'(),-.1289:;?ABCDEFGHIJKLMNOPRSTUVWYabcdefghijklmnopqrstuvwxyz| ïँंःअआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह़ऽािीुूृॄॆेैॊोौ्ॐ।৷‌’

# Vocab Size: 137
# The length of training set is 267766
# The length of validation set is 29752