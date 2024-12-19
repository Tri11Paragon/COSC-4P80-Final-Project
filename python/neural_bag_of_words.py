import collections

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import tqdm

seed = 1234
validate_size = 0.25

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

print(nltk.pos_tag(nltk.word_tokenize("Hello there you stupid fucking whore mr parker")))

def tokenize(input):
    return {"tokens": nltk.word_tokenize(input["text"])}

train_data = train_data.map(tokenize)
test_data = test_data.map(tokenize)

train_valid_data = train_data.train_test_split(test_size=validate_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]

print(train_data)
print(test_data)
print(valid_data)