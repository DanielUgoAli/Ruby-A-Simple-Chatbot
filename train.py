import json
import numpy as np
from nltk_utils import tokenizer, stemmer, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import model
# from tqdm.notebook import tqdm

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = [] # Collect all of the words and patterns
tags = [] # Collect all of the different tags
xy = [] # For collecting both tags and patterns

# TOKENIZATION
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenizer(pattern)
        all_words.extend(w) # w is an array so we dont need to nested arrays by adding array in arrays
        xy.append((w, tag))

# EXCLUDING SPECIAL CHARACTERS/ STEMMING
words_to_ignore = ['?', '!', '.', ',', '&', '*', '`', '~', '/', '^', '$']
all_words = [stemmer(w) for w in all_words if w not in words_to_ignore]
all_words = sorted(set(all_words))
# tags = sorted(set(tags))

X_train = []
y_train = []
for (sentence_patterns, tag) in xy:
    bag = bag_of_words(sentence_patterns, all_words)
    X_train.append(bag)

    labels = tags.index(tag)
    y_train.append(labels) # IT WAS SUPPOSED TO BE ONE HOT ENCODING BUT CROSS ENTROPY DOES NOT FIT IN WOTH ONE HOT ENCODED DATA

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = torch.from_numpy(X_train).type(torch.float32)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)

# Creating the chat dataset
class Dataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# HyperParameters
BATCH_SIZE = 10
hidden_units = 8
input_features = len(X_train[0])
output_features = len(tags)
lr = 0.001 # Learnimg Rate
epochs = 3000

dataset = Dataset()
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model(input_features=input_features, 
             output_features=output_features, 
             hidden_units=hidden_units).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}/{epochs}\n------")
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        model.train()
        preds = model(words)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"loss: {loss.item():.4f}")
        
print(f"Final Loss: {loss.item():.4f}")

ruby_data = {"model_state": model.state_dict(),
             "input_features": input_features,
             "output_features": output_features,
             "hidden_units": hidden_units,
             "all_words": all_words, 
             "tags": tags}

FILE_SAVE_PATH = "data.pth"
torch.save(ruby_data, FILE_SAVE_PATH)

print(f"Completed Training || Saving File: {FILE_SAVE_PATH}")
