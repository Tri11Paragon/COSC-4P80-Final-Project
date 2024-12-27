import collections

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.probability import FreqDist
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import tqdm

seed = 1234
validate_size = 0.25
min_freq = 5
batch_size = 512
n_epochs = 10
PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<UNK>"

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

print(nltk.pos_tag(nltk.word_tokenize("Hello there you stupid fucking whore mr parker")))

def tokenize(input):
    return {"tokens": nltk.word_tokenize(input["text"])}

def vocabulary(dataset):
    # tokens is a list of lists containing the tokens, so loop over the main list then loop over the tokens
    all_tokens = [token for list_of_tokens in dataset["tokens"] for token in list_of_tokens]
    freq_dist = FreqDist(all_tokens)
    vocab = [word for word,freq in freq_dist.items() if freq >= min_freq]
    vocab.append(PAD_TOKEN)
    vocab.append(UNKNOWN_TOKEN)
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    return word_to_index, index_to_word

def tokens_to_indices(example, word_to_index):
    unk_index = word_to_index[UNKNOWN_TOKEN]
    ids = [word_to_index.get(token, unk_index) for token in example["tokens"]]
    return {"ids": ids}

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

train_data = train_data.map(tokenize)
test_data = test_data.map(tokenize)

train_valid_data = train_data.train_test_split(test_size=validate_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]

print(train_data)
print(test_data)
print(valid_data)

word_to_index, index_to_word = vocabulary(train_data)

train_data = train_data.map(tokens_to_indices, fn_kwargs={"word_to_index": word_to_index})
test_data = test_data.map(tokens_to_indices, fn_kwargs={"word_to_index": word_to_index})
valid_data = valid_data.map(tokens_to_indices, fn_kwargs={"word_to_index": word_to_index})

print(train_data)
print(test_data)
print(valid_data)

train_data = train_data.with_format(type="torch", columns=["ids", "label"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])

pad_index = word_to_index[PAD_TOKEN]

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

class NBoW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, ids):
        # ids = [batch size, seq len]
        embedded = self.embedding(ids)
        # embedded = [batch size, seq len, embedding dim]
        pooled = embedded.mean(dim=1)
        # pooled = [batch size, embedding dim]
        prediction = self.fc(pooled)
        # prediction = [batch size, output dim]
        return prediction

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

vocab_size = len(word_to_index)
embedding_dim = 300
output_dim = len(train_data.unique("label"))

model = NBoW(vocab_size, embedding_dim, output_dim, pad_index)

print(f"The model has {count_parameters(model):,} trainable parameters")

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = model.to(device)
criterion = criterion.to(device)

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


best_valid_loss = float("inf")

metrics = collections.defaultdict(list)

model.load_state_dict(torch.load("nbow.pt"))

for epoch in range(n_epochs):
    train_loss, train_acc = train(
        train_data_loader, model, criterion, optimizer, device
    )
    valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "nbow.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_losses"], label="train loss")
ax.plot(metrics["valid_losses"], label="valid loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()
fig.show()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_accs"], label="train accuracy")
ax.plot(metrics["valid_accs"], label="valid accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()
fig.show()

test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)
print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")

def predict_sentiment(text, model, device):
    unk_index = word_to_index[UNKNOWN_TOKEN]
    tokens = nltk.word_tokenize(text)
    ids = [word_to_index.get(token, unk_index) for token in tokens]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability

text = "This film is terrible!"

print(predict_sentiment(text, model, device))

text = "This film is great!"

print(predict_sentiment(text, model, device))

text = "This film is not terrible, it's great!"

print(predict_sentiment(text, model, device))

text = "This film is not great, it's terrible!"

print(predict_sentiment(text, model, device))