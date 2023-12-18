import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    tag = intent['tag']
    classes.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        documents.append((w, tag))

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
classes = sorted(set(classes))

X_train = []
y_train = []

for (pattern_sentence, tag) in documents:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = classes.index(tag)
    y_train.append(float(label))

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


input_size = len(X_train[0])
hidden_size = 8
output_size = len(classes)
batch_size = 8
learning_rate = 0.001
num_epochs = 1000
print(input_size, output_size)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# print(list(train_loader))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:

            words = words.to(device)
            labels = labels.to(device)
            outputs = model(words)

            loss = criterion(outputs, labels.to(torch.long))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
    print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": classes

}
FILE = "data.pth"
torch.save(data, FILE)
print(f"training complete. file saved to {FILE}")