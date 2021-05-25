import pandas as pd
import numpy as np
from gensim import downloader
import torch
from nltk.tokenize import word_tokenize

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 500)
        self.l2 = torch.nn.Linear(500, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.sigmoid(x)
        return x


ball_train = pd.read_csv('train/train.tsv', sep='\t', error_bad_lines=False, header=None)
y_train = pd.DataFrame(ball_train[0])
x_train = pd.DataFrame(ball_train[1])
x_np=x_train.to_numpy()
x_np = [str(item) for item in x_np]
x_train=[word_tokenize(i) for i in x_np]

ball_dev = pd.read_csv('dev-0/in.tsv', sep='\t', error_bad_lines=False, header=None)
X_dev = pd.DataFrame(ball_dev)
X_dev_np=X_dev.to_numpy()
X_dev_np = [str(item) for item in X_dev_np]
X_dev=[word_tokenize(i) for i in X_dev_np]

ball_test=pd.read_csv('test-A/in.tsv', sep='\t', error_bad_lines=False, header=None)
X_test = pd.DataFrame(ball_test)
X_test_np=X_test.to_numpy()
X_test_np = [str(item) for item in X_test_np]
X_test=[word_tokenize(i) for i in X_test_np]

w2v = downloader.load('word2vec-google-news-300')

x_train = [np.mean([w2v[word] for word in content if word in w2v] or [np.zeros(300)], axis=0) for content in x_train]

X_dev = [np.mean([w2v[word] for word in content if word in w2v] or [np.zeros(300)], axis=0) for content in X_dev]

X_test = [np.mean([w2v[word] for word in content if word in w2v] or [np.zeros(300)], axis=0) for content in X_test]


lr_model = LogisticRegressionModel(300)

BATCH_SIZE = 5
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(lr_model.parameters(), lr = 0.1)
loss_score = 0
acc_score = 0
items_total = 0
lr_model.train()
for i in range(0, y_train.shape[0], BATCH_SIZE):
    X = x_train[i:i + BATCH_SIZE]
    X = torch.tensor(X)
    Y = y_train[i:i + BATCH_SIZE]
    Y = torch.tensor(Y.astype(np.float32).to_numpy()).reshape(-1, 1)
    Y_predictions = lr_model(X.float())
    acc_score += torch.sum((Y_predictions > 0.5) == Y).item()
    items_total += Y.shape[0]
    optimizer.zero_grad()
    loss = criterion(Y_predictions, Y)
    loss.backward()
    optimizer.step()
    loss_score += loss.item() * Y.shape[0]


Y_dev_predicted, Y_test_predicted = [], []
lr_model.eval()
with torch.no_grad():
    for i in range(0, len(X_dev), BATCH_SIZE):
        X = X_dev[i:i+BATCH_SIZE]
        X = torch.tensor(X)
        outputs = lr_model(X.float())
        prediction = (outputs > 0.5)
        Y_dev_predicted += prediction.tolist()
    for i in range(0, len(X_test), BATCH_SIZE):
        X = X_test[i:i+BATCH_SIZE]
        X = torch.tensor(X)
        outputs = lr_model(X.float())
        prediction = (outputs > 0.5)
        Y_test_predicted += prediction.tolist()

for i in range(0, len(Y_dev_predicted)):
    if Y_dev_predicted[i]==[True]:
        Y_dev_predicted[i]=1
    else:
        Y_dev_predicted[i]=0

for i in range(0, len(Y_test_predicted)):
    if Y_test_predicted[i]==[True]:
        Y_test_predicted[i]=1
    else:
        Y_test_predicted[i]=0


pd.DataFrame(Y_dev_predicted).to_csv('dev-0/out.tsv', sep='\t', index=False, header=False)
pd.DataFrame(Y_test_predicted).to_csv('test-A/out.tsv', sep='\t', index=False, header=False)