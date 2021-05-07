from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
from stop_words import get_stop_words

stop_words = get_stop_words('polish')

v = TfidfVectorizer(stop_words=None)
naive_bayes=MultinomialNB()

ball_train = pd.read_csv('train/train.tsv', sep='\t', error_bad_lines=False, header=None)

y_train = pd.DataFrame(ball_train[0])
x_train = pd.DataFrame(ball_train[1])
x_np=x_train.to_numpy()
x_np = [str(item) for item in x_np]

x_train=v.fit_transform(x_np)

naive_bayes.fit(x_train, y_train)

ball_dev = pd.read_csv('dev-0/in.tsv', sep='\t', error_bad_lines=False, header=None)

X_dev = pd.DataFrame(ball_dev)
X_dev_np=X_dev.to_numpy()
X_dev_np = [str(item) for item in X_dev_np]
X_dev=v.transform(X_dev_np)

Y_dev_predicted = naive_bayes.predict(X_dev)
pd.DataFrame(Y_dev_predicted).to_csv('dev-0/out.tsv', sep='\t', index=False, header=False)


ball_test=pd.read_csv('test-A/in.tsv', sep='\t', error_bad_lines=False, header=None)
X_test = pd.DataFrame(ball_test)
X_test_np=X_test.to_numpy()
X_test_np = [str(item) for item in X_test_np]
X_test=v.transform(X_test_np)

Y_test_predicted = naive_bayes.predict(X_test)
pd.DataFrame(Y_test_predicted).to_csv('test-A/out.tsv', sep='\t', index=False, header=False)