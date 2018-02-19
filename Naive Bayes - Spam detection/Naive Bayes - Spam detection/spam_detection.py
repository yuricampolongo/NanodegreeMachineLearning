from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


df = pd.read_table('smsspamcollection/SMSSpamCollection', sep='\t', header=None, names=['label', 'sms_message'])
df.head()
df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
df.head() # retorna (linhas, colunas)

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print('Número de linhas no dataset: {}'.format(df.shape[0]))
print('Número de linhas no conjunto de treinamento: {}'.format(X_train.shape[0]))
print('Número de linhas no conjunto de teste: {}'.format(X_test.shape[0]))

# Instancie o método CountVectorizer method
count_vector = CountVectorizer()

# Ajuste os dados de treinamento e retorne a matriz
training_data = count_vector.fit_transform(X_train)

# Transforme dados de teste e retorne a matriz. Note que não estamos ajustando os dados de texto no CountVectorizer()
testing_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)

print('Acurácia: ', format(accuracy_score(y_test, predictions)))
print('Precisão: ', format(precision_score(y_test, predictions)))
print('Recall: ', format(recall_score(y_test, predictions)))
print('Escore F1: ', format(f1_score(y_test, predictions)))