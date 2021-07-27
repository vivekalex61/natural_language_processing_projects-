
import numpy as np
import pandas as pd
import emoji
import itertools

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, SimpleRNN,LSTM, Activation
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from scipy import spatial
import matplotlib.pyplot as plt

#traindataset
train = pd.read_csv('./train.csv',header=None)
#testdataset
test = pd.read_csv('./test.csv',header=None)


#first 5 training examples
train.head(5)

#first 5 testing examples
test.head()

#we are converting numbers into it's suitable 
emoji_dict = { 0 : ":heart:", 1 : ":baseball:", 2 : ":smile:", 3 : ":disappointed:", 4 : ":berger:"}



for ix in emoji_dict.keys():
    print (ix,end=" ")
    print (emoji.emojize(emoji_dict[ix], use_aliases=True))



# Creating training and testing data
X_train = train[0]
Y_train = train[1]

X_test = test[0]
Y_test = test[1]

print (X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)



for ix in range(X_train.shape[0]):
    X_train[ix] = X_train[ix].split()

for ix in range(X_test.shape[0]):
    X_test[ix] = X_test[ix].split()
    
Y_train = to_categorical(Y_train)


# Now checking the above conversion by printing train and test data at 0th index
print (X_train[0],Y_train[0])




np.unique(np.array([len(ix) for ix in X_train]) , return_counts=True)

np.unique(np.array([len(ix) for ix in X_test]) , return_counts=True)


embeddings_index = {}
'''glove embeddingsGloVe stands for global vectors for word representation. It is an unsupervised learning algorithm developed by Stanford for generating word embeddings by aggregating global word-word co-occurrence matrix from a corpus. 
'''
f = open('./glove.6B.50d.txt', encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# Checking length of a particular word
embeddings_index["i"].shape


# Checking cosine similarity of words happy and sad
spatial.distance.cosine(embeddings_index["happy"], embeddings_index["sad"])


# Checking cosine similarity of words India and Delhi
spatial.distance.cosine(embeddings_index["india"], embeddings_index["delhi"])


# Checking cosine similarity of words france and paris
spatial.distance.cosine(embeddings_index["france"], embeddings_index["paris"])


# Filling the embedding matrix
embedding_matrix_train = np.zeros((X_train.shape[0], 10, 50))
embedding_matrix_test = np.zeros((X_test.shape[0], 10, 50))

for ix in range(X_train.shape[0]):
    for ij in range(len(X_train[ix])):
        embedding_matrix_train[ix][ij] = embeddings_index[X_train[ix][ij].lower()]
        
for ix in range(X_test.shape[0]):
    for ij in range(len(X_test[ix])):
        embedding_matrix_test[ix][ij] = embeddings_index[X_test[ix][ij].lower()]        




print (embedding_matrix_train.shape, embedding_matrix_test.shape)



# A simple LSTM network
def train_model():

	model = Sequential()
	model.add(LSTM(128, input_shape=(10,50), return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LSTM(64, return_sequences=False))
	model.add(Dropout(0.5))
	model.add(Dense(5))
	model.add(Activation('softmax'))
        return model
model=train_model()
model.summary()


# Setting Loss ,Optimiser for model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training model
hist = model.fit(embedding_matrix_train,
                 Y_train,
                 epochs = 50, 
                 batch_size=32,
                 shuffle=True
                )



# Prediction of trained model
pred = model.predict_classes(embedding_matrix_test)


# ### ACCURACY

# Calculating accuracy / score  of the model
float(sum(pred==Y_test))/embedding_matrix_test.shape[0]


# Printing the sentences with the predicted and the labelled emoji
for ix in range(embedding_matrix_test.shape[0]):
    
    if pred[ix] != Y_test[ix]:
        print(ix)
        print (test[0][ix],end=" ")
        print (emoji.emojize(emoji_dict[pred[ix]], use_aliases=True),end=" ")
        print (emoji.emojize(emoji_dict[Y_test[ix]], use_aliases=True))






