import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

#Initialize empty lists for words, classes, documents and characters to ignore
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#Lemmatize words, sort words/classes alphabetically, remove ingoreLetters/duplicates, and send to files
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Initialize empty list, create a list of 0's with length equal to classes and sort in outputEmpty variable
training = []
outputEmpty = [0] * len(classes)

'''Loop through each document and intialize empty list back, then tokenize each word in document and lemmatize it
Create a bag of words for the representation of the ducment by setting the value of each word to one if appears in document'''
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words: bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty) 
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

#Create new sequential keras model 
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(128, input_shape = (len(trainX[0]),), activation = 'relu'))
#Dropout layer, randomly set 50% of inputs to 0 at each update during training to reduce overfitting
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation = 'softmax'))

#Create stochastic gradient descent optimizer with learning rate of 0.01, momentum of 0.9 and stroke momentum maneuver
sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.9, nesterov = True)

#Compile model specifying loss function as categorical crossentropy, optimzer as SGD optimzer and the metric used to evaluate model as accuracy 
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#Cereate variable hist to train model
hist = model.fit(np.array(trainX), np.array(trainY), epochs = 200, batch_size = 5, verbose = 1)

#Save train model
model.save('chatbot_model.h5', hist)
print("Executed")