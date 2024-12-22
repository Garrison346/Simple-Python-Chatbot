import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import load_model

#Intialize lemmatizer object
lemmatizer = WordNetLemmatizer()

#Opens and reads intents.json file into a python dictionary using json library, words/classes.pkl are loaded into lists uding pickle
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

#Load pre-trained chatbot model
model = tf.keras.models.load_model('chatbot_model.h5')

#Take intents as input, tokenize, lemmatize each word and return list of lemmatized words  
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

'''Takes sentence as input, called clean up function to get list of lemmatized words then creates a bag of words to represent
the sentence and return as numpy array'''
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word ==w:
                bag[i] = 1
    return np.array(bag)

'''Takes sentence as input then uses bad of words function to create representation of the sentence then feeds the
words to the pre trained chatbot model.Returns a list of intents with the probabilities and error underscore threshold
variable that will set the minimum probability required for an intent to be considered. Thr list of intents
is sorted in descending order of probability and each intent is returned as a dictionary containing intent'''
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key = lambda x:x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

'''Take parameters instents_list(contain predicted intents of user input) and intents_json(json file that lists
all the possible intents and the corresponding responses), them match predicted intent with intent in json file
selects response from list of responses and returns as chatbot response'''
def get_response(intents_list, intents_json):
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intent']
    for i in list_of_intents:
        if i['tag']==tag:
            result = random.choice(i['responses'])
            break

    return result
print("Great! Chatbot is Live!")

while True:
    message = input("You: ")
    if message.lower() == 'quit':
        print('Goodbye!')
        break

    ints = predict_class(message)
    res = get_response(ints, intents)
    print(f'Chatbot: {res}')
