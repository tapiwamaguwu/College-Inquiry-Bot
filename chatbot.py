import os
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle

import numpy as np

nltk.download('punkt')
nltk.download('wordnet')

from keras.models import load_model


import json
import random


model_filepath= os.path.join('chatbot','da_model.h5')
intents_filepath= os.path.join('chatbot','test.json')
words_pickle_filepath= os.path.join('chatbot','words.pkl')
classes_pickle_filepath= os.path.join('chatbot','classes.pkl')

model = load_model(model_filepath)
intents = json.loads(open(intents_filepath, encoding="utf8").read())
words = pickle.load(open(words_pickle_filepath,'rb'))
classes = pickle.load(open(classes_pickle_filepath,'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'].split(","))
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

response = chatbot_response("Hi")
print(response)
# print("Hi I\'m Cole, a bot designed to help you with any enquiries about the university of Zimbabwe \n I can help with Undergraduate and Postgraduate programs and the requirements for each \n feel free to ask me anything ")
# msg = ''
# while msg != 'quit':
#   msg = input()
#   if(msg != 'quit'):


