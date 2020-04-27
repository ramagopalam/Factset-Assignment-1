# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 23:51:51 2020
"""
#import statements for all the libraries required
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier
import string
from sklearn.model_selection import train_test_split



#importing bothe excel sheets
df1=pd.read_excel(r'Assignment-1.xlsx',sheet_name=0)
df2=pd.read_excel(r'Assignment-1.xlsx',sheet_name=1)
df1=pd.DataFrame(df1)
df2=pd.DataFrame(df2)
df2=df2[['company_name','url']]

#making a training feature matrix by tokenizing the company name
x_train1 = [word_tokenize(i) for i in df1["company_name"]]
x_train2 = [word_tokenize(i) for i in df2["company_name"]]

x_train = np.concatenate((x_train1,x_train2),axis = 0)
print(len(x_train))

#making response vector
y_train1 = np.zeros(len(df1))
y_train2 = np.ones(len(df2))

y_train = np.concatenate((y_train1,y_train2),axis = 0)
print(len(y_train))

#splitting the dataset
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train)
print(len(x_train),len(x_test),len(y_train),len(y_test))

#making a tuple (x,y) and adding it to the list
documents_train=[]
for i in range(len(x_train)):
    documents_train.append((x_train[i],y_train[i]))
print(documents_train[0])

#Lemmitization Process starts
lemmatizer=WordNetLemmatizer()

#this function returns what the word's part of speech indicate
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

#making a list of stop words and punctuations    
stops=set(stopwords.words('english'))
punctuations=list(string.punctuation)
stops.update(punctuations)
stops,punctuations

#this function removes the stop words,punctuation marks and lemmitizes each word
def clean_tweet(words):
    output_words=[]
    for w in words:
        if w.lower() not in stops:
            pos=pos_tag([w]) #returns the part of speech of the word
            clean_word=lemmatizer.lemmatize(w,pos=get_simple_pos(pos[0][1]))
            output_words.append(clean_word.lower())
    return output_words


#cleaning process
i=0
for doc,sentiment in documents_train:
    #print(doc,sentiment)
    documents_train[i]=(clean_tweet(doc),sentiment)
    #print(i)
    i=i+1

i=0
for doc in x_test:
    #print(doc)
    x_test[i]=clean_tweet(doc)
    #print(x_test[i])
    #print(i)
    i=i+1

#puts all the words present after cleaning process in a list    
all_words=[]
for doc in documents_train:
    all_words+=doc[0]
    
freq=nltk.FreqDist(all_words) #finding the frequency of each word
common=freq.most_common(2625) #making a list of tuples where each tuple is in the form (word, it's frequency)
print(common)
features=[i[0] for i in common] # storing all the unique words i.e. all words in the list of tuple
features
    
def get_feature_dict(words): # checks if the features variable contains the words present in 'words' that is passed and returns true or false for each of thw word present in features variable
    current_features={} # dictionary created
    words_set=set(words)
    for w in features:
        current_features[w] = w in words_set
    return current_features

#makes a tuple consisting of (dictionary retrieved from get_features_dict() , 0/1)
    
training_data=[(get_feature_dict(doc),category) for doc,category in documents_train]
testing_data=[get_feature_dict(doc) for doc in x_test]

#applying naive bayes on trainig data to create a model
classifier=NaiveBayesClassifier.train(training_data)

#predicting the value of testing data using the model
y_pred=[]
for i in range(len(testing_data)):
    #print(i)
    y_pred.append(classifier.classify(testing_data[i]))
    
#print(y_test)
#print(y_pred)

#checking the values of test dat with the values our model predicted to calculate accuracy
accuracy = (sum(y_pred == y_test)/len(y_test))
print(accuracy)


#extracting important keywords from the dataset that can be used to decide whether a company is vc or non vc 
vc=[]
nvc=[]
for tuples in documents_train:
    if tuples[1]==1.0:
        nvc.append(tuples[0])
    else:
        vc.append(tuples[0])

for i in range(len(x_test)):
    if(y_test[i]==1.0):
        nvc.append(x_test[i])
    else:
        vc.append(x_test[i])
       
#puts all the words present in the vc list
all_words_vc=[]
for word in vc:
    all_words_vc+=word

#puts all the words present in the nvc list
all_words_nvc=[]
for word in nvc:
    all_words_nvc+=word
   
freq_vc=nltk.FreqDist(all_words_vc)  #finding the frequency of each word
kw_vc=freq_vc.most_common(5)      #making a list of tuples for five most freqent words

freq_nvc=nltk.FreqDist(all_words_nvc)  #finding the frequency of each word
kw_nvc=freq_nvc.most_common(5)      #making a list of tuples for five most freqent words

#print(freq_vc['group'])

#creating a list of comparitive keywords
for i in kw_vc:
    value=freq_nvc[i[0]]
    a=(i[0],value)
    if a not in kw_nvc:
        kw_nvc.append(a)

for i in kw_nvc:
     value_1=freq_vc[i[0]]
     b=(i[0],value_1)
     if b not in kw_vc:
         kw_vc.append(b)










