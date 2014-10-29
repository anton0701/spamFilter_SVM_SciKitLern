__author__ = 'apple'

from scipy.io import *
from stemming.porter2 import stem
from sklearn import svm
import numpy
import re

def processEmail (s):
    #Preprocessing text of email
    # Return array of stemed words

    s = s.lower()
    s=re.sub(r'<[^<>]+>', ' ', s)
    s=re.sub(r'[0-9]+', ' number',s)
    s=re.sub(r'(http|https)://[^\s]*', 'httpaddr', s)
    s=re.sub(r'[^\s]+@[^\s]+', 'emailaddr', s)
    s=re.sub(r'[$]+', ' dollar', s)
    s=re.sub(r'[^a-zA-Z0-9]',' ',s)
    a = s.split()
    b = []

    for i in a:
        b.append(stem(i))
    for i in b:
        if len(i) < 2:
            b.remove(i)
    return b

#Loading memory of most common words from spam
vocabulary = {}
vocab = open('vocab.txt')
n = 0

for vocabLine in vocab.readlines():
    n += 1
    array = vocabLine.split()
    vocabTemp = {n : array[1]}
    vocabulary.update(vocabTemp)

#Load train data
dataTrain = loadmat('spamTrain.mat')

trainX = dataTrain.get('X')
trainY = dataTrain.get('y')
trainY = trainY.ravel()

#Part 1: Here you can check email messsage if it is spam
spamFile = open('emailSample3.txt')
s = spamFile.read()
s = processEmail(s)
wordIndices = []
for i in s:
    for j in xrange(len(vocabulary.values())):
        if i == vocabulary.values()[j]:
            wordIndices.append(j)
            break

wordsArray = [0 for z in xrange(1899)]
for i in wordIndices:
    wordsArray[i] = 1
#print wordsArray

w = numpy.array(trainY)
clf = svm.SVC(kernel='linear')
clf.fit(trainX, trainY)
#Print '1' if email is a spam
print clf.predict(wordsArray)


#Part 2.
#Getting statistics about spam filter accuracy.
#Load test data.
dataTest = loadmat('spamTest.mat')

testX = dataTrain.get('X')
testY = dataTrain.get('y')
testY = trainY.ravel()
correct = 0
total = 0
#Making prediction and counting results.
for i in xrange(len(testY)):
    ans = clf.predict(testX[i])[0]
    print 'Prediction:' + str(ans) + '. Answer:' + str(testY[i])
    if ans == testY[i]:
        correct += 1
    total += 1
print len(testY)
print 'Total:' + str(total) + '. Correct:' + str(correct)
