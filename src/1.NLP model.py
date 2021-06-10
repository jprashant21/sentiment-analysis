

import os
import re
import numpy as np
import pickle as pk
from nltk import word_tokenize, FreqDist, pos_tag
from nltk.metrics import ConfusionMatrix
from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.classify import accuracy
from nltk.tokenize import word_tokenize as wt
import nltk
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string

folder_path = "F:\\Analytics\\Sentiment Mining"
os.chdir(folder_path) 

trainset = pk.load(open("trainset.pk","rb"),encoding="latin1")
testset = pk.load(open("testset.pk","rb"),encoding="latin1")
neededTag= ["JJ","JJR","JJS","RB","RBR","RBS"]    
neededTag= ["O"]    

stop = stopwords.words('english')
WNlemma = nltk.WordNetLemmatizer()
def preprocess(sentence):
    #sent_pos = nlp.ner(sentence)
    sent_pos = word_tokenize(sentence)
    #toks = [ w for w,t in sent_pos if t in neededTag]
    toks = [ t.lower() for t in sent_pos  if t not in string.punctuation ]
    toks=[WNlemma.lemmatize(t) for t in toks]
    toks = [t for t in toks if t not in stop ]
    toks_clean = [ t for t in toks if len(t) >= 3 ]
    return toks_clean

def neg_tag(text):
    transformed = re.sub(r"\b(?:never|nothing|nowhere|noone|none|not|haven't|hasn't|hasnt|hadn't|hadnt|can't|cant|couldn't|couldnt|shouldn't|shouldnt|won't|wont|wouldn't|wouldnt|don't|dont|doesn't|doesnt|didn't|didnt|isnt|isn't|aren't|arent|aint|ain't|hardly|seldom)\b[\w\s]+[^\w\s]", lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)), text, flags=re.IGNORECASE)
    return(transformed)

# Create a training list which will now contain reviews with Negatively tagged words and their labels
train_set_neg = []

# Append elements to the list
for doc in trainset:
    trans = neg_tag(doc[0])
    lab = doc[1]
    train_set_neg.append([trans, lab])

# Create a testing list which will now contain reviews with Negatively tagged words and their labels
test_set_neg = []

# Append elements to the list
for doc in testset:
    trans = neg_tag(doc[0])
    lab = doc[1]
    test_set_neg.append([trans, lab])

train_nolab = [preprocess(t[0]) for t in train_set_neg]
test_nolab = [preprocess(t[0]) for t in test_set_neg]

train_lab = [t[1] for t in train_set_neg]
test_lab = [t[1] for t in test_set_neg]

trainJoined= [ ' '.join(f) for f in train_nolab ]
tewstJoined= [ ' '.join(f) for f in test_nolab ]
vectorizer = TfidfVectorizer()

# this is used below for training the SVM
train_vectors = vectorizer.fit_transform(trainJoined)
test_vectors = vectorizer.transform(tewstJoined)

# SVM Classifier from sklearn
def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=10000.0, gamma='auto', kernel='rbf')
    svm.fit(X, y)
    return svm

# Pickled model as it takes ahwile for generation
classifier_svm = train_svm(train_vectors, train_lab)
pk.dump(classifier_svm, open("classifier_svm.pk","wb"))
classifier_svm = pk.load(open("classifier_svm.pk", "rb"))
predSVM = classifier_svm.predict(test_vectors)
#type(predSVM)
pred = list(predSVM)
cm = ConfusionMatrix(pred, test_lab)
print(cm)
print(classification_report(pred,  test_lab))
# 89.617
# 89.32



def preprocess_rev(sentence):
    #sent_pos = nlp.ner(sentence)
    sent_pos = word_tokenize(sentence)
    #toks = [ w for w,t in sent_pos if t in neededTag]
    toks = [ t.lower() for t in sent_pos  if t not in string.punctuation ]
    toks=[WNlemma.lemmatize(t) for t in toks]
    toks = [t for t in toks if t not in stop ]
    toks_clean = [ t for t in toks if len(t) >= 3 ]
    return toks_clean

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


train_proc = []
for doc in trainset:
    trans = preprocess(doc[0])
    lab = doc[1]
    train_proc.append([trans, lab])

train_proc_neg = []
train_proc_pos = []
for t in range(len(train_proc)):
    if train_proc[t][1] == -1:
        train_proc_neg.append(train_proc[t])
    else:
        train_proc_pos.append(train_proc[t])

train_proc_neg_join= [' '.join(f[0]) for f in train_proc_neg]
train_proc_pos_join= [' '.join(f[0]) for f in train_proc_pos]

Neg_bag_of_words = ''
for l in range(len(train_proc_neg_join)):
    Neg_bag_of_words = Neg_bag_of_words + train_proc_neg_join[l]

Pos_bag_of_words = ''
for l in range(len(train_proc_neg_join)):
    Pos_bag_of_words = Pos_bag_of_words + train_proc_pos_join[l]

all_words = list()    
all_words.append(Pos_bag_of_words)
all_words.append(Neg_bag_of_words)
    
# And tfidf indexing
vec_tfidf = TfidfVectorizer()
allwords_tfidf = vec_tfidf.fit_transform(all_words)
features = vec_tfidf.get_feature_names()


top_feats_in_doc(allwords_tfidf,features,0,25)
top_feats_in_doc(allwords_tfidf,features,1,25)


vec_tf = CountVectorizer()
grain_tf = vec_tf.fit_transform(grain_text)



neg_words = vec_tfidf.inverse_transform(neg_tfidf)
neg_tfidf.toarray()[:10]


## Deep learning model for sentiment classification

from keras.models import Sequential
from keras.layers import Dense
import numpy

model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(train_nolab, train_lab, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

