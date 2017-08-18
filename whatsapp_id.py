#coding:utf8
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import string
from time import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from nltk.corpus import stopwords
def parseOutText(all_text):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """

    # from nltk.stem import SnowballStemmer
    # stemmer = SnowballStemmer('turkish')

    words = ""
    ### remove punctuation
    text_string = all_text.translate(string.maketrans("", ""), string.punctuation)
    ### project part 2: comment out the line below
    #words = text_string

    ### split the text string into individual words, stem each word,
    ### and append the stemmed word to words (make sure there's a single
    ### space between each stemmed word)
    #text_string = text_string.replace('\n', '') 
    wordList = text_string.split()
    wordList = [i for i in wordList if not hasNumbers(i) and not 'aha' in i]
    wordList = filter(None, wordList)
    
    # wordList = [stemmer.stem(word) for word in wordList]

    words = ' '.join(wordList)     
    return words

def getPersonAndMessage(line):
  line_splitted = line.split(':')
  message = line_splitted[4].strip()
  message = parseOutText(message)
  person = line_splitted[3].strip()
  return message, person

def hasNumbers(inputString):
  return any(char.isdigit() for char in inputString)

def saveData():
  messages = []
  labels = []
  
  with open('_chat.txt') as chat_file:
    counter = 0
    for line in chat_file:
      if counter > 10000:
        break

      if(len(line.split(':')) == 5):
        message, person = getPersonAndMessage(line)
        if len(person) > 20 or message == '':
          continue
        messages.append(message)
        labels.append(person)
      # counter += 1

  messages = np.array(messages)
  pickle.dump( messages, open("messages_data.pkl", "w") )
  pickle.dump( labels, open("labels_data.pkl", "w") )

with open('messages_data.pkl') as messages_data:
  messages = pickle.load(messages_data)

with open('labels_data.pkl') as labels_data:
  labels = pickle.load(labels_data)

label_encoder = LabelEncoder()
labels_categorized = labels #label_encoder.fit_transform(labels)

# Split train and test sets
messages_train, messages_test, labels_train, labels_test = train_test_split(messages, labels_categorized, test_size=0.1, random_state=42)

# Text vectorization
tr_stopwords = stopwords.words('turkish')
vectorizer = TfidfVectorizer(stop_words=tr_stopwords, max_df=0.5, sublinear_tf=True)
messages_train_transformed = vectorizer.fit_transform(messages_train)
messages_test_transformed = vectorizer.transform(messages_test)

# Feature selection
selector = SelectPercentile(f_classif, percentile=1)
selector.fit(messages_train_transformed, labels_train)
features_train_transformed = selector.transform(messages_train_transformed).toarray()
features_test_transformed = selector.transform(messages_test_transformed).toarray()


# Train and fit the model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

print len(features_train_transformed)
# clf = SVC(C=1000.0, kernel='rbf', random_state=42)
clf = GaussianNB()
# clf = DecisionTreeClassifier(random_state=42, min_samples_split=20)
t0 = time()
clf.fit(features_train_transformed, labels_train)
print "Training time: ", round(time() - t0, 3), "s"

pred = clf.predict(features_test_transformed)

print accuracy_score(labels_test, pred)

print "Test it !"
while(1):
  print "Input:",
  input = raw_input()
  input = vectorizer.transform([input])
  input = selector.transform(input)
  input = input.toarray()
  label = clf.predict(input)
  print 'Output:',
  print label[0]