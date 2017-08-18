import numpy as np

from sklearn.preprocessing import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

def preprocess(messages, labels):
  messages_train, messages_test, labels_train, labels_test = cross_validation.train_test_split