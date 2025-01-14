import pandas as pd
import  numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report
import re

# loading the sample dataset used for the project
def load_data(file):
    data = pd.read_csv(file, sep='\t', header=None, names=['label', 'message'])
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

# tokenization
def tokenize(text):
    return re.findall(r'\b\w+\b' , text.lower())

# preprocessing the dataset for training purpose
def train_classifier(train_data):
    vectorizer = CountVectorizer(analyzer=tokenize)
    X_train = vectorizer.fit_transform(train_data['message'])
    y_train = train_data['label']
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    return vectorizer, classifier

def calculate_priors(data):
    total_messages = len(data)
    spam_messages = len(data[data['label'] == 1])
    non_spam_messages = len(data[data['label'] == 0])
    return spam_messages / total_messages, non_spam_messages / total_messages


def predict_message(message, classifier, vectorizer, priors):
    message_vector = vectorizer.transform([message])
    log_likelihoods = classifier.feature_log_prob_
    feature_names = vectorizer.get_feature_names_out()
    words = tokenize(message)

    log_likelihood_spam = np.log(priors[0])
    log_likelihood_ham = np.log(priors[1])

    for word in words:
        if word in feature_names:
            idx = np.where(feature_names == word)[0][0]
            log_likelihood_spam += log_likelihoods[1][idx]
            log_likelihood_ham += log_likelihoods[0][idx]

    posterior_spam = np.exp(log_likelihood_spam)
    posterior_ham = np.exp(log_likelihood_ham)
    if posterior_spam > posterior_ham:
        print('this is email is classified as SPAM message')
    else:
        print('this email is the legitimate message/ ham')


def posterior():
    file = 'SMSSpamCollection'
    data = load_data(file)
    train_data , test_data = train_test_split(data , test_size= 0.3 , random_state= 0)
    priors= calculate_priors(train_data)
    vectorizer , classifier = train_classifier(train_data)

    x_test = vectorizer.transform(test_data['message'])
    y_test =  test_data['label']
    y_prediction = classifier.predict(x_test)

    print(classification_report(y_test , y_prediction))
    test_message =" please subscribe to this channel for 3 dollar daily to win 1 million dollar prize"
    predict_message(test_message, classifier , vectorizer , priors)
if __name__ == "__main__":
    posterior()




