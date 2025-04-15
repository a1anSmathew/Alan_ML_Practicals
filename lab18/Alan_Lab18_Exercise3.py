import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# # Add correct path to NLTK data
# nltk.data.path.append("/home/ibab/nltk_data")
#
# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)  # Remove URLs, mentions, hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)        # Remove punctuation
    tokens = word_tokenize(text.lower())  #

    return " ".join([word for word in tokens if word not in stop_words])

def load_data():
    tweet = pd.read_csv('Tweets.csv')
    tweet = tweet[tweet['airline_sentiment'].isin(['positive', 'negative'])]
    tweet['label'] = tweet['airline_sentiment'].map({'negative': 0, 'positive': 1})
    tweet['clean_text'] = tweet['text'].apply(preprocess)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tweet['clean_text'])
    y = tweet['label']
    return X,y

def kernel_func(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    # Train and evaluate SVM with different kernels
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    for k in kernels:
        model = SVC(kernel=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Kernel: {k} - Accuracy: {acc:.4f}")

def main():
    X,y = load_data()
    kernel_func(X,y)

if __name__ == '__main__':
    main()
