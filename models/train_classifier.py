import sys
import re
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score
import xgboost as xgb
import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    ''' Load data from database '''
    engine=create_engine(database_filepath)
    df = pd.read_sql_table(database_filepath,engine)
    X=df['message'] # Leave original message apart to avoid multi correlation
    y = df.drop(['id','message','original','genre'],axis=1) # Drop non categories columns
    category_names = y.columns.to_list()
    return X, y, category_names


def tokenize(text):
    """ Tokenize and lemmatize text data """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf',MultiOutputClassifier(xgb.XGBClassifier(objective="binary:logistic")))
    ])
    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    ''' Evaluate the three metrics and print results '''
    accuracy = []
    precision = []
    recall = []
    opt_preds= model.predict(X_test)
    for i in range(len(category_names.columns)):
        accuracy.append(accuracy_score(opt_preds[:,i],Y_test.iloc[:,i]))
        precision.append(precision_score(opt_preds[:,i],Y_test.iloc[:,i],average='macro',zero_division=0))
        recall.append(recall_score(opt_preds[:,i],Y_test.iloc[:,i],average='macro',zero_division=0))
    print(f'Accuracy score is {sum(accuracy)/len(accuracy)}')
    print(f'Precision score is {sum(precision)/len(precision)}')
    print(f'Recall score is {sum(recall)/len(recall)}')
    


def save_model(model, model_filepath):
    ''' Save as a picke file '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()