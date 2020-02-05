from pymagnitude import Magnitude
import nltk
from nltk import word_tokenize
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import pandas as pd
import spacy
from prettytable import PrettyTable
import numpy as np

    
# Compute Average W2V for each datapoint
def avg_glove(df, glove):
    vectors = []
    for title in tqdm(df.data.values):
        vectors.append(np.average(glove.query(word_tokenize(title)), axis = 0))
    return np.array(vectors), df.intent.values


# Perform GridSearch for SVM and return best model
def train_model(features, y):
    svm = SVC(gamma = 'scale', probability = True)
    params = {'C' : [0.01, 0.1, 1, 10, 50, 100, 1000]}          # Try these values for regularization
    grid = GridSearchCV(svm, params, cv = 3, n_jobs = -1, scoring = 'roc_auc_ovr', verbose = 2, refit = True)    # Change to cv=7
    grid.fit(features,y)
    print(grid.best_params_)
    return grid.best_estimator_


# Dummy Responses 
def reply(intent, entities):
    responses = {
        'affirmative' : 'Alright! I\'ll do that', 
        'greet' : 'Hello!',
        'bye' : 'See you later!',
        'negative' : 'Cancelled!',
        'bookMeeting' : 'Ok scheduling a meeting',
        'applyLeave' : 'Ok apply for leave'
    }
    print('BOT: ' + responses[intent] + '\n')


def print_confidence_table(intents, confidence_scores):
    x = PrettyTable()
    x.field_names = ['Intent', 'Confidence']
    for intent, score in zip(intents, confidence_scores):
        x.add_row([intent, score])
    print(x)

# Pass the user input through the pipeline
#             -> Tuned SVM Model -> Intent 
# User Input |
#             -> Spacy Entity Model -> Entities 
def test_pipeline(user_input, model, vectors):
    nlp = spacy.load('custom_ner')
    entities = nlp(user_input)

    feature = np.average(vectors.query(word_tokenize(user_input)), axis = 0)
    intent = model.predict(feature.reshape(1, -1))
    confidence_scores = model.predict_proba(feature.reshape(1,-1))
    confidence_scores = [round(score, 2) for score in confidence_scores[0]]
    print_confidence_table(model.classes_, confidence_scores)
    print('Intent: ', intent)
    print('Entities: ', [(X.text, X.label_) for X in entities.ents])
    reply(intent[0], entities.ents)


if __name__  == '__main__':
    #nltk.download('punkt')
    #vectors = Magnitude("glove.twitter.27B.100d.magnitude")
    vectors = Magnitude("crawl-300d-2M.magnitude")
    data = pd.read_csv('chatito_train.csv')
    features, y = avg_glove(data, vectors)
    model = train_model(features, y)

    print('Ready!\n')
    while True:
        test_pipeline(input(), model, vectors)

