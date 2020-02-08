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
import pkg_resources
from symspellpy import SymSpell, Verbosity
import time 
# npx chatito chatito --outputPath='./' --trainingFileName='training_dataset.json'

    
# Compute Average W2V for each datapoint
def avg_glove(df, glove):
    vectors = []
    for title in tqdm(df.data.values):
        vectors.append(np.average(glove.query(word_tokenize(title)), axis = 0))
    return np.array(vectors), df.intent.values

def run_grid_search(model, features, y):
    params = {'C' : [0.1, 1, 10, 50, 100]}          # Try these values for regularization
    grid = GridSearchCV(model, params, cv = 3, n_jobs = -1, scoring = 'roc_auc_ovr', verbose = 2, refit = True)    # Change to cv=7
    grid.fit(features,y)
    print(grid.best_params_)
    return grid.best_estimator_


# Perform GridSearch for SVM and return best model
def train_smalltalk_model(smalltalk_path):
    smalltalk = pd.read_csv(smalltalk_path)
    features, y = avg_glove(smalltalk, vectors)

    svm = SVC(gamma = 'scale', probability = True)
    return run_grid_search(svm, features, y)
     

# Perform GridSearch for SVM and return best model
def train_model(dataset_path, smalltalk_path):
    data = pd.read_csv(dataset_path)
    smalltalk = pd.read_csv(smalltalk_path)
    smalltalk['intent'] = ['smalltalk' for _ in smalltalk.data.values]
    data = pd.concat([data, smalltalk])
    features, y = avg_glove(data, vectors)


    svm = SVC(gamma = 'scale', probability = True)
    return run_grid_search(svm, features, y)

def smalltalk_reply(intent):
    responses = {
        'affirmative' : 'Alright! I\'ll do that', 
        'greet' : 'Hello!',
        'bye' : 'See you later!',
        'negative' : 'Cancelled!'
    }
    print('BOT: ' + responses[intent] + '\n')

# Dummy Responses 
def reply(intent, entities):
    responses = {
        'leaveBalance' : 'Checking your balance!!',
        'bookMeeting' : 'Ok scheduling a meeting',
        'applyLeave' : 'Ok apply for leave',
        'smalltalk' : 'smalltalk'
    }
    print('BOT: ' + responses[intent] + '\n')


def print_confidence_table(intents, confidence_scores):
    x = PrettyTable()
    x.field_names = ['Intent', 'Confidence']
    for intent, score in zip(intents, confidence_scores):
        x.add_row([intent, score])
    print(x)


def spell_correct(sym_spell, user_input):
    suggestions = sym_spell.lookup_compound(user_input, max_edit_distance=2)
    user_input = [suggest._term for suggest in suggestions][0]
    return user_input




# Pass the user input through the pipeline
#             -> Tuned SVM Model -> Intent 
# User Input |
#             -> Spacy Entity Model -> Entities 
def test_pipeline(user_input, model, smalltalk_model, vectors, sym_spell):
    start_time = time.time()
    nlp = spacy.load('custom_ner')
    #user_input = spell_correct(user_input)
    input_tokens = word_tokenize(user_input)
    entities = nlp(''.join(input_tokens))
    feature = np.average(vectors.query(input_tokens), axis = 0)
    intent = model.predict(feature.reshape(1, -1))
    confidence_scores = model.predict_proba(feature.reshape(1,-1))
    confidence_scores = [round(score, 2) for score in confidence_scores[0]]
    print_confidence_table(model.classes_, confidence_scores)

    print('Intent: ', intent)
    print('Entities: ', [(X.text, X.label_) for X in entities.ents])
    if intent[0] == 'smalltalk':
        smalltalk_intent = smalltalk_model.predict(feature.reshape(1, -1))
        confidence_scores = smalltalk_model.predict_proba(feature.reshape(1,-1))
        confidence_scores = [round(score, 2) for score in confidence_scores[0]]
        print_confidence_table(smalltalk_model.classes_, confidence_scores)
        smalltalk_reply(smalltalk_intent[0])
    else:
        reply(intent[0], entities.ents)
    print('Exec time : ', time.time()-start_time)

def init_dictionary(corpus_path):
    sym_spell = SymSpell()
    dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.create_dictionary(corpus_path)
    return sym_spell

if __name__  == '__main__':
    #nltk.download('punkt')
    #vectors = Magnitude("elmo_2x1024_128_2048cnn_1xhighway_weights_GoogleNews_vocab.magnitude")
    #vectors = Magnitude('elmo_2x1024_128_2048cnn_1xhighway_weights.magnitude')
    #vectors = Magnitude("glove.twitter.27B.100d.magnitude")
    dataset_path = 'chatito_train.csv'
    smalltalk_path = 'smalltalk.csv'
    vectors = Magnitude("crawl-300d-2M.magnitude")
    

    sym_spell = init_dictionary(dataset_path)
    model = train_model(dataset_path, smalltalk_path)
    smalltalk_model = train_smalltalk_model(smalltalk_path)

    print('Ready!\n')
    while True:
        test_pipeline(input(), model, smalltalk_model, vectors, sym_spell)

