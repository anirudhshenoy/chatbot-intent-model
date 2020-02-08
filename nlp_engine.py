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
from intent import *
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
def train_sub_intent_model(dataset_path):
    data = pd.read_csv(dataset_path)
    features, y = avg_glove(data, vectors)

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



# Dummy Responses 
def reply(intent, entities, models, feature, context):
    responses = {
        'leaveBalance' : 'Checking your balance!!',
        'bookMeeting' : 'Ok scheduling a meeting',
        'applyLeave' : applyLeave,
        'smalltalk' : smalltalk
    }
    #print('BOT: ' + responses[intent] + '\n')
    responses[intent](models, feature, context)


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
def test_pipeline(user_input, model, vectors, context):
    start_time = time.time()
    nlp = spacy.load('custom_ner')
    #user_input = spell_correct(user_input)
    input_tokens = word_tokenize(user_input)
    entities = nlp(''.join(input_tokens))
    feature = np.average(vectors.query(input_tokens), axis = 0).reshape(1,-1)
    intent = models['mainModel'].predict(feature)
    confidence_scores = models['mainModel'].predict_proba(feature)
    confidence_scores = [round(score, 2) for score in confidence_scores[0]]
    print_confidence_table(models['mainModel'].classes_, confidence_scores)
    context.append(intent)
    print('Intent: ', intent)
    print('Entities: ', [(X.text, X.label_) for X in entities.ents])
    reply(intent[0], entities.ents, models, feature, context)
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
    vectors = Magnitude("glove.twitter.27B.100d.magnitude")
    #vectors = Magnitude("crawl-300d-2M.magnitude")
    basedir = 'dataset/'
    dataset_path = basedir + 'chatito_train.csv'

    subintents = ['smalltalk', 'typeofleave']    

    sym_spell = init_dictionary(dataset_path)
    model = train_model(dataset_path, basedir + subintents[0] + '.csv')

    models = {'mainModel' : model}
    for subintent in subintents:
        models[subintent] = train_sub_intent_model(basedir + subintent + '.csv')

    print('Ready!\n')
    context = []
    while True:
        test_pipeline(input(), models, vectors, context)

