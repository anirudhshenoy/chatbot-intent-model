from pymagnitude import *
import nltk
from nltk import word_tokenize
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import pandas as pd
import spacy

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns
sns.set_palette("muted")
    

def calc_f1(p_and_r):
    p, r = p_and_r
    return (2*p*r)/(p+r)


# Print the F1, Precision, Recall, ROC-AUC, and Accuracy Metrics 
# Since we are optimizing for F1 score - we will first calculate precision and recall and 
# then find the probability threshold value that gives us the best F1 score

def print_model_metrics(y_test, y_test_prob, confusion = False, verbose = True, return_metrics = False):

    precision, recall, threshold = precision_recall_curve(y_test, y_test_prob, pos_label = 1)
    
    #Find the threshold value that gives the best F1 Score
    best_f1_index =np.argmax([calc_f1(p_r) for p_r in zip(precision, recall)])
    best_threshold, best_precision, best_recall = threshold[best_f1_index], precision[best_f1_index], recall[best_f1_index]
    
    # Calulcate predictions based on the threshold value
    y_test_pred = np.where(y_test_prob > best_threshold, 1, 0)

    print(y_test_pred)
    
    # Calculate all metrics
    f1 = f1_score(y_test, y_test_pred, pos_label = 1, average = 'binary')
    roc_auc = roc_auc_score(y_test, y_test_prob)
    acc = accuracy_score(y_test, y_test_pred)
          
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(f1, best_precision, best_recall, roc_auc, acc))
    
    if return_metrics:
        return np.array([f1, best_precision, best_recall, roc_auc, acc])





def run_svc(train_features, test_features, y_train, y_test, alpha = 1e-6, return_f1 = True, verbose = True):
    #svm = SGDClassifier(loss = 'log', alpha = alpha, n_jobs = -1, penalty = 'l2') # change to hinge loss
    svm = LinearSVC()
    svm.fit(train_features, y_train) 
    y_test_prob = svm.predict(test_features)
    print(accuracy_score(y_test_prob, y_test))
    #print_model_metrics(y_test, y_test_prob, confusion = False, verbose = True, return_metrics = False)
    #return y_test_prob

def avg_glove(df, glove):
    vectors = []
    for title in tqdm(df.data.values):
        vectors.append(np.average(glove.query(word_tokenize(title)), axis = 0))
    return np.array(vectors), df.intent.values

def train_model(features, y):
    svm = SVC(gamma = 'scale', probability = True)
    params = {'C' : [0.01, 0.1, 1, 10, 100, 1000]}
    grid = GridSearchCV(svm, params, cv = 3, n_jobs = -1, scoring = 'accuracy', verbose = 2, refit = True)
    grid.fit(features,y)
    print(grid.best_params_)
    return grid.best_estimator_


def reply(intent):
    responses = {
        'findRestaurantsByCity' : 'I see you\'re hungry! Do you want me to look for restaurants',
        'affirmative' : 'Alright! I\'ll do that', 
        'greet' : 'Hello!',
        'bye' : 'See you later!',
        'negative' : 'Cancelled!',
        'bookMeeting' : 'Ok scheduling a meeting',
        'applyLeave' : 'Ok apply for leave'
    }

    print('BOT: ' + responses[intent])



def test_pipeline(user_input, model):
    nlp = spacy.load('en_core_web_sm')
    entities = nlp(user_input)

    feature = np.average(glove.query(word_tokenize(user_input)), axis = 0)
    intent = model.predict(feature.reshape(1, -1))
    print(intent)
    print(entities.ents)
    print(model.predict_proba(feature.reshape(1, -1)))
    reply(intent[0])


if __name__  == '__main__':
    #nltk.download('punkt')
    glove = Magnitude("glove.twitter.27B.50d.magnitude")
    data = pd.read_csv('chatito_train.csv')
    features, y = avg_glove(data, glove)
    model = train_model(features, y)


    while True:
        test_pipeline(input(), model)

    #print(run_svc(train_features, test_features, y_train, y_test))