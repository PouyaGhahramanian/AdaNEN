
"""
    Use this module to run experiments on the selected data stream
    with AdaNEN and the baseline models.
    -----------------------
    Pouya Ghahramanian
    """

import os
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('baselines')
import numpy as np
import pandas as pd
import pandas as pd
import logging
import random
import pickle
import argparse
import torch
import time
from AdaNEN import AdaNEN
from hbp import HBP
from mlp import MLP

import skmultiflow
print(skmultiflow.__version__)

# Classic machine learning baselines
from skmultiflow.lazy.knn_adwin import KNNAdwin
from skmultiflow.meta import OzaBaggingAdwin
from skmultiflow.lazy import KNN
from skmultiflow.meta.dynamic_weighted_majority import DynamicWeightedMajority
from sklearn.neural_network import MLPClassifier
from skmultiflow.meta import AccuracyWeightedEnsemble
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from skmultiflow.meta import AdditiveExpertEnsemble
from skmultiflow.meta.learn_nse import LearnNSE
from skmultiflow.meta.learn_pp import LearnPP
from skmultiflow.meta.leverage_bagging import LeverageBagging
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from sklearn import tree
from Goowe import Goowe

parser = argparse.ArgumentParser(description='Get experiment info.')
parser.add_argument('--stream', '-s', default = 'nyt',
                    help = 'Data Stream.')
parser.add_argument('--dataset_type', '-d', default = 'text',
                    help = 'Dataset type (Text or Numerical).')
parser.add_argument('--embedding_type', '-e', default = 'bert',
                    help = 'Embedding type (W2V or BERT).')
parser.add_argument('--results_file', '-o', default = 'default',
                    help = 'Address to the results file to save the accuracy info.')
parser.add_argument('--eval_window', '-w', default = 10,
                    help = 'Evaluation window size.')
parser.add_argument('--sample_size', '-p', default = 0,
                    help = 'Data size selected for the experiment. If not specified, all data will be included.')
parser.add_argument('--exclude_models', '-x', nargs = '+', default = [],
                    help = 'Baseline models to be excluded.')

args = parser.parse_args()
STREAM = args.stream
RESULTS_FILE = args.results_file
EVAL_WINDOW = int(args.eval_window)
DATASET_TYPE = args.dataset_type
EXCLUDE_MODELS = args.exclude_models
EMBD = args.embedding_type
SAMPLE_SIZE = int(args.sample_size)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Globally Define Value of Variables and Parameters to Get From Selected Data Stream Later
ids = None
texts = None
embeddings = None
labels = None
embedding_size = 300
data_size = 0
num_c = 2
num_f = 300
# data = None

if(STREAM == 'nyt'):
    # Get data stream
    logging.info('\n\tReading data stream: {}'.format(str.upper(STREAM)))
    df = pd.read_pickle('data/nyt_modified.csv')
    data = df.values
    # Data ===> 0: Text, 1: Embedding_W2V, 2: Embedding_BERT, 3: Labels
    mean = np.mean(df['Label'].values)
    median = np.median(df['Label'].values)
    data_size = data.shape[0]
    texts = data[:, 0]
    embeddings_w2v = data[:, 1]
    embeddings_bert = data[:, 2]
    labels_ = data[:, 3]
    embeddings = embeddings_w2v
    if(EMBD == 'bert'):
        embeddings = embeddings_bert
        embedding_size = 768
    labels = labels_
    if RESULTS_FILE == 'default': RESULTS_FILE = 'nyt'
    num_f = embedding_size

elif(STREAM == 'elec'):
    # Get data stream
    logging.info('\n\tReading data stream: {}'.format(str.upper(STREAM)))
    df = pd.read_csv('data/elec.csv')
    data = df.values
    # Data ===> 0-9: Features, 10: Labels
    data_size = data.shape[0]
    texts = []
    num_f = 6
    num_c = 2
    labels_ = data[:,num_f]
    embeddings = data[:,:num_f]
    labels = labels_
    if RESULTS_FILE == 'default': RESULTS_FILE = 'elec'

elif(STREAM == 'phishing'):
    # Get data stream
    logging.info('\n\tReading data stream: {}'.format(str.upper(STREAM)))
    df = pd.read_csv('data/phishing.csv')
    data = df.values
    # Data ===> 0-5: Features, 6: Labels
    data_size = data.shape[0]
    texts = []
    num_f = 46
    num_c = 2
    labels_ = data[:,num_f]
    embeddings = data[:,:num_f]
    labels = labels_
    if(RESULTS_FILE == 'default'): RESULTS_FILE = 'phishing'

elif(STREAM == 'hyperplane'):
    # Get data stream
    logging.info('\n\tReading data stream: {}'.format(str.upper(STREAM)))
    df = pd.read_csv('data/rotatingHyperplane.csv')
    data = df.values
    # Data ===> 0-5: Features, 6: Labels
    data_size = data.shape[0]
    print(data.shape)
    texts = []
    num_f = 10
    num_c = 2
    labels_ = data[:,num_f]
    embeddings = data[:,:num_f]
    labels = labels_
    if(RESULTS_FILE == 'default'): RESULTS_FILE = 'hyperplane'

elif(STREAM == 'squares'):
    # Get data stream
    logging.info('\n\tReading data stream: {}'.format(str.upper(STREAM)))
    df = pd.read_csv('data/moving_squares.csv')
    data = df.values
    # Data ===> 0-5: Features, 6: Labels
    data_size = data.shape[0]
    print(data.shape)
    texts = []
    num_f = 2
    num_c = 4
    labels_ = data[:,num_f]
    embeddings = data[:,:num_f]
    labels = labels_
    if(RESULTS_FILE == 'default'): RESULTS_FILE = 'squares'

elif(STREAM == 'mg2c2d'):
    # Get data stream
    logging.info('\n\tReading data stream: {}'.format(str.upper(STREAM)))
    data = np.loadtxt('data/MG_2C_2D.txt', delimiter=',', dtype=float)    # Data ===> 0-5: Features, 6: Labels
    data_size = data.shape[0]
    print(data.shape)
    texts = []
    num_f = 2
    num_c = 2
    labels_ = data[:,num_f] - 1.
    embeddings = data[:,:num_f]
    labels = labels_
    if(RESULTS_FILE == 'default'): RESULTS_FILE = 'mg2c2d'

elif(STREAM == 'spam'):
    # Get data stream
    logging.info('\n\tReading data stream: {}'.format(str.upper(STREAM)))
    df = pd.read_csv('data/spam.csv')
    data = df.values
    data_size = data.shape[0]
    print(data.shape)
    texts = []
    num_f = 499
    num_c = 2
    labels_ = data[:,num_f]
    embeddings = data[:,:num_f]
    labels = labels_
    if(RESULTS_FILE == 'default'): RESULTS_FILE = 'spam'

elif(STREAM == 'usenet'):
    # Get data stream
    logging.info('\n\tReading data stream: {}'.format(str.upper(STREAM)))
    df = pd.read_csv('data/usenet.csv')
    data = df.values
    data_size = data.shape[0]
    print(data.shape)
    texts = []
    num_f = 99
    num_c = 2
    labels_ = data[:,num_f]
    embeddings = data[:,:num_f]
    labels = labels_
    if(RESULTS_FILE == 'default'): RESULTS_FILE = 'usenet'

elif(STREAM == 'email'):
    # Get data stream
    logging.info('\n\tReading data stream: {}'.format(str.upper(STREAM)))
    data = np.load('data/email_data_numpy.npy')
    data_size = data.shape[0]
    print(data.shape)
    texts = []
    num_f = 913
    num_c = 2
    labels_ = data[:,num_f]
    embeddings = data[:,:num_f]
    labels = labels_
    if(RESULTS_FILE == 'default'): RESULTS_FILE = 'email'

else:
    print('Error in the stream name: ', STREAM)
    exit()

# Initialize Parameters and the Baseline Models
# Parameters
architecture_ = (128, 64, 32, 16)
architecture = [128, 64, 32, 16]
layers = [1e-3, 1e-2, 1e-1]
no = len(layers)
learning_rate = 1e-2
target_values = [i for i in range(num_c)]

# GPU Device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Baseline Models
# Neural Models
adanen = AdaNEN(feature_size = num_f, arch = architecture, num_classes = num_c, etha = 1e-4,
                 betha = 0.8, s = 0.2, num_outs = no, lrs = layers, optimizer = 'rmsprop')
adam = MLPClassifier(solver='adam', alpha=1e-5, learning_rate_init=1e-5,
            hidden_layer_sizes=architecture_, random_state=1)
sgd = MLPClassifier(solver='sgd', alpha=1e-2, learning_rate_init=1e-2,
            hidden_layer_sizes=architecture_, random_state=1)
hbp = HBP(feature_size =num_f, etha = 1e-2, hidden_size = 16, L = 4, classes_num = num_c)
adanen.to(device)
hbp.to(device)

# Ensemble Models
knnadwin = KNNAdwin(n_neighbors=5, max_window_size=2000, leaf_size=30)
ozaadwin = OzaBaggingAdwin(base_estimator=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=2)
dwm = DynamicWeightedMajority()
awe = AccuracyWeightedEnsemble()
arf = AdaptiveRandomForest()
aee = AdditiveExpertEnsemble()
learn_pp = LearnPP(base_estimator=KNN(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=30)
learn_pp_nse = LearnNSE(base_estimator=tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0., min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, random_state=None, splitter='best'), window_size=1, slope=0.5, crossing_point=10, pruning=None)
lbg = LeverageBagging(base_estimator = KNN(n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=2)
hat = HAT()
# Version of ScikitMultiflow to be used with GOOWE is: 0.4.1, Other models: 0.5.3
# GOOWE Parameters
N_MAX_CLASSIFIERS = 15
CHUNK_SIZE = 500
WINDOW_SIZE = 100
num_targets = 1
CHUNK_SIZE = 500
goowe = Goowe(n_max_components=N_MAX_CLASSIFIERS,
              chunk_size=CHUNK_SIZE,
              window_size=WINDOW_SIZE,
              logging = False)
goowe.prepare_post_analysis_req(num_f, num_targets, num_c, target_values)

# Define and Initialize Evaluation Variables
# FIXME: Should fix 'GOOWE': dwm here
models = {'AdaNEN': adanen, 'SGD': sgd, 'HBP': hbp,
          'Adam': adam, 'HAT': hat, 'AEE': aee, 'AWE': awe, 'ARF': arf,
          'DWM': dwm, 'Learn++': learn_pp, 'Learn++.NSE': learn_pp_nse,
          'KNN-Adwin': knnadwin, 'Oza-Adwin': ozaadwin,
          'GOOWE': goowe}
crrs = {'AdaNEN': 0.0, 'Adam': 0.0, 'SGD': 0.0, 'HBP': 0.0, 'DWM': 0.0,
             'AEE': 0.0, 'AWE': 0.0, 'HAT': 0.0, 'Learn++': 0.0,
             'Learn++.NSE': 0.0, 'ARF': 0.0, 'Oza-Adwin': 0.0, 'KNN-Adwin': 0.0,
             'GOOWE': 0.0}
crrs_all = {'AdaNEN': 0.0, 'Adam': 0.0, 'SGD': 0.0, 'HBP': 0.0, 'DWM': 0.0,
             'AEE': 0.0, 'AWE': 0.0, 'HAT': 0.0, 'Learn++': 0.0,
             'Learn++.NSE': 0.0, 'ARF': 0.0, 'Oza-Adwin': 0.0, 'KNN-Adwin': 0.0,
             'GOOWE': 0.0}
preds = {'AdaNEN': 0.0, 'Adam': 0.0, 'SGD': 0.0, 'HBP': 0.0, 'DWM': 0.0,
             'AEE': 0.0, 'AWE': 0.0, 'HAT': 0.0, 'Learn++': 0.0,
             'Learn++.NSE': 0.0, 'ARF': 0.0, 'Oza-Adwin': 0.0, 'KNN-Adwin': 0.0,
             'GOOWE': 0.0}
accs = {'AdaNEN': [], 'Adam': [], 'SGD': [], 'HBP': [], 'DWM': [],
             'AEE': [], 'AWE': [], 'HAT': [], 'Learn++': [],
             'Learn++.NSE': [], 'ARF': [], 'Oza-Adwin': [], 'KNN-Adwin': [],
             'GOOWE': []}
ensemble_weights = {'AdaNEN': [], 'HBP': []}
losses = {'AdaNEN': [], 'Adam': [], 'SGD': [], 'HBP': []}
ensemble_weights_tmp = {'AdaNEN': 0, 'HBP': 0}
losses_tmp = {'AdaNEN': 0, 'Adam': 0, 'SGD': 0, 'HBP': 0}
times_all = {'AdaNEN': 0.0, 'Adam': 0.0, 'SGD': 0.0, 'HBP': 0.0, 'DWM': 0.0,
             'AEE': 0.0, 'AWE': 0.0, 'HAT': 0.0, 'Learn++': 0.0, 'Learn++.NSE': 0.0,
             'ARF': 0.0, 'Oza-Adwin': 0.0, 'KNN-Adwin': 0.0, 'GOOWE': 0.0}

# Exclude User-Specified Models
for model in EXCLUDE_MODELS:
    models.pop(model)
# models.pop('GOOWE')

model_keys = models.keys()
clf_num = len(model_keys)

# Experiments goes here...
# Partial Fit Ensemble Models on a Small Portion of Data as Required
init_size = 10
for i in range(init_size):
    X_init, y_init = np.asarray(embeddings[i]).reshape(-1, num_f), np.asarray(int(labels[i])).reshape(-1,)
    lbg = lbg.partial_fit(X_init, y_init, classes = target_values)
    knnadwin.partial_fit(X_init, y_init)
    ozaadwin.partial_fit(X_init, y_init, classes = target_values)
    learn_pp = learn_pp.partial_fit(X_init, y_init, classes = target_values)
    arf.partial_fit(X_init, y_init, classes = target_values)
    adam.partial_fit(X_init, y_init, classes = target_values)
    sgd.partial_fit(X_init, y_init, classes = target_values)
    learn_pp_nse.partial_fit(X_init, y_init, classes = target_values)
    goowe.partial_fit(X_init, y_init)

# Test-Then-Train on Data One by One
total = 0.
s_time = time.time()
if(SAMPLE_SIZE != 0):
    data_size = SAMPLE_SIZE
for i in range(data_size):
    if((i + 1) % EVAL_WINDOW == 0):
        for model in models.keys():
            acc = 100.0 * (crrs[model] / total)
            accs[model].append(acc)
        for model in losses.keys():
            losses[model].append(losses_tmp[model])
        for model in ensemble_weights.keys():
            ensemble_weights[model].append(ensemble_weights_tmp[model])
        e_time = round(time.time() - s_time, 2)
        logging.info('\t===============================================================')
        logging.info('\tData Instance: {0}'
                     '\n\t\tElapsed Time: {1}'
                     '\n\t\t====== Overall Accuracies ======'
                     .format(i+1, e_time))
        for key in models.keys():
            acc = 100.0 * (crrs_all[key] / (i + 1))
            print('\t\t' + key + ': ' + str(round(acc, 2)))
            crrs[key] = 0.
        logging.info('\t===============================================================')
        logging.info('\n\t\t====== Overall Times ======')
        for key in models.keys():
            print('\t\t' + key + ': ' + str(round(times_all[key], 2)))
        logging.info('\t===============================================================')
        if('AdaNEN' in models.keys()):
            logging.info('\tAdaNEN Ensemble Weights: {0}'.format(adanen.get_weights()))
        total = 0.

    total += 1
    X_t, Y_t = embeddings[i].reshape(-1, num_f), np.array(labels[i]).reshape(-1,)
    for m in model_keys:
        model = models[m]
        if(m == 'AdaNEN'):
            time_1 = time.time()
            preds[m] = 0.
            losses_tmp[m] = 0.
            try:
                preds[m] = models[m].predict(X_t)
                losses_tmp[m] = models[m].partial_fit(X_t, Y_t)
            except:
                print("Error in the prediction and partial fit of Model: {}.".format(m))
            times_all[m] += (time.time() - time_1)
            crrs[m] += np.sum(preds[m] == Y_t)
            crrs_all[m] += np.sum(preds[m] == Y_t)
            ensemble_weights_tmp[m] = models[m].get_weights()
        else:
            time_1 = time.time()
            preds[m] = 0.
            try:
                preds[m] = model.predict(X_t)
                model.partial_fit(X_t, Y_t)
            except:
                print ("Error in the prediction and partial fit of Model: {}.".format(m))
            times_all[m] += (time.time() - time_1)
            crrs[m] += np.sum(preds[m] == Y_t)
            crrs_all[m] += np.sum(preds[m] == Y_t)

# Create directory if it does not exist to store the results
RESULTS_ADDR = 'results/' + RESULTS_FILE + '/'
try:
   os.makedirs(RESULTS_ADDR)
except FileExistsError:
   logging.info('\n\t\tResults directory already exists. Overwriting results data...')
   pass

# Save Results for AdaNEN and the Baseline Models
with open(RESULTS_ADDR + '/times.data', 'wb') as f:
    pickle.dump(times_all, f)
with open(RESULTS_ADDR + '/accuracies_all.data', 'wb') as f:
    pickle.dump(accs, f)
with open(RESULTS_ADDR + '/ensemble_weights.data', 'wb') as f:
    pickle.dump(ensemble_weights, f)
with open(RESULTS_ADDR + '/losses.data', 'wb') as f:
    pickle.dump(losses, f)
