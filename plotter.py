import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import argparse
import os
import pickle

#dataset_size = 45312
#dataset_size = 11055
#dataset_size = 200000
dataset_size = 1500
multiply = 100.

    ##################################################
    ### Helper methods to calculate Moving Average ####
    ##################################################

def ma(arr, ws = 10):
    s = arr.shape[0]
    s_new = int(s/ws)
    arr_new = np.zeros(s_new)
    for i in range(s_new-1):
        a1 = int(i*ws)
        a2 = int((i+1)*ws)
        arr_new[i] = np.mean(arr[a1:a2])
    a1 = int(-1 * ws)
    if(ws == 1): arr_new[-1] = arr[-1]
    else: arr_new[-1] = np.mean(arr[a1:-1])
    return arr_new

    ######################################
    ### Get user specified parameters ####
    ######################################

parser = argparse.ArgumentParser(description='Get data stream results and output file name.')
parser.add_argument('--results_path', '-p', default = 'nyt',
                    help = 'Results file containing prequential accuracy values.')
parser.add_argument('--output_file', '-o', default = 'none',
                    help = 'Output file name for the generated accuracy plot.')
parser.add_argument('--window_size', '-w', default = 30,
                    help = 'Number of evaluation windows.')

args = parser.parse_args()
data_stream = args.results_path
output_file = 'figs/' + args.output_file
WINDOW_SIZE = int(args.window_size)
if output_file == 'figs/none': output_file = 'figs/' + data_stream

accuracies_top_models = ['AdaNEN', 'Adam', 'AWE', 'KNN-Adwin', 'GOOWE']
# accuracies_top_models = ['AdaNEN', 'Adam', 'DWM', 'KNN-Adwin', 'HAT']

lines_1 = ["-","--","-.",":", 'dashdot', ':', '--']
lines_2 = ["-","-","-","-", '-', '-', '-']
colors = ['pink', 'c', 'gray', 'orangered', 'darkred', 'm', 'blue', 'navy',
          'teal', 'salmon', 'grey', 'midnightblue', 'violet']
colors_2 = ['midnightblue', 'orangered', 'darkred', 'dimgray', 'g', 'k', 'm', 'c', 'blue', 'navy',
          'teal', 'salmon', 'grey', 'pink', 'violet']
markers = ['v', '1', '2', '3', '4', '8', '.', '<', '>',
           's', 'p', 'o', 'D', '+']
lines = lines_2

    #############################################
    ### Load Time, Accuracy and Weights data ####
    #############################################

with open('results/' + data_stream + '/accuracies_all.data', 'rb') as f:
    accuracies_all = pickle.load(f)
with open('results/' + data_stream + '/times.data', 'rb') as f:
    times_all = pickle.load(f)

    #####################################
    ### Print Time and Accuracy data ####
    #####################################

models = []
accuracies_avg = {}
print('\n\t-------------------------------------------------------------')
print('\tData Stream: === ' + data_stream.upper() + ' === ')
print('\t-------------------------------------------------------------')
print('\t========== Accuracy values ==========\n\t-----------------------------')
for m in accuracies_all.keys():
    if(accuracies_all[m] != []):
        models.append(m)
        accuracies_avg[m] = np.mean(accuracies_all[m])
        print('\t' + str(m) + ': ' + str(np.round(np.mean(accuracies_all[m]), 2)))
print('\n\t========== Time values ==========\n\t-----------------------------')
for m in times_all.keys():
    if(times_all[m] != 0.0):
        print('\t' + str(m) + ': ' + str(np.round((np.mean(times_all[m])/dataset_size)*multiply, 2)))

    ######################################
    ### Part 1: Accuracy Data Plotter ####
    ######################################

font_size = 90
font_size_legend = 60
#plt.rcParams.update({'font.size': 27})
plt.rcParams.update({'font.size': 60})
plt.rcParams["font.family"] = "serif"
fig_size = (40, 24)
fig, ax = plt.subplots(figsize = fig_size, nrows = 1)
x = np.arange(1, WINDOW_SIZE + 1)
c = 0
line_w = 8
marker_s = 36
for m in accuracies_top_models:
    if(m != 'ADANN-MS'):
        if(m == 'AdaNEN'):
            ax.plot(x, ma(np.asarray(accuracies_all[m]), int(len(accuracies_all[m]) / x.shape[0]))[:len(x)],
                color = 'midnightblue', label='AdaNEN', linewidth=line_w, marker = '*', markersize = marker_s, linestyle = '-')
        elif(m == 'AEE'):
            ax.plot(x, ma(np.asarray(accuracies_all[m]), int(len(accuracies_all[m]) / x.shape[0]))[:len(x)],
                color = colors[c], label='AddExp', linewidth=line_w, marker = '.', markersize = marker_s, linestyle = lines[c])
        elif(m == 'BERT'):
            ax.plot(x, ma(np.asarray(accuracies_all[m]), int(len(accuracies_all[m]) / x.shape[0]))[:len(x)],
                color = colors[c], label=models[i], linewidth=line_w, marker = '.', markersize = marker_s, linestyle = lines[c])
        else:
            ax.plot(x, ma(np.asarray(accuracies_all[m]), int(len(accuracies_all[m]) / x.shape[0]))[:len(x)],
                color = colors[c], label=m, linewidth=line_w, marker = '.', markersize = marker_s, linestyle = lines[c])
        c += 1

ax.grid(True)
plt.xticks(np.arange(0, WINDOW_SIZE + 1, step = 5))
plt.xlabel('Evaluation Window #', fontsize = font_size)
plt.ylabel('Accuracy (%)', fontsize = font_size)
ax.legend(loc = 'lower left', prop={"size":font_size_legend})
fig.savefig(output_file + '_accuracies.eps', format = 'eps')
plt.show()
