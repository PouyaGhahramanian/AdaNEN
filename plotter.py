import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import argparse
import os
import pickle

    ##################################################
    ### Helper method to calculate Moving Average ####
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
                    help = 'Results directory containing accuracy values.')
parser.add_argument('--output_file', '-o', default = 'none',
                    help = 'File name to store the generated prequential accuracy plots.')
parser.add_argument('--window_size', '-w', default = 20,
                    help = 'Number of evaluation windows.')

args = parser.parse_args()
data_stream = args.results_file
output_file = args.output_file
WINDOW_SIZE = int(args.window_size)
if output_file == 'none': output_file = data_stream

accuracies_top_models = ['AdaNEN', 'Adam', 'DWM', 'KNN-Adwin', 'HAT']
models_p2 = ['AdaNEN', 'Adam', 'SGD', 'HBP']

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

with open('Results/' + data_stream + '/accuracies_all.data', 'rb') as f:
    accuracies_all = pickle.load(f)
with open('Results/' + data_stream + '/times.data', 'rb') as f:
    times_all = pickle.load(f)
with open('Results/' + data_stream + '/losses.data', 'rb') as f:
    losses = pickle.load(f)
with open('Results/' + data_stream + '/ensemble_weights.data', 'rb') as f:
    ensemble_weights = pickle.load(f)

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
        print('\t' + str(m) + ': ' + str(np.round(np.mean(times_all[m]), 2)))

    ######################################
    ### Part 1: Accuracy Data Plotter ####
    ######################################

plt.rcParams.update({'font.size': 27})
plt.rcParams["font.family"] = "serif"
fig_size = (40, 24)
fig, ax = plt.subplots(figsize = fig_size, nrows = 1)
x = np.arange(1, WINDOW_SIZE + 1)
c = 0
line_w = 8
marker_s = 36
for m in accuracies_top_models:
    if(m == 'AdaNEN'):
        ax.plot(x, ma(np.asarray(accuracies_all[m]), int(len(accuracies_all[m]) / x.shape[0]))[:len(x)],
            color = colors[c], label='AdaNEN', linewidth=line_w, marker = markers[c], markersize = marker_s, linestyle = '-')
    elif(m == 'AEE'):
        ax.plot(x, ma(np.asarray(accuracies_all[m]), int(len(accuracies_all[m]) / x.shape[0]))[:len(x)],
            color = colors[c], label='AddExp', linewidth=line_w, marker = markers[c], markersize = marker_s, linestyle = lines[c])
    else:
        ax.plot(x, ma(np.asarray(accuracies_all[m]), int(len(accuracies_all[m]) / x.shape[0]))[:len(x)],
            color = colors[c], label=m, linewidth=line_w, marker = markers[c], markersize = marker_s, linestyle = lines[c])
    c += 1

ax.grid(True)
plt.xticks(np.arange(0, WINDOW_SIZE + 1, step = 5))
plt.xlabel('Evaluation Window #', fontsize = 60)
plt.ylabel('Accuracy (%)', fontsize = 60)
ax.legend(loc = 'lower left', prop={"size":60})
fig.savefig(output_file + '_accuracies.svg')
plt.show()

    #########################################
    ### Part 2: Ensemble Weights Plotter ####
    #########################################

fig_size = (40, 72)
line_w = 10
marker_s = 42
figs, axs = plt.subplots(figsize = fig_size, nrows=3)
figs.tight_layout(pad = 4.0)
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]
font_size = 60
c = 0
plt.rcParams.update({'font.size': 27})
plt.rcParams["font.family"] = "serif"
plt.xlabel('Evaluation Window #', fontsize = font_size)
figs.tight_layout(pad = 4.0)

for m in models_p2:
    if(m == 'AdaNEN'):
        ax1.plot(x, ma(np.asarray(accuracies_all[m]), int(len(accuracies_all[m]) / x.shape[0]))[:len(x)],
            color = colors_2[c], label='ADANN', linewidth=line_w, marker = markers[c], markersize = marker_s, linestyle = '-')
    elif(m == 'AEE'):
        ax1.plot(x, ma(np.asarray(accuracies_all[m]), int(len(accuracies_all[m]) / x.shape[0]))[:len(x)],
            color = colors_2[c], label='AddExp', linewidth=line_w, marker = markers[c], markersize = marker_s, linestyle = lines[c])
    else:
        ax1.plot(x, ma(np.asarray(accuracies_all[m]), int(len(accuracies_all[m]) / x.shape[0]))[:len(x)],
            color = colors_2[c], label=m, linewidth=line_w, marker = markers[c], markersize = marker_s, linestyle = lines[c])
    c += 1

ax1.set_title('Accuracy values', fontsize = font_size)
ax1.set_ylabel('Accuracy (%)', fontsize = font_size)
ax1.legend(loc = 'lower left', prop={"size":font_size})
ax1.set_xticks(np.arange(0, WINDOW_SIZE + 1, step = 5))
ax1.grid(True)

weights_adanen = np.asarray(ensemble_weights['AdaNEN'])
weights_a = weights_adanen
weights_ma_1 = ma(weights_a[:, 0], int(len(accuracies_all[m]) / x.shape[0]))
weights_ma_2 = ma(weights_a[:, 1], int(len(accuracies_all[m]) / x.shape[0]))
weights_ma_3 = ma(weights_a[:, 2], int(len(accuracies_all[m]) / x.shape[0]))
weights_ma = np.concatenate([weights_ma_1.reshape(-1, 1), weights_ma_2.reshape(-1, 1), weights_ma_3.reshape(-1, 1)], axis = 1)

ax2.set_ylabel('Weight', fontsize = font_size)
ax2.set_title('Output layer weights', fontsize = font_size)
ax2.plot(x, weights_ma_1, linewidth=line_w, markersize = marker_s, marker='1', color = 'royalblue', label='1e-3')
ax2.plot(x, weights_ma_2, linewidth=line_w, markersize = marker_s, marker='2', color = 'dimgray', label='1e-2')
ax2.plot(x, weights_ma_3, linewidth=line_w, markersize = marker_s, marker='3', color = 'crimson', label='1e-1')
ax2.set_xticks(np.arange(0, WINDOW_SIZE + 1, step = 5))
ax2.grid(True)
ax2.legend(loc = 'lower left', prop={"size":font_size})

lrs = np.array([1e-3, 1e-2, 1e-1])
avg_lr = weights_ma.dot(lrs.T)
ax3.set_title('Avg. lr of output layer', fontsize = font_size)
ax3.set_ylabel('Avg. lr', fontsize = font_size)
ax3.plot(x, avg_lr, marker='.', color = 'steelblue', linewidth=line_w, markersize = marker_s, label = 'Avg. lr')
#ax3.yaxis.set_ticks(np.arange(np.round(np.min(avg_lr), 2), np.round(np.max(avg_lr), 2), np.round(np.max(avg_lr)/5, 2)))
ax3.grid(True)
ax3.legend(loc = 'lower left', prop={"size":font_size})

plt.xticks(np.arange(0, WINDOW_SIZE + 1, step = 5))
figs.savefig(output_file + '_weights.eps', format = 'eps')
plt.show()
