
"""
    This code is a Pytorch implementation of the paper 'Online Deep Learning:
    Learning Deep Neural Networks on the Fly', Sahoo et. al.
    which introduces Hedge Backpropagation algorithm.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HBP(nn.Module):

    def __init__(self, feature_size = 300, hidden_size = 16, L = 8, classes_num = 5, etha = 0.2, betha = 0.99, s = 0.2):

        ##########################################################################
        ################# Initializing HBP model and parameters. #################
        ##########################################################################
        super(HBP, self).__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.betha = nn.Parameter(torch.tensor(betha), requires_grad=False).to(self.device)
        self.etha = nn.Parameter(torch.tensor(etha), requires_grad=False).to(self.device)
        self.s = nn.Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.L = L
        self.classes_num = classes_num
        self.hidden_layers_ = []
        self.output_layers = []
        self.losses = []
        self.alpha = nn.Parameter(torch.Tensor(self.L).fill_(1 / (self.L + 1.0)), requires_grad=False)
        self.hidden_layers_.append(nn.Linear(self.feature_size, self.hidden_size))
        for i in range(L-1): self.hidden_layers_.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.hidden_layers = nn.ModuleList(self.hidden_layers_)
        self.output_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.classes_num) for i in range(L)])

    def forward(self, x):

        ##########################################################################
        ################### Claculating output of classifiers. ###################
        ##########################################################################
        x = torch.from_numpy(x).float().to(self.device)
        fs = []
        h_t = F.relu(self.hidden_layers[0](x))
        fs.append(self.output_layers[0](h_t))
        for i in range(1, self.L):
            h_t = F.relu(self.hidden_layers[i](h_t))
            fs.append(self.output_layers[i](h_t))
        fs = torch.stack(fs)
        return fs

    def predict(self, x):

        ##########################################################################
        ######### Prediction data instance or a batch of data instances. #########
        ##########################################################################
        fs = self.forward(x)
        F = torch.sum(torch.mul(self.alpha.view(self.L, 1).repeat(1, len(x)).view(self.L, len(x), 1), fs), 0)
        pred = torch.argmax(F, dim=1).cpu().numpy()
        return pred

    def partial_fit(self, x, y):

        y = torch.from_numpy(y).to(self.device)
        fs = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        losses = []
        ##########################################################################
        ### Updating weights of output layers (Theta parameter) using equ. (2) ###
        ##########################################################################
        weights = []
        biases = []

        for i in range(self.L):

            losses.append(criterion(fs[i], y.long()))
            losses[i].backward(retain_graph = True)
            self.output_layers[i].weight.data -= self.etha * self.alpha[i] * self.output_layers[i].weight.grad.data
            self.output_layers[i].bias.data -= self.etha * self.alpha[i] * self.output_layers[i].bias.grad.data
            weights.append(self.alpha[i] * self.hidden_layers[i].weight.grad.data)
            biases.append(self.alpha[i] * self.hidden_layers[i].bias.grad.data)
            #self.zero_grad()

        Loss = np.dot((self.alpha).cpu().numpy().T, np.asarray(losses)).item()

        ##########################################################################
        ##### Updating weights of hidden layers (W parameter) using equ. (3) #####
        ##########################################################################
        for i in range(1, self.L):
            self.hidden_layers[i].weight.data -= self.etha * torch.sum(torch.cat(weights[i:]))
            self.hidden_layers[i].bias.data -= self.etha * torch.sum(torch.cat(biases[i:]))

        ##########################################################################
        ### Updating weights of classifiers (alpha parameter) using Hedge algo ###
        ##########################################################################
        for i in range(self.L):
            '''
            losses_sum = sum(losses)
            for ii in range(len(losses)):
                losses[ii] = losses[ii] / losses_sum
            '''
            self.alpha[i] *= torch.pow(self.betha, losses[i])
            self.alpha[i] = torch.max(self.alpha[i], self.s / self.L)

        ###########################################################################
        ############ Normalizing weights of classifiers (alpha values) ############
        ###########################################################################
        alpha_sum = torch.sum(self.alpha)
        self.alpha = nn.Parameter(self.alpha / alpha_sum, requires_grad=False)

        ##########################################################################
        ###### Total Loss is a weighted combinatoutpution of classifiers' losses. ######
        ######     Loss = Sum(alpha_i * loss_i) for i in range(self.L)      ######
        ##########################################################################
        return(Loss)

    def zero_grad(self):
        for i in range(self.L):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def get_weights(self):
        weights = self.alpha.cpu().numpy()
        return weights

    def set_weights(self, weights):
        self.alpha.data = torch.from_numpy(weights).float().to(self.device)

if __name__ == '__main__':

    ###########################################################################
    ####### Initializing a toy model and training it on a data instance #######
    ###########################################################################
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    hbp = HBP(L = 5)
    hbp.to(device)
    input = np.random.randn(10, 300)
    labels = np.random.randint(low = 0, high=5, size=(10,))
    output = hbp(input)
    preds_1 = hbp.predict(input)
    epochs = 100
    for e in range(epochs): print('Loss: ', hbp.partial_fit(input, labels))
    preds_2 = hbp.predict(input)
    print('Classifier weights: ', hbp.alpha)
    print('Actual labels: ', labels)
    print('Predictions before training: ', preds_1)
    print('Predictions after training: ', preds_2)
