
"""
    This code is a Pytorch implementation of the paper
    'A Novel Neural Ensemble Architecture for On-The-Fly Classification of Evolving Text Streams'.
    Submitted to ACM TKDD (ACM Transactions on Knowledge Discovery from Data).
    ---------------------------
    Pouya Ghahramanian
    ---------------------------
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaNEN(nn.Module):

    def __init__(self, feature_size = 300, arch = [64], num_classes = 2, etha = 1e-5,
                 p_drop = 0.2, betha = 0.99, s = 0.2, num_outs = 3, lrs = [1e-3, 1e-2, 1e-1],
                 optimizer = 'rmsprop'):

        ##########################################################################
        ################# Initializing model and parameters.  ####################
        ##########################################################################
        super(AdaNEN, self).__init__()
        if len(arch) < 1:
            raise ValueError('Parameter arch should be a list of at least one element to specify the number of neurons in the first hidden layer.')
        self.device = torch.device('cpu')
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.betha = nn.Parameter(torch.tensor(betha), requires_grad=False).to(self.device)
        self.etha = nn.Parameter(torch.tensor(etha), requires_grad=False).to(self.device)
        self.s = nn.Parameter(torch.tensor(s), requires_grad=False).to(self.device)
        self.num_outs = num_outs
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.architecture = arch
        self.L = len(arch)
        self.losses = []
        self.FREEZE_HIDDEN = False
        Hidden_Layers = []
        self.lrs = lrs
        self.lr = etha
        self.alpha = nn.Parameter(torch.Tensor(self.num_outs).fill_(1 / (self.num_outs + 1.0)), requires_grad=False)
        Hidden_Layers.append(nn.Linear(feature_size, arch[0]))
        for i in range(len(arch)-1):
            Hidden_Layers.append(nn.Linear(arch[i], arch[i+1]))
        self.hidden_layers = nn.ModuleList(Hidden_Layers)
        self.output_layers = nn.ModuleList([nn.Linear(sum(arch), num_classes) for i in range(num_outs)])
        self.dropout = nn.Dropout(p = p_drop)
        params = []
        params.append({'params': self.hidden_layers.parameters(), 'lr': etha})
        for k in range(self.num_outs):
            params.append({'params': self.output_layers[k].parameters(), 'lr': lrs[k]})
        if optimizer == 'sgd':
            self.optim = torch.optim.SGD(params, momentum=0.0)
        elif optimizer == 'adam':
                    self.optim = torch.optim.Adam(params, betas=(0.9, 0.999), eps=1e-08,
                                            weight_decay=0, amsgrad=False)
        elif optimizer == 'rmsprop':
            self.optim = torch.optim.RMSprop(params, alpha=0.99, eps=1e-08, weight_decay=0,
                                             momentum=0, centered=False)
        elif optimizer == 'adagrad':
            self.optim = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0,
                                             initial_accumulator_value=0, eps=1e-10)
        elif optimizer == 'adadelta':
            self.optim = torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        elif optimizer == 'adagrad':
            self.optim = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0,
                                             initial_accumulator_value=0, eps=1e-10)
        elif optimizer == 'adamax':
            self.optim = torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            self.optim = torch.optim.SGD(params, momentum=0.0)

    def forward(self, x):

        ##########################################################################
        ################### Claculating output of classifiers. ###################
        ##########################################################################
        x = torch.from_numpy(x).float().to(self.device)
        fs = []
        h_t = self.hidden_layers[0](x)
        hts = h_t
        for i in range(1, self.L):
            h_t = self.hidden_layers[i](h_t)
            hts = torch.cat((hts, h_t), dim = 1)
        for i in range(self.num_outs):
            fs.append(self.output_layers[i](self.dropout(hts)))
        fs = torch.stack(fs)
        return fs

    def predict(self, x):

        ##########################################################################
        ######### Prediction data instance or a batch of data instances. #########
        ##########################################################################
        self.eval()
        fs = self.forward(x)
        F = torch.sum(torch.mul(self.alpha.view(self.num_outs, 1).repeat(1, len(x)).view(self.num_outs, len(x), 1), fs), 0)
        pred = torch.argmax(F, dim=1).cpu().numpy()
        return pred

    def partial_fit(self, x, y):

        self.train()
        y = torch.from_numpy(y.astype(np.float32)).to(self.device)
        fs = self.forward(x)
        criterion = nn.CrossEntropyLoss()
        losses = []
        ##################################################################################
        ### Updating weights of hidden and output layers using predefined optimizer.   ###
        ##################################################################################
        weights = []
        biases = []
        losses = []
        self.output_layers.requires_grad = True
        self.hidden_layers.requires_grad = False
        t_loss = criterion(fs[0], y.long())
        losses.append(t_loss)
        self.optim.zero_grad()
        t_loss = t_loss * self.alpha[0]
        for i in range(1, self.num_outs):
            losses.append(criterion(fs[i], y.long()))
            lss = losses[i]
            t_loss += lss * self.alpha[i]
        self.losses = losses
        self.output_layers.requires_grad = True
        self.hidden_layers.requires_grad = not(self.FREEZE_HIDDEN)
        t_loss.backward()
        self.optim.step()
        Loss = np.dot(self.alpha.cpu().numpy(), torch.FloatTensor(losses).cpu().numpy()).item()
        ################################################################################
        ### Updating weights of classifiers (alpha parameter) using ensemble method. ###
        ################################################################################
        losses_sum = sum(losses)
        for i in range(self.num_outs):
            self.alpha[i] *= torch.pow(self.betha, (losses[i]))
            self.alpha[i] = torch.max(self.alpha[i], self.s / self.num_outs)

        ###########################################################################
        ############ Normalizing weights of classifiers (alpha values) ############
        ###########################################################################
        alpha_sum = torch.sum(self.alpha)
        self.alpha = nn.Parameter(self.alpha / alpha_sum, requires_grad=False)

        #################################################################################
        ###### Total Loss is a weighted combinatoutpution of classifiers' losses.  ######
        ######     Loss = Sum(alpha_i * loss_i) for i in range(self.num_outs)      ######
        #################################################################################
        return(Loss)

    def zero_grad(self):
        for i in range(self.num_outs):
            self.output_layers[i].weight.grad.data.fill_(0)
            self.output_layers[i].bias.grad.data.fill_(0)
        for i in range(self.L):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def zero_grad_hidden(self):
        for i in range(self.L):
            self.hidden_layers[i].weight.grad.data.fill_(0)
            self.hidden_layers[i].bias.grad.data.fill_(0)

    def get_weights(self):
        weights = self.alpha.detach().cpu().numpy()
        return weights

    def get_losses(self):
        lsss = [np.max(l.data.cpu().numpy()) for l in self.losses]
        return lsss

    def set_weights(self, weights):
        self.alpha.data = torch.from_numpy(weights).float().to(self.device)

    def freeze_hidden_layer(self):
        self.FREEZE_HIDDEN = True

    def defreeze_hidden_layer(self):
        self.FREEZE_HIDDEN = False

    def reset_weights(self):
        weights = np.full((self.num_outs), 1 / (self.num_outs + 1.0))
        self.alpha.data = torch.from_numpy(weights).float().to(self.device)
