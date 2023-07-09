
"""
    This code is a Pytorch implementation of a Multi Layer Perceptron (MLP).
    Architecture of the network and optimizer can be specified by the user.
    ---------------------------
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class MLP(nn.Module):

    def __init__(self, feature_size = 300, arch = [64], num_classes = 2, optimizer = 'sgd', lr = 1e-2):
        super(MLP, self).__init__()
        if len(arch) < 1:
            raise ValueError('Parameter arch should be a list of at least one element to specify the number of neurons in the first hidden layer.')
        Hidden_Layers = []
        Hidden_Layers.append(nn.Linear(feature_size, arch[0]))
        for i in range(len(arch)-1):
            Hidden_Layers.append(nn.Linear(arch[i], arch[i+1]))
        self.hidden_layers = nn.ModuleList(Hidden_Layers)
        self.output_layer = nn.Linear(arch[-1], num_classes)
        self.num_classes = num_classes
        '''
        self.params = [{'params': self.hidden_layers.parameters(), 'lr': lr},
                       {'params': self.output_layer.parameters(), 'lr': lr}]
        '''
        self.params = [{'params': self.parameters(), 'lr': lr}]
        if optimizer == 'sgd':
            self.optim = torch.optim.SGD(params = self.params, momentum=0.0)
        if optimizer == 'rmsprop':
            self.optim = torch.optim.RMSprop(lr = lr, params = self.params)
        elif optimizer == 'adam':
            self.optim = torch.optim.Adam(lr = lr, params = self.params)
        else:
            self.optim = torch.optim.SGD(params = self.params, momentum=0.9)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.hidden_num = len(arch)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        y_prime = F.relu(self.hidden_layers[0](x))
        for i in range(1, self.hidden_num):
            y_prime = F.relu(self.hidden_layers[i](y_prime))
        y_prime = self.output_layer(y_prime)
        return y_prime

    def predict(self, x):
        pred = self.forward(x)
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        return pred

    def partial_fit(self, x, y):
        y = torch.from_numpy(y).to(self.device)
        criterion = nn.CrossEntropyLoss()
        y_prime = self.forward(x)
        loss = criterion(y_prime, y.long())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        loss_np = loss.detach().cpu().numpy()
        return loss_np

if __name__ == '__main__':

    ###########################################################################
    ####### Initializing a toy model and training it on a data instance #######
    ###########################################################################
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    mlp = MLP(feature_size = 300, arch = [64], num_classes = 2, optimizer = 'rmsprop', lr = 1e-2)
    mlp.to(device)
    input = np.random.randn(1, 300)
    labels = np.random.randint(low = 0, high=2, size=(1,))
    output = mlp(input)
    preds_1 = mlp.predict(input)
    epochs = 100
    for e in range(epochs): print('Loss: ', mlp.partial_fit(input, labels))
    preds_2 = mlp.predict(input)
    print('Actual labels: ', labels)
    print('Predictions before training: ', preds_1)
    print('Predictions after training: ', preds_2)
