import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualInferenceNetwork(nn.Module):
    
    def __init__(self, input_size, bert_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2, label_size=0):
        super(ContextualInferenceNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        
        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "relu":
            self.activation = nn.ReLu()
            
        # Input layer
        self.input_layer = nn.Linear(bert_size + label_size, hidden_sizes[0])
        
        # Hidden layers
        layers = []
        for i in range(len(hidden_sizes) - 1):
            h_in = hidden_sizes[i]
            h_out = hidden_sizes[i + 1]
            layers.append(nn.Linear(h_in, h_out))
            layers.append(self.activation)
        self.hiddens = nn.Sequential(*layers)
        
        # mu
        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        
        # sigma
        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        
        # Dropout
        self.dropout_enc = nn.Dropout(p=self.dropout)
        
    def forward(self, x, x_bert, labels=None):
        x = x_bert
        
        if labels:
            x = torch.cat((x_bert, labels), 1)
        
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
        
        return mu, log_sigma
        
class CombinedInferenceNetwork(nn.Module):
    
    def __init__(self, input_size, bert_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2, label_size=0):
        super(CombinedInferenceNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        
        if activation == "softplus":
            self.actitvation = nn.Softplus()
        elif activation == "relu":
            self.actitvation = nn.ReLU()
        
        # Input layer 
        self.adapt_bert = nn.Linear(bert_size, input_size)
        self.input_layer = nn.Linear(input_size + input_size + label_size, hidden_sizes[0])
        
        # Hidden layers
        layers = []
        for i in range(len(hidden_sizes) - 1):
            h_in = hidden_sizes[i]
            h_out = hidden_sizes[i + 1]
            layers.append(nn.Linear(h_in, h_out))
            layers.append(self.actitvation)
        self.hiddens = nn.Sequential(*layers)
        
        # mu
        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        
        # sigma
        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        
        # Dropout
        self.dropout_enc = nn.Dropout(p=self.dropout)
        
    def forward(self, x, x_bert, labels=None):
        x_bert = self.adapt_bert(x)
        
        x = torch.cat((x, x_bert), 1)
        
        if labels:
            x = torch.cat((x, labels), 1)
            
        x = self.input_layer(x)
        x = self.actitvation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
        
        return mu, log_sigma