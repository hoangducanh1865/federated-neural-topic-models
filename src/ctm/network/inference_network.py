import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualInferenceNetwork(nm.Module):
    
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
        self.input_layer = 
        
        # Hidden layers
        self.hiddens = 
        
        # mu
        self.f_mu y 
        self.f_mu_batchnorm = 
        
        # sigma
        self.f_sigma = 
        self.f_sigma_batchnorm = 
        
        # Dropout
        self.dropout_enc = 
        
    def forward(self, x, x_bert, labels=None):
        x = 
        
        if labels:
            x = 
        
        x = 
        x = 
        x = 
        
        mu = 
        log_sigma = 
        
        return mu, log_sigma
        
class CombinedInferenceNetwork(nn.Module):
    
    def __init__(self):
        pass
        
    def forward(self):
        pass