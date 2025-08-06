import torch
from torch import nn
from torch.nn import functional as F
from src.ctm.network.inference_network import CombinedInferenceNetwork, ContextualInferenceNetwork


class DecoderNetwork(nn.Module):
    
    def __init__(self, ):
        super(DecoderNetwork, self).__init__()
        
        self.input_size = input_size
        self.n_components = components # num_topic
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation 
        self.dropout = dropout
        self.learn_priors = learn_priors 
        self.topic_word_matrix = topic_word_matrix
        
        if infnet == "zeroshot":
            self.infnet = ContextualInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size)
        elif infnet == "combined": # QUESTION: Why is bow_size not passed here?
            self.infnet = CombinedInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size)
        else:
            raise Exception("Missing infnet parameter, options are zeroshot and combined")
        
        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)
        
        # prior_mean   
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor([topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)
        
        # prior_variance
        topic_prior_variance = 1.0 - (1.0 / self.n_components)
        self.prior_variance = torch.tensor([topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)
        
        # beta
        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)
        
        # theta
        self.drop_theta = nn.Dropout(p=self.dropout)
        
        
        
        
        
        