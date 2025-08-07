import torch
from torch import nn
from torch.nn import functional as F
from src.ctm.network.inference_network import CombinedInferenceNetwork, ContextualInferenceNetwork


class DecoderNetwork(nn.Module):
    
    def __init__(self, input_size, bert_size, infnet, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True, label_size=0):
        super(DecoderNetwork, self).__init__()
        
        self.input_size = input_size
        self.n_components = n_components # num_topic
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation 
        self.dropout = dropout
        self.learn_priors = learn_priors 
        self.topic_word_matrix = None
        
        if infnet == "zeroshot":
            self.inf_net = ContextualInferenceNetwork(
                input_size, bert_size, n_components, hidden_sizes, activation, label_size=label_size)
        elif infnet == "combined": # QUESTION: Why is bow_size not passed here?
            self.inf_net = CombinedInferenceNetwork(
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
        
        # Drop out on theta
        self.drop_theta = nn.Dropout(p=self.dropout)
        
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x, x_bert, labels=None):
        # posterior_mu and posterior_sigma
        posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
        posterior_sigma = torch.exp(posterior_log_sigma)
        
        # theta
        theta = F.softmax(self.reparameterize(posterior_mu, 2 * posterior_log_sigma), dim=1) # QUESTION: In the original code, they use posterior_log_sigma without multiple by 2?
        theta = self.drop_theta(theta)
        
        # beta
        if self.model_type == "prodLDA":
            self.topic_word_matrix = self.beta
            word_dist = F.softmax(self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
        elif self.model_type == "LDA":
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            self.topic_word_matrix = beta
            word_dist = torch.matmul(theta, beta)
        else:
            raise NotImplementedError("Model Type Not Implemented")
        # QUESTION: Discuss more on the difference between prodLDA and LDA above
            
        estimated_labels = None
        if labels:
            estimated_labels = self.label_classification(theta) # docs-topics -> label 
            
        return self.prior_mean, self.prior_variance, \
               posterior_mu, posterior_sigma, posterior_log_sigma, \
               word_dist, estimated_labels 
        
    def get_theta(self, x, x_bert, labels=None):
        with torch.no_grad():
            posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
            theta = F.softmax(self.reparameterize(posterior_mu, 2 * posterior_log_sigma), dim=1) # QUESTION: In the original code, they use posterior_log_sigma without multiple by 2?
            return theta
        
        
        
        
        