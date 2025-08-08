import torch 
import torch.nn as nn
import math
class InputEmbeddings(nn.Module):
    
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
        
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    
    def __init__(self,d_model:int,seq_len:int,dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) 
        
        #Create a matrix of shape (seq,len,d_model)
        
        pe = torch.zeros(seq_len,d_model)
        #Create a vector of shape(seq_len,1)
        position = torch.arrange(0,seq_len,dtype= torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        #Apply the sign to even pos 
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe',pe)
    def forward(self,x):
        x= x+ (self.pe[:,:x.shape[[1],:]]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    
    def __init__(self,eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) 
        self.bias = nn.Parameter(torch.zeroes(1))
        
    def forward(self,x):
        mean = x.mean(dim = -1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        return self.alpha * (x-mean) / (std +self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    
    def __init__(self,d_model:int,d_ff :int,dropout:float)->None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)
    
    def forward(self,x):

        return self.linear_3(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float) ->None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model %h ==0, "d_model is not divisible by h"
        
        self.d_k = d_model //h
        self.w_q = nn.Linear(d_model,d_model) 
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        
    def
         