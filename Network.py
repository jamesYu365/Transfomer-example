import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import MultiHeadAttention,PositionalEncoding
from utils import clone_module_list,get_training_data,subsequent_mask,FeedForward


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff:int,
                 heads:int,
                 bias:bool=True,
                 is_gated: bool = False,
                 bias_gate:bool=True,
                 activation=nn.ELU(),
                 dropout_prob: float=0.1):
        
        super(EncoderLayer, self).__init__()
        
        self.attn = MultiHeadAttention(heads,d_model,dropout_prob,bias)
        self.feed_forward = FeedForward(d_model,d_ff,dropout_prob,activation,
                                         is_gated,bias,bias_gate)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        z=self.norm1(x)
        a = self.attn(query=z,key=z,value=z)
        x = x+self.dropout(a)
        
        z=self.norm2(x)
        a = self.feed_forward(z)
        x = x+self.dropout(a)

        return x

class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff:int,
                 heads:int,
                 bias:bool=True,
                 is_gated: bool = False,
                 bias_gate:bool=True,
                 activation=nn.ELU(),
                 dropout_prob: float=0.1):
        
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttention(heads,d_model,dropout_prob,bias)
        self.attn2 = MultiHeadAttention(heads,d_model,dropout_prob,bias)
        self.feed_forward = FeedForward(d_model,d_ff,dropout_prob,activation,
                                         is_gated,bias,bias_gate)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc,mask_out,mask_enc):
        
        z=self.norm1(x)
        a = self.attn1(query=z,key=z,value=z,mask=mask_out)
        x = x+self.dropout(a)

        
        z=self.norm2(x)
        #出来的是query的大小，而且不需要再做masked attention
        a = self.attn2(query=z,key=enc,value=enc)
        x = x+self.dropout(a)
                
        z=self.norm3(x)
        a = self.feed_forward(z)
        x = x+self.dropout(a)
        return x


class Transformer(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff:int,
                 heads:int,
                 input_size:int,
                 output_size:int,
                 enc_seq_len: int,
                 dec_seq_len:int,
                 max_len:int=5000,
                 bias:bool=True,
                 is_gated: bool = False,
                 bias_gate:bool=True,
                 activation=nn.ELU(),
                 dropout_prob: float=0.1,
                 n_layers: int=6
                 ):
        
        super(Transformer, self).__init__()
        
        self.enc_seq_len = enc_seq_len
        
        #Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_layers):
            self.encs.append(EncoderLayer(d_model,d_ff,heads,bias,is_gated,bias_gate,
                                          activation,dropout_prob))
        
        self.decs = nn.ModuleList()
        for i in range(n_layers):
            self.decs.append(DecoderLayer(d_model,d_ff,heads,bias,is_gated,bias_gate,
                                          activation,dropout_prob))
        
        self.pos = PositionalEncoding(d_model,dropout_prob,max_len)
        
        #Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, d_model)
        self.dec_input_fc = nn.Linear(output_size, d_model)
        self.out_fc = nn.Linear(d_model,output_size)
    
    def forward(self, x,y,mask_out,mask_enc):
        #encoder
        e = self.encs[0](self.pos(self.enc_input_fc(x)))
        for enc in self.encs[1:]:
            e = enc(e)
        
        #decoder
        d = self.decs[0](self.dec_input_fc(y),e,mask_out,mask_enc)
        for dec in self.decs[1:]:
            d = dec(d, e,mask_out,mask_enc)
            
        #d shape is [dec_seq_len,batch,d_model]
        x = self.out_fc(d)
        
        return x.squeeze(-1)
