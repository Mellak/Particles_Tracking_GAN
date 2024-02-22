import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
import torchvision
import time
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print('self.pe', x.shape, self.pe[:x.size(0), :].shape, self.pe[:x.size(1), :].unsqueeze(0).shape)
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)
    
class Transformer2(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=0.1, layer_norm_eps=1e-15, norm_first=True) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x, mask=None):
        # x should have shape (seq_len, batch_size, dim)
        x = x.permute(1, 0, 2)  # Adjust the dimension for nn.TransformerEncoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.permute(1, 0, 2)  # Change back the dimension to (batch_size, seq_len, dim)
        return x
    
# define a function that map a mask of 0 and 1 to a mask of -inf and 0:
def generate_sa_mask(mask):
    mask = mask.masked_fill(mask==0, float('-inf'))
    mask = mask.masked_fill(mask==1, 0)
    return mask

def generate_padding_mask(mask):
    mask = 1 - mask
    # make it type BoolTensor:
    mask = mask.bool()
    return mask


class GeneratorNumIntEnergyDirection2(nn.Module):
    def __init__(self, seq_len=18, patch_size=1, channels=4, latent_dim=100, embed_dim=64, depth=3,
                 num_heads=2, mlp_dim=128, forward_drop_rate=0.3, attn_drop_rate=0.3, lstm=False):
        super(GeneratorNumIntEnergyDirection2, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        self.condtion_embed_dim = 10
        self.mask_lenght = seq_len
        
        self.embedding_num_inter = nn.Embedding(seq_len+1 ,self.condtion_embed_dim)
        self.l1 = nn.Linear(self.latent_dim + self.condtion_embed_dim + self.mask_lenght + 3 + 1 , self.seq_len * self.embed_dim)
        #self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.positional_encoding = PositionalEncoding(self.embed_dim, self.seq_len)
        
        self.blocks = Transformer2(self.embed_dim, self.depth, num_heads, mlp_dim)
        
        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        )
        
        #self.deconv_mlp = nn.Sequential(nn.Linear(self.embed_dim, 128),nn.PReLU(),nn.Linear(128, self.channels))

    def forward(self, z, num_inter, energy, masks, start_vector):
        num_inter_embed = self.embedding_num_inter(num_inter)
        
        z = torch.cat((z, num_inter_embed, masks, start_vector, energy.unsqueeze(-1)), dim=1)

        x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        x = self.positional_encoding(x)

        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        #output = self.deconv_mlp(x)
        #output = output.permute(0, 3, 1, 2)
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, H, W)

        #---Post processing---
        output[:, 0, 0, 0]  = energy

        output[:, 1:, :, 0] = torch.zeros_like(output[:, 1:, :, 0])
        
        final_energy_mask = torch.ones(num_inter.size(0), self.mask_lenght, dtype=torch.int).to('cuda')
        _num_inter = num_inter - 1
        final_energy_mask[torch.arange(_num_inter.size(0)).unsqueeze(1), _num_inter.unsqueeze(1)] = 0

        output[:, 0, 0, :] = output[:, 0, 0, :] * final_energy_mask

        output = output[:,:,0,:].transpose(1,2) * masks.unsqueeze(-1)

        output = output.transpose(1,2).reshape(-1, self.channels, H, W)

        return output
'''     
class GeneratorNumIntEnergyDirection2(nn.Module):
    def __init__(self, seq_len=18, patch_size=1, channels=4, latent_dim=100, embed_dim=64, depth=3,
                 num_heads=2, mlp_dim=128, forward_drop_rate=0.3, attn_drop_rate=0.3, lstm=False):
        super(GeneratorNumIntEnergyDirection2, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        self.condtion_embed_dim = 10
        self.mask_lenght = seq_len
        
        self.embedding_num_inter = nn.Embedding(seq_len+1 ,self.condtion_embed_dim)
        self.l1 = nn.Linear(self.latent_dim + self.condtion_embed_dim + self.mask_lenght + 3 +1 , self.seq_len * self.embed_dim)
        #self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.positional_encoding = PositionalEncoding(self.embed_dim, self.seq_len)
        
        self.blocks = Transformer2(self.embed_dim, self.depth, num_heads, mlp_dim)
        
        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        )
        
        #self.deconv_mlp = nn.Sequential(nn.Linear(self.embed_dim, 128),nn.PReLU(),nn.Linear(128, self.channels))

    def forward(self, z, num_inter, energy, masks, start_vector):
        num_inter_embed = self.embedding_num_inter(num_inter)
        z = torch.cat((z, num_inter_embed, masks, start_vector, energy.unsqueeze(-1)), dim=1)

        x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        x = self.positional_encoding(x)

        H, W = 1, self.seq_len
        x = self.blocks(x, mask=generate_padding_mask(masks))
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        #output = self.deconv_mlp(x)
        #output = output.permute(0, 3, 1, 2)
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, H, W)

        #---Post processing---
        output[:, 0, 0, 0]  = 0*output[:, 0, 0, 0] + 1*energy

        output[:, 1:, :, 0] = output[:, 1:, :, 0] * 0 #torch.zeros_like(output[:, 1:, :, 0])
        
        final_energy_mask = torch.ones(num_inter.size(0), self.mask_lenght, dtype=torch.int).to('cuda')
        _num_inter = num_inter - 1
        final_energy_mask[torch.arange(_num_inter.size(0)).unsqueeze(1), _num_inter.unsqueeze(1)] = 0

        output[:, 0, 0, :] = output[:, 0, 0, :] * final_energy_mask

        output = output[:,:,0,:].transpose(1,2) * masks.unsqueeze(-1)

        output = output.transpose(1,2).reshape(-1, self.channels, H, W)

        return output'''
 
class ViTwMask2(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) #** 2
        patch_dim = (channels+2) * patch_size #** 2
        #self.cdt_embed_dim = 

        self.patch_size = patch_size
        self.embed_energy_vector = nn.Linear(1, image_size)
        self.embedding_cdt = nn.Embedding(image_size+1 ,image_size)
        self.positional_encoding = PositionalEncoding(dim, image_size+1)
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer2(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.PReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, cdt, energy, mask=None):

        cdt = self.embedding_cdt(cdt)
        energy_vector = self.embed_energy_vector(energy.unsqueeze(1))
        img = torch.cat((img, cdt.unsqueeze(1).unsqueeze(1), energy_vector.unsqueeze(1).unsqueeze(1)),dim=1)


        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=p)
        x = self.patch_to_embedding(x)


        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # add one on the top of the mask by cat:
        mask = torch.cat((torch.ones(mask.size(0),1, dtype=torch.int).to('cuda'), mask), dim=1)
        x = self.positional_encoding(x)
        x = self.transformer(x, mask=generate_padding_mask(mask))

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
    
def main():
    pass

def generate_random_tensor(batch_number):
    lower_bound = 3
    upper_bound = 18
    tensor = torch.randint(lower_bound, upper_bound + 1, size=(batch_number,), dtype=torch.int)
    return tensor

# Function to generate masks of shape (batch_size, 18)
def generate_random_mask(batch_number):
    tensor = torch.randint(0, 2, size=(batch_number, 18), dtype=torch.int)
    return tensor

if __name__ == '__main__':
    pass