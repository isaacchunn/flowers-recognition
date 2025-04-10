import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from tqdm import tqdm
import numpy as np

class VPT(nn.Module):
    def __init__(self, prompt_length=50,  num_classes=102, model_name='vit_b_16',prompt_dropout=0.0):
        super(VPT, self).__init__()

        # create vit model
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

        # Learanble parameters
        self.cls_token = self.vit.cls_token # (1,1,768); trainable weights
        self.prompt_tokens = nn.Parameter(torch.zeros(1, prompt_length, self.vit.embed_dim)) # (1, prompt_length, 768)
        nn.init.uniform_(self.prompt_tokens, -0.1, 0.1)

        # 
        self.prompt_dropout = nn.Dropout(prompt_dropout)

        # replace the head with a new fully connected layer
        self.vit.head = nn.Linear(self.vit.embed_dim,  num_classes)


        # Freeze vit parameters
        for param in self.vit.parameters():
            param.requires_grad = False
        # Unfreeze the head for fine-tuning
        for param in self.vit.head.parameters():
            param.requires_grad = True

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings


        B = x.shape[0]

        # Process the input through patch embedding 
        x = self.vit.patch_embed(x) # x shape = (B, 196, 768); 196 = (224/16)**2 = 14**2 = 196; 768 = d

        # Get cls_token
        cls_token = self.vit.cls_token.expand(B,-1,-1) # cls_token = (B, 1, 768)

        # concate cls_token with patch_embedding
        x = torch.cat((cls_token, x), dim = 1) # x shape = (B, 196 + 1, 768)

        # Add position embeddings
        x = x + self.vit.pos_embed # x shape = (B, 197, 768)


        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_tokens).expand(B, -1, -1),
                x[:, 1:, :]
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        # (B, 1 + 50 + 196, 768)

        
        return x

    def forward(self,x):
        # e.g. x.shape = (B, 3, 224, 224)

        x = self.incorporate_prompt(x)

       # Pass through the transformer blocks
        for blk in self.vit.blocks:
            x = blk(x)
        
        # Apply normalization
        x = self.vit.norm(x)
        
        # Extract CLS token
        x = x[:, 0]

        # Classification head
        logits = self.vit.head(x)

        return logits
    
    

