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
    
    



# class VisualPromptTransformer(nn.Module):
#     def __init__(self, 
#                  model_name='vit_b_16', 
#                  num_classes=102, 
#                  prompt_length=5,
#                  prompt_dropout=0.0,
#                  frozen=True):
#         """
#         Visual Prompt Tuning for Vision Transformer
#         Args:
#             model_name: Name of the ViT model (default: 'vit_base_patch16_224')
#             num_classes: Number of output classes
#             prompt_length: Number of prompt tokens to add
#             prompt_dropout: Dropout probability for prompt tokens
#             frozen: Whether to freeze the pre-trained model
#         """
#         if model_name == 'vit_b_16':
#             model_name = 'vit_base_patch16_224'
#         super().__init__()
        
#         # Load pretrained ViT model
#         self.model = timm.create_model(model_name, pretrained=True)
        
#         # Get embedding dimension from model
#         embed_dim = self.model.embed_dim
        
#         # Initialize prompt tokens (learnable parameters)
#         self.prompt_tokens = nn.Parameter(torch.zeros(1, prompt_length, embed_dim))
#         # Initialize with random values (better than zeros)
#         nn.init.normal_(self.prompt_tokens, std=0.02)
        
#         self.prompt_dropout = nn.Dropout(prompt_dropout)
#         self.prompt_length = prompt_length
        
#         # Replace the head with a new one for our task
#         self.model.head = nn.Linear(embed_dim, num_classes)
        
#         # Freeze parameters of the original model if required
#         if frozen:
#             for param in self.model.parameters():
#                 param.requires_grad = False
#             # Unfreeze the head (classification layer)
#             for param in self.model.head.parameters():
#                 param.requires_grad = True
    
#     def forward(self, x):
#         # Get batch size
#         B = x.shape[0]
        
#         # Extract patch embeddings and position embeddings from the model
#         x = self.model.patch_embed(x)  # (B, N, D)
        
#         # Add position embeddings to patch embeddings
#         cls_token = self.model.cls_token.expand(B, -1, -1)
        
#         # Expand prompt tokens to batch size and apply dropout
#         prompt_tokens = self.prompt_tokens.expand(B, -1, -1)
#         prompt_tokens = self.prompt_dropout(prompt_tokens)
        
#         # Concatenate [CLS] token, prompt tokens, and patch embeddings
#         x = torch.cat([cls_token, prompt_tokens, x], dim=1)
        
#         # Add positional embeddings (need to handle the extra prompt tokens)
#         pos_embed = self.model.pos_embed
        
#         # For the [CLS] token and patch embeddings, we use the original positional embeddings
#         cls_pos_embed = pos_embed[:, 0:1, :]
#         patch_pos_embed = pos_embed[:, 1:, :]
        
#         # For prompt tokens, we don't add positional embeddings (learnable prompts)
#         x = torch.cat([
#             cls_pos_embed + x[:, 0:1, :],  # [CLS] token + its pos embedding
#             x[:, 1:self.prompt_length+1, :],  # Prompt tokens (no pos embedding)
#             patch_pos_embed + x[:, self.prompt_length+1:, :]  # Patch embeddings + pos embeddings
#         ], dim=1)
        
#         # Continue with the rest of the ViT model
#         x = self.model.pos_drop(x)
        
#         # Apply transformer blocks
#         for blk in self.model.blocks:
#             x = blk(x)
        
#         x = self.model.norm(x)
        
#         # Take [CLS] token output for classification
#         x = x[:, 0]
        
#         # Apply classification head
#         x = self.model.head(x)
        
#         return x


