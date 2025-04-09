import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from tqdm import tqdm
import numpy as np

class VisualPromptTransformer(nn.Module):
    def __init__(self, 
                 model_name='vit_b_16', 
                 num_classes=102, 
                 prompt_length=5,
                 prompt_dropout=0.0,
                 frozen=True):
        """
        Visual Prompt Tuning for Vision Transformer
        Args:
            model_name: Name of the ViT model (default: 'vit_base_patch16_224')
            num_classes: Number of output classes
            prompt_length: Number of prompt tokens to add
            prompt_dropout: Dropout probability for prompt tokens
            frozen: Whether to freeze the pre-trained model
        """
        if model_name == 'vit_b_16':
            model_name = 'vit_base_patch16_224'
        super().__init__()
        
        # Load pretrained ViT model
        self.model = timm.create_model(model_name, pretrained=True)
        
        # Get embedding dimension from model
        embed_dim = self.model.embed_dim
        
        # Initialize prompt tokens (learnable parameters)
        self.prompt_tokens = nn.Parameter(torch.zeros(1, prompt_length, embed_dim))
        # Initialize with random values (better than zeros)
        nn.init.normal_(self.prompt_tokens, std=0.02)
        
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        self.prompt_length = prompt_length
        
        # Replace the head with a new one for our task
        self.model.head = nn.Linear(embed_dim, num_classes)
        
        # Freeze parameters of the original model if required
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze the head (classification layer)
            for param in self.model.head.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        # Get batch size
        B = x.shape[0]
        
        # Extract patch embeddings and position embeddings from the model
        x = self.model.patch_embed(x)  # (B, N, D)
        
        # Add position embeddings to patch embeddings
        cls_token = self.model.cls_token.expand(B, -1, -1)
        
        # Expand prompt tokens to batch size and apply dropout
        prompt_tokens = self.prompt_tokens.expand(B, -1, -1)
        prompt_tokens = self.prompt_dropout(prompt_tokens)
        
        # Concatenate [CLS] token, prompt tokens, and patch embeddings
        x = torch.cat([cls_token, prompt_tokens, x], dim=1)
        
        # Add positional embeddings (need to handle the extra prompt tokens)
        pos_embed = self.model.pos_embed
        
        # For the [CLS] token and patch embeddings, we use the original positional embeddings
        cls_pos_embed = pos_embed[:, 0:1, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        
        # For prompt tokens, we don't add positional embeddings (learnable prompts)
        x = torch.cat([
            cls_pos_embed + x[:, 0:1, :],  # [CLS] token + its pos embedding
            x[:, 1:self.prompt_length+1, :],  # Prompt tokens (no pos embedding)
            patch_pos_embed + x[:, self.prompt_length+1:, :]  # Patch embeddings + pos embeddings
        ], dim=1)
        
        # Continue with the rest of the ViT model
        x = self.model.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.model.blocks:
            x = blk(x)
        
        x = self.model.norm(x)
        
        # Take [CLS] token output for classification
        x = x[:, 0]
        
        # Apply classification head
        x = self.model.head(x)
        
        return x
