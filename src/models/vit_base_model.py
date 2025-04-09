import torch
import torch.nn as nn
import torchvision.models as models


class VisualPromptTransformer(nn.Module):
    def __init__(self, num_prompts=10, embedding_dim=768, num_classes=10):
        super().__init__()
        
        # Load pre-trained ViT model
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Freeze all parameters of the model
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Create learnable prompt tokens
        self.prompt_tokens = nn.Parameter(torch.zeros(1, num_prompts, embedding_dim))
        # Initialize with random values
        nn.init.normal_(self.prompt_tokens, std=0.02)
        
        # Replace the classification head for the new task
        self.vit.heads = nn.Linear(embedding_dim, num_classes)
        # Only unfreeze the classification head
        for param in self.vit.heads.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        # Get batch size
        batch_size = x.shape[0]
        
        # The original ViT processes the input through a patch embedding layer
        # and adds a class token at the beginning
        x = self.vit.patch_embed(x)
        cls_token = self.vit.class_token.expand(batch_size, -1, -1)
        
        # Expand prompts to batch size and concatenate with input sequence
        prompts = self.prompt_tokens.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, prompts, x), dim=1)
        
        # Add position embeddings (we may need to interpolate position embeddings
        # since we've changed sequence length)
        if self.vit.pos_embed is not None:
            pos_embed = self.vit.interpolate_pos_encoding(x, self.vit.pos_embed)
            x = x + pos_embed
            
        # Pass through the transformer encoder
        x = self.vit.encoder(x)
        
        # Classification based on class token
        x = x[:, 0]
        x = self.vit.heads(x)
        return x