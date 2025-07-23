import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms
import argparse
import pandas as pd
import torchvision.datasets as datasets
import os
from PIL import Image
from coco_dataset import CustomCLIPDataset  # Import your custom dataset class
# --- Import your models ---
# Make sure your image_encoder.py and text_encoder.py files are in the same directory
# or are properly installed in your environment.
from image_encoder import ViT # Replace with your actual file/class name
from text_encoder import TextEncoder   # Replace with your actual file/class name
# --- Import the official CLIP Tokenizer from transformers ---
from transformers import CLIPTokenizer

# --- The Combined CLIP Model ---

class CLIPModel(nn.Module):
    """
    The main CLIP model that combines the image and text encoders.
    """
    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # The learnable temperature parameter `t`, as described in the paper.
        # Initialized to the equivalent of 0.07 from the paper.
        # We store it as a log-parameterized value for training stability.
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, texts):
        # Get the embeddings from each encoder
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)

        # L2 normalize the features to have unit length.
        # This is essential for calculating cosine similarity.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Clamp the temperature parameter to prevent it from becoming too large,
        # which can cause training instability.
        logit_scale = self.logit_scale.exp().clamp(max=100)

        return image_features, text_features, logit_scale

# --- Symmetric Cross-Entropy Loss (InfoNCE) ---

def symmetric_loss(image_features, text_features, logit_scale):
    """
    Calculates the symmetric InfoNCE loss for a batch of image and text features.
    """
    # Calculate the cosine similarity logits
    # The shape of logits will be [batch_size, batch_size]
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # Create the ground truth labels. For a batch of size N, the correct pairings
    # are at indices (0,0), (1,1), ..., (N-1, N-1), which are the diagonal elements.
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    # Calculate the cross-entropy loss for both directions
    loss_i = nn.functional.cross_entropy(logits_per_image, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)

    # The final loss is the average of the two directional losses
    return (loss_i + loss_t) / 2


# --- The Training Loop ---

def main(args):
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Instantiate Models ---
    # Instantiate your actual encoders here, passing any necessary arguments.
    # For example:
    # image_enc = ImageEncoder(embed_dim=args.embed_dim).to(device)
    # text_enc = TextEncoder(embed_dim=args.embed_dim, context_length=76, ...).to(device)
    image_enc = ViT(image_size=224,patch_size=16,dim=args.embed_dim,layer=12,heads=8,mlp_dim=1024,).to(device)
    text_enc = TextEncoder(context_length=77,vocab_size=49408,embed_dim=args.embed_dim,transformer_width=768,transformer_heads=8,transformer_layers=12).to(device)


    model = CLIPModel(image_enc, text_enc).to(device)



    # --- Dataloader ---
    # Replace YourDataset with your actual dataset class
    coco_raw_dataset = datasets.CocoCaptions(root=args.image_folder,
                                         annFile=args.captions_json,)
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    dataset=CustomCLIPDataset(
        coco_dataset=coco_raw_dataset,
        tokenizer=tokenizer
            )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # --- Optimizer and Scheduler ---
    # Use AdamW optimizer as mentioned in the paper
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Use a cosine learning rate schedule as mentioned in the paper
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader) * args.epochs)

    # --- Mixed-Precision Training Setup ---
    # This scaler helps prevent underflow issues with gradients in fp16.
    scaler = torch .GradScaler()

    # --- Training ---
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (images, texts) in enumerate(dataloader):
            images = images.to(device)
            texts = texts.to(device)
            optimizer.zero_grad()

            # Use torch.cuda.amp.autocast for the mixed-precision forward pass
            with torch.autocast(device_type=device):
                image_features, text_features, logit_scale = model(images, texts)
                loss = symmetric_loss(image_features, text_features, logit_scale)

            # Use the gradient scaler to scale the loss
            # This helps with stability in mixed-precision training.
            # Backward pass with the gradient scaler
            scaler.scale(loss).backward()  # Previously: loss.backward()     
            scaler.step(optimizer)       # Previously: optimizer.step()

            # Update the scaler for the next iteration
            scaler.update()  
            scheduler.step() # Update learning rate at each step

            total_loss += loss.item()
            
            if (i + 1) % args.log_steps == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- End of Epoch {epoch+1}, Average Loss: {avg_loss:.4f} ---")
        
        # Optional: Save a checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"clip_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Training Script")
    
    parser.add_argument("epochs",type=int,default=1, help="Number of training epochs")
    parser.add_argument("batch_size",type=int,default=128, help="Batch size for training")
    parser.add_argument("lr",type=float,default=5e-4, help="Learning rate")
    parser.add_argument("weight_decay",type=float,default=0.01, help="Weight decay for optimizer")
    parser.add_argument("embed_dim",type=int,default=512, help="Embedding dimension for both encoders")
    parser.add_argument("log_steps",type=int,default=100, help="Log training progress every N steps")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images")
    parser.add_argument("captions_json", type=str, help="Path to the CSV file containing captions")
    args = parser.parse_args()
    main(args)
