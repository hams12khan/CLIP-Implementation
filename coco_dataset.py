import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomCLIPDataset(Dataset):
    """
    A PyTorch Dataset for MS COCO, compatible with CLIP training.
    It takes a pre-initialized torchvision CocoCaptions dataset.
    """
    def __init__(self, coco_dataset, tokenizer, image_transform=None):
        """
        Args:
            coco_dataset: An instance of torchvision.datasets.CocoCaptions.
            tokenizer: An instance of the CLIPTokenizer.
            image_transform (callable, optional): Optional transform for images.
        """
        self.coco_dataset = coco_dataset
        self.tokenizer = tokenizer
        
        # Use a default image transformation if none is provided.
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                     std=(0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.image_transform = image_transform

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Get image and captions from the CocoCaptions dataset
        # The image is already a PIL Image object.
        # captions is a list of strings.
        image, captions = self.coco_dataset[idx]

        # 2. Randomly select one caption from the list
        caption = random.choice(captions)
        
        # 3. Tokenize the selected caption
        tokenized_output = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        text_tokens = tokenized_output['input_ids'].squeeze()

        # 4. Apply image transformation
        # The original image from COCO is already an RGB PIL Image
        image_tensor = self.image_transform(image)

        return image_tensor, text_tokens