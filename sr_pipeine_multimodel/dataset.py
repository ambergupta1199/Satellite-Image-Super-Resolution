import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

class SatelliteDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(config.captions_path)
        
        # CLIP text processing
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = AutoModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load images
        lr = torch.load(row['lr_path'].replace('.png', '.pt'))
        hr = torch.load(row['hr_path'].replace('.png', '.pt'))
        
        # Text embedding
        inputs = self.tokenizer(
            row['caption'],
            return_tensors='pt',
            max_length=77,
            padding='max_length',
            truncation=True
        )
        with torch.no_grad():
            text_emb = self.text_encoder(**inputs).last_hidden_state.mean(dim=1)
            
        return lr, hr, text_emb.squeeze()