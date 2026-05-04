import torch
from torch.utils.data import Dataset, DataLoader
from data.sample_data import load_data
from utils.preprocessing import preprocess
from utils.stylometry import extract_features

class TriTaskDataset(Dataset):
    def __init__(self, df, vocab, max_len=50):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        
        self.topic_map = {t: i for i, t in enumerate(df['topic'].unique())}
        self.sent_map = {s: i for i, s in enumerate(df['sentiment'].unique())}
        self.auth_map = {a: i for i, a in enumerate(df['author'].unique())}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        
        tokens = preprocess(text)
        token_ids = [self.vocab.get(t, 1) for t in tokens[:self.max_len]]
        # Pad sequence
        token_ids += [0] * (self.max_len - len(token_ids))
        
        stylo = extract_features(text)
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "stylo": torch.tensor(stylo, dtype=torch.float32),
            "topic": torch.tensor(self.topic_map[row['topic']], dtype=torch.long),
            "sentiment": torch.tensor(self.sent_map[row['sentiment']], dtype=torch.long),
            "author": torch.tensor(self.auth_map[row['author']], dtype=torch.long),
            "text": text
        }

def get_dataloaders(batch_size=32):
    df = load_data()
    
    # Simple Vocab builder
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in df['text']:
        for t in preprocess(text):
            if t not in vocab:
                vocab[t] = len(vocab)
                
    dataset = TriTaskDataset(df, vocab)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Optimized Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    return train_loader, val_loader, vocab, dataset.topic_map, dataset.sent_map, dataset.auth_map
