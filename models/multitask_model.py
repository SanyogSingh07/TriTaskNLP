import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_topics, num_sentiments, num_authors, stylo_dim=3):
        super(MultiTaskModel, self).__init__()
        
        # Shared Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Shared Encoder (BiLSTM)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Attention Layer (New addition)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # The shared representation combines the BiLSTM output with Stylometric features
        shared_dim = hidden_dim * 2 + stylo_dim
        
        # Task-Specific Heads
        self.topic_head = nn.Sequential(nn.Linear(shared_dim, 64), nn.ReLU(), nn.Linear(64, num_topics))
        self.sentiment_head = nn.Sequential(nn.Linear(shared_dim, 64), nn.ReLU(), nn.Linear(64, num_sentiments))
        self.author_head = nn.Sequential(nn.Linear(shared_dim, 64), nn.ReLU(), nn.Linear(64, num_authors))
        
    def forward(self, x, stylo):
        # x: (batch, seq_len), stylo: (batch, stylo_dim)
        embedded = self.embedding(x)
        lstm_out, _ = self.encoder(embedded)
        
        # Self-Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Inject stylometric features into the shared representation
        combined = torch.cat((context_vector, stylo), dim=1)
        
        # Task Predictions
        topic_out = self.topic_head(combined)
        sentiment_out = self.sentiment_head(combined)
        author_out = self.author_head(combined)
        
        return topic_out, sentiment_out, author_out, combined
