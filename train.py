import torch
import torch.nn as nn
from models.multitask_model import MultiTaskModel
from data.dataset import get_dataloaders

from utils.dashboard import train_with_dashboard

def train():
    from utils.init_system import state
    from utils.dashboard import train_with_dashboard
    
    device = state.device
    model = state.model
    train_loader = state.train_loader
    vocab = state.vocab
    optimizer = state.optimizer
    criterion = state.criterion
    
    assert device is not None
    assert model is not None
    assert train_loader is not None
    assert vocab is not None
    assert optimizer is not None
    assert criterion is not None
    assert state.topic_map is not None
    assert state.sent_map is not None
    assert state.auth_map is not None
    
    print(f"Training initiated on: {device}")
    
    train_with_dashboard(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=5,
        device=device,
        vocab=vocab,
        maps=(state.topic_map, state.sent_map, state.auth_map)
    )
