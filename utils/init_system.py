import torch
from rich.console import Console

console = Console()

class SystemState:
    def __init__(self):
        self.device = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.criterion = None
        self.vocab = None
        self.topic_map = None
        self.sent_map = None
        self.auth_map = None

state = SystemState()

def initialize_system():
    console.print("\n⚙️  [bold green]Initializing System...[/bold green]\n")

    # -----------------------------
    # Device Check
    # -----------------------------
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"✔ Device: {state.device}")

    # -----------------------------
    # Load Data
    # -----------------------------
    from data.dataset import get_dataloaders
    state.train_loader, state.val_loader, state.vocab, state.topic_map, state.sent_map, state.auth_map = get_dataloaders(batch_size=32)
    console.print("✔ Dataset Ready")

    # -----------------------------
    # Load Model
    # -----------------------------
    from models.multitask_model import MultiTaskModel
    state.model = MultiTaskModel(
        vocab_size=len(state.vocab), 
        embed_dim=100, 
        hidden_dim=256, 
        num_topics=len(state.topic_map), 
        num_sentiments=len(state.sent_map), 
        num_authors=len(state.auth_map)
    ).to(state.device)

    try:
        checkpoint = torch.load("models/model.pth", map_location=state.device, weights_only=False)
        state.model.load_state_dict(checkpoint['model_state_dict'])
        console.print("✔ Pre-trained Model Loaded")
    except FileNotFoundError:
        console.print("✔ Initialized New Model (Untrained)")

    # -----------------------------
    # Optimizer + Loss
    # -----------------------------
    import torch.optim as optim
    import torch.nn as nn

    state.optimizer = optim.Adam(state.model.parameters(), lr=0.001)
    state.criterion = nn.CrossEntropyLoss()
    console.print("✔ Optimizer Ready\n")

    return state
