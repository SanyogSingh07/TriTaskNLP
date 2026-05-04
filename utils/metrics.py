import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix", filename="confusion_matrix.png"):
    # Apply dark background matching our NVIDIA styling if available
    try:
        from utils.visualization import apply_nvidia_theme
        apply_nvidia_theme()
    except ImportError:
        pass

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Try to use NVIDIA Greens, fallback to default 'Greens'
    try:
        from utils.visualization import get_nvidia_cmap
        cmap = get_nvidia_cmap()
    except ImportError:
        cmap = "Greens"
        
    im = ax.imshow(cm, cmap=cmap)

    # Labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    plt.xlabel("Predicted", color="#76B900", fontweight="bold")
    plt.ylabel("Actual", color="#76B900", fontweight="bold")
    plt.title(title, color="#76B900", fontweight="bold")

    # Add values inside cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            # If dark background, use white text. If light cell, use black.
            color = "black" if cm[i, j] > cm.max() / 2 else "white"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontweight="bold")

    os.makedirs("plots", exist_ok=True)
    save_path = os.path.abspath(f"plots/{filename}")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    if os.name == 'nt':
        os.startfile(save_path)
