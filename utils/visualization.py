import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from models.multitask_model import MultiTaskModel
from data.dataset import get_dataloaders

def get_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, vocab, topic_map, sent_map, auth_map = get_dataloaders(batch_size=32)
    
    try:
        checkpoint = torch.load("models/model.pth", map_location=device, weights_only=False)
    except FileNotFoundError:
        print("❌ Error: Model not found. Train the model first to extract embeddings.")
        return None, None, None
        
    model = MultiTaskModel(
        vocab_size=len(vocab), embed_dim=100, hidden_dim=256, 
        num_topics=len(topic_map), num_sentiments=len(sent_map), num_authors=len(auth_map)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    embeddings = []
    topic_labels = []
    sentiment_labels = []
    
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            stylo = batch['stylo'].to(device)
            topic = batch['topic']
            sentiment = batch['sentiment']
            
            _, _, _, shared_repr = model(input_ids, stylo)
            embeddings.append(shared_repr.cpu().numpy())
            topic_labels.append(topic.numpy())
            sentiment_labels.append(sentiment.numpy())
            if len(embeddings) > 10: break # Keep it fast
            
    return np.vstack(embeddings), np.concatenate(topic_labels), np.concatenate(sentiment_labels)

# -----------------------------
# NVIDIA Theme Setup
# -----------------------------
def apply_nvidia_theme():
    plt.style.use('dark_background')  # dark base

    plt.rcParams.update({
        "axes.edgecolor": "#76B900",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "figure.facecolor": "#0b0b0b",
        "axes.facecolor": "#0b0b0b",
        "grid.color": "#444444"
    })

# -----------------------------
# Label Coloring (Green Gradient)
# -----------------------------
def get_green_colors(labels):
    unique_labels = np.unique(labels)
    color_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Normalize into values for colormap
    colors = [color_map[l] / max(1, len(unique_labels) - 1) for l in labels]
    return colors

def get_nvidia_cmap():
    # Ranging from deep green to bright NVIDIA green for visibility on dark background
    return mcolors.LinearSegmentedColormap.from_list("NVIDIAGreens", ["#1a4d00", "#76B900", "#d1ff66"])

# -----------------------------
# 1. PCA (NVIDIA Style)
# -----------------------------
def plot_pca(X, labels, label_name="Label", filename="pca_visualization.png"):
    print(f"📉 Running PCA for {label_name}...")
    apply_nvidia_theme()

    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)

    colors = get_green_colors(labels)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap=get_nvidia_cmap(), alpha=0.8, edgecolors='#0b0b0b', linewidth=0.5)

    plt.title(f"PCA Visualization ({label_name})", color="#76B900", fontweight="bold")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    
    unique = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=get_nvidia_cmap()(i / max(1, len(unique) - 1)), markeredgecolor='#0b0b0b', markersize=10, label=str(u)) for i, u in enumerate(unique)]
    plt.legend(handles=handles, title=label_name, facecolor='#0b0b0b', edgecolor='#76B900', labelcolor='white')

    os.makedirs("plots", exist_ok=True)
    save_path = os.path.abspath(f"plots/{filename}")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    if os.name == 'nt':
        os.startfile(save_path)

# -----------------------------
# 2. t-SNE (BEST VISUAL - NVIDIA Style)
# -----------------------------
def plot_tsne(X, labels, label_name="Label", filename="tsne_visualization.png"):
    print(f"🔥 Running t-SNE for {label_name}...")
    apply_nvidia_theme()

    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    colors = get_green_colors(labels)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap=get_nvidia_cmap(), alpha=0.8, edgecolors='#0b0b0b', linewidth=0.5)

    plt.title(f"t-SNE Visualization ({label_name})", color="#76B900", fontweight="bold")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, alpha=0.3)
    
    unique = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=get_nvidia_cmap()(i / max(1, len(unique) - 1)), markeredgecolor='#0b0b0b', markersize=10, label=str(u)) for i, u in enumerate(unique)]
    plt.legend(handles=handles, title=label_name, facecolor='#0b0b0b', edgecolor='#76B900', labelcolor='white')

    os.makedirs("plots", exist_ok=True)
    save_path = os.path.abspath(f"plots/{filename}")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    if os.name == 'nt':
        os.startfile(save_path)

# -----------------------------
# 3. K-Means (NVIDIA Style)
# -----------------------------
def plot_kmeans(X, k=4):
    print("📊 Running K-Means Clustering...")
    apply_nvidia_theme()

    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    colors = get_green_colors(labels)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors, cmap=get_nvidia_cmap(), alpha=0.7, edgecolors='#0b0b0b', linewidth=0.5)

    # Centroids in bright NVIDIA green
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker='X',
        s=300,
        c='#d1ff66',
        edgecolors='white',
        linewidths=2,
        zorder=10
    )

    plt.title(f"K-Means Clustering (k={k})", color="#76B900", fontweight="bold")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, alpha=0.3)

    os.makedirs("plots", exist_ok=True)
    save_path = os.path.abspath("plots/kmeans_nvidia.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    if os.name == 'nt':
        os.startfile(save_path)

# -----------------------------
# 4. Combined Visualization
# -----------------------------
def plot_all():
    print("Generating Visual Dashboard...")

    print("Extracting representations from PyTorch model...")
    X, topic_labels, sentiment_labels = get_embeddings()
    if X is None: return

    print("Applying NVIDIA-Styled Visualizations")
    apply_nvidia_theme()

    print("\n--- Topic Visualizations ---")
    plot_pca(X, topic_labels, "Topic", "pca_topic.png")
    plot_tsne(X, topic_labels, "Topic", "tsne_topic.png")

    print("\n--- Sentiment Visualizations ---")
    plot_pca(X, sentiment_labels, "Sentiment", "pca_sentiment.png")
    plot_tsne(X, sentiment_labels, "Sentiment", "tsne_sentiment.png")

    print("\n--- Clustering Comparison ---")
    plot_kmeans(X)
    print("All NVIDIA-style visualisations successfully plotted and saved!")
