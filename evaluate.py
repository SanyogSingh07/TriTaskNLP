import torch
from models.multitask_model import MultiTaskModel
from data.dataset import get_dataloaders
from sklearn.metrics import accuracy_score

def evaluate():
    from utils.init_system import state
    from sklearn.metrics import accuracy_score
    
    device = state.device
    model = state.model
    val_loader = state.val_loader
    topic_map = state.topic_map
    sent_map = state.sent_map
    auth_map = state.auth_map
    
    assert device is not None
    assert model is not None
    assert val_loader is not None
    assert topic_map is not None
    assert sent_map is not None
    assert auth_map is not None
    
    print(f"Evaluating model on {device}...")
    
    if not hasattr(model, 'load_state_dict'):
        print("Error: Model initialization failed.")
        return
        
    model.eval()
    t_preds, s_preds, a_preds = [], [], []
    t_true, s_true, a_true = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            stylo = batch['stylo'].to(device)
            
            t_out, s_out, a_out, _ = model(input_ids, stylo)
            
            t_preds.extend(torch.argmax(t_out, dim=1).cpu().numpy())
            s_preds.extend(torch.argmax(s_out, dim=1).cpu().numpy())
            a_preds.extend(torch.argmax(a_out, dim=1).cpu().numpy())
            
            t_true.extend(batch['topic'].numpy())
            s_true.extend(batch['sentiment'].numpy())
            a_true.extend(batch['author'].numpy())
            
    topic_acc = accuracy_score(t_true, t_preds)*100
    sent_acc = accuracy_score(s_true, s_preds)*100
    auth_acc = accuracy_score(a_true, a_preds)*100
    
    print(f"Topic Accuracy:     {topic_acc:.2f}%")
    print(f"Sentiment Accuracy: {sent_acc:.2f}%")
    print(f"Author Accuracy:    {auth_acc:.2f}%")

    from utils.metrics import plot_confusion_matrix
    
    # Invert the maps to get class names
    inv_topic_map = {v: k for k, v in topic_map.items()}
    inv_sent_map = {v: k for k, v in sent_map.items()}
    inv_auth_map = {v: k for k, v in auth_map.items()}
    
    # Sort class names by index
    topic_classes = [inv_topic_map[i] for i in range(len(inv_topic_map))]
    sent_classes = [inv_sent_map[i] for i in range(len(inv_sent_map))]
    auth_classes = [inv_auth_map[i] for i in range(len(inv_auth_map))]

    print("\nPlotting Confusion Matrices...")
    plot_confusion_matrix(t_true, t_preds, topic_classes, title="Topic Confusion Matrix", filename="cm_topic.png")
    plot_confusion_matrix(s_true, s_preds, sent_classes, title="Sentiment Confusion Matrix", filename="cm_sentiment.png")
    plot_confusion_matrix(a_true, a_preds, auth_classes, title="Author Confusion Matrix", filename="cm_author.png")
    
    return {
        "Topic Accuracy": f"{topic_acc:.2f}%",
        "Sentiment Accuracy": f"{sent_acc:.2f}%",
        "Author Accuracy": f"{auth_acc:.2f}%"
    }
