import torch
import sys
import os
# Dynamically add the parent TriTaskNLP folder to the Python path so this file can be debugged directly!
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multitask_model import MultiTaskModel
from utils.preprocessing import preprocess
from utils.stylometry import extract_features

def predict(text):
    from utils.init_system import state
    
    device = state.device
    model = state.model
    vocab = state.vocab
    topic_map = state.topic_map
    sent_map = state.sent_map
    auth_map = state.auth_map
    
    if not hasattr(model, 'load_state_dict'):
        print("❌ Error: Model initialization failed.")
        return {"Error": "Model initialization failed"}
        
    model.eval()
    
    tokens = preprocess(text)
    token_ids = [vocab.get(t, 1) for t in tokens[:50]]
    token_ids += [0] * (50 - len(token_ids))
    stylo = extract_features(text)
    
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    stylo_tensor = torch.tensor([stylo], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        t_out, s_out, a_out, _ = model(input_ids, stylo_tensor)
        
    rev_topic = {v: k for k, v in topic_map.items()}
    rev_sent = {v: k for k, v in sent_map.items()}
    rev_auth = {v: k for k, v in auth_map.items()}
    
    t_pred = rev_topic[torch.argmax(t_out, dim=1).item()]
    s_pred = rev_sent[torch.argmax(s_out, dim=1).item()]
    a_pred = rev_auth[torch.argmax(a_out, dim=1).item()]
    
    result = {
        "Topic": f"{t_pred.capitalize()} ({torch.softmax(t_out, dim=1).max().item()*100:.1f}%)",
        "Sentiment": f"{s_pred.capitalize()} ({torch.softmax(s_out, dim=1).max().item()*100:.1f}%)",
        "Author": f"{a_pred.capitalize()} ({torch.softmax(a_out, dim=1).max().item()*100:.1f}%)"
    }
    
    return result
