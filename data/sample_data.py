import pandas as pd
from sklearn.datasets import fetch_20newsgroups

def load_data():
    categories = [
        'comp.sys.ibm.pc.hardware', 
        'misc.forsale',
        'rec.sport.baseball', 
        'talk.politics.misc'
    ]
    
    # fetch_20newsgroups caches the dataset automatically
    newsgroups_data, newsgroups_target = fetch_20newsgroups(
        subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), return_X_y=True
    )
    
    topic_map = {
        0: "tech",          # comp.sys.ibm.pc.hardware
        1: "entertainment", # misc.forsale (used as a proxy for entertainment/misc)
        2: "sports",        # rec.sport.baseball
        3: "politics"       # talk.politics.misc
    }
    
    data = []
    for i, text in enumerate(newsgroups_data):
        if not text.strip():
            continue
            
        topic = topic_map[newsgroups_target[i]]
        
        # Synthetic Sentiment
        pos_words = ["good", "great", "excellent", "win", "like", "love"]
        neg_words = ["bad", "terrible", "awful", "loss", "problem", "hate"]
        text_lower = text.lower()
        if sum(1 for w in pos_words if w in text_lower) > sum(1 for w in neg_words if w in text_lower):
            sentiment = "positive"
        elif sum(1 for w in neg_words if w in text_lower) > sum(1 for w in pos_words if w in text_lower):
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        # Synthetic Author
        words_len = len(text.split())
        if words_len < 50: author = "author1"
        elif words_len < 150: author = "author2"
        else: author = "author3"
        
        data.append((text, topic, sentiment, author))
        
        if len(data) >= 1000: # Limit size to keep demo fast
            break
            
    df = pd.DataFrame(data, columns=["text", "topic", "sentiment", "author"])
    return df
