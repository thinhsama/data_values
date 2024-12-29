"""
NLP Datasets

Uses HuggingFace `transformers` and `datasets` as dependencies.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import DistilBertModel, DistilBertTokenizerFast

# Cache dictionary for loaded DataFrames
data_cache = {}

def ensure_directory_exists(directory):
    """Ensure the directory exists, create it if not."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_bert_embeddings(data_loader, model, tokenizer, device, save_path):
    """Convert text data to pooled DistilBERT embeddings."""
    embeddings = []

    for batch in tqdm(data_loader):
        # Tokenize input text
        inputs = tokenizer(batch, max_length=200, padding=True, truncation=True, return_tensors="pt").to(device)

        # Generate embeddings without gradients
        with torch.no_grad():
            output = model(**inputs)[0]
            pooled_embeddings = torch.mean(output, dim=1).cpu()

        embeddings.append(pooled_embeddings)

    # Save embeddings
    embeddings = torch.cat(embeddings, dim=0)
    torch.save(embeddings, save_path)
    return embeddings

def download_ag_news(cache_dir, force_download=False):
    """Download and prepare the AG News classification dataset."""
    ensure_directory_exists(cache_dir)
    file_path = Path(cache_dir) / "ag_news.csv"

    if "ag_news" in data_cache and not force_download:
        return data_cache["ag_news"]

    dataset = load_dataset("ag_news", split="train")
    texts = dataset["text"]
    labels = np.array(dataset["label"])

    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv(file_path, index=False)
    data_cache["ag_news"] = (df["text"].values, df["label"].values)
    return data_cache["ag_news"]

def download_yelp_reviews(cache_dir, force_download=False):
    """Download and prepare the Yelp Review dataset."""
    ensure_directory_exists(cache_dir)
    file_path = Path(cache_dir) / "yelp_reviews.csv"

    if "yelp_reviews" in data_cache and not force_download:
        return data_cache["yelp_reviews"]

    dataset = load_dataset("yelp_review_full", split="train")
    texts = dataset["text"]
    labels = np.array(dataset["label"])

    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv(file_path, index=False)
    data_cache["yelp_reviews"] = (df["text"].values, df["label"].values)
    return data_cache["yelp_reviews"]

def download_squad(cache_dir, force_download=False):
    """Download and prepare the SQuAD question answering dataset."""
    ensure_directory_exists(cache_dir)
    file_path = Path(cache_dir) / "squad.csv"

    if "squad" in data_cache and not force_download:
        return data_cache["squad"]

    dataset = load_dataset("squad", split="train")
    texts = [f"Context: {item['context']} Question: {item['question']}" for item in dataset]
    labels = [item['answers']['text'][0] if item['answers']['text'] else "" for item in dataset]

    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv(file_path, index=False)
    data_cache["squad"] = (df["text"].values, df["label"].values)
    return data_cache["squad"]

def download_trec(cache_dir, force_download=False):
    """Download and prepare the TREC question classification dataset."""
    ensure_directory_exists(cache_dir)
    file_path = Path(cache_dir) / "trec.csv"

    if "trec" in data_cache and not force_download:
        return data_cache["trec"]

    dataset = load_dataset("trec", split="train")
    texts = dataset["text"]
    labels = np.array(dataset["coarse_label"])

    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv(file_path, index=False)
    data_cache["trec"] = (df["text"].values, df["label"].values)
    return data_cache["trec"]

def download_amazon_reviews(cache_dir, force_download=False):
    """Download and prepare the Amazon Reviews polarity dataset."""
    ensure_directory_exists(cache_dir)
    file_path = Path(cache_dir) / "amazon_reviews.csv"

    if "amazon_reviews" in data_cache and not force_download:
        return data_cache["amazon_reviews"]

    dataset = load_dataset("amazon_polarity", split="train")
    texts = dataset["content"]
    labels = np.array(dataset["label"])

    df = pd.DataFrame({"text": texts, "label": labels})
    df.to_csv(file_path, index=False)
    data_cache["amazon_reviews"] = (df["text"].values, df["label"].values)
    return data_cache["amazon_reviews"]

def subsample_data(data, labels, num_samples=1000, random_state=None):
    """Randomly subsample a specified number of data points with a fixed random state."""
    rng = np.random.default_rng(seed=random_state)
    indices = rng.choice(len(data), num_samples, replace=False)
    return data[indices], labels[indices]

def prepare_embeddings(dataset, model_name="distilbert-base-uncased", cache_dir=".", batch_size=128):
    """Generate DistilBERT embeddings for a given dataset."""
    ensure_directory_exists(cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name).to(device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    save_path = Path(cache_dir) / "distilbert_embeddings.pt"
    return load_bert_embeddings(data_loader, model, tokenizer, device, save_path)

# Load and prepare AG News embeddings
cache_dir = "./cache"
ag_news_data, ag_news_labels = download_ag_news(cache_dir)
ag_news_data, ag_news_labels = subsample_data(ag_news_data, ag_news_labels, random_state=42)
ag_news_embeddings = prepare_embeddings(ag_news_data, cache_dir=cache_dir)

# Load and prepare Yelp Review embeddings
yelp_data, yelp_labels = download_yelp_reviews(cache_dir)
yelp_data, yelp_labels = subsample_data(yelp_data, yelp_labels, random_state=42)
yelp_embeddings = prepare_embeddings(yelp_data, cache_dir=cache_dir)

# Load and prepare SQuAD embeddings
squad_data, squad_labels = download_squad(cache_dir)
squad_data, squad_labels = subsample_data(squad_data, squad_labels, random_state=42)
squad_embeddings = prepare_embeddings(squad_data, cache_dir=cache_dir)

# Load and prepare TREC embeddings
trec_data, trec_labels = download_trec(cache_dir)
trec_data, trec_labels = subsample_data(trec_data, trec_labels, random_state=42)
trec_embeddings = prepare_embeddings(trec_data, cache_dir=cache_dir)

# Load and prepare Amazon Reviews embeddings
amazon_data, amazon_labels = download_amazon_reviews(cache_dir)
amazon_data, amazon_labels = subsample_data(amazon_data, amazon_labels, random_state=42)
amazon_embeddings = prepare_embeddings(amazon_data, cache_dir=cache_dir)
