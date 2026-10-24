import sys
import os
from os.path import join, exists
import pickle
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import arxiv
import faiss
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import data_dir, thumbnail_dir

os.makedirs(data_dir, exist_ok=True)
os.makedirs(thumbnail_dir, exist_ok=True)

def parse_authors(authors):
    parsed = []
    for a in authors:
        parts = a.name.strip().split()
        if len(parts) == 0:
            continue
        last_name = parts[-1]
        first_name = " ".join(parts[:-1]) if len(parts) > 1 else ""
        parsed.append([last_name, first_name, ""])
    return str(parsed)

def fetch_papers():
    client = arxiv.Client(page_size=100, delay_seconds=1.0, num_retries=3)
    papers = []
    seen_ids = set()

    # Landmark seminal papers
    seminal_ids = [
        "1706.03762",  # Attention Is All You Need
        "1512.03385",  # ResNet
        "1810.04805",  # BERT
        "1412.6980",   # Adam Optimizer
        "1406.2661",   # GANs
        "2005.14165",  # GPT-3
        "2006.11239",  # DDPM Diffusion
        "2106.09685",  # LoRA
        "2302.13971",  # LLaMA
        "1908.10084",  # Sentence-BERT
        "2103.00020",  # CLIP
        "2112.10752",  # Latent Diffusion / Stable Diffusion
        "2203.02155",  # InstructGPT
        "1301.3781",   # Word2Vec
        "1506.01497",  # Faster R-CNN
        "1910.10683",  # T5
        "2205.01068",  # OPT
        "2210.03629",  # ReAct: Synergizing Reasoning and Acting
        "2305.18290",  # DPO: Direct Preference Optimization
        "2001.08361",  # Scaling Laws for Neural Language Models
    ]

    print("[+] Fetching seminal papers...")
    try:
        search = arxiv.Search(id_list=seminal_ids)
        for r in client.results(search):
            clean_id = r.get_short_id().split("v")[0]
            if clean_id not in seen_ids:
                seen_ids.add(clean_id)
                papers.append({
                    "id": clean_id,
                    "title": r.title.strip().replace("\n", " "),
                    "abstract": r.summary.strip().replace("\n", " "),
                    "doi": r.doi if r.doi else "",
                    "categories": " ".join(r.categories),
                    "update_date": r.updated.strftime("%Y-%m-%d"),
                    "authors_parsed": parse_authors(r.authors)
                })
    except Exception as e:
        print(f"[-] Warning fetching seminal papers: {e}")

    # Category searches to populate diverse scientific domains
    categories = [
        ("cat:cs.AI", 15),
        ("cat:cs.LG", 15),
        ("cat:cs.CV", 15),
        ("cat:cs.CL", 15),
        ("cat:cs.RO", 12),
        ("cat:cs.CR", 12),
        ("cat:quant-ph", 12),
        ("cat:astro-ph", 12),
        ("cat:hep-th", 12),
        ("cat:stat.ML", 12),
    ]

    for cat_query, max_res in categories:
        print(f"[+] Fetching up to {max_res} papers for {cat_query}...")
        try:
            search = arxiv.Search(
                query=cat_query,
                max_results=max_res,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            for r in client.results(search):
                clean_id = r.get_short_id().split("v")[0]
                if clean_id not in seen_ids:
                    seen_ids.add(clean_id)
                    papers.append({
                        "id": clean_id,
                        "title": r.title.strip().replace("\n", " "),
                        "abstract": r.summary.strip().replace("\n", " "),
                        "doi": r.doi if r.doi else "",
                        "categories": " ".join(r.categories),
                        "update_date": r.updated.strftime("%Y-%m-%d"),
                        "authors_parsed": parse_authors(r.authors)
                    })
        except Exception as e:
            print(f"[-] Warning fetching {cat_query}: {e}")

    df = pd.DataFrame(papers)
    print(f"[+] Total papers fetched: {len(df)}")
    return df

def build_index(df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Loading SentenceTransformer model on {device}...")
    model = SentenceTransformer("all-mpnet-base-v2", device=device)

    print("[+] Generating embeddings for papers...")
    text_to_embed = (df["title"] + "\n " + df["abstract"]).tolist()
    embeddings = model.encode(text_to_embed, show_progress_bar=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    print(f"[+] Creating FAISS index (dim={dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    ids = df["id"].tolist()

    # Save to data directory
    csv_path = join(data_dir, "arxiv_processed.csv")
    csv_test_path = join(data_dir, "arxiv_processed_test.csv")
    index_path = join(data_dir, "index.faiss")
    index_test_path = join(data_dir, "index_test.faiss")
    ids_path = join(data_dir, "ids.pkl")
    ids_test_path = join(data_dir, "ids_test.pkl")

    df.to_csv(csv_path, index=False)
    df.to_csv(csv_test_path, index=False)
    faiss.write_index(index, index_path)
    faiss.write_index(index, index_test_path)

    with open(ids_path, "wb") as f:
        pickle.dump(ids, f)
    with open(ids_test_path, "wb") as f:
        pickle.dump(ids, f)

    print(f"[+] Successfully saved {len(df)} papers and FAISS index into '{data_dir}'!")

if __name__ == "__main__":
    df = fetch_papers()
    if len(df) == 0:
        print("[-] Failed to fetch papers from arXiv. Exiting.")
        sys.exit(1)
    build_index(df)
