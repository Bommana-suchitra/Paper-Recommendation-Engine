# Streamlit app - Enhanced Modern UI
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import streamlit as st
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import time
from os.path import join, exists
import json
import html
from concurrent.futures import ThreadPoolExecutor

from config import data_dir, thumbnail_dir, testing
from utils import get_thumbnail, get_url


# Explicitly set the environment variable TOKENIZERS_PARALLELISM to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Page Configuration
st.set_page_config(
    page_title="Paper Intelligence - Neural Recommendation Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/mcpeixoto/PaperRecomenderSystem",
        "About": "Modern Semantic Academic Paper Recommendation Engine powered by FAISS & Sentence-Transformers",
    },
)

# Load External Stylesheet
css_path = join(os.path.dirname(__file__), "style.css")
if exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cache decorator compatibility
cache_resource = getattr(st, "cache_resource", getattr(st, "experimental_singleton", lambda f: f))

def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

if "selected_categories" not in st.session_state:
    st.session_state.selected_categories = []

if "search_query" not in st.session_state:
    st.session_state.search_query = ""


@cache_resource
def load_data():
    target = join(data_dir, "arxiv_processed_test.csv") if testing else join(data_dir, "arxiv_processed.csv")
    if not exists(target):
        st.info("Dataset not found. Generating initial dataset from arXiv...")
        from data_handling.fetch_and_build import fetch_papers, build_index
        df = fetch_papers()
        build_index(df)
    return pd.read_csv(target, dtype=str)


@cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-mpnet-base-v2", device=device)
    return model


@cache_resource
def load_index():
    target = join(data_dir, "index_test.faiss") if testing else join(data_dir, "index.faiss")
    if not exists(target):
        st.info("FAISS index not found. Generating index...")
        from data_handling.fetch_and_build import fetch_papers, build_index
        df = fetch_papers()
        build_index(df)
    return faiss.read_index(target)


def load_user_tags():
    if not exists("tags.json"):
        return {}
    try:
        with open("tags.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_user_tags_from_query(query, user_tag_keys, model):
    if not user_tag_keys:
        return []

    query_embedding = model.encode(query, convert_to_tensor=True).cpu()
    all_tags = load_user_tags()
    to_encode = [f"{key} - {','.join(all_tags[key])}" for key in user_tag_keys if key in all_tags]
    if not to_encode:
        return []
    user_tags_embeddings = model.encode(to_encode, convert_to_tensor=True).cpu()
    dot_product = np.dot(query_embedding, user_tags_embeddings.T)

    return [(tag, float(prob)) for tag, prob in zip(user_tag_keys, dot_product)]


# Load models and data
data = load_data()
model = load_model()
index = load_index()
user_tags = load_user_tags()


def draw_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 10px 0 20px 0;">
            <h2 style="margin: 0; font-size: 1.4rem; background: linear-gradient(135deg, #a78bfa, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚙️ Engine Control
            </h2>
            <p style="font-size: 0.85rem; color: #64748b; margin: 4px 0 0 0;">Neural filters and metadata tagging</p>
        </div>
        """, unsafe_allow_html=True)

        # Category Filter
        all_cats = set()
        for cat_str in data["categories"].dropna().unique():
            for c in str(cat_str).split():
                all_cats.add(c)
        categories = sorted(list(all_cats))

        st.subheader("Filter by Category")
        selected_categories = st.multiselect(
            "Select specific arXiv disciplines",
            categories,
            default=st.session_state.selected_categories,
            placeholder="All disciplines..."
        )

        col_apply, col_clear = st.columns(2)
        with col_apply:
            if st.button("Apply Filter", use_container_width=True):
                st.session_state.selected_categories = selected_categories
                rerun_app()
        with col_clear:
            if st.button("Reset", use_container_width=True):
                st.session_state.selected_categories = []
                rerun_app()

        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 24px 0;'>", unsafe_allow_html=True)

        # User Tags
        st.subheader(f"User Tags ({len(user_tags)})")
        tag_pills_html = " ".join([
            f"<span class='user-tag-pill'>{tag}</span>" for tag in list(user_tags.keys())[:12]
        ])
        st.markdown(f"<div class='pills-row' style='margin-bottom: 15px;'>{tag_pills_html}</div>", unsafe_allow_html=True)

        with st.expander("➕ Add Custom Tag", expanded=False):
            new_tag = st.text_input("Tag Name", placeholder="e.g. LLM Reasoning")
            new_tag_values = st.text_input("Keywords", placeholder="chain of thought, reasoning, deduction")
            if st.button("Save Tag", use_container_width=True):
                if new_tag and new_tag_values:
                    user_tags[new_tag] = [v.strip() for v in new_tag_values.split(",") if v.strip()]
                    with open("tags.json", "w", encoding="utf-8") as f:
                        json.dump(user_tags, f, indent=2)
                    st.success(f"Added tag '{new_tag}'!")
                    rerun_app()

        if user_tags:
            with st.expander("🗑️ Manage Tags", expanded=False):
                remove_tag = st.selectbox("Select tag to remove", list(user_tags.keys()))
                if st.button("Remove Selected Tag", use_container_width=True):
                    user_tags.pop(remove_tag, None)
                    with open("tags.json", "w", encoding="utf-8") as f:
                        json.dump(user_tags, f, indent=2)
                    rerun_app()


def main():
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">
            <span>✨</span> Next-Gen Academic Intelligence
        </div>
        <h1 class="hero-title">Paper Recommendation Engine</h1>
        <p class="hero-subtitle">
            Instant semantic discovery across arXiv academic papers. Leveraging dense 768-dimensional sentence transformer embeddings and accelerated vector search.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Telemetry System Grid
    device_name = "NVIDIA RTX 4050 (CUDA)" if torch.cuda.is_available() else "Standard CPU"
    st.markdown(f"""
    <div class="telemetry-grid">
        <div class="telemetry-card">
            <div class="telemetry-icon">📚</div>
            <div class="telemetry-content">
                <span class="telemetry-label">Corpus Size</span>
                <span class="telemetry-value">{len(data)} Verified Papers</span>
            </div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-icon">⚡</div>
            <div class="telemetry-content">
                <span class="telemetry-label">Embedding Model</span>
                <span class="telemetry-value">all-mpnet-base-v2 (768-d)</span>
            </div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-icon">🚀</div>
            <div class="telemetry-content">
                <span class="telemetry-label">Compute Engine</span>
                <span class="telemetry-value">{device_name}</span>
            </div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-icon">🔍</div>
            <div class="telemetry-content">
                <span class="telemetry-label">Vector Database</span>
                <span class="telemetry-value">FAISS IndexFlatIP</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    draw_sidebar()

    def set_query(new_q):
        st.session_state.search_query = new_q

    # Quick Suggestion Chips
    st.markdown("<div class='quick-search-label'>💡 Quick Explore Prompts:</div>", unsafe_allow_html=True)
    chip_cols = st.columns(5)
    suggestions = [
        "Attention & Transformers",
        "Latent Diffusion Models",
        "Reinforcement Learning (RLHF)",
        "Quantum Entanglement",
        "Residual Neural Networks"
    ]
    for idx, (col, sug) in enumerate(zip(chip_cols, suggestions)):
        with col:
            st.button(f"🔎 {sug}", key=f"sug_{idx}", use_container_width=True, on_click=set_query, args=(sug,))

    # Search Bar & Result Controls
    search_col, count_col = st.columns([4, 1])
    with search_col:
        query = st.text_input(
            "Search Query",
            key="search_query",
            placeholder="Search topics, methods, equations, or concepts (e.g. self-attention, generative diffusion, RLHF)...",
            label_visibility="collapsed"
        )
    with count_col:
        k = st.number_input("Results limit", 1, min(50, len(data)), min(10, len(data)), label_visibility="collapsed")


    if query:
        start_time = time.time()
        query_vec = model.encode(query, normalize_embeddings=True).reshape(1, -1)
        D, I = index.search(query_vec.astype(np.float32), k)
        end_time = time.time()
        elapsed = end_time - start_time

        rows = []
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(data):
                row_dict = data.iloc[idx].to_dict()
                row_dict["score"] = float(score)
                rows.append(row_dict)

        df = pd.DataFrame(rows)

        # Concurrently fetch any available thumbnails
        futures = []
        with ThreadPoolExecutor(max_workers=min(max(len(df), 1), 8)) as executor:
            for _, row in df.iterrows():
                futures.append(executor.submit(get_thumbnail, get_url(row["id"])))

        # Header Results Banner
        st.markdown(f"""
        <div class="results-header-banner">
            <span>✨ Neural match complete: retrieved top <b>{len(df)}</b> papers</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">⚡ {elapsed*1000:.1f}ms latency</span>
        </div>
        """, unsafe_allow_html=True)

        i = 0
        displayed = 0
        for _, row in df.iterrows():
            title = str(row.get("title", "")).replace("\n", " ").replace("\t", " ").strip()
            url = get_url(row["id"])
            pdf_url = url.replace("abs", "pdf")

            # Parse Authors
            raw_authors = str(row.get("authors_parsed", "[]"))
            try:
                authors_list = eval(raw_authors)
                authors = [x[1] + " " + x[0] if (len(x) > 1 and x[1]) else x[0] for x in authors_list]
            except Exception:
                authors = []

            # Category filter check
            categories = str(row.get("categories", "")).split()
            if len(st.session_state.selected_categories) > 0:
                has_match = any(cat in st.session_state.selected_categories for cat in categories)
                if not has_match:
                    i += 1
                    continue

            authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else "Authors Unspecified"
            clean_abstract = " ".join(str(row.get('abstract', '')).split())
            escaped_title = html.escape(title)
            escaped_abstract = html.escape(clean_abstract)
            escaped_authors = html.escape(authors_str)
            score_val = round(row['score'] * 100)

            # Score badge class
            if score_val >= 45:
                score_class = "score-high"
                score_label = f"🎯 {score_val}% Match"
            elif score_val >= 35:
                score_class = "score-mid"
                score_label = f"⚡ {score_val}% Match"
            else:
                score_class = "score-low"
                score_label = f"🔍 {score_val}% Match"

            # Category badges
            cat_pills = "".join([f"<span class='category-pill'>{html.escape(c)}</span>" for c in categories])

            # User Tags
            user_tag_list = get_user_tags_from_query(
                title + " " + clean_abstract,
                list(user_tags.keys()),
                model
            )
            matching_user_tags = [
                f"<span class='user-tag-pill'>🏷️ {html.escape(tag)} ({round(prob*100)}%)</span>"
                for tag, prob in user_tag_list if prob > 0.5
            ]
            user_tags_html = "".join(matching_user_tags)

            card_html = (
                f'<div class="paper-card">'
                f'<div class="card-top-bar">'
                f'<span class="score-badge {score_class}">{score_label}</span>'
                f'<span class="meta-date">📅 {row.get("update_date", "N/A")} &bull; arXiv:{row["id"]}</span>'
                f'</div>'
                f'<div class="paper-title">'
                f'<a href="{url}" target="_blank">{escaped_title} <span style="font-size: 0.9rem; opacity: 0.7;">↗</span></a>'
                f'</div>'
                f'<div class="paper-authors">'
                f'<span>👥</span> <span>{escaped_authors}</span>'
                f'</div>'
                f'<div class="pills-row">'
                f'{cat_pills}'
                f'{user_tags_html}'
                f'</div>'
                f'<div class="abstract-box">'
                f'{escaped_abstract}'
                f'</div>'
                f'<div class="card-actions">'
                f'<a class="action-btn btn-primary" href="{url}" target="_blank">📄 View on arXiv</a>'
                f'<a class="action-btn btn-secondary" href="{pdf_url}" target="_blank">📥 Direct PDF</a>'
                f'</div>'
                f'</div>'
            )
            if hasattr(st, "html"):
                st.html(card_html)
            else:
                st.markdown(card_html, unsafe_allow_html=True)

            # Thumbnail preview if available
            try:
                thumbnail = futures[i].result()
            except Exception:
                thumbnail = None
            i += 1

            if thumbnail:
                try:
                    st.image(thumbnail, use_container_width=True)
                except TypeError:
                    st.image(thumbnail, use_column_width=True)

            displayed += 1

        if displayed == 0:
            st.warning("No papers matched your query under the currently active category filters. Try clearing or expanding your filter selection.")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; color: #64748b;">
            <p style="font-size: 1.1rem; margin-bottom: 8px;">Enter a topic, query, or select one of the Quick Explore suggestions above.</p>
            <p style="font-size: 0.85rem;">The neural engine will compute cosine similarity across 768-dimensional paper embeddings in milliseconds.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
