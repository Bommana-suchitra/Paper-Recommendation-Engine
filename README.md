# Paper Recommendation Engine

An intelligent semantic paper discovery platform designed to help researchers find relevant academic literature on **arXiv** quickly and accurately using dense vector embeddings and vector similarity search.

---

## 🚀 About the Project

Searching through academic literature using simple keyword matching often misses critical papers that use different terminology. 

The **Paper Recommendation Engine** solves this by converting paper titles and abstracts into **768-dimensional dense semantic vectors**. When you search with a topic, concept, or query, the system computes cosine similarity across the entire vector database using **FAISS**, instantly returning the most relevant papers with match confidence scores.

### Key Features
- **Semantic Vector Search**: Retrieve papers by meaning and conceptual similarity rather than exact keyword matches.
- **Dynamic Relevance Scoring**: Every paper displays a percentage match score (e.g., `🎯 60% Match`).
- **Discipline & Category Filtering**: Filter papers by arXiv disciplines (`cs.AI`, `cs.LG`, `cs.CV`, `cs.CL`, `quant-ph`, `stat.ML`, etc.).
- **User Tags & Custom Taxonomy**: Add and manage custom research tags (e.g., *Machine Learning*, *Quantum Computing*, *Computer Vision*, *Deep Learning*,).
- **Direct Links**: One-click navigation to arXiv abstract pages and direct PDF downloads.
- **Hardware Acceleration**: Automatic GPU acceleration via PyTorch and CUDA (with CPU fallback).

---

## 🛠️ Built With

* **[Streamlit](https://streamlit.io/)** - Interactive web interface with a dark glassmorphic UI.
* **[Sentence-Transformers](https://sbert.net/)** - Semantic embeddings using the `all-mpnet-base-v2` model.
* **[Facebook AI Similarity Search (FAISS)](https://github.com/facebookresearch/faiss)** - High-speed dense vector indexing and retrieval.
* **[PyTorch](https://pytorch.org/)** - Deep learning backend with CUDA support.
* **[arXiv API](https://pypi.org/project/arxiv/)** - Retrieval of academic research papers, authors, categories, and abstracts.
* **[Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)** - Data processing and vector computations.

---

## ⚙️ Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- *(Optional)* NVIDIA GPU with CUDA for faster embedding and inference.

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bommana-suchitra/Paper-Recommendation-Engine.git
   cd Paper-Recommendation-Engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Running the Application

To start the Streamlit web application, run:

```bash
streamlit run website/app.py
```

Once started, open your browser and navigate to:
```
http://localhost:8501
```

---

## 🔄 Updating or Re-indexing Papers (Optional)

The project includes an automated script to fetch fresh papers from the arXiv API and generate new FAISS vector embeddings:

```bash
python data_handling/fetch_and_build.py
```

This updates the dataset (`data/arxiv_processed.csv`) and the FAISS index (`data/index.faiss`) automatically.

---

## 📄 License

Distributed under the MIT License. See [LICENSE.md](LICENSE.md) for details.
