# Emergent Ventures Semantic Map

A structured dataset, semantic search system, and interactive map of Emergent Ventures winners.

This project builds a **semantic map** of winners of **Emergent Ventures**, Tyler Cowen's fast-grant style program funding high-variance talent and ideas. The goal is to create a living dataset and visualization that reveals:

- key themes and domains across cohorts
- relationships between winners, projects, and ideas
- clusters of innovation (AI, biotech, education, hardware, etc.)
- geographic and temporal patterns
- connections that aren't obvious from reading cohort lists

## Phase 1 Status: Complete ✅

Phase 1 implements a minimal viable semantic map using the EV India 13 cohort (19 entries) with:

- ✅ Data cleaning and normalization with controlled vocabularies
- ✅ Embedding generation using OpenAI's text-embedding-3-large
- ✅ Clustering analysis (UMAP + HDBSCAN/K-means)
- ✅ Semantic search prototype with cosine similarity

## Project Structure

```
emergent-ventures-semantic-map/
├── data/
│   ├── raw/
│   │   └── data.json              # Original EV India 13 data
│   ├── processed/
│   │   ├── cleaned_data.json      # Cleaned and normalized data
│   │   └── embeddings.npy         # Generated embeddings
│   └── vocabularies/
│       ├── domains.json           # Canonical domain vocabulary
│       ├── categories.json        # Category vocabulary
│       ├── locations.json         # Location vocabulary
│       └── domain_mapping.json    # Domain normalization mapping
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Data quality analysis
│   ├── 02_data_cleaning.ipynb     # Data cleaning and normalization
│   ├── 03_embeddings.ipynb        # Embedding generation
│   ├── 04_clustering.ipynb        # Clustering analysis
│   └── 05_semantic_search.ipynb   # Semantic search demo
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py           # Data cleaning utilities
│   ├── embeddings.py               # OpenAI embedding generation
│   ├── clustering.py               # UMAP and clustering
│   └── search.py                  # Semantic search engine
├── requirements.txt
└── README.md
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up OpenAI API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys).

## Usage

### Running the Pipeline

The notebooks should be run in order:

1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`)
   - Analyzes data quality and identifies normalization needs

2. **Data Cleaning** (`notebooks/02_data_cleaning.ipynb`)
   - Normalizes domains, categories, and locations
   - Creates controlled vocabularies
   - Enhances embedding text
   - Saves cleaned data to `data/processed/cleaned_data.json`

3. **Generate Embeddings** (`notebooks/03_embeddings.ipynb`)
   - Generates embeddings using OpenAI API
   - Saves embeddings to `data/processed/embeddings.npy`
   - **Note:** Requires OpenAI API key and will incur API costs

4. **Clustering Analysis** (`notebooks/04_clustering.ipynb`)
   - Performs UMAP dimensionality reduction
   - Clusters using HDBSCAN and K-means
   - Generates cluster visualizations
   - Adds cluster assignments to data

5. **Semantic Search** (`notebooks/05_semantic_search.ipynb`)
   - Demonstrates semantic search functionality
   - Tests example queries

### Using the Python Modules

You can also use the modules programmatically:

```python
from src.search import SemanticSearch
import json
import numpy as np

# Load data and embeddings
with open('data/processed/cleaned_data.json', 'r') as f:
    data = json.load(f)
embeddings = np.load('data/processed/embeddings.npy')

# Initialize search engine
search_engine = SemanticSearch(data, embeddings)

# Search
results = search_engine.search("AI and machine learning projects", top_k=5)
for result in results:
    print(f"{result['name']}: {result['similarity_score']:.4f}")
```

## Phase 1 Results

### Dataset
- **19 entries** from EV India 13 cohort
- **Cleaned and normalized** with controlled vocabularies
- **Enhanced embedding text** for better semantic representation

### Clustering
- **UMAP** for 2D/3D visualization
- **HDBSCAN** and **K-means** clustering methods
- **3-6 meaningful clusters** identified based on domains and categories

### Semantic Search
- **Cosine similarity** search over embeddings
- **Top-k retrieval** with similarity scores
- Tested with queries like:
  - "AI and machine learning projects"
  - "Healthcare and medical devices"
  - "Hardware and robotics"
  - "Education and learning platforms"

## Technologies

- **Python 3.8+**
- **OpenAI API** (text-embedding-3-large)
- **UMAP** (dimensionality reduction)
- **HDBSCAN** (density-based clustering)
- **scikit-learn** (K-means, cosine similarity)
- **Jupyter Notebooks** (exploratory analysis)

## Next Steps (Phase 2+)

See [PLAN.md](PLAN.md) for the full project roadmap:

- **Phase 2:** Expand dataset, automate scraping, build knowledge graph
- **Phase 3:** Interactive visualization, web UI, narrative generation
- **Phase 4:** Large-scale analytics, link prediction, public API

## Contributing

This project is open to:
- Adding more cohorts
- Improving tagging and normalization
- Extending the knowledge graph
- Experimenting with visualizations
- Building semantic search demos

PRs and issue discussions are welcome.

## License

See [LICENSE](LICENSE) file for details.
