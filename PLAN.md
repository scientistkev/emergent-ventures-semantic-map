# 🍁 Emergent Ventures Semantic Map

_A structured dataset, semantic search system, and interactive map of Emergent Ventures winners._

This project builds a **semantic map** of winners of **Emergent Ventures**, Tyler Cowen’s fast-grant style program funding high-variance talent and ideas. The goal is to create a living dataset and visualization that reveals:

- key themes and domains across cohorts
- relationships between winners, projects, and ideas
- clusters of innovation (AI, biotech, education, hardware, etc.)
- geographic and temporal patterns
- connections that aren’t obvious from reading cohort lists

The project begins small, then expands into a full knowledge graph with embeddings, search, and analytics.

---

## 🚀 Project Goals

- Build a clean, structured dataset of all EV cohorts
- Generate embeddings for semantic search and clustering
- Visualize clusters and relationships across winners
- Provide a simple search interface
- Enable time-series analysis of EV’s evolution
- Maintain an ongoing, auto-updated semantic map

---

# 🧱 Phase 1 — Minimal Viable Map (Week 1)

**Goal:** a small but functional semantic map using a single cohort.

### 1. Collect initial dataset

Start with 5–20 entries (e.g., EV India 13).
Fields include name, cohort, project, tags, domains, summary, etc.

### 2. Clean & normalize data

Create controlled vocabularies for:

- domains
- project types
- geographies
- outcomes

### 3. Generate embeddings

Use `text-embedding-3-large` or similar.
Embed the summary + project description + domain tags.

### 4. First clustering

- UMAP for dimensionality reduction
- HDBSCAN or K-means for grouping
- Produce initial thematic clusters

### 5. Minimal semantic search prototype

- cosine similarity search over embedding vectors
- return top 5 nearest winners for any query

This becomes the first version of the semantic map.

---

# 🧭 Phase 2 — Expand Dataset & Automate (Weeks 2–4)

### 6. Scrape additional cohorts

Extract winners from:

- Marginal Revolution posts
- Mercatus Center PDF pages
- Press announcements
- Guest posts mentioning EV winners

Store raw → clean → summarize → structure.

### 7. Build full data schema (knowledge-graph ready)

Nodes:

- Person, Project, Domain, Geography, Outcome, Institution

Edges:

- Person → Project
- Person → Domain
- Domain ↔ Domain
- Project → Outcome
- Cohort → Person

### 8. Automated LLM labeling

Tag each entry with:

- domains
- project category
- geography
- problem area
- likely outcomes (startup, research, content, policy)

### 9. Incremental update pipeline

When new cohorts appear:

- scrape → summarize → embed → update → re-cluster
- detect new themes (“emergent clusters”)

---

# 🌐 Phase 3 — Full Semantic Map (Months 2–3)

### 10. Build interactive visualization

Technologies: D3.js, ObservableHQ, Plotly, or Streamlit.

Views:

- 2D semantic map of people/projects
- domain co-occurrence graph
- geography map
- timeline of cohorts
- filter by domain, cohort, geography, age, category

### 11. Build a simple web UI

Features:

- semantic search bar
- cluster browser
- winner profile pages
- list & timeline view

### 12. Narrative generation

LLMs can produce:

- thematic summaries (“biotech innovation across cohorts”)
- comparisons (“evolution of AI governance themes”)
- reading lists for specific interests

### 13. Integrate external data

- Twitter/X
- personal websites
- Google Scholar
- LinkedIn bios
- startup funding data
- Substack articles
- interviews and podcasts

All of this enriches the graph.

---

# 🧠 Phase 4 — Large-Scale Graph Analytics (Month 3+)

### 14. Temporal analytics

- how EV’s focus shifts over time
- domain drift
- cluster growth & collapse
- rising vs declining themes

### 15. Link prediction & graph ML

Using Node2Vec, DeepWalk, or GraphSAGE:

- identify potential collaborations
- locate “idea clusters”
- find underexplored domains
- predict future winners’ thematic neighborhoods

### 16. Public API

Expose:

- search
- clusters
- node metadata
- embeddings
- similarity scores

### 17. Automated weekly updates

Cron-driven:

- check for new winners
- update dataset
- re-run embeddings
- publish updated dashboards and analytics

---

# 📦 Data Schema (v1.0)

```json
{
  "name": "",
  "age": null,
  "education": "",
  "location": "",
  "project_name": "",
  "project_description": "",
  "domains": [],
  "category": "",
  "funding_type": "grant",
  "cohort": "",
  "links": [],
  "embedding_text": ""
}
```

---

# 📊 Example: EV India 13th Cohort (Included)

The repository includes structured JSON for the **EV India 13th Cohort**, which serves as the seed dataset for the semantic map.

---

# 🛠️ Technologies

**Core:**

- Python
- Jupyter / notebooks
- OpenAI or Voyage embeddings
- FAISS / Pinecone / Weaviate
- Neo4j or ArangoDB
- UMAP, HDBSCAN
- Streamlit or D3.js

**Optional:**

- GitHub Actions for weekly updates
- Docker for reproducible pipelines
- Fly.io/Vercel for hosting

---

# 🤝 Contributing

This project is open to:

- adding more cohorts
- improving tagging
- extending the knowledge graph
- experimenting with visualizations
- building semantic search demos

PRs and issue discussions are welcome.
