"""
Clustering analysis using UMAP and HDBSCAN/K-means.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path
import umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 5,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embedding dimensions using UMAP.
    
    Args:
        embeddings: High-dimensional embeddings
        n_components: Number of dimensions for reduction
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        random_state: Random seed
        
    Returns:
        Reduced-dimensional embeddings
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    reduced = reducer.fit_transform(embeddings)
    return reduced


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2
) -> np.ndarray:
    """
    Cluster embeddings using HDBSCAN.
    
    Args:
        embeddings: Embedding vectors
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples in cluster
        
    Returns:
        Array of cluster labels (-1 for noise)
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    
    labels = clusterer.fit_predict(embeddings)
    return labels


def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Cluster embeddings using K-means.
    
    Args:
        embeddings: Embedding vectors
        n_clusters: Number of clusters
        random_state: Random seed
        
    Returns:
        Array of cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: range = range(3, 9),
    random_state: int = 42
) -> Tuple[int, Dict[int, float]]:
    """
    Find optimal number of clusters using elbow method and silhouette score.
    
    Args:
        embeddings: Embedding vectors
        k_range: Range of k values to test
        random_state: Random seed
        
    Returns:
        Tuple of (optimal_k, scores_dict)
    """
    scores = {}
    inertias = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        inertias[k] = kmeans.inertia_
        if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette
            scores[k] = silhouette_score(embeddings, labels)
        else:
            scores[k] = -1
    
    # Find optimal k (highest silhouette score)
    optimal_k = max(scores, key=scores.get)
    
    return optimal_k, scores


def analyze_clusters(
    data: List[Dict[str, Any]],
    cluster_labels: np.ndarray,
    embeddings_reduced: np.ndarray = None
) -> Dict[str, Any]:
    """
    Analyze clusters and generate summaries.
    
    Args:
        data: List of entry dictionaries
        cluster_labels: Cluster assignments
        embeddings_reduced: Optional 2D reduced embeddings for visualization
        
    Returns:
        Dictionary with cluster analysis results
    """
    df = pd.DataFrame(data)
    df['cluster'] = cluster_labels
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    
    cluster_summaries = []
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            continue
        
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Get top domains
        all_domains = []
        for domains in cluster_data.get('domains_normalized', cluster_data.get('domains', [])):
            if isinstance(domains, list):
                all_domains.extend(domains)
        
        domain_counts = pd.Series(all_domains).value_counts()
        top_domains = domain_counts.head(5).to_dict()
        
        # Get top categories
        category_counts = cluster_data['category'].value_counts()
        top_categories = category_counts.head(3).to_dict()
        
        cluster_summaries.append({
            'cluster_id': int(cluster_id),
            'size': len(cluster_data),
            'top_domains': top_domains,
            'top_categories': top_categories,
            'names': cluster_data['name'].tolist()
        })
    
    return {
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'cluster_summaries': cluster_summaries,
        'cluster_labels': cluster_labels.tolist()
    }


def visualize_clusters(
    embeddings_reduced: np.ndarray,
    cluster_labels: np.ndarray,
    data: List[Dict[str, Any]],
    output_path: Path = None,
    title: str = "Cluster Visualization"
) -> None:
    """
    Visualize clusters in 2D space.
    
    Args:
        embeddings_reduced: 2D reduced embeddings
        cluster_labels: Cluster assignments
        data: List of entry dictionaries
        output_path: Optional path to save figure
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Get unique clusters
    unique_clusters = sorted(set(cluster_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        label = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
        plt.scatter(
            embeddings_reduced[mask, 0],
            embeddings_reduced[mask, 1],
            c=[colors[i]],
            label=label,
            alpha=0.6,
            s=100
        )
    
    plt.title(title, fontsize=16)
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    plt.show()


def perform_clustering_analysis(
    data: List[Dict[str, Any]],
    embeddings: np.ndarray,
    method: str = "both",
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Perform complete clustering analysis.
    
    Args:
        data: List of entry dictionaries
        embeddings: High-dimensional embeddings
        method: "hdbscan", "kmeans", or "both"
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with clustering results
    """
    results = {}
    
    # Reduce dimensions
    print("Reducing dimensions with UMAP...")
    embeddings_2d = reduce_dimensions(embeddings, n_components=2)
    embeddings_3d = reduce_dimensions(embeddings, n_components=3)
    
    if method in ["hdbscan", "both"]:
        print("\nClustering with HDBSCAN...")
        hdbscan_labels = cluster_hdbscan(embeddings, min_cluster_size=3, min_samples=2)
        hdbscan_analysis = analyze_clusters(data, hdbscan_labels, embeddings_2d)
        
        print(f"Found {hdbscan_analysis['n_clusters']} clusters")
        print(f"Noise points: {hdbscan_analysis['n_noise']}")
        
        results['hdbscan'] = {
            'labels': hdbscan_labels,
            'analysis': hdbscan_analysis,
            'embeddings_2d': embeddings_2d
        }
        
        if output_dir:
            visualize_clusters(
                embeddings_2d,
                hdbscan_labels,
                data,
                output_dir / "hdbscan_clusters.png",
                "HDBSCAN Clustering"
            )
    
    if method in ["kmeans", "both"]:
        print("\nFinding optimal k for K-means...")
        optimal_k, scores = find_optimal_k(embeddings, k_range=range(3, 9))
        print(f"Optimal k: {optimal_k} (silhouette scores: {scores})")
        
        print(f"\nClustering with K-means (k={optimal_k})...")
        kmeans_labels = cluster_kmeans(embeddings, n_clusters=optimal_k)
        kmeans_analysis = analyze_clusters(data, kmeans_labels, embeddings_2d)
        
        print(f"Found {kmeans_analysis['n_clusters']} clusters")
        
        results['kmeans'] = {
            'labels': kmeans_labels,
            'analysis': kmeans_analysis,
            'optimal_k': optimal_k,
            'scores': scores,
            'embeddings_2d': embeddings_2d
        }
        
        if output_dir:
            visualize_clusters(
                embeddings_2d,
                kmeans_labels,
                data,
                output_dir / "kmeans_clusters.png",
                f"K-means Clustering (k={optimal_k})"
            )
    
    return results

