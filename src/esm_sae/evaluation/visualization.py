"""
Visualization utilities for SAE feature evaluation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import os

from esm_sae.evaluation.evaluation_config import OUTPUT_DIR


def plot_top_concepts_heatmap(
    metrics_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    top_n_concepts: int = 20,
    top_n_features: int = 20,
):
    """
    Create heatmap of top concepts and features.

    Args:
        metrics_df: DataFrame with metrics
        output_dir: Directory to save plot
        top_n_concepts: Number of top concepts to include
        top_n_features: Number of top features to include
    """
    # Get best F1 score for each concept-feature pair
    best_df = metrics_df.sort_values("f1_score", ascending=False)
    best_df = best_df.drop_duplicates(["concept", "feature_id"])

    # Get top concepts by maximum F1 score
    top_concepts = best_df.groupby("concept")["f1_score"].max().nlargest(top_n_concepts).index.tolist()

    # Get top features that appear in these concepts
    concept_features = best_df[best_df["concept"].isin(top_concepts)]
    top_features = concept_features.sort_values("f1_score", ascending=False)["feature_id"].head(top_n_features).unique()

    # Create matrix for heatmap
    heatmap_data = pd.pivot_table(
        concept_features[concept_features["feature_id"].isin(top_features)],
        values="f1_score",
        index="concept",
        columns="feature_id",
        fill_value=0
    )

    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f", linewidths=.5)
    plt.title(f"F1 Scores for Top {top_n_concepts} Concepts and {top_n_features} Features")
    plt.tight_layout()

    # Save plot
    output_path = output_dir / "top_concepts_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved concept heatmap to {output_path}")


def plot_concept_category_performance(
    metrics_df: pd.DataFrame,
    concept_categories: Dict[str, List[str]],
    output_dir: Path = OUTPUT_DIR,
):
    """
    Plot performance by concept category.

    Args:
        metrics_df: DataFrame with metrics
        concept_categories: Dictionary mapping categories to concepts
        output_dir: Directory to save plot
    """
    # Get best F1 score for each concept
    best_scores = metrics_df.groupby("concept")["f1_score"].max().reset_index()

    # Assign category to each concept
    category_data = []
    for category, concepts in concept_categories.items():
        for concept in concepts:
            # Find concepts that contain this concept name
            matching_concepts = best_scores[best_scores["concept"].str.contains(concept)]

            for _, row in matching_concepts.iterrows():
                category_data.append({
                    "category": category,
                    "concept": row["concept"],
                    "f1_score": row["f1_score"]
                })

    # Convert to DataFrame
    category_df = pd.DataFrame(category_data)

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="category", y="f1_score", data=category_df)
    plt.title("F1 Score Distribution by Concept Category")
    plt.xlabel("Concept Category")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    output_path = output_dir / "concept_category_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved category performance plot to {output_path}")


def plot_threshold_impact(
    metrics_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
):
    """
    Plot impact of threshold on metrics.

    Args:
        metrics_df: DataFrame with metrics
        output_dir: Directory to save plot
    """
    # Calculate average metrics per threshold
    threshold_metrics = metrics_df.groupby("threshold").agg({
        "precision": "mean",
        "recall": "mean",
        "f1_score": "mean"
    }).reset_index()

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="threshold", y="precision", data=threshold_metrics, marker="o", label="Precision")
    sns.lineplot(x="threshold", y="recall", data=threshold_metrics, marker="s", label="Recall")
    sns.lineplot(x="threshold", y="f1_score", data=threshold_metrics, marker="^", label="F1 Score")

    plt.title("Impact of Activation Threshold on Metrics")
    plt.xlabel("Activation Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Save plot
    output_path = output_dir / "threshold_impact.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved threshold impact plot to {output_path}")


def plot_feature_frequency(
    metrics_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    top_n: int = 50,
):
    """
    Plot frequency of top features across concepts.

    Args:
        metrics_df: DataFrame with metrics
        output_dir: Directory to save plot
        top_n: Number of top features to plot
    """
    # Count how many concepts each feature appears in with F1 > 0.5
    significant_features = metrics_df[metrics_df["f1_score"] > 0.5]
    feature_counts = significant_features["feature_id"].value_counts().head(top_n)

    # Create plot
    plt.figure(figsize=(12, 8))
    feature_counts.plot(kind="bar")
    plt.title(f"Top {top_n} Features by Number of Associated Concepts (F1 > 0.5)")
    plt.xlabel("Feature ID")
    plt.ylabel("Number of Concepts")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save plot
    output_path = output_dir / "feature_frequency.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature frequency plot to {output_path}")


def plot_feature_concept_network(
    metrics_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    min_f1: float = 0.6,
    max_concepts: int = 30,
    max_features: int = 50,
):
    """
    Create a network visualization of feature-concept relationships.

    Args:
        metrics_df: DataFrame with metrics
        output_dir: Directory to save plot
        min_f1: Minimum F1 score to include
        max_concepts: Maximum number of concepts to include
        max_features: Maximum number of features to include
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed. Skipping network visualization.")
        return

    # Filter to significant relationships
    significant = metrics_df[metrics_df["f1_score"] >= min_f1]

    # Get top concepts and features
    top_concepts = significant.groupby("concept")["f1_score"].max().nlargest(max_concepts).index
    concept_df = significant[significant["concept"].isin(top_concepts)]

    top_features = concept_df["feature_id"].value_counts().head(max_features).index
    filtered_df = concept_df[concept_df["feature_id"].isin(top_features)]

    # Create graph
    G = nx.Graph()

    # Add nodes
    for concept in filtered_df["concept"].unique():
        G.add_node(concept, node_type="concept")

    for feature in filtered_df["feature_id"].unique():
        G.add_node(f"F{feature}", node_type="feature")

    # Add edges
    for _, row in filtered_df.iterrows():
        G.add_edge(
            row["concept"],
            f"F{row['feature_id']}",
            weight=row["f1_score"]
        )

    # Create plot
    plt.figure(figsize=(12, 12))

    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, seed=42, k=0.3)

    # Draw nodes
    concept_nodes = [n for n, attr in G.nodes(data=True) if attr.get("node_type") == "concept"]
    feature_nodes = [n for n, attr in G.nodes(data=True) if attr.get("node_type") == "feature"]

    nx.draw_networkx_nodes(G, pos, nodelist=concept_nodes, node_color="lightblue", node_size=300, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=feature_nodes, node_color="lightgreen", node_size=200, alpha=0.8)

    # Draw edges with width proportional to F1 score
    edges = [(u, v) for u, v, d in G.edges(data=True)]
    weights = [G[u][v]["weight"] * 3 for u, v in edges]

    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.5)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Feature-Concept Relationship Network (F1 >= {:.1f})".format(min_f1))
    plt.axis("off")
    plt.tight_layout()

    # Save plot
    output_path = output_dir / "feature_concept_network.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved network visualization to {output_path}")


def create_visualizations(
    metrics_path: Path = OUTPUT_DIR / "all_metrics.csv",
    concept_categories: Optional[Dict[str, List[str]]] = None,
):
    """
    Create all visualizations.

    Args:
        metrics_path: Path to metrics CSV
        concept_categories: Dictionary mapping categories to concepts
    """
    # Load metrics
    metrics_df = pd.read_csv(metrics_path)

    # Create output directory if needed
    output_dir = metrics_path.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)

    # Create plots
    plot_top_concepts_heatmap(metrics_df, output_dir)
    plot_threshold_impact(metrics_df, output_dir)
    plot_feature_frequency(metrics_df, output_dir)

    if concept_categories is not None:
        plot_concept_category_performance(metrics_df, concept_categories, output_dir)

    # Create network visualization (requires networkx)
    try:
        plot_feature_concept_network(metrics_df, output_dir)
    except Exception as e:
        print(f"Error creating network visualization: {e}")

    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    from esm_sae.evaluation.evaluation_config import CONCEPT_CATEGORIES

    parser = argparse.ArgumentParser(description="Create visualizations for SAE evaluation")
    parser.add_argument("--metrics_file", type=Path, default=OUTPUT_DIR / "all_metrics.csv",
                        help="Path to metrics CSV file")

    args = parser.parse_args()

    create_visualizations(args.metrics_file, CONCEPT_CATEGORIES)
