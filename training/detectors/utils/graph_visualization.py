"""
Visualization utilities for NeSyDeFake
- Causal DAG visualization with concept names
- Semantic concept interpretation
- Violation score analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

import torch


def visualize_causal_dag(
    dag: torch.Tensor,
    concept_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
    title: str = "Learned Causal DAG"
):
    """
    Visualize the learned causal DAG as a heatmap
    
    Args:
        dag: (N, N) adjacency matrix
        concept_names: List of concept names
        save_path: Path to save figure (optional)
        figsize: Figure size
        title: Plot title
    """
    dag_np = dag.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        dag_np,
        xticklabels=concept_names,
        yticklabels=concept_names,
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        cbar_kws={'label': 'Causal Strength'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Effect (Child Concept)', fontsize=12, fontweight='bold')
    plt.ylabel('Cause (Parent Concept)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Causal DAG saved to {save_path}")
    
    plt.show()


def visualize_causal_graph_network(
    dag: torch.Tensor,
    concept_names: List[str],
    threshold: float = 0.3,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10)
):
    """
    Visualize causal DAG as a network graph with arrows
    
    Args:
        dag: (N, N) adjacency matrix
        concept_names: List of concept names
        threshold: Only show edges with strength > threshold
        save_path: Path to save figure
        figsize: Figure size
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed. Install with: pip install networkx")
        return
    
    dag_np = dag.detach().cpu().numpy()
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, name in enumerate(concept_names):
        G.add_node(i, label=name)
    
    # Add edges above threshold
    edge_weights = []
    for i in range(dag_np.shape[0]):
        for j in range(dag_np.shape[1]):
            if dag_np[i, j] > threshold:
                G.add_edge(i, j, weight=dag_np[i, j])
                edge_weights.append(dag_np[i, j])
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Draw nodes
    node_colors = ['lightblue' if i < 7 else 'lightgreen' for i in range(len(concept_names))]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=3000,
        alpha=0.9
    )
    
    # Draw node labels
    labels = {i: name for i, name in enumerate(concept_names)}
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=9,
        font_weight='bold'
    )
    
    # Draw edges with varying thickness
    if edge_weights:
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # Normalize weights for visualization
        max_weight = max(weights) if weights else 1
        widths = [3 * w / max_weight for w in weights]
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            width=widths,
            alpha=0.6,
            edge_color=weights,
            edge_cmap=plt.cm.Blues,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1'
        )
    
    plt.title(f"Causal Network (threshold={threshold})", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Causal network saved to {save_path}")
    
    plt.show()


def visualize_semantic_concepts(
    semantic_concepts: torch.Tensor,
    concept_names: List[str],
    sample_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Visualize semantic concept values for a sample
    
    Args:
        semantic_concepts: (B, num_concepts) tensor
        concept_names: List of concept names
        sample_idx: Which sample to visualize
        save_path: Path to save figure
        figsize: Figure size
    """
    concepts = semantic_concepts[sample_idx].detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    # Create bar plot
    colors = ['skyblue' if 'emotion' in name else 
              'lightgreen' if 'gender' in name else
              'coral' if 'age' in name else
              'plum' for name in concept_names]
    
    bars = plt.bar(range(len(concepts)), concepts, color=colors, alpha=0.7, edgecolor='black')
    
    plt.xlabel('Semantic Concepts', fontsize=12, fontweight='bold')
    plt.ylabel('Value / Probability', fontsize=12, fontweight='bold')
    plt.title(f'Semantic Concepts (Sample {sample_idx})', fontsize=14, fontweight='bold')
    plt.xticks(range(len(concepts)), concept_names, rotation=45, ha='right')
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, concepts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.02,
            f'{value:.2f}',
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Semantic concepts saved to {save_path}")
    
    plt.show()


def compare_real_vs_fake_concepts(
    real_concepts: torch.Tensor,
    fake_concepts: torch.Tensor,
    concept_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6)
):
    """
    Compare semantic concept distributions for real vs fake videos
    
    Args:
        real_concepts: (N, num_concepts) tensor for real videos
        fake_concepts: (M, num_concepts) tensor for fake videos
        concept_names: List of concept names
        save_path: Path to save figure
        figsize: Figure size
    """
    real_mean = real_concepts.mean(dim=0).detach().cpu().numpy()
    fake_mean = fake_concepts.mean(dim=0).detach().cpu().numpy()
    
    real_std = real_concepts.std(dim=0).detach().cpu().numpy()
    fake_std = fake_concepts.std(dim=0).detach().cpu().numpy()
    
    x = np.arange(len(concept_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width/2, real_mean, width, label='Real', 
                   yerr=real_std, capsize=5, alpha=0.7, color='green')
    bars2 = ax.bar(x + width/2, fake_mean, width, label='Fake',
                   yerr=fake_std, capsize=5, alpha=0.7, color='red')
    
    ax.set_xlabel('Semantic Concepts', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Value ± Std', fontsize=12, fontweight='bold')
    ax.set_title('Real vs Fake: Semantic Concept Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(concept_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Concept comparison saved to {save_path}")
    
    plt.show()


def analyze_violation_scores(
    violation_scores: torch.Tensor,
    labels: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Analyze and visualize causal violation scores
    
    Args:
        violation_scores: (B,) tensor of violation scores
        labels: (B,) tensor of labels (0=real, 1=fake)
        save_path: Path to save figure
        figsize: Figure size
    """
    violations = violation_scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    real_violations = violations[labels_np == 0]
    fake_violations = violations[labels_np == 1]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(real_violations, bins=30, alpha=0.6, label='Real', color='green', density=True)
    axes[0].hist(fake_violations, bins=30, alpha=0.6, label='Fake', color='red', density=True)
    axes[0].set_xlabel('Causal Violation Score', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Distribution of Violation Scores', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    data = [real_violations, fake_violations]
    axes[1].boxplot(data, labels=['Real', 'Fake'], patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('Causal Violation Score', fontsize=12)
    axes[1].set_title('Violation Score by Class', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Violation analysis saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print("\n=== Violation Score Statistics ===")
    print(f"Real videos - Mean: {real_violations.mean():.4f}, Std: {real_violations.std():.4f}")
    print(f"Fake videos - Mean: {fake_violations.mean():.4f}, Std: {fake_violations.std():.4f}")
    print(f"Separation: {abs(real_violations.mean() - fake_violations.mean()):.4f}")


def create_summary_report(
    pred_dict: Dict,
    data_dict: Dict,
    config: Dict,
    save_dir: str
):
    """
    Create a comprehensive visualization report
    
    Args:
        pred_dict: Model prediction dictionary
        data_dict: Input data dictionary
        config: Model configuration
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nCreating visualization report in {save_dir}...")
    
    # 1. Causal DAG heatmap
    if pred_dict.get('causal_dag') is not None:
        from nesydefake_detector import NeSyDeFakeHybridDetector
        model = NeSyDeFakeHybridDetector(config)
        concept_names = model.causal_module.get_concept_names(config)
        
        visualize_causal_dag(
            pred_dict['causal_dag'],
            concept_names,
            save_path=os.path.join(save_dir, 'causal_dag_heatmap.png')
        )
        
        visualize_causal_graph_network(
            pred_dict['causal_dag'],
            concept_names,
            threshold=0.3,
            save_path=os.path.join(save_dir, 'causal_network.png')
        )
    
    # 2. Semantic concepts
    if pred_dict.get('semantic_concepts') is not None:
        from semantic_grounding import DeepFaceSemanticExtractor
        extractor = DeepFaceSemanticExtractor(config)
        concept_names = extractor.get_concept_names()
        
        visualize_semantic_concepts(
            pred_dict['semantic_concepts'],
            concept_names,
            sample_idx=0,
            save_path=os.path.join(save_dir, 'semantic_concepts.png')
        )
    
    # 3. Violation scores
    if pred_dict.get('violation_score') is not None and data_dict.get('label') is not None:
        analyze_violation_scores(
            pred_dict['violation_score'],
            data_dict['label'],
            save_path=os.path.join(save_dir, 'violation_analysis.png')
        )
    
    print(f"Visualization report created successfully!")


# Example usage
if __name__ == '__main__':
    # Test with dummy data
    num_concepts = 10
    concept_names = [f'concept_{i}' for i in range(num_concepts)]
    
    # Random DAG
    dag = torch.rand(num_concepts, num_concepts)
    dag = dag * (torch.rand_like(dag) > 0.7).float()  # Sparsify
    
    visualize_causal_dag(dag, concept_names)
    visualize_causal_graph_network(dag, concept_names, threshold=0.2)