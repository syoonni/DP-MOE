import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from pathlib import Path

class FeatureVisualizer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_correlation_heatmap(self, corr_matrix, feature_names):
        """Plot correlation heatmap of features"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   xticklabels=feature_names,
                   yticklabels=feature_names,
                   cmap='coolwarm', 
                   center=0,
                   annot=True,
                   fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png')
        plt.close()

    def plot_feature_network(self, corr_matrix, feature_names, threshold=0.5):
        """Create network graph of feature relationships"""
        plt.figure(figsize=(15, 15))
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for name in feature_names:
            G.add_node(name)
        
        # Add edges based on correlation
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if abs(corr_matrix[i, j]) > threshold:
                    G.add_edge(feature_names[i], 
                             feature_names[j], 
                             weight=abs(corr_matrix[i, j]))
        
        # Draw the network
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=1000, alpha=0.6)
        
        # Draw edges with width based on correlation strength
        edges = G.edges()
        weights = [G[u][v]['weight'] * 2 for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title('Feature Relationship Network (Correlations > {})'.format(threshold))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_network.png')
        plt.close()

    def plot_expert_performance(self, scores_history, n_experts):
        """Plot performance scores for different expert configurations"""
        plt.figure(figsize=(10, 6))
        
        for i in range(n_experts):
            plt.plot(scores_history[i], label=f'Expert {i+1}')
        
        plt.plot(np.mean(scores_history, axis=0), 
                label='Mean Performance', 
                linewidth=2, 
                color='black', 
                linestyle='--')
        
        plt.xlabel('Configuration Iteration')
        plt.ylabel('Performance Score')
        plt.title('Expert Performance Across Different Configurations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'expert_performance.png')
        plt.close()

    def plot_feature_distribution(self, expert_features):
        """Visualize how features are distributed among experts"""
        plt.figure(figsize=(12, 6))
        
        n_experts = len(expert_features)
        n_features = max(len(features) for features in expert_features.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_experts))
        
        for i, (expert, features) in enumerate(expert_features.items()):
            plt.barh([j + i/n_experts for j in range(len(features))], 
                    [1] * len(features),
                    height=1/n_experts,
                    color=colors[i],
                    alpha=0.7,
                    label=expert)
            
            # Add feature names
            for j, feature in enumerate(features):
                plt.text(0.5, j + i/n_experts, feature, 
                        ha='center', va='center')
        
        plt.yticks([])
        plt.xlabel('Features')
        plt.title('Feature Distribution Across Experts')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distribution.png')
        plt.close()

    def plot_optimization_progress(self, scores, n_experts_range):
        """Plot optimization progress for different numbers of experts"""
        plt.figure(figsize=(10, 6))
        
        for n_exp in n_experts_range:
            plt.plot(scores[n_exp], label=f'{n_exp} Experts')
            
        plt.xlabel('Optimization Step')
        plt.ylabel('Configuration Score')
        plt.title('Optimization Progress by Number of Experts')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimization_progress.png')
        plt.close()