import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from matplotlib.collections import LineCollection
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from torch_geometric.utils import k_hop_subgraph, to_networkx

if torch.cuda.is_available():
    device = torch.device('cuda')
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#    device = torch.device('mps')
else:
    device = torch.device('cpu')

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True),
])
dataset = Planetoid(path, dataset, transform=transform)
train_data, val_data, test_data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = model.encode(x, edge_index)
        return model.decode(z, edge_label_index).view(-1)


model = GCN(dataset.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()

    out = model(train_data.x, train_data.edge_index,
                train_data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_label_index).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


for epoch in range(1, 501):
    loss = train()
    if epoch % 20 == 0:
        val_auc = test(val_data)
        test_auc = test(test_data)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

model_config = ModelConfig(
    mode='binary_classification',
    task_level='edge',
    return_type='raw',
)

# Explain model output for a single edge:

# edge_label_index = val_data.edge_label_index[:, 0]
#
# explainer = Explainer(
#     model=model,
#     explanation_type='model',
#     algorithm=GNNExplainer(epochs=200),
#     node_mask_type='attributes',
#     edge_mask_type='object',
#     model_config=model_config,
# )
# explanation = explainer(
#     x=train_data.x,
#     edge_index=train_data.edge_index,
#     edge_label_index=edge_label_index,
# )
# print(f'Generated model explanations in {explanation.available_explanations}')

# Explain a selected target (phenomenon) for a single edge:

edge_label_index = val_data.edge_label_index[:, 0]
target = val_data.edge_label[0].unsqueeze(dim=0).long()

explainer = Explainer(
    model=model,
    explanation_type='phenomenon',
    algorithm=GNNExplainer(epochs=200),
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=model_config,
)
explanation = explainer(
    x=train_data.x,
    edge_index=train_data.edge_index,
    target=target,
    edge_label_index=edge_label_index,
)


available_explanations = explanation.available_explanations
print(f'Generated phenomenon explanations in {available_explanations}')

edge_importance = explanation.edge_mask.cpu().detach().numpy()
print("Edge importance values:", edge_importance)
print("Max edge importance:", np.max(edge_importance))
print("Min edge importance:", np.min(edge_importance))
print("Non-zero edge importance count:", np.sum(edge_importance > 0))

# exit(0)


# edge_mask = explanation.edge_mask
# subgraph = explanation.edge_index

G = to_networkx(data=train_data, to_undirected=True)

# Draw the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # Positions for all nodes
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color='lightblue',
    edge_color='gray',
    node_size=500,
    font_size=10,
)
plt.title("Graph Visualization")
plt.show()


# Assign edge importance from explanation
edge_mask = explanation.edge_mask.cpu().detach().numpy()
for i, (u, v) in enumerate(train_data.edge_index.t().tolist()):
    G[u][v]['importance'] = edge_mask[i]

# Prepare the graph layout
pos = nx.spring_layout(G, seed=42)  # Layout for consistent visualization

# Extract edge segments and weights
edges = [(u, v) for u, v in G.edges]
edge_segments = [(pos[u], pos[v]) for u, v in edges]
edge_weights = [G[u][v]['importance'] for u, v in edges]

# Create a LineCollection for edges with a colormap
lc = LineCollection(edge_segments, cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=max(edge_weights)))
lc.set_array(edge_weights)

# Plot the graph
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=ax)
ax.add_collection(lc)

# Add labels
nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

# Add a colorbar
plt.colorbar(lc, ax=ax, label="Edge Importance")
plt.title("Subgraph with Edge Importance (Cora Dataset)")
plt.axis("off")
plt.show()


if explanation.node_mask is not None:

    # Aggregate node importance into a single value per node
    node_importance_aggregated = explanation.node_mask.cpu().detach().numpy().mean(axis=1)

    # Normalize node importance
    norm = Normalize(vmin=0, vmax=np.max(node_importance_aggregated))
    sm = ScalarMappable(cmap=plt.cm.Reds, norm=norm)

    # Get node colors based on aggregated importance
    node_colors = [node_importance_aggregated[node] for node in G.nodes]

    # Draw the graph with node importance
    plt.figure(figsize=(10, 8))
    ax = plt.gca()  # Get the current axes
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        cmap=plt.cm.Reds,
        node_size=500,
        font_size=10,
        ax=ax,  # Associate the graph with the axes
    )

    # Add colorbar associated with the current axes
    plt.colorbar(sm, ax=ax, label="Node Importance")
    plt.title("Subgraph with Node Importance")
    plt.show()

# Select an edge to explain
edge_idx_to_explain = 0  # First edge in validation set
src, dst = val_data.edge_label_index[:, edge_idx_to_explain]

# Extract the k-hop subgraph around the source node of the edge
num_hops = 2
subset, sub_edge_index, mapping, edge_mask_in_subgraph = k_hop_subgraph(
    src.item(), num_hops=num_hops, edge_index=train_data.edge_index, relabel_nodes=True
)

# Convert the subgraph to a NetworkX graph
subgraph_data = train_data.clone()
subgraph_data.edge_index = sub_edge_index
G = to_networkx(subgraph_data, to_undirected=True, node_attrs=['x'])

# Extract node importance
node_mask = explanation.node_mask.cpu().detach().numpy()
node_importance_aggregated = node_mask.mean(axis=1)  # Aggregate features

# Initialize node colors for all nodes in G
node_colors = np.zeros(len(G.nodes))  # Default color for nodes outside the subgraph
for i, node in enumerate(subset):
    node_colors[node] = node_importance_aggregated[node]

# Extract edge importance
edge_importance = explanation.edge_mask.cpu().detach().numpy()
edge_colors = [edge_importance[i] for i in edge_mask_in_subgraph]

# Prepare edge segments for visualization
pos = nx.spring_layout(G, seed=42)
edge_segments = [(pos[u], pos[v]) for u, v in G.edges]

# Normalize and dynamically set edge importance threshold
edge_colors = np.array(edge_colors)
edge_colors = (edge_colors - np.min(edge_colors)) / (np.max(edge_colors) - np.min(edge_colors) + 1e-8)

# Dynamically determine threshold for sparse edge importance
non_zero_values = edge_colors[edge_colors > 0]
if len(non_zero_values) > 0:
    edge_threshold = np.percentile(non_zero_values, 95)  # 95th percentile of non-zero values
else:
    edge_threshold = 0  # Default fallback for zero edge importance

important_edges = [i for i, val in enumerate(edge_colors) if val > edge_threshold]
if not important_edges:
    print("No edges passed the threshold. Visualizing all edges.")
    filtered_edge_segments = edge_segments
    filtered_edge_colors = edge_colors
else:
    filtered_edge_segments = [edge_segments[i] for i in important_edges]
    filtered_edge_colors = [edge_colors[i] for i in important_edges]

print(f"Important Edge Count: {len(important_edges)}")

# Update LineCollection for edges
lc = LineCollection(filtered_edge_segments, cmap=plt.cm.Blues, norm=Normalize(vmin=0, vmax=max(filtered_edge_colors)))
lc.set_array(filtered_edge_colors)

# Use Kamada-Kawai layout for better spacing
pos = nx.kamada_kawai_layout(G)

# Scale node size by importance
node_sizes = [500 + 5000 * val for val in node_colors]

# Highlight source and target nodes
highlight_nodes = [src.item(), dst.item()]
ax = plt.gca()  # Ensure an axis is defined for matplotlib
nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Reds, node_size=node_sizes, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color="yellow", node_size=800, edgecolors="black", ax=ax)

# Add filtered edges
ax.add_collection(lc)

# Draw important node labels
important_nodes = {node: f"{node}" for node, importance in enumerate(node_colors) if importance > 0.05}
nx.draw_networkx_labels(G, pos, labels=important_nodes, font_size=8, ax=ax)

# Add colorbar for edges
plt.colorbar(lc, ax=ax, label="Edge Importance")
plt.title(f"{num_hops}-Hop Subgraph (Explaining Link: {src.item()} -> {dst.item()})")
plt.axis("off")
plt.show()
