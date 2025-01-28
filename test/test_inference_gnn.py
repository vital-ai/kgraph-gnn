import torch
import pandas as pd
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.utils import add_self_loops

data_dir = "../test_data"
MODEL_PATH = "../gnn_models/hetero_gnn.pth"


def load_data():
    data = HeteroData()

    # --- Person nodes ---
    person_df = pd.read_csv(f"{data_dir}/person_nodes.csv")
    data['Person'].x = torch.tensor(person_df[['budget_per_task']].values, dtype=torch.float)

    data['Person'].x = data['Person'].x / 10000.0

    # --- AI Agent nodes ---
    agent_df = pd.read_csv(f"{data_dir}/agent_nodes.csv")
    data['AI Agent'].x = torch.tensor(agent_df[['cost_per_task']].values, dtype=torch.float)

    # --- Skill nodes ---
    skill_df = pd.read_csv(f"{data_dir}/skill_nodes.csv")
    data['Skill'].num_nodes = len(skill_df)

    data['Skill'].x = torch.zeros((data['Skill'].num_nodes, 1), dtype=torch.float)

    # Assign a trivial feature vector for each skill if not done in training:
    if 'Skill' not in data.node_types or data['Skill'].x is None:
        data['Skill'].x = torch.zeros((len(skill_df), 1), dtype=torch.float)

    # --- Person-Skill edges ---
    person_skill_edges_df = pd.read_csv(f"{data_dir}/person_skill_edges.csv")
    data['Person', 'has_skill', 'Skill'].edge_index = torch.tensor(
        person_skill_edges_df.values.T, dtype=torch.long
    )

    # --- AI Agent-Skill edges ---
    agent_skill_edges_df = pd.read_csv(f"{data_dir}/agent_skill_edges.csv")
    data['AI Agent', 'offers_skill', 'Skill'].edge_index = torch.tensor(
        agent_skill_edges_df.values.T, dtype=torch.long
    )

    # --- Link-prediction edges ---
    link_prediction_edges_df = pd.read_csv(f"{data_dir}/link_prediction_edges.csv")
    data['Person', 'link_prediction', 'AI Agent'].edge_index = torch.tensor(
        link_prediction_edges_df[['person_id', 'agent_id']].values.T, dtype=torch.long
    )
    data['Person', 'link_prediction', 'AI Agent'].edge_label = torch.tensor(
        link_prediction_edges_df['link_score'].values, dtype=torch.float
    )

    # Optional self-loops for skill edges only (matching training script)
    data['Person', 'has_skill', 'Skill'].edge_index, _ = add_self_loops(
        data['Person', 'has_skill', 'Skill'].edge_index,
        num_nodes=data['Person'].num_nodes
    )
    data['AI Agent', 'offers_skill', 'Skill'].edge_index, _ = add_self_loops(
        data['AI Agent', 'offers_skill', 'Skill'].edge_index,
        num_nodes=data['AI Agent'].num_nodes
    )

    # Create reverse edges (same as training)
    data['Skill', 'rev_has_skill', 'Person'].edge_index = \
        data['Person', 'has_skill', 'Skill'].edge_index.flip(0)

    data['Skill', 'rev_offers_skill', 'AI Agent'].edge_index = \
        data['AI Agent', 'offers_skill', 'Skill'].edge_index.flip(0)

    data['AI Agent', 'rev_link_prediction', 'Person'].edge_index = \
        data['Person', 'link_prediction', 'AI Agent'].edge_index.flip(0)

    return data


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # First HeteroConv
        self.conv1 = HeteroConv({
            ('Person', 'has_skill', 'Skill'): SAGEConv((-1, -1), hidden_channels),
            ('Skill', 'rev_has_skill', 'Person'): SAGEConv((-1, -1), hidden_channels),
            ('AI Agent', 'offers_skill', 'Skill'): SAGEConv((-1, -1), hidden_channels),
            ('Skill', 'rev_offers_skill', 'AI Agent'): SAGEConv((-1, -1), hidden_channels),
            ('Person', 'link_prediction', 'AI Agent'): SAGEConv((-1, -1), hidden_channels),
            ('AI Agent', 'rev_link_prediction', 'Person'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        # Second HeteroConv
        self.conv2 = HeteroConv({
            ('Person', 'has_skill', 'Skill'): SAGEConv((-1, -1), hidden_channels),
            ('Skill', 'rev_has_skill', 'Person'): SAGEConv((-1, -1), hidden_channels),
            ('AI Agent', 'offers_skill', 'Skill'): SAGEConv((-1, -1), hidden_channels),
            ('Skill', 'rev_offers_skill', 'AI Agent'): SAGEConv((-1, -1), hidden_channels),
            ('Person', 'link_prediction', 'AI Agent'): SAGEConv((-1, -1), hidden_channels),
            ('AI Agent', 'rev_link_prediction', 'Person'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        # Optional linear for final link prediction
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        # First layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # Second layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        return x_dict


def main():
    # Load data
    data = load_data()

    # Build the same model architecture
    hidden_channels = 32  # Must match training
    model = HeteroGNN(hidden_channels=hidden_channels)

    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Get updated embeddings for the *entire* graph in one forward pass
    out = model(data.x_dict, data.edge_index_dict)

    # Retrieve all Person->Agent edges
    edge_index = data['Person', 'link_prediction', 'AI Agent'].edge_index
    edge_label = data['Person', 'link_prediction', 'AI Agent'].edge_label

    src_idx = edge_index[0]  # Person node indices
    dst_idx = edge_index[1]  # Agent node indices

    # Access updated node embeddings
    src_emb = out['Person'][src_idx]
    dst_emb = out['AI Agent'][dst_idx]

    # Compute scores (dot product or your chosen function)
    scores = (src_emb * dst_emb).sum(dim=-1)

    print("=== Link Prediction Scores ===")
    for i in range(scores.size(0)):
        p_id = src_idx[i].item()
        a_id = dst_idx[i].item()
        label = edge_label[i].item()
        score = scores[i].item()
        print(f"Person {p_id} -> Agent {a_id}: label={label}, score={score:.4f}")


if __name__ == "__main__":
    main()