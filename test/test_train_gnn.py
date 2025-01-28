import torch
from torch_geometric.data import HeteroData
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import add_self_loops

data_dir = "../test_data"

def load_data():
    data = HeteroData()

    # --- Person nodes ---
    person_df = pd.read_csv(f"{data_dir}/person_nodes.csv")
    # Person features: budget_per_task
    data['Person'].x = torch.tensor(person_df[['budget_per_task']].values, dtype=torch.float)

    data['Person'].x = data['Person'].x / 10000.0

    # --- AI Agent nodes ---
    agent_df = pd.read_csv(f"{data_dir}/agent_nodes.csv")
    # Agent features: cost_per_task
    data['AI Agent'].x = torch.tensor(agent_df[['cost_per_task']].values, dtype=torch.float)

    # --- Skill nodes ---
    skill_df = pd.read_csv(f"{data_dir}/skill_nodes.csv")
    data['Skill'].num_nodes = len(skill_df)
    # We’ll give each Skill a trivial feature vector for now:
    # If you wish, you can encode skill_label into a vector here (e.g., one-hot or embedding).
    data['Skill'].x = torch.zeros((data['Skill'].num_nodes, 1), dtype=torch.float)

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

    # --- Link prediction edges (Person -> AI Agent) ---
    link_prediction_edges_df = pd.read_csv(f"{data_dir}/link_prediction_edges.csv")
    data['Person', 'link_prediction', 'AI Agent'].edge_index = torch.tensor(
        link_prediction_edges_df[['person_id', 'agent_id']].values.T, dtype=torch.long
    )
    data['Person', 'link_prediction', 'AI Agent'].edge_label = torch.tensor(
        link_prediction_edges_df['link_score'].values, dtype=torch.float
    )

    # --- Add self-loops ONLY for the skill-related edges ---
    # This can help the 'Skill' node to retain its own info when updating.
    data['Person', 'has_skill', 'Skill'].edge_index, _ = add_self_loops(
        data['Person', 'has_skill', 'Skill'].edge_index,
        num_nodes=data['Person'].num_nodes
    )
    data['AI Agent', 'offers_skill', 'Skill'].edge_index, _ = add_self_loops(
        data['AI Agent', 'offers_skill', 'Skill'].edge_index,
        num_nodes=data['AI Agent'].num_nodes
    )

    # --- Do NOT add self-loops to link prediction edges ---
    # That caused a mismatch in edge_label size vs. edge_index.

    # --- Create reverse edges so Person/Agent also become destinations ---
    # Person -> Skill => Skill -> Person
    data['Skill', 'rev_has_skill', 'Person'].edge_index = \
        data['Person', 'has_skill', 'Skill'].edge_index.flip(0)

    # AI Agent -> Skill => Skill -> AI Agent
    data['Skill', 'rev_offers_skill', 'AI Agent'].edge_index = \
        data['AI Agent', 'offers_skill', 'Skill'].edge_index.flip(0)

    # Person -> AI Agent => AI Agent -> Person
    data['AI Agent', 'rev_link_prediction', 'Person'].edge_index = \
        data['Person', 'link_prediction', 'AI Agent'].edge_index.flip(0)


    return data


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # We define a single HeteroConv layer with multiple SAGEConv sub-modules.
        # Each tuple (src, rel, dst) is added in both directions, e.g., Person->Skill and Skill->Person, etc.
        # First HeteroConv
        self.conv1 = HeteroConv({
            ('Person', 'has_skill', 'Skill'): SAGEConv((-1, -1), hidden_channels),
            ('Skill', 'rev_has_skill', 'Person'): SAGEConv((-1, -1), hidden_channels),
            ('AI Agent', 'offers_skill', 'Skill'): SAGEConv((-1, -1), hidden_channels),
            ('Skill', 'rev_offers_skill', 'AI Agent'): SAGEConv((-1, -1), hidden_channels),
            ('Person', 'link_prediction', 'AI Agent'): SAGEConv((-1, -1), hidden_channels),
            ('AI Agent', 'rev_link_prediction', 'Person'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

        # Second HeteroConv
        self.conv2 = HeteroConv({
            ('Person', 'has_skill', 'Skill'): SAGEConv((-1, -1), hidden_channels),
            ('Skill', 'rev_has_skill', 'Person'): SAGEConv((-1, -1), hidden_channels),
            ('AI Agent', 'offers_skill', 'Skill'): SAGEConv((-1, -1), hidden_channels),
            ('Skill', 'rev_offers_skill', 'AI Agent'): SAGEConv((-1, -1), hidden_channels),
            ('Person', 'link_prediction', 'AI Agent'): SAGEConv((-1, -1), hidden_channels),
            ('AI Agent', 'rev_link_prediction', 'Person'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

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


def train_model(data, hidden_channels=32, epochs=100, batch_size=2):
    model = HeteroGNN(hidden_channels=hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.BCEWithLogitsLoss()

    # We'll train link prediction with a LinkNeighborLoader on the Person->Agent edges:
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1, -1], # num_neighbors=[10, 10],
        edge_label_index=('Person', 'link_prediction', 'AI Agent'),
        edge_label=data['Person', 'link_prediction', 'AI Agent'].edge_label,
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()

            # Forward pass to get updated embeddings for all node types in the subgraph batch
            out = model(batch.x_dict, batch.edge_index_dict)

            # Extract Person->Agent edges from the batch
            src_idx = batch['Person', 'link_prediction', 'AI Agent'].edge_label_index[0]
            dst_idx = batch['Person', 'link_prediction', 'AI Agent'].edge_label_index[1]

            # Updated embeddings for Person and Agent
            src_emb = out['Person'][src_idx]
            dst_emb = out['AI Agent'][dst_idx]

            # Compute a link score (dot product, could do an MLP instead)
            pred = (src_emb * dst_emb).sum(dim=-1)

            target = batch['Person', 'link_prediction', 'AI Agent'].edge_label
            loss = criterion(pred, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    return model


def save_model(model, path="../gnn_models/hetero_gnn.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def main():
    # Load our heterogeneous data with reversed edges
    data = load_data()

    # Print shapes for sanity check
    print("Person->Agent edge index shape:",
          data['Person', 'link_prediction', 'AI Agent'].edge_index.shape)
    print("Link label shape:",
          data['Person', 'link_prediction', 'AI Agent'].edge_label.shape)

    # Train the model
    trained_model = train_model(data)

    # Save the trained model, just the weights
    save_model(trained_model)

    # complete model including architecture
    # but has code dependencies by pickling the instance
    torch.save(trained_model, "../gnn_models/hetero_gnn_full.pth")

if __name__ == "__main__":
    main()
