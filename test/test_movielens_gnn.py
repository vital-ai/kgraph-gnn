import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.datasets import MovieLens
from torch_geometric.transforms import RandomLinkSplit
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from torch_geometric.explain.explainer import Explainer
from torch_geometric.explain.algorithm import GNNExplainer

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Get the dimensions from the original model's movie encoder
        first_layer = model.movie_encoder[0]
        self.movie_feature_dim = first_layer.in_features
        self.hidden_channels = first_layer.out_features

        # Add prediction head for binary classification
        self.pred_head = torch.nn.Sequential(
            torch.nn.Linear(16, 8),  # Original output is 16-dim
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )

    def forward(self, x, edge_index, **kwargs):
        """Forward pass converting embeddings to binary prediction"""

        movie_features = None

        try:
            # Get number of movies from edge_index
            num_movies = edge_index[1].max().item() + 1

            # Create movie features with correct input dimension
            movie_features = torch.zeros(
                (num_movies, self.movie_feature_dim),
                device=x.device,
                dtype=x.dtype
            )

            # Create heterogeneous dictionary
            x_dict = {
                'user': x,
                'movie': movie_features
            }

            edge_index_dict = {
                ('user', 'rates', 'movie'): edge_index,
                ('movie', 'rev_rates', 'user'): edge_index.flip(0)
            }

            # Get embeddings from original model
            embeddings = self.model(x_dict, edge_index_dict)
            user_embeddings = embeddings['user']

            # Use embeddings for target user only if index is provided
            if 'index' in kwargs:
                index = kwargs['index']
                if torch.is_tensor(index) and index.numel() == 1:
                    user_embeddings = user_embeddings[index]
                elif isinstance(index, int):
                    user_embeddings = user_embeddings[index]

            # Convert to binary prediction
            pred = self.pred_head(user_embeddings)
            return pred

        except Exception as e:
            print(f"\nError in model forward pass: {str(e)}")
            print("\nShape information:")
            print(f"Input user features: {x.shape}")
            print(f"Generated movie features: {movie_features.shape}")
            print(f"Edge index: {edge_index.shape}")
            if 'index' in kwargs:
                print(f"Index: {kwargs['index']}")
            raise


class ExplanationAnalyzer:
    def __init__(self, model, data, movie_df):
        self.movielens_to_graph = None
        self.graph_to_movielens = None
        self.model = model
        self.data = data
        self.movie_df = movie_df
        self.edge_type = ('user', 'rates', 'movie')
        self.edge_index = data[self.edge_type].edge_index

        # Create mapping between graph indices and MovieLens IDs
        self.setup_movie_mapping()

    def setup_movie_mapping(self):
        """Create mappings between graph indices and MovieLens IDs"""
        try:
            # Get unique movie IDs from the graph
            unique_movies = torch.unique(self.edge_index[1]).tolist()
            print(f"\nFound {len(unique_movies)} unique movies in the graph")

            # Create mapping dictionaries
            self.graph_to_movielens = {}
            self.movielens_to_graph = {}

            # First, print some debug info about the movie_df
            print("\nMovie DataFrame info:")
            print(f"Number of movies in DataFrame: {len(self.movie_df)}")
            print("First few movieIds:", self.movie_df['movieId'].head().tolist())

            # Create the mappings
            for graph_idx in unique_movies:
                try:
                    movie_row = self.movie_df.iloc[graph_idx]
                    movielens_id = movie_row.name
                    self.graph_to_movielens[graph_idx] = movielens_id
                    self.movielens_to_graph[movielens_id] = graph_idx
                except IndexError:
                    print(f"Warning: No movie found for graph index {graph_idx}")

            print(f"Created mappings for {len(self.graph_to_movielens)} movies")

        except Exception as e:
            print(f"Error setting up movie mapping: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_movie_info(self, graph_idx):
        """Get movie information from graph index"""
        try:
            if graph_idx in self.graph_to_movielens:
                movielens_id = self.graph_to_movielens[graph_idx]
                movie_info = self.movie_df.iloc[movielens_id]
                return {
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'id': movielens_id,
                    'graph_idx': graph_idx
                }
            else:
                return {'title': f'Unknown (Graph ID: {graph_idx})', 'genres': 'N/A'}
        except Exception as e:
            return {'title': f'Error looking up movie {graph_idx}: {str(e)}', 'genres': 'N/A'}

    def analyze_explanation(self, explanation, user_id):
        """Analyze a generated explanation"""
        if explanation is None:
            print("No explanation generated")
            return

        print(f"\nExplanation Analysis for User {user_id}")
        print("=" * 50)

        # Feature importance analysis
        if hasattr(explanation, 'node_mask'):
            print("\nFeature Importance:")
            mask = explanation.node_mask
            if isinstance(mask, torch.Tensor):
                if mask.dim() > 1:
                    mask = mask.mean(dim=0)
                values, indices = mask.sort(descending=True)
                for val, idx in zip(values[:5], indices[:5]):
                    print(f"Feature {idx.item()}: {val.item():.4f}")

        # Edge importance analysis
        if hasattr(explanation, 'edge_mask'):
            print("\nMost Influential Movies:")
            mask = explanation.edge_mask
            if isinstance(mask, torch.Tensor):
                edge_imp, edge_idx = mask.topk(5)
                edge_indices = self.edge_index[:, edge_idx]

                for imp, movie_idx in zip(edge_imp, edge_indices[1]):
                    movie_info = self.get_movie_info(movie_idx.item())
                    print(f"\n{movie_info['title']}")
                    print(f"Genres: {movie_info['genres']}")
                    print(f"Influence Score: {imp.item():.4f}")

        # Show user's actual interactions
        print("\nUser's Recent Movies:")
        user_movies = self.edge_index[1][self.edge_index[0] == user_id]
        for i, movie_idx in enumerate(user_movies[:5]):
            movie_info = self.get_movie_info(movie_idx.item())
            print(f"{i + 1}. {movie_info['title']} ({movie_info['genres']})")


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, movie_feature_dim):
        super().__init__()

        self.movie_encoder = torch.nn.Sequential(
            torch.nn.Linear(movie_feature_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Dropout(0.2)
        )

        self.user_encoder = torch.nn.Sequential(
            torch.nn.Linear(16, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Dropout(0.2)
        )

        self.conv1 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv(hidden_channels, hidden_channels),
            ('movie', 'rev_rates', 'user'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv(hidden_channels, out_channels),
            ('movie', 'rev_rates', 'user'): SAGEConv(hidden_channels, out_channels),
        }, aggr='mean')

        self.bn1 = torch.nn.ModuleDict({
            'user': torch.nn.LayerNorm(hidden_channels),
            'movie': torch.nn.LayerNorm(hidden_channels)
        })

        self.bn2 = torch.nn.ModuleDict({
            'user': torch.nn.LayerNorm(out_channels),
            'movie': torch.nn.LayerNorm(out_channels)
        })

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'user': self.user_encoder(x_dict['user']),
            'movie': self.movie_encoder(x_dict['movie'])
        }

        x_dict1 = self.conv1(x_dict, edge_index_dict)
        x_dict1 = {key: F.relu(self.bn1[key](x)) for key, x in x_dict1.items()}

        x_dict2 = self.conv2(x_dict1, edge_index_dict)
        out_dict = {key: self.bn2[key](x) for key, x in x_dict2.items()}

        return out_dict

    def decode(self, z_dict, edge_label_index):
        row, col = edge_label_index
        user_emb = F.normalize(z_dict['user'][row], dim=-1, eps=1e-8)
        movie_emb = F.normalize(z_dict['movie'][col], dim=-1, eps=1e-8)
        return (user_emb * movie_emb).sum(dim=-1).sigmoid()


def create_movie_features(movies_df):
    """Create movie feature matrix from genres and year."""
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)

    genres_set = set()
    for genres in movies_df['genres'].str.split('|'):
        if isinstance(genres, (list, pd.Series)):
            genres_set.update(genres)
    genres_list = sorted(list(genres_set))

    num_movies = len(movies_df)
    num_genres = len(genres_list)
    genre_features = np.zeros((num_movies, num_genres))

    for i, genres in enumerate(movies_df['genres'].str.split('|')):
        if isinstance(genres, (list, pd.Series)):
            for genre in genres:
                genre_idx = genres_list.index(genre)
                genre_features[i, genre_idx] = 1

    years = movies_df['year'].values.reshape(-1, 1)
    year_mean = np.nanmean(years)
    year_std = np.nanstd(years)
    years = np.where(np.isnan(years), year_mean, years)
    years = (years - year_mean) / (year_std + 1e-8)

    movie_features = np.hstack([genre_features, years])
    return torch.FloatTensor(movie_features), genres_list


def process_edge_features(data):
    edge_type = ('user', 'rates', 'movie')
    edge_store = data[edge_type]

    if hasattr(edge_store, 'rating') and edge_store.rating is not None:
        ratings = edge_store.rating.float()
        ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-8)
        edge_store.edge_attr = ratings.view(-1, 1)
        edge_store.edge_weight = ratings

    if edge_type in data.edge_types:
        edge_index = edge_store.edge_index
        rev_edge_store = data['movie', 'rev_rates', 'user']
        rev_edge_store.edge_index = edge_index.flip(0)
        if hasattr(edge_store, 'edge_attr'):
            rev_edge_store.edge_attr = edge_store.edge_attr
        if hasattr(edge_store, 'edge_weight'):
            rev_edge_store.edge_weight = edge_store.edge_weight

    return data


def train(model, train_data, optimizer):
    model.train()
    optimizer.zero_grad()

    z_dict = model(train_data.x_dict, train_data.edge_index_dict)
    edge_label_index = train_data['user', 'rates', 'movie'].edge_label_index
    edge_label = train_data['user', 'rates', 'movie'].edge_label

    edge_label = edge_label.float()
    if edge_label.max() > 1 or edge_label.min() < 0:
        edge_label = (edge_label > 0).float()

    pred = model.decode(z_dict, edge_label_index).view(-1)
    pred = torch.clamp(pred, min=1e-6, max=1 - 1e-6)

    loss = F.binary_cross_entropy(pred, edge_label)

    if torch.isnan(loss).any():
        print("Warning: NaN in loss")
        return float('inf')

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return float(loss)


def test(model, data):
    model.eval()
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)
        edge_label_index = data['user', 'rates', 'movie'].edge_label_index
        edge_label = data['user', 'rates', 'movie'].edge_label

        edge_label = edge_label.float()
        if edge_label.max() > 1 or edge_label.min() < 0:
            edge_label = (edge_label > 0).float()

        pred = model.decode(z_dict, edge_label_index).view(-1)
        pred = torch.clamp(pred, min=1e-6, max=1 - 1e-6)

        try:
            return roc_auc_score(edge_label.cpu().numpy(), pred.cpu().numpy())
        except ValueError as e:
            print(f"Error computing AUC: {str(e)}")
            return 0.5

def save_model(model, optimizer, epoch, best_val_auc, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_auc': best_val_auc,
    }, save_path)
    print(f"Model saved to {save_path}")

def explain_recommendation(model, data, user_id, movie_id, movie_df, genres_list):
    model.eval()
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)

        user_emb = z_dict['user'][user_id]
        movie_emb = z_dict['movie'][movie_id]

        user_emb_norm = F.normalize(user_emb, dim=-1, eps=1e-8)
        movie_emb_norm = F.normalize(movie_emb, dim=-1, eps=1e-8)

        score = (user_emb_norm * movie_emb_norm).sum().item()

        movie_info = movie_df.iloc[movie_id]

        feature_weights = model.movie_encoder[0].weight.detach()
        movie_features = data.x_dict['movie'][movie_id]

        feature_names = genres_list + ['year']
        feature_contribs = []
        for i, (name, feature_val) in enumerate(zip(feature_names, movie_features)):
            if feature_val > 0:
                importance = feature_weights[:, i].abs().mean().item()
                feature_contribs.append((name, float(feature_val), importance))

        feature_contribs.sort(key=lambda x: x[2], reverse=True)

        edge_index = data['user', 'rates', 'movie'].edge_index
        user_movies = edge_index[1][edge_index[0] == user_id].tolist()

        similar_movies = []
        if user_movies:
            user_movie_embs = z_dict['movie'][user_movies]
            user_movie_embs = F.normalize(user_movie_embs, dim=-1, eps=1e-8)
            similarities = (user_movie_embs * movie_emb_norm).sum(dim=-1)

            top_k = min(3, len(user_movies))
            top_sims, top_indices = similarities.topk(top_k)
            similar_movies = [
                (movie_df.iloc[user_movies[idx]]['title'], float(sim))
                for idx, sim in zip(top_indices, top_sims)
            ]

        return {
            'movie_title': movie_info['title'],
            'genres': movie_info['genres'],
            'score': score,
            'feature_contributions': feature_contribs,
            'similar_movies': similar_movies
        }


def print_recommendation_explanation(explanation):
    print("\nRecommendation Explanation")
    print("-" * 50)
    print(f"Movie: {explanation['movie_title']}")
    print(f"Genres: {explanation['genres']}")
    print(f"Recommendation Score: {explanation['score']:.4f}")

    print("\nFeature Analysis:")
    print("-" * 30)
    for feature, value, importance in explanation['feature_contributions']:
        if feature == 'year':
            print(f"{feature:15s}: {value:6.2f} (importance: {importance:.4f})")
        else:
            print(f"{feature:15s}: {'✓':6s} (importance: {importance:.4f})")

    if explanation['similar_movies']:
        print("\nSimilar Movies User Has Interacted With:")
        print("-" * 30)
        for title, sim in explanation['similar_movies']:
            print(f"{title:50s} (similarity: {sim:.4f})")


def recommend_for_user(model, data, user_id, movie_df=None, top_k=5):
    model.eval()
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)
        user_emb = z_dict['user'][user_id]
        movie_embs = z_dict['movie']

        user_emb = F.normalize(user_emb, dim=-1, eps=1e-8)
        movie_embs = F.normalize(movie_embs, dim=-1, eps=1e-8)

        scores = (movie_embs * user_emb).sum(dim=-1)
        scores = torch.nan_to_num(scores, nan=float('-inf'))

        top_scores, top_indices = scores.topk(top_k)

        recommendations = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            if movie_df is not None:
                try:
                    movie_info = movie_df.iloc[idx]
                    recommendations.append({
                        'movie_id': idx,
                        'title': movie_info['title'],
                        'genres': movie_info['genres'],
                        'score': score
                    })
                except IndexError:
                    recommendations.append({
                        'movie_id': idx,
                        'title': f"Unknown Movie (ID: {idx})",
                        'genres': 'N/A',
                        'score': score
                    })
            else:
                recommendations.append({
                    'movie_id': idx,
                    'score': score
                })

        return recommendations

def explain_user_recommendations(model, data, user_id, movie_df, genres_list, top_k=5):
    recommendations = recommend_for_user(model, data, user_id, movie_df, top_k)

    print(f"\nExplanations for User {user_id}'s Recommendations:")
    print("=" * 80)

    for i, rec in enumerate(recommendations, 1):
        movie_id = rec.get('movie_id')
        if movie_id is not None:
            explanation = explain_recommendation(
                model, data, user_id, movie_id, movie_df, genres_list
            )
            print(f"\nRecommendation #{i}")
            print_recommendation_explanation(explanation)

def analyze_model_weights(model, genres_list):
    """Analyze the learned weights of the model to understand feature importance."""
    with torch.no_grad():
        # Analyze movie encoder weights
        movie_weights = model.movie_encoder[0].weight.detach()  # First layer weights

        # Calculate feature importance based on weight magnitudes
        feature_importance = torch.abs(movie_weights).mean(dim=0)

        # Create feature names (genres + year)
        feature_names = genres_list + ['year']

        # Sort features by importance
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, feature_importance)}
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        return sorted_importance


def analyze_genre_relationships(model, data, genres_list):
    """Analyze how different genres are related in the learned embeddings."""
    with torch.no_grad():
        # Get movie embeddings
        movie_embeddings = model(data.x_dict, data.edge_index_dict)['movie']
        movie_embeddings = F.normalize(movie_embeddings, dim=1)

        # Get genre masks
        genre_masks = []
        for genre in genres_list:
            mask = data.x_dict['movie'][:, genres_list.index(genre)] > 0
            genre_masks.append(mask)

        # Calculate average embeddings per genre
        genre_embeddings = []
        for mask in genre_masks:
            if mask.any():
                genre_emb = movie_embeddings[mask].mean(dim=0)
                genre_embeddings.append(genre_emb)
            else:
                genre_embeddings.append(torch.zeros_like(movie_embeddings[0]))

        genre_embeddings = torch.stack(genre_embeddings)

        # Calculate genre similarities
        genre_similarities = torch.mm(genre_embeddings, genre_embeddings.t())

        # Get top relationships for each genre
        relationships = []
        for i, genre in enumerate(genres_list):
            sims, indices = genre_similarities[i].topk(4)  # Get top 3 similar genres (excluding self)
            similar_genres = [(genres_list[idx], float(sim)) for idx, sim in zip(indices[1:], sims[1:])]
            relationships.append((genre, similar_genres))

        return relationships

def analyze_temporal_patterns(model, data, movie_df):
    """Analyze how the model handles different time periods."""
    with torch.no_grad():
        # Get movie embeddings
        movie_embeddings = model(data.x_dict, data.edge_index_dict)['movie']
        movie_embeddings = F.normalize(movie_embeddings, dim=1)

        # Extract years from movie titles
        years = movie_df['title'].str.extract(r'\((\d{4})\)').astype(float).iloc[:, 0]

        # Define decade bins
        decades = {}
        for year in years.dropna().unique():
            decade = int(year) // 10 * 10
            if decade not in decades:
                decades[decade] = []
            mask = years == year
            if mask.any():
                decade_embs = movie_embeddings[mask]
                decades[decade].extend(decade_embs)

        # Calculate average embedding per decade
        decade_embeddings = {
            decade: torch.stack(embs).mean(dim=0)
            for decade, embs in decades.items()
            if embs
        }

        # Calculate decade similarities
        decade_sims = {}
        decades_list = sorted(decade_embeddings.keys())
        for d1 in decades_list:
            sims = []
            for d2 in decades_list:
                sim = F.cosine_similarity(
                    decade_embeddings[d1].unsqueeze(0),
                    decade_embeddings[d2].unsqueeze(0)
                ).item()
                sims.append((d2, sim))
            decade_sims[d1] = sorted(sims, key=lambda x: x[1], reverse=True)

        return decade_sims

def print_model_analysis(feature_importance, genre_relationships, temporal_patterns):
    """Print the model analysis results in a readable format."""
    print("\nModel Analysis Results")
    print("=" * 50)

    print("\nFeature Importance:")
    print("-" * 30)
    for feature, importance in feature_importance:
        print(f"{feature:15s}: {importance:.4f}")

    print("\nGenre Relationships:")
    print("-" * 30)
    for genre, similar_genres in genre_relationships:
        print(f"\n{genre} is most similar to:")
        for similar_genre, similarity in similar_genres:
            print(f"  - {similar_genre:15s}: {similarity:.4f}")

    print("\nTemporal Patterns:")
    print("-" * 30)
    for decade, similarities in temporal_patterns.items():
        print(f"\nMovies from {decade}s are most similar to:")
        # Print top 3 most similar decades (excluding self)
        for other_decade, similarity in similarities[1:4]:
            print(f"  - {other_decade}s: {similarity:.4f}")


def analyze_user_influence(model, data, genres_list, user_id, movie_df):
    """Analyze how user similarities affect recommendations compared to direct features."""
    model.eval()
    with torch.no_grad():
        # Get embeddings
        z_dict = model(data.x_dict, data.edge_index_dict)

        # Get user embeddings and normalize
        user_embs = F.normalize(z_dict['user'], dim=-1)
        target_user_emb = user_embs[user_id]

        # Calculate user similarities
        user_sims = torch.mm(user_embs, target_user_emb.unsqueeze(1)).squeeze()

        # Find most similar users
        top_k_users = 5
        similar_user_scores, similar_user_indices = user_sims.topk(top_k_users + 1)  # +1 for self

        # Get movie preferences for similar users
        edge_index = data['user', 'rates', 'movie'].edge_index

        # Analyze movies rated by similar users
        similar_users_movies = {}
        for idx, sim_user_idx in enumerate(similar_user_indices[1:]):  # Skip self
            user_movies = edge_index[1][edge_index[0] == sim_user_idx].tolist()
            if user_movies:
                similar_users_movies[sim_user_idx.item()] = {
                    'similarity': similar_user_scores[idx + 1].item(),
                    'movies': user_movies
                }

        # Get recommendations for target user
        movie_embs = F.normalize(z_dict['movie'], dim=-1)
        scores = torch.mm(movie_embs, target_user_emb.unsqueeze(1)).squeeze()

        # Get top recommendations
        top_k_recs = 5
        top_scores, top_indices = scores.topk(top_k_recs)

        # Analyze feature vs. user similarity influence
        recommendations_analysis = []
        for movie_idx, score in zip(top_indices, top_scores):
            movie_idx = movie_idx.item()

            # Get movie info
            movie_info = movie_df.iloc[movie_idx]

            # Find similar users who rated this movie
            users_who_rated = []
            for sim_user_idx, user_data in similar_users_movies.items():
                if movie_idx in user_data['movies']:
                    users_who_rated.append({
                        'user_id': sim_user_idx,
                        'user_similarity': user_data['similarity']
                    })

            # Calculate feature similarity
            movie_features = data.x_dict['movie'][movie_idx]
            feature_names = genres_list + ['year']
            active_features = [
                (name, float(val))
                for name, val in zip(feature_names, movie_features)
                if val > 0
            ]

            recommendations_analysis.append({
                'movie_title': movie_info['title'],
                'movie_id': movie_idx,
                'score': float(score),
                'genres': movie_info['genres'],
                'similar_users_who_rated': users_who_rated,
                'active_features': active_features,
                'user_influence_score': sum(u['user_similarity'] for u in users_who_rated) / (
                    len(users_who_rated) if users_who_rated else 1),
                'feature_influence_score': len(active_features) / len(feature_names)
            })

        return recommendations_analysis

def print_user_influence_analysis(analysis):
    """Print the user similarity vs. feature influence analysis."""
    print("\nUser Similarity vs. Feature Influence Analysis")
    print("=" * 80)

    for rec in analysis:
        print(f"\nMovie: {rec['movie_title']}")
        print(f"Overall Score: {rec['score']:.4f}")
        print(f"Genres: {rec['genres']}")

        print("\nUser Influence:")
        if rec['similar_users_who_rated']:
            print(f"Number of similar users who rated: {len(rec['similar_users_who_rated'])}")
            print(f"Average user similarity: {rec['user_influence_score']:.4f}")
            print("\nSimilar users who rated this movie:")
            for user in rec['similar_users_who_rated']:
                print(f"  User {user['user_id']}: similarity = {user['user_similarity']:.4f}")
        else:
            print("No similar users rated this movie")

        print("\nFeature Influence:")
        print(f"Feature coverage: {rec['feature_influence_score']:.4f}")
        print("Active features:")
        for feature, value in rec['active_features']:
            if feature == 'year':
                print(f"  {feature}: {value:.2f}")
            else:
                print(f"  {feature}: ✓")

        print(f"\nInfluence Balance:")
        print(f"User-based influence: {rec['user_influence_score']:.4f}")
        print(f"Feature-based influence: {rec['feature_influence_score']:.4f}")
        print("-" * 40)

def analyze_gnn_structure_influence(model, data, movie_df):
    """Analyze how the GNN balances message passing vs direct features."""
    model.eval()
    with torch.no_grad():
        # Get final embeddings after message passing
        final_embeddings = model(data.x_dict, data.edge_index_dict)
        final_user_emb = final_embeddings['user']
        final_movie_emb = final_embeddings['movie']

        # Normalize embeddings for comparison
        final_user_emb_norm = F.normalize(final_user_emb, dim=1)
        final_movie_emb_norm = F.normalize(final_movie_emb, dim=1)

        # Analyze graph structure influence
        edge_index = data['user', 'rates', 'movie'].edge_index
        user_degrees = torch.bincount(edge_index[0], minlength=final_user_emb.size(0))
        movie_degrees = torch.bincount(edge_index[1], minlength=final_movie_emb.size(0))

        # Calculate user similarity matrix
        user_sims = torch.mm(final_user_emb_norm, final_user_emb_norm.t())

        # Calculate movie similarity matrix
        movie_sims = torch.mm(final_movie_emb_norm, final_movie_emb_norm.t())

        # Analyze neighborhood influence
        def get_neighborhood_stats(sims, degrees):
            avg_neighbor_sim = []
            for i in range(len(degrees)):
                if degrees[i] > 0:
                    # Get similarities excluding self
                    node_sims = sims[i]
                    node_sims[i] = 0  # exclude self-similarity
                    top_sims, _ = node_sims.topk(min(5, len(node_sims)))
                    avg_neighbor_sim.append(top_sims.mean().item())
            return {
                'mean': np.mean(avg_neighbor_sim),
                'std': np.std(avg_neighbor_sim),
                'max': np.max(avg_neighbor_sim),
                'min': np.min(avg_neighbor_sim)
            }

        user_neigh_stats = get_neighborhood_stats(user_sims, user_degrees)
        movie_neigh_stats = get_neighborhood_stats(movie_sims, movie_degrees)

        # Analyze genre-based clustering
        genres = movie_df['genres'].str.get_dummies('|')
        genre_names = genres.columns.tolist()
        genre_stats = {}

        for genre in genre_names:
            genre_mask = genres[genre] == 1
            if genre_mask.sum() > 1:  # need at least 2 movies
                genre_movies_emb = final_movie_emb_norm[genre_mask]
                genre_sims = torch.mm(genre_movies_emb, genre_movies_emb.t())

                # Remove self-similarities
                genre_sims.fill_diagonal_(0)

                genre_stats[genre] = {
                    'mean_similarity': float(genre_sims.mean()),
                    'max_similarity': float(genre_sims.max()),
                    'num_movies': int(genre_mask.sum())
                }

        # Calculate degree influence
        degree_influence = {
            'users': float(torch.corrcoef(torch.stack([
                user_degrees.float(),
                torch.norm(final_user_emb_norm, dim=1)
            ]))[0, 1]),
            'movies': float(torch.corrcoef(torch.stack([
                movie_degrees.float(),
                torch.norm(final_movie_emb_norm, dim=1)
            ]))[0, 1])
        }

        return {
            'user_neighborhood': user_neigh_stats,
            'movie_neighborhood': movie_neigh_stats,
            'genre_clustering': genre_stats,
            'degree_influence': degree_influence,
            'user_degree_stats': {
                'mean': float(user_degrees[user_degrees > 0].float().mean()),
                'std': float(user_degrees[user_degrees > 0].float().std()),
                'max': int(user_degrees.max()),
                'min': int(user_degrees[user_degrees > 0].min())
            },
            'movie_degree_stats': {
                'mean': float(movie_degrees[movie_degrees > 0].float().mean()),
                'std': float(movie_degrees[movie_degrees > 0].float().std()),
                'max': int(movie_degrees.max()),
                'min': int(movie_degrees[movie_degrees > 0].min())
            }
        }

def print_structure_analysis(analysis):
    """Print the GNN structure analysis results."""
    print("\nGNN Structure Analysis")
    print("=" * 50)

    print("\nUser Neighborhood Influence:")
    user_stats = analysis['user_neighborhood']
    print(f"Mean similarity: {user_stats['mean']:.4f}")
    print(f"Std dev       : {user_stats['std']:.4f}")
    print(f"Max similarity: {user_stats['max']:.4f}")
    print(f"Min similarity: {user_stats['min']:.4f}")

    print("\nMovie Neighborhood Influence:")
    movie_stats = analysis['movie_neighborhood']
    print(f"Mean similarity: {movie_stats['mean']:.4f}")
    print(f"Std dev       : {movie_stats['std']:.4f}")
    print(f"Max similarity: {movie_stats['max']:.4f}")
    print(f"Min similarity: {movie_stats['min']:.4f}")

    print("\nGenre Clustering:")
    genres = analysis['genre_clustering']
    sorted_genres = sorted(genres.items(), key=lambda x: x[1]['mean_similarity'], reverse=True)
    for genre, stats in sorted_genres:
        print(f"\n{genre}:")
        print(f"Mean similarity: {stats['mean_similarity']:.4f}")
        print(f"Max similarity : {stats['max_similarity']:.4f}")
        print(f"Number of movies: {stats['num_movies']}")

    print("\nDegree Influence:")
    degree_inf = analysis['degree_influence']
    print(f"User degree correlation : {degree_inf['users']:.4f}")
    print(f"Movie degree correlation: {degree_inf['movies']:.4f}")

    print("\nConnectivity Statistics:")
    print("\nUsers:")
    user_deg = analysis['user_degree_stats']
    print(f"Mean degree: {user_deg['mean']:.2f}")
    print(f"Std dev   : {user_deg['std']:.2f}")
    print(f"Max degree: {user_deg['max']}")
    print(f"Min degree: {user_deg['min']}")

    print("\nMovies:")
    movie_deg = analysis['movie_degree_stats']
    print(f"Mean degree: {movie_deg['mean']:.2f}")
    print(f"Std dev   : {movie_deg['std']:.2f}")
    print(f"Max degree: {movie_deg['max']}")
    print(f"Min degree: {movie_deg['min']}")

def explain_prediction(model, data, user_id, movie_df):
    """Get explanation for a specific user's recommendations"""
    model_config = {
        'mode': 'binary_classification',
        'task_level': 'node',
        'return_type': 'raw'
    }

    try:
        # Wrap the model
        wrapped_model = ModelWrapper(model)

        # Create explainer
        explainer = Explainer(
            model=wrapped_model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type='model',
            model_config=model_config,
            node_mask_type='attributes',
            edge_mask_type='object'
        )

        # Get user features
        x = data.x_dict['user']
        edge_type = ('user', 'rates', 'movie')
        edge_index = data[edge_type].edge_index

        print(f"\nDebug Information for User {user_id}:")
        print(f"User features shape: {x.shape}")
        print(f"Edge index shape: {edge_index.shape}")

        # Format node index
        node_idx = int(user_id) if not torch.is_tensor(user_id) else user_id.item()

        # Get explanation
        explanation = explainer(
            x=x,
            edge_index=edge_index,
            # don't need to pass target for a "model" explanation
            # target=torch.tensor([1], dtype=torch.float),  # Binary target
            index=torch.tensor([node_idx], dtype=torch.long)
        )

        return explanation

    except Exception as e:
        print(f"Error in explanation generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def explain_recommendations(model, data, movie_df, genres_list):
    """Run explanation analysis for multiple users"""
    print("\nGenerating explanations for recommendations...")

    analyzer = ExplanationAnalyzer(model, data, movie_df)

    test_users = [5, 10, 15]
    for user_id in test_users:
        explanation = explain_prediction(model, data, user_id, movie_df)
        analyzer.analyze_explanation(explanation, user_id)
        print("-" * 50)

def main():
    torch.manual_seed(42)
    data_dir = './data/MovieLens'

    print("Loading data...")
    dataset = MovieLens(root=data_dir)
    data = dataset[0]

    movie_df = pd.read_csv(os.path.join(data_dir, 'raw', 'ml-latest-small', 'movies.csv'))
    movie_features, genres_list = create_movie_features(movie_df)
    data['movie'].x = movie_features

    num_users = data['user'].num_nodes
    data['user'].x = torch.nn.init.xavier_normal_(torch.empty(num_users, 16)) * 0.1

    data = process_edge_features(data)

    transform = RandomLinkSplit(
        num_val=0.15,
        num_test=0.1,
        is_undirected=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        edge_types=[('user', 'rates', 'movie')],
    )

    train_data, val_data, test_data = transform(data)

    # note: code is assuming CPU
    # for small dataset, MPS seems slower than CPU due to overhead
    # if torch.cuda.is_available():
    #    device = torch.device('cuda')
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #    device = torch.device('mps')
    # else:
    #    device = torch.device('cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    movie_feature_dim = train_data['movie'].x.size(1)
    print(f"Movie feature dimension: {movie_feature_dim}")

    # Create save directory
    save_dir = os.path.join(data_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'movie_recommender.pt')

    model = HeteroGNN(
        hidden_channels=32,
        out_channels=16,
        movie_feature_dim=movie_feature_dim
    ).to(device)

    optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    )

    best_val_auc = 0
    best_epoch = 0
    patience = 20
    epochs_without_improvement = 0

    print("Starting training...")
    for epoch in range(1, 100):
        loss = train(model, train_data, optimizer)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

        if epoch % 5 == 0:
            val_auc = test(model, val_data)
            print(f"Validation AUC: {val_auc:.4f}")

            scheduler.step(loss)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_without_improvement = 0
                # Save best model
                save_model(model, optimizer, epoch, best_val_auc, model_path)
            else:
                epochs_without_improvement += 5

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining finished. Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    test_auc = test(model, test_data)
    print(f"Final Test AUC: {test_auc:.4f}")

    print("\nAnalyzing trained model...")
    feature_importance = analyze_model_weights(model, genres_list)
    genre_relationships = analyze_genre_relationships(model, test_data, genres_list)
    temporal_patterns = analyze_temporal_patterns(model, test_data, movie_df)
    print_model_analysis(feature_importance, genre_relationships, temporal_patterns)

    print("\nAnalyzing GNN structure influence...")
    structure_analysis = analyze_gnn_structure_influence(model, test_data, movie_df)
    print_structure_analysis(structure_analysis)

    # Generate and explain recommendations for example users
    example_users = [5, 10, 15]
    for user_id in example_users:
        print(f"\nRecommendations for user {user_id}:")
        print("=" * 50)
        recommendations = recommend_for_user(model, test_data, user_id, movie_df)
        for rec in recommendations:
            print(f"Title: {rec['title']}")
            print(f"Genres: {rec['genres']}")
            print(f"Score: {rec['score']:.4f}")
            print("---")

        # Generate detailed explanations
        print("\nDetailed explanations:")
        explain_user_recommendations(model, test_data, user_id, movie_df, genres_list)

    print("\nAnalyzing user similarity influence...")
    for user_id in [5, 10, 15]:
        print(f"\nAnalyzing recommendations for User {user_id}")
        user_influence = analyze_user_influence(model, test_data, genres_list, user_id, movie_df)
        print_user_influence_analysis(user_influence)

    print("\nRunning explanation analysis...")
    explain_recommendations(model, test_data, movie_df, genres_list)


if __name__ == "__main__":
    main()

