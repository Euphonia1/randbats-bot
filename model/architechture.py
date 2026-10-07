import tools
import torch
import torch.nn as nn
import config
from psjax import consts as C
from psjax.data import load_data, names

def build_move_feature_table() -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute explicit features for every move.
 
    Returns:
        features: (num_moves, num_features) float tensor, roughly scaled to [0, 1]
        type_ids: (num_moves,) long tensor with each move's type ID
    """
    num_moves = len(tools.MOVE_NAMES)
    num_features = C.NUM_NUMERIC + len(C.CATEGORIES) + len(C.MOVE_FLAGS)
    features = torch.zeros(num_moves, num_features)
    type_ids = torch.zeros(num_moves, dtype=torch.long)
 
    for move_id, name in enumerate(tools.MOVE_NAMES):
        m = tools.MOVE_DATA[name]
        type_ids[move_id] = C.TYPE_TO_ID[m["type"]]
 
        always_hits = m["accuracy"] is True
        numeric = [
            m["basePower"] / 250.0,
            1.0 if always_hits else m["accuracy"] / 100.0,
            float(always_hits),
            m["pp"] / 64.0,
            m.get("priority", 0) / 5.0,
            # Moves like Low Kick list 0 power but still deal damage
            float(m["basePower"] == 0 and m["category"] != "Status"),
        ]
        category = [float(m["category"] == c) for c in C.CATEGORIES]
        flags = [float(f in m.get("flags", {})) for f in C.MOVE_FLAGS]
 
        features[move_id] = torch.tensor(numeric + category + flags)
 
    return features, type_ids


class TypeEmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(C.NUM_TYPES, config.TYPE_EMBEDDING_OUTPUT_DIM) 

    def forward(self, type_id: int):
        return self.embedding(torch.tensor(type_id))

    
class MoveEmbeddingLayer(nn.Module):
    def __init__(self, type_embedding: TypeEmbeddingLayer, output_dim: int = 32):
        super().__init__()
        self.type_embedding = type_embedding  # shared with Pokémon types
        self.move_embedding = nn.Embedding(len(tools.MOVE_NAMES), config.MOVE_EMBEDDING_OUTPUT_DIM)
 
        features, type_ids = build_move_feature_table()
        # Buffers move with the model to GPU and are saved in state_dict,
        # but are not trained.
        self.register_buffer("move_features", features)
        self.register_buffer("move_type_ids", type_ids)
 
        in_dim = (
            config.MOVE_EMBEDDING_OUTPUT_DIM
            + config.TYPE_EMBEDDING_OUTPUT_DIM
            + features.shape[1]
        )
        self.output_dim = output_dim
        self.project = nn.Sequential(nn.Linear(in_dim, output_dim), nn.ReLU())
 
    def forward(self, move_ids: torch.Tensor) -> torch.Tensor:
        """move_ids: long tensor of any shape, e.g. (batch,) or (batch, 4).
        Returns a tensor of shape (*move_ids.shape, output_dim)."""
        learned = self.move_embedding(move_ids)
        move_type = self.type_embedding(self.move_type_ids[move_ids])
        explicit = self.move_features[move_ids]
        return self.project(torch.cat([learned, move_type, explicit], dim=-1))

def Pokemon_Embedding_Layer(self, species: str, item: str, ability: str, moves: list[str]):
    pass



type_emb = TypeEmbeddingLayer()
move_layer = MoveEmbeddingLayer(type_emb)
n = tools.names()
ids = n.move_id("close combat")
out = move_layer(ids)
print(out.shape)  # should be (32,)