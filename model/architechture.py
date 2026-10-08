import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from psjax import consts as C
from psjax.data import load_data, names
from psjax.fog import HISTORY_LEN

# Flag names in bit order, so flag column i is FLAG_NAMES[i].
FLAG_NAMES = sorted(C.FLAG_BITS, key=C.FLAG_BITS.get)
# Statuses in one-hot column order; no status is all zeros.
STATUSES = (C.BRN, C.PAR, C.SLP, C.FRZ, C.PSN, C.TOX)
# Volatiles with a fixed duration, so both players can count the turns left.
# The rest only show as present or absent: confusion and Outrage, for example,
# last a random number of turns that neither player is told.
PUBLIC_VOLATILE_TURNS = (C.V_PERISHSONG, C.V_TAUNT, C.V_ENCORE, C.V_DISABLE, C.V_YAWN,
                         C.V_SLOWSTART, C.V_MAGNETRISE, C.V_THROATCHOP, C.V_SYRUPBOMB)
# ID inputs use -1 for "none" (an empty move slot, a mono-type's second type),
# which embeds as zeros, and UNKNOWN for something the opponent has not
# revealed, which embeds as a learned vector of its own.
UNKNOWN = -2
# How far back the history tells turns apart: an event this turn, last turn,
# ..., and anything older all alike.
HISTORY_TURNS = 8


def build_move_feature_table() -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute explicit features for every move.

    Rows are indexed by the engine's move IDs (`names().move_id(...)`), and
    values come from the compiled tables (`gen9.npz`).

    Returns:
        features: (num_moves, num_features) float tensor, roughly scaled to [0, 1]
        type_ids: (num_moves,) long tensor with each move's type ID
    """
    data = load_data()
    base_power = np.asarray(data["move_base_power"])
    accuracy = np.asarray(data["move_accuracy"])
    category = np.asarray(data["move_category"])

    always_hits = accuracy == -1
    numeric = np.stack([
        base_power / 250.0,
        np.where(always_hits, 1.0, accuracy / 100.0),
        always_hits,
        np.asarray(data["move_pp"]) / 64.0,
        np.asarray(data["move_priority"]) / 5.0,
        # Moves like Low Kick list 0 power but still deal damage
        (base_power == 0) & (category != C.CAT_STATUS),
    ], axis=1)
    category_one_hot = np.eye(len(C.CATEGORIES))[category]
    bits = np.array([C.FLAG_BITS[f] for f in FLAG_NAMES])
    flags = np.asarray(data["move_flags"])[:, None] >> bits & 1

    features = np.concatenate([numeric, category_one_hot, flags, _move_effects(data)], axis=1)
    type_ids = np.asarray(data["move_type"])
    return (torch.tensor(features, dtype=torch.float32),
            torch.tensor(type_ids, dtype=torch.long))


def _move_effects(data) -> np.ndarray:
    """(num_moves, num_effects) features for what each move does beyond its damage.

    Secondary effects count by their chance, so Scald's burn is 0.3 and Fiery
    Dance's Sp. Atk boost is 0.5 of a stage. Stat changes, statuses and
    volatiles land on the move's target; `hits_foe` says whether that is the
    opponent (Growl, Spore) or the user (Swords Dance, Substitute).
    """
    col = lambda k: np.asarray(data[k])
    chance = col("move_sec_chance") / 100.0  # (num_moves, 2), one per secondary
    statuses = np.array(STATUSES)

    hits_foe = np.isin(col("move_target"), C.FOE_TARGETS)
    status = col("move_status")[:, None] == statuses
    sec_status = (chance[..., None] * (col("move_sec_status")[..., None] == statuses)).sum(1)
    flinch = (chance * (col("move_sec_volatile") == C.V_FLINCH)).sum(1)
    confuse = (chance * (col("move_sec_volatile") == C.V_CONFUSION)).sum(1)
    # Expected stat stages, from the move itself plus its secondaries
    boosts = col("move_boosts") + (chance[..., None] * col("move_sec_boosts")).sum(1)
    self_boosts = col("move_self_boosts") + (chance[..., None] * col("move_sec_self_boosts")).sum(1)
    fraction = lambda k: col(k)[:, 0] / np.maximum(col(k)[:, 1], 1)

    side_conditions = np.arange(C.NUM_SIDE_CONDITIONS)
    volatiles = np.arange(C.NUM_VOLATILES)
    return np.concatenate([
        np.stack([hits_foe, flinch, confuse], axis=1),
        status, sec_status,
        boosts / 2.0, self_boosts / 2.0,
        np.stack([
            fraction("move_drain"), fraction("move_recoil"), fraction("move_heal"),
            col("move_multihit").mean(1) / 5.0,  # average hits
            col("move_ohko"), col("move_crit_ratio") > 1, col("move_will_crit"),
            col("move_self_switch") > 0,  # U-turn, Volt Switch, Baton Pass, Shed Tail
            col("move_force_switch"),  # Roar, Dragon Tail
            col("move_selfdestruct") > 0,
            col("move_stalling"),  # Protect and friends
        ], axis=1),
        (col("move_side_condition")[:, None] == side_conditions)
        | (col("move_self_side_condition")[:, None] == side_conditions),
        col("move_weather")[:, None] == np.arange(1, C.NUM_WEATHER),
        col("move_terrain")[:, None] == np.arange(1, C.NUM_TERRAIN),
        (col("move_volatile")[:, None] == volatiles)
        | (col("move_self_volatile")[:, None] == volatiles),
    ], axis=1).astype(np.float32)


def build_species_table() -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute each species' types and base stats.

    Returns:
        type_ids: (num_species, 2) long tensor; the second type is
            C.TYPE_NONE (-1) for a mono-type species
        base_stats: (num_species, 6) float tensor, one column per stat in
            C.STAT_NAMES order (hp, atk, def, spa, spd, spe), each scaled to
            [0, 1] by 255, the highest a base stat can be
    """
    data = load_data()
    type_ids = np.asarray(data["species_types"])
    base_stats = np.asarray(data["species_base_stats"]) / 255.0
    return (torch.tensor(type_ids, dtype=torch.long),
            torch.tensor(base_stats, dtype=torch.float32))


def unknown_vector(dim: int) -> nn.Parameter:
    """The learned embedding for an UNKNOWN ID."""
    return nn.Parameter(torch.randn(dim) * 0.02)


def embed(layer: nn.Module, ids: torch.Tensor,
          unknown: torch.Tensor | None = None) -> torch.Tensor:
    """Apply `layer` to `ids`, with -1 (none) as zeros and UNKNOWN as the
    learned `unknown` vector."""
    out = layer(ids.clamp(min=0)) * (ids >= 0).unsqueeze(-1)
    if unknown is not None:
        out = torch.where((ids == UNKNOWN).unsqueeze(-1), unknown.to(out.dtype), out)
    return out


def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Two layers, normalized at the end: every embedding's final step."""
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                         nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim))


def slot_one_hot(slot: torch.Tensor) -> torch.Tensor:
    """A move slot, (batch,), as a (batch, 4) one-hot; -1 (none) is all zeros."""
    return slot.unsqueeze(-1) == torch.arange(C.MOVES_PER_POKEMON, device=slot.device)


def embed_move_slots(move_embedding: "MoveEmbeddingLayer", move_ids: torch.Tensor,
                     pp: torch.Tensor, maxpp: torch.Tensor) -> torch.Tensor:
    """Each move slot's embedding with the fraction of its PP left appended:
    (..., 4) IDs give (..., 4, move_embedding.output_dim + 1). An empty slot
    (-1) is all zeros; an unrevealed one has full PP, as it has never been used."""
    moves = embed(move_embedding, move_ids, move_embedding.unknown)
    pp_left = torch.where(move_ids == -1, 0.0, pp / maxpp.clamp(min=1))
    return torch.cat([moves, pp_left.unsqueeze(-1).to(moves.dtype)], dim=-1)


class TypeEmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(C.NUM_TYPES, config.TYPE_EMBEDDING_OUTPUT_DIM)
        self.unknown = unknown_vector(config.TYPE_EMBEDDING_OUTPUT_DIM)  # an unseen Tera type

    def forward(self, type_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(torch.as_tensor(type_ids))


class MoveEmbeddingLayer(nn.Module):
    def __init__(self, type_embedding: TypeEmbeddingLayer,
                 output_dim: int = config.MOVE_LAYER_OUTPUT_DIM):
        super().__init__()
        self.type_embedding = type_embedding  # shared with Pokémon types
        features, type_ids = build_move_feature_table()
        self.move_embedding = nn.Embedding(features.shape[0], config.MOVE_EMBEDDING_OUTPUT_DIM)

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
        self.project = mlp(in_dim, output_dim)
        self.unknown = unknown_vector(output_dim)  # a move the opponent has not used
 
    def forward(self, move_ids: torch.Tensor) -> torch.Tensor:
        """move_ids: long tensor of any shape, e.g. (batch,) or (batch, 4).
        Returns a tensor of shape (*move_ids.shape, output_dim)."""
        learned = self.move_embedding(move_ids)
        move_type = self.type_embedding(self.move_type_ids[move_ids])
        explicit = self.move_features[move_ids]
        return self.project(torch.cat([learned, move_type, explicit], dim=-1))


class PokemonEmbeddingLayer(nn.Module):
    def __init__(self, type_embedding: TypeEmbeddingLayer, move_embedding: MoveEmbeddingLayer,
                 output_dim: int = config.POKEMON_EMBEDDING_OUTPUT_DIM):
        super().__init__()
        self.type_embedding = type_embedding  # shared with moves
        self.move_embedding = move_embedding  # one layer shared by all four slots
        type_ids, base_stats = build_species_table()
        n = names()
        self.species_embedding = nn.Embedding(type_ids.shape[0], config.SPECIES_EMBEDDING_OUTPUT_DIM)
        self.item_embedding = nn.Embedding(len(n.items), config.ITEM_EMBEDDING_OUTPUT_DIM)
        self.ability_embedding = nn.Embedding(len(n.abilities), config.ABILITY_EMBEDDING_OUTPUT_DIM)
        self.unknown_species = unknown_vector(config.SPECIES_EMBEDDING_OUTPUT_DIM)
        self.unknown_item = unknown_vector(config.ITEM_EMBEDDING_OUTPUT_DIM)
        self.unknown_ability = unknown_vector(config.ABILITY_EMBEDDING_OUTPUT_DIM)

        # Buffers move with the model to GPU and are saved in state_dict,
        # but are not trained.
        self.register_buffer("species_type_ids", type_ids)
        self.register_buffer("species_base_stats", base_stats)
        self.register_buffer("statuses", torch.tensor(STATUSES), persistent=False)

        in_dim = (
            config.SPECIES_EMBEDDING_OUTPUT_DIM
            + config.ITEM_EMBEDDING_OUTPUT_DIM
            + config.ABILITY_EMBEDDING_OUTPUT_DIM
            + 3 * config.TYPE_EMBEDDING_OUTPUT_DIM  # two species types and the Tera type
            + 1  # terastallized
            + 1  # fraction of HP left
            + len(STATUSES)  # status one-hot
            + 2  # failed attempts so far while asleep, and whether Rest caused the sleep
            + C.MOVES_PER_POKEMON * (move_embedding.output_dim + 1)  # each move and its PP left
            + base_stats.shape[1]
            + 1  # level
            + C.NUM_STATS - 1  # Attack through Speed at that level
            + 1  # toxic counter
        )
        self.output_dim = output_dim
        self.project = mlp(in_dim, output_dim)

    def forward(self, species_ids: torch.Tensor, item_ids: torch.Tensor,
                ability_ids: torch.Tensor, move_ids: torch.Tensor,
                pp: torch.Tensor, maxpp: torch.Tensor,
                tera_type_ids: torch.Tensor, terastallized: torch.Tensor,
                hp: torch.Tensor, maxhp: torch.Tensor,
                status: torch.Tensor, sleep_attempts: torch.Tensor,
                rest_sleep: torch.Tensor, level: torch.Tensor, stats: torch.Tensor,
                toxic_counter: torch.Tensor) -> torch.Tensor:
        """Every argument shares one batch shape, e.g. (batch,) or (batch, 6),
        except move_ids, pp and maxpp, which add a trailing dimension of 4.

        species_ids: long, species IDs
        item_ids: long, item IDs; 0 is no item (none held, or consumed)
        ability_ids: long, ability IDs; 0 is none, or one the engine does not model
        move_ids: long, (..., 4) move IDs
        pp: (..., 4) PP left in each move slot
        maxpp: (..., 4) each move slot's maximum PP
        tera_type_ids: long, Tera type IDs in C.TYPES order
        terastallized: bool, whether the Pokemon has Terastallized
        hp: current HP; 0 is fainted
        maxhp: maximum HP (100 for an opponent under fog, where hp is a percent)
        status: long, major status (C.STATUS_NONE, C.BRN, ... C.TOX)
        sleep_attempts: times it has tried to move and stayed asleep
        rest_sleep: bool, whether Rest caused the sleep
            (both are public, and read only while status is C.SLP)
        level: level, 1-100 (Random Battles scales it by how strong the set is)
        stats: (..., 5) Attack, Defense, Sp. Atk, Sp. Def and Speed
        toxic_counter: turns the toxic poison has built up; 0 unless badly poisoned

        UNKNOWN in species_ids, item_ids, ability_ids, move_ids or
        tera_type_ids marks something the opponent has not revealed yet; -1 is
        an empty move slot. An unrevealed species has unknown types too, and
        zeros for its base stats, level and stats.

        Returns a tensor of shape (*species_ids.shape, output_dim)."""
        species = embed(self.species_embedding, species_ids, self.unknown_species)
        known = (species_ids >= 0).unsqueeze(-1)
        species_ids = species_ids.clamp(min=0)
        item = embed(self.item_embedding, item_ids, self.unknown_item)
        ability = embed(self.ability_embedding, ability_ids, self.unknown_ability)
        types = embed(self.type_embedding,
                      torch.where(known, self.species_type_ids[species_ids], UNKNOWN),
                      self.type_embedding.unknown)
        tera_type = embed(self.type_embedding, tera_type_ids, self.type_embedding.unknown)
        moves = embed_move_slots(self.move_embedding, move_ids, pp, maxpp)
        base_stats = self.species_base_stats[species_ids] * known
        hp_left = (hp / maxhp.clamp(min=1)).unsqueeze(-1).to(species.dtype)
        asleep = status == C.SLP
        sleep = torch.stack([
            torch.where(asleep, sleep_attempts / 3.0, 0.0),
            asleep & rest_sleep,
        ], dim=-1).to(species.dtype)
        status = (status.unsqueeze(-1) == self.statuses).to(species.dtype)
        level = (level.unsqueeze(-1) / 100.0 * known).to(species.dtype)
        stats = (stats / 500.0 * known).to(species.dtype)
        toxic_counter = (toxic_counter.unsqueeze(-1) / 15.0).to(species.dtype)
        return self.project(torch.cat([
            species, item, ability, types.flatten(-2), tera_type,
            terastallized.unsqueeze(-1).to(species.dtype), hp_left, status, sleep,
            moves.flatten(-2), base_stats, level, stats, toxic_counter,
        ], dim=-1))


class PlayerEmbeddingLayer(nn.Module):
    """What belongs to a player rather than to one Pokemon: how many are left,
    whether Tera is spent, and the active slot's state, which clears when its
    Pokemon switches out. The Pokemon themselves are tokens of their own."""

    def __init__(self, move_embedding: MoveEmbeddingLayer,
                 output_dim: int = config.PLAYER_LAYER_OUTPUT_DIM):
        super().__init__()
        self.move_embedding = move_embedding  # for the last move used
        in_dim = (
            1  # how many Pokemon are left
            + 1  # whether Tera has been used
            + C.NUM_BOOSTS  # the active Pokemon's stat stages
            + C.NUM_VOLATILES  # and which volatile conditions it has
            + len(PUBLIC_VOLATILE_TURNS)  # turns left on the ones anyone can count
            + move_embedding.output_dim  # the last move it used
            + 1  # whether this is its first turn out (Fake Out, First Impression)
            + 4 * C.MOVES_PER_POKEMON  # Choice-locked, Encored, Disabled, locked-in slot
        )
        self.output_dim = output_dim
        self.project = mlp(in_dim, output_dim)
        self.register_buffer("public_volatile_turns", torch.tensor(PUBLIC_VOLATILE_TURNS),
                             persistent=False)

    def forward(self, pokemon_left: torch.Tensor, tera_used: torch.Tensor,
                boosts: torch.Tensor, volatiles: torch.Tensor, last_move: torch.Tensor,
                moves_since_switch: torch.Tensor, choice_slot: torch.Tensor,
                encore_slot: torch.Tensor, disabled_slot: torch.Tensor,
                locked_slot: torch.Tensor) -> torch.Tensor:
        """pokemon_left: (batch,) how many of the six have not fainted
        tera_used: bool (batch,), whether any of them has Terastallized
        boosts: (batch, C.NUM_BOOSTS) stat stages, -6..6, in C.BOOST_NAMES order
        volatiles: (batch, C.NUM_VOLATILES) turns left, 0 absent, in C.V_* order;
            only presence is used, plus turns left for PUBLIC_VOLATILE_TURNS
        last_move: long, (batch,) move ID of the last move it used; -1 none
        moves_since_switch: (batch,) moves it has made since switching in
        choice_slot, encore_slot, disabled_slot, locked_slot: long, (batch,)
            the move slot it is held to by a Choice item, Encore, Disable, or a
            move like Outrage; -1 none, and -1 for an opponent's unrevealed
            Choice lock

        Returns a tensor of shape (batch, output_dim)."""
        dtype = self.move_embedding.unknown.dtype
        held_slots = torch.cat([slot_one_hot(s) for s in
                                (choice_slot, encore_slot, disabled_slot, locked_slot)], dim=-1)
        return self.project(torch.cat([
            (pokemon_left / C.TEAM_SIZE).unsqueeze(-1).to(dtype),
            tera_used.unsqueeze(-1).to(dtype),
            (boosts / 6.0).to(dtype), (volatiles > 0).to(dtype),
            (volatiles[..., self.public_volatile_turns] / 5.0).to(dtype),
            embed(self.move_embedding, last_move),
            (moves_since_switch == 0).unsqueeze(-1).to(dtype),
            held_slots.to(dtype),
        ], dim=-1))


class FieldEmbeddingLayer(nn.Module):
    def __init__(self, output_dim: int = config.FIELD_LAYER_OUTPUT_DIM):
        super().__init__()
        # Hazards count layers; every other side condition counts turns left,
        # at most 8 (screens with Light Clay).
        scale = [C.SIDE_CONDITION_MAX.get(i, 8) for i in range(C.NUM_SIDE_CONDITIONS)]
        self.register_buffer("side_condition_scale",
                             torch.tensor(scale, dtype=torch.float32), persistent=False)
        in_dim = (
            C.NUM_WEATHER + C.NUM_TERRAIN
            + 2  # Trick Room and Gravity turns left
            + 2 * C.NUM_SIDE_CONDITIONS  # your side, then the opponent's
            + 3  # decision phase: C.PHASE_MOVE, C.PHASE_SWITCH or C.PHASE_END
            + 2  # who owes a replacement: you, then the opponent
        )
        self.output_dim = output_dim
        self.project = mlp(in_dim, output_dim)

    def forward(self, weather: torch.Tensor, terrain: torch.Tensor,
                trick_room: torch.Tensor, gravity: torch.Tensor,
                side_conditions: torch.Tensor, phase: torch.Tensor,
                force_switch: torch.Tensor) -> torch.Tensor:
        """weather, terrain: long (batch,), C.WEATHER_* and C.TERRAIN_* IDs
        trick_room, gravity: (batch,) turns left; 0 is inactive
        side_conditions: (batch, 2, C.NUM_SIDE_CONDITIONS), your side first;
            hazard layers or turns left, 0 is absent
        phase: long (batch,), what this decision is: C.PHASE_MOVE for a normal
            turn, C.PHASE_SWITCH while a replacement is owed
        force_switch: bool (batch, 2), you first: who owes that replacement,
            after a faint or a U-turn

        Returns a tensor of shape (batch, output_dim)."""
        dtype = self.side_condition_scale.dtype
        return self.project(torch.cat([
            F.one_hot(weather.long(), C.NUM_WEATHER).to(dtype),
            F.one_hot(terrain.long(), C.NUM_TERRAIN).to(dtype),
            (torch.stack([trick_room, gravity], dim=-1) / 5.0).to(dtype),
            (side_conditions / self.side_condition_scale).flatten(-2).to(dtype),
            F.one_hot(phase.long(), 3).to(dtype), force_switch.to(dtype),
        ], dim=-1))


class HistoryEmbeddingLayer(nn.Module):
    """One event from the battle log: a Pokemon, yours or the opponent's,
    used a move or switched in, some turns ago."""

    def __init__(self, species_embedding: nn.Embedding, move_embedding: MoveEmbeddingLayer,
                 output_dim: int = config.HISTORY_LAYER_OUTPUT_DIM):
        super().__init__()
        self.species_embedding = species_embedding  # shared with the Pokemon
        self.move_embedding = move_embedding
        in_dim = (
            species_embedding.embedding_dim  # who acted
            + move_embedding.output_dim  # the move it used; zeros for a switch-in
            + 1  # yours, not the opponent's
            + 1  # a switch-in
            + HISTORY_TURNS  # turns ago, one-hot
        )
        self.output_dim = output_dim
        self.project = mlp(in_dim, output_dim)

    def forward(self, present: torch.Tensor, mine: torch.Tensor, species_ids: torch.Tensor,
                move_ids: torch.Tensor, turns_ago: torch.Tensor) -> torch.Tensor:
        """Each (batch, events):
        present: bool, whether there is an event here at all
        mine: bool, whether it was yours
        species_ids: long, the Pokemon that acted: yours as it is, the
            opponent's as they showed it (an Illusion's disguise)
        move_ids: long, the move it used; -1 for a switch-in
        turns_ago: how many turns back it happened; 0 is this turn

        Returns a tensor of shape (batch, events, output_dim)."""
        dtype = self.move_embedding.unknown.dtype
        turns_ago = turns_ago.long().clamp(0, HISTORY_TURNS - 1)
        return self.project(torch.cat([
            embed(self.species_embedding, species_ids),
            embed(self.move_embedding, move_ids),
            mine.unsqueeze(-1).to(dtype),
            (present & (move_ids == -1)).unsqueeze(-1).to(dtype),
            F.one_hot(turns_ago, HISTORY_TURNS).to(dtype),
        ], dim=-1))


# What each transformer token is. Every token has its type's embedding added,
# which is all the content the summary token has. The history's events come
# between the state and the actions, oldest first. The last 14 are the
# actions, in the engine's encoding: 0-3 use a move, 4-7 Terastallize and use
# it, 8-13 switch to that team slot -- scored from your Pokemon's own tokens.
(TOKEN_SUMMARY, TOKEN_ME, TOKEN_OPPONENT, TOKEN_FIELD, TOKEN_THEIR_POKEMON,
 TOKEN_HISTORY, TOKEN_MOVE, TOKEN_TERA_MOVE, TOKEN_MY_POKEMON) = range(9)
TOKEN_TYPES = (
    [TOKEN_SUMMARY, TOKEN_ME, TOKEN_OPPONENT, TOKEN_FIELD]
    + [TOKEN_THEIR_POKEMON] * C.TEAM_SIZE
    + [TOKEN_HISTORY] * HISTORY_LEN
    + [TOKEN_MOVE] * C.MOVES_PER_POKEMON
    + [TOKEN_TERA_MOVE] * C.MOVES_PER_POKEMON
    + [TOKEN_MY_POKEMON] * C.TEAM_SIZE
)
# Matchup features from `game_inputs._matchups`. A move token gets (min roll,
# max roll, present) against the opponent's active Pokemon and up to three of
# their others, then who moves first. Each of your Pokemon gets (min, max) for
# its best move into their active Pokemon, (min, max, known) for their best
# revealed move into it, then who would move first.
MOVE_MATCHUPS = 3 * 4 + 1
SWITCH_MATCHUPS = 2 + 3 + 1


class GameNetwork(nn.Module):
    """Both players, the field, every Pokemon, the recent history and every
    action as tokens, through a transformer.

    The action head scores each action token, so every move and switch is
    judged in the context of the whole game; a switch is scored from the
    token of the Pokemon it brings in. The win head reads the summary token.
    Everything is from your point of view.

    The history is the battle log's last HISTORY_LEN moves and switch-ins, a
    token each, in the order they happened. That order is the one thing the
    state does not show: who moved first, and so who is faster than its
    Speed says (a Choice Scarf) or slower.
    """

    def __init__(self):
        super().__init__()
        self.type_embedding = TypeEmbeddingLayer()
        self.move_embedding = MoveEmbeddingLayer(self.type_embedding)
        # Both shared by the two players
        self.pokemon_embedding = PokemonEmbeddingLayer(self.type_embedding, self.move_embedding)
        self.player_embedding = PlayerEmbeddingLayer(self.move_embedding)
        self.field_embedding = FieldEmbeddingLayer()
        self.history_embedding = HistoryEmbeddingLayer(
            self.pokemon_embedding.species_embedding, self.move_embedding)

        d = config.TRANSFORMER_DIM
        pokemon_dim = self.pokemon_embedding.output_dim
        move_dim = self.move_embedding.output_dim + 1  # with its PP left
        # Each kind of token has its own projection into the transformer's width.
        self.to_player = nn.Linear(self.player_embedding.output_dim, d)
        self.to_field = nn.Linear(self.field_embedding.output_dim, d)
        self.to_their_pokemon = nn.Linear(pokemon_dim, d)
        self.to_move = nn.Linear(move_dim + MOVE_MATCHUPS, d)
        self.to_tera_move = nn.Linear(
            move_dim + config.TYPE_EMBEDDING_OUTPUT_DIM + MOVE_MATCHUPS, d)
        self.to_my_pokemon = nn.Linear(pokemon_dim + SWITCH_MATCHUPS, d)
        self.to_history = nn.Linear(self.history_embedding.output_dim, d)
        self.token_type = nn.Embedding(len(set(TOKEN_TYPES)), d)
        self.register_buffer("token_types", torch.tensor(TOKEN_TYPES), persistent=False)
        # Which Pokemon on each side is on the field
        self.on_field = nn.Embedding(2, d)
        # Each event's place in the history, the newest last, so events in
        # the same turn are in the order they happened.
        self.history_position = nn.Embedding(HISTORY_LEN, d)
        # Added to each action token, so the transformer can weigh the options
        # it actually has (a Choice lock, an Encore, a forced switch).
        self.action_legality = nn.Embedding(2, d)

        layer = nn.TransformerEncoderLayer(
            d, config.TRANSFORMER_HEADS, config.TRANSFORMER_FF_DIM,
            dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, config.TRANSFORMER_LAYERS, norm=nn.LayerNorm(d),
            enable_nested_tensor=False)
        self.action_head = nn.Linear(d, 1)
        self.win_head = nn.Linear(d, 1)

    def forward(self, me: dict, opponent: dict, field: dict[str, torch.Tensor],
                legal_actions: torch.Tensor, matchups: dict[str, torch.Tensor],
                history: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """me, opponent: each a dict of
            team: PokemonEmbeddingLayer.forward's arguments by name, each with
                a team dimension of 6, e.g. species_ids (batch, 6) and move_ids
                (batch, 6, 4)
            active_slot: long (batch,), team index of the active Pokemon
            and PlayerEmbeddingLayer.forward's arguments by name, apart from
            pokemon_left and tera_used, which are read off the team
        field: FieldEmbeddingLayer.forward's arguments by name
        legal_actions: bool (batch, C.NUM_ACTIONS), the engine's legal action mask
        matchups: moves and tera_moves (batch, 4, MOVE_MATCHUPS), switches
            (batch, 6, SWITCH_MATCHUPS)
        history: HistoryEmbeddingLayer.forward's arguments by name, each
            (batch, HISTORY_LEN), oldest first; rows with no event in them
            (until the battle has that many) are masked out

        `game_inputs.game_inputs` builds all of it from a battle, as one player
        is allowed to see it.

        Returns (action_logits, win_logit):
            action_logits: (batch, C.NUM_ACTIONS); illegal actions are set to
                the lowest float, so a softmax gives them probability 0
            win_logit: (batch,); its sigmoid is the probability that you win
        """
        my_team, my_active = me["team"], me["active_slot"].long()
        slots = torch.arange(C.TEAM_SIZE, device=my_active.device)

        def pokemon_and_player(player):
            team = player["team"]
            pokemon = self.pokemon_embedding(**team)  # (batch, 6, pokemon_dim)
            on_field = self.on_field((slots == player["active_slot"].unsqueeze(-1)).long())
            side = {k: v for k, v in player.items() if k not in ("team", "active_slot")}
            side = self.player_embedding(pokemon_left=(team["hp"] > 0).sum(-1),
                                         tera_used=team["terastallized"].any(-1), **side)
            return pokemon, on_field, side

        my_pokemon, my_on_field, my_side = pokemon_and_player(me)
        their_pokemon, their_on_field, their_side = pokemon_and_player(opponent)
        field = self.field_embedding(**field)

        # The active Pokemon's move slots, for the move and Tera-move actions
        rows = torch.arange(my_active.shape[0], device=my_active.device)
        active = {k: v[rows, my_active] for k, v in my_team.items()}
        moves = embed_move_slots(self.move_embedding, active["move_ids"],
                                 active["pp"], active["maxpp"])
        tera_type = embed(self.type_embedding, active["tera_type_ids"],
                          self.type_embedding.unknown)
        tera_moves = torch.cat(
            [moves, tera_type.unsqueeze(-2).expand(-1, C.MOVES_PER_POKEMON, -1)], dim=-1)

        events = self.to_history(self.history_embedding(**history)) \
            + self.history_position.weight  # (batch, HISTORY_LEN, d)

        summary = my_side.new_zeros(my_side.shape[0], 1, self.token_type.embedding_dim)
        state = torch.cat([
            summary,
            self.to_player(my_side).unsqueeze(1),
            self.to_player(their_side).unsqueeze(1),
            self.to_field(field).unsqueeze(1),
            self.to_their_pokemon(their_pokemon) + their_on_field,
        ], dim=1)
        legality = self.action_legality(legal_actions.long())  # (batch, 14, d)
        actions = torch.cat([
            self.to_move(torch.cat([moves, matchups["moves"]], dim=-1)),
            self.to_tera_move(torch.cat([tera_moves, matchups["tera_moves"]], dim=-1)),
            self.to_my_pokemon(torch.cat([my_pokemon, matchups["switches"]], dim=-1))
            + my_on_field,
        ], dim=1) + legality
        tokens = torch.cat([state, events, actions], dim=1) \
            + self.token_type(self.token_types)  # (batch, 24 + HISTORY_LEN, d)
        # Attention skips the history rows with no event in them.
        empty = ~history["present"]
        padding = torch.cat([empty.new_zeros(state.shape[:2]), empty,
                             empty.new_zeros(actions.shape[:2])], dim=1)
        out = self.transformer(tokens, src_key_padding_mask=padding)

        action_logits = self.action_head(out[:, -C.NUM_ACTIONS:]).squeeze(-1)
        action_logits = action_logits.masked_fill(
            ~legal_actions, torch.finfo(action_logits.dtype).min)
        win_logit = self.win_head(out[:, 0]).squeeze(-1)
        return action_logits, win_logit


if __name__ == "__main__":
    type_emb = TypeEmbeddingLayer()
    move_layer = MoveEmbeddingLayer(type_emb)
    n = names()
    ids = torch.tensor(n.move_id("close combat"))
    out = move_layer(ids)
    print(out.shape)  # (config.MOVE_LAYER_OUTPUT_DIM,)

    pokemon_layer = PokemonEmbeddingLayer(type_emb, move_layer)
    species = torch.tensor(n.species_id("great tusk"))
    item = torch.tensor(n.item_id("booster energy"))
    ability = torch.tensor(n.ability_id("protosynthesis"))
    moves = torch.tensor([n.move_id(m) for m in
                          ["headlong rush", "close combat", "ice spinner", "rapid spin"]])
    maxpp = torch.tensor([8, 8, 24, 64])
    pp = torch.tensor([8, 7, 24, 64])
    tera_type = torch.tensor(n.type_id("ground"))
    terastallized = torch.tensor(False)
    hp = torch.tensor(250)
    maxhp = torch.tensor(311)
    status = torch.tensor(C.STATUS_NONE)
    sleep_attempts = torch.tensor(0)
    rest_sleep = torch.tensor(False)
    level = torch.tensor(79)
    stats = torch.tensor([253, 253, 129, 129, 183])  # Atk, Def, SpA, SpD, Spe at level 79
    toxic_counter = torch.tensor(0)
    out = pokemon_layer(species, item, ability, moves, pp, maxpp, tera_type, terastallized,
                        hp, maxhp, status, sleep_attempts, rest_sleep, level, stats,
                        toxic_counter)
    print(out.shape)  # (config.POKEMON_EMBEDDING_OUTPUT_DIM,)

    player_layer = PlayerEmbeddingLayer(move_layer)
    team = dict(species_ids=species, item_ids=item, ability_ids=ability, move_ids=moves,
                pp=pp, maxpp=maxpp, tera_type_ids=tera_type, terastallized=terastallized,
                hp=hp, maxhp=maxhp, status=status, sleep_attempts=sleep_attempts,
                rest_sleep=rest_sleep, level=level, stats=stats, toxic_counter=toxic_counter)
    # Six copies of the Great Tusk, with slot 0 active, +2 Attack after a Swords Dance
    team = {k: torch.stack([v] * C.TEAM_SIZE) for k, v in team.items()}
    boosts = torch.zeros(C.NUM_BOOSTS)
    boosts[C.B_ATK] = 2
    volatiles = torch.zeros(C.NUM_VOLATILES)
    none = torch.tensor(-1)
    player = dict(team=team, active_slot=torch.tensor(0), boosts=boosts, volatiles=volatiles,
                  last_move=torch.tensor(n.move_id("swords dance")),
                  moves_since_switch=torch.tensor(1), choice_slot=none, encore_slot=none,
                  disabled_slot=none, locked_slot=none)
    out = player_layer(pokemon_left=torch.tensor(6), tera_used=torch.tensor(False),
                       **{k: v for k, v in player.items() if k not in ("team", "active_slot")})
    print(out.shape)  # (config.PLAYER_LAYER_OUTPUT_DIM,)

    game = GameNetwork()
    batch_of_one = lambda p: dict(team={k: v.unsqueeze(0) for k, v in p["team"].items()},
                                  **{k: v.unsqueeze(0) for k, v in p.items() if k != "team"})
    field = dict(weather=torch.tensor([C.WEATHER_NONE]), terrain=torch.tensor([C.TERRAIN_NONE]),
                 trick_room=torch.tensor([0]), gravity=torch.tensor([0]),
                 side_conditions=torch.zeros(1, 2, C.NUM_SIDE_CONDITIONS),
                 phase=torch.tensor([C.PHASE_MOVE]), force_switch=torch.zeros(1, 2, dtype=torch.bool))
    legal = torch.ones(1, C.NUM_ACTIONS, dtype=torch.bool)
    legal[0, C.ACTION_SWITCH_BASE + 0] = False  # slot 0 is already active
    matchups = dict(moves=torch.zeros(1, C.MOVES_PER_POKEMON, MOVE_MATCHUPS),
                    tera_moves=torch.zeros(1, C.MOVES_PER_POKEMON, MOVE_MATCHUPS),
                    switches=torch.zeros(1, C.TEAM_SIZE, SWITCH_MATCHUPS))
    # The leads came in last turn, then Great Tusk used Swords Dance this turn
    history = dict(present=torch.zeros(1, HISTORY_LEN, dtype=torch.bool),
                   mine=torch.zeros(1, HISTORY_LEN, dtype=torch.bool),
                   species_ids=torch.full((1, HISTORY_LEN), -1),
                   move_ids=torch.full((1, HISTORY_LEN), -1),
                   turns_ago=torch.zeros(1, HISTORY_LEN, dtype=torch.long))
    history["present"][0, -3:] = True
    history["mine"][0, [-3, -1]] = True
    history["species_ids"][0, -3:] = species
    history["move_ids"][0, -1] = n.move_id("swords dance")
    history["turns_ago"][0, -3:] = torch.tensor([1, 1, 0])
    action_logits, win_logit = game(batch_of_one(player), batch_of_one(player), field, legal,
                                    matchups, history)
    print(action_logits.softmax(-1).shape)  # should be (1, 14)
    print(torch.sigmoid(win_logit))  # probability of winning
