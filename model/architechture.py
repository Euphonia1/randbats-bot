import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from psjax import consts as C
from psjax.data import load_data, names

# Flag names in bit order, so flag column i is FLAG_NAMES[i].
FLAG_NAMES = sorted(C.FLAG_BITS, key=C.FLAG_BITS.get)
# Statuses in one-hot column order; no status is all zeros.
STATUSES = (C.BRN, C.PAR, C.SLP, C.FRZ, C.PSN, C.TOX)
# Volatiles with a fixed duration, so both players can count the turns left.
# The rest only show as present or absent: confusion and Outrage, for example,
# last a random number of turns that neither player is told.
PUBLIC_VOLATILE_TURNS = (C.V_PERISHSONG, C.V_TAUNT, C.V_ENCORE, C.V_DISABLE, C.V_YAWN,
                         C.V_SLOWSTART, C.V_MAGNETRISE, C.V_THROATCHOP, C.V_SYRUPBOMB)


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


def embed_or_zero(layer: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Apply `layer` to `ids`, giving a zero vector wherever an ID is -1
    (a mono-type's missing second type, an empty move slot, or anything the
    opponent has not revealed yet)."""
    valid = ids >= 0
    return layer(ids.clamp(min=0)) * valid.unsqueeze(-1)


def slot_one_hot(slot: torch.Tensor) -> torch.Tensor:
    """A move slot, (batch,), as a (batch, 4) one-hot; -1 (none) is all zeros."""
    return slot.unsqueeze(-1) == torch.arange(C.MOVES_PER_POKEMON, device=slot.device)


def embed_move_slots(move_embedding: nn.Module, move_ids: torch.Tensor,
                     pp: torch.Tensor, maxpp: torch.Tensor) -> torch.Tensor:
    """Each move slot's embedding with the fraction of its PP left appended:
    (..., 4) IDs give (..., 4, move_embedding.output_dim + 1). An empty or
    unrevealed slot (-1) is all zeros."""
    moves = embed_or_zero(move_embedding, move_ids)
    pp_left = torch.where(move_ids >= 0, pp / maxpp.clamp(min=1), 0.0)
    return torch.cat([moves, pp_left.unsqueeze(-1).to(moves.dtype)], dim=-1)


class TypeEmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(C.NUM_TYPES, config.TYPE_EMBEDDING_OUTPUT_DIM) 

    def forward(self, type_id: int):
        return self.embedding(torch.as_tensor(type_id))

    
class MoveEmbeddingLayer(nn.Module):
    def __init__(self, type_embedding: TypeEmbeddingLayer, output_dim: int = 32):
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
        self.project = nn.Sequential(nn.Linear(in_dim, output_dim), nn.ReLU())
 
    def forward(self, move_ids: torch.Tensor) -> torch.Tensor:
        """move_ids: long tensor of any shape, e.g. (batch,) or (batch, 4).
        Returns a tensor of shape (*move_ids.shape, output_dim)."""
        learned = self.move_embedding(move_ids)
        move_type = self.type_embedding(self.move_type_ids[move_ids])
        explicit = self.move_features[move_ids]
        return self.project(torch.cat([learned, move_type, explicit], dim=-1))


class PokemonEmbeddingLayer(nn.Module):
    def __init__(self, type_embedding: TypeEmbeddingLayer,
                 move_embedding: MoveEmbeddingLayer, output_dim: int = 32):
        super().__init__()
        self.type_embedding = type_embedding  # shared with moves
        self.move_embedding = move_embedding  # one layer shared by all four slots
        type_ids, base_stats = build_species_table()
        n = names()
        self.species_embedding = nn.Embedding(type_ids.shape[0], config.SPECIES_EMBEDDING_OUTPUT_DIM)
        self.item_embedding = nn.Embedding(len(n.items), config.ITEM_EMBEDDING_OUTPUT_DIM)
        self.ability_embedding = nn.Embedding(len(n.abilities), config.ABILITY_EMBEDDING_OUTPUT_DIM)

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
        self.project = nn.Sequential(nn.Linear(in_dim, output_dim), nn.ReLU())

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

        -1 in species_ids, item_ids, ability_ids, move_ids or tera_type_ids
        marks an empty move slot or something the opponent has not revealed
        yet, and is embedded as zeros. An unrevealed species also zeroes its
        types, base stats, level and stats.

        Returns a tensor of shape (*species_ids.shape, output_dim)."""
        species = embed_or_zero(self.species_embedding, species_ids)
        known = (species_ids >= 0).unsqueeze(-1)
        species_ids = species_ids.clamp(min=0)
        item = embed_or_zero(self.item_embedding, item_ids)
        ability = embed_or_zero(self.ability_embedding, ability_ids)
        types = embed_or_zero(self.type_embedding,
                              torch.where(known, self.species_type_ids[species_ids], -1))
        tera_type = embed_or_zero(self.type_embedding, tera_type_ids)
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
    def __init__(self, pokemon_embedding: PokemonEmbeddingLayer, output_dim: int = 64):
        super().__init__()
        self.pokemon_embedding = pokemon_embedding  # one layer shared by all six slots
        in_dim = (
            2 * pokemon_embedding.output_dim  # the active Pokemon and the bench
            + 1  # how many bench Pokemon are left
            + 1  # whether Tera has been used
            + C.NUM_BOOSTS  # the active Pokemon's stat stages
            + C.NUM_VOLATILES  # and which volatile conditions it has
            + len(PUBLIC_VOLATILE_TURNS)  # turns left on the ones anyone can count
            + pokemon_embedding.move_embedding.output_dim  # the last move it used
            + 1  # whether this is its first turn out (Fake Out, First Impression)
            + 4 * C.MOVES_PER_POKEMON  # Choice-locked, Encored, Disabled, locked-in slot
        )
        self.output_dim = output_dim
        self.project = nn.Sequential(nn.Linear(in_dim, output_dim), nn.ReLU())
        self.register_buffer("public_volatile_turns", torch.tensor(PUBLIC_VOLATILE_TURNS),
                             persistent=False)

    def forward(self, team: dict[str, torch.Tensor], active_slot: torch.Tensor,
                boosts: torch.Tensor, volatiles: torch.Tensor, last_move: torch.Tensor,
                moves_since_switch: torch.Tensor, choice_slot: torch.Tensor,
                encore_slot: torch.Tensor, disabled_slot: torch.Tensor,
                locked_slot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """team: PokemonEmbeddingLayer.forward's arguments by name, each with a
            team dimension of 6, e.g. species_ids (batch, 6) and move_ids
            (batch, 6, 4)
        active_slot: long, (batch,) team index of the active Pokemon
        boosts: (batch, C.NUM_BOOSTS) stat stages, -6..6, in C.BOOST_NAMES order
        volatiles: (batch, C.NUM_VOLATILES) turns left, 0 absent, in C.V_* order
        last_move: long, (batch,) move ID of the last move it used; -1 none
        moves_since_switch: (batch,) moves it has made since switching in
        choice_slot, encore_slot, disabled_slot, locked_slot: long, (batch,)
            the move slot it is held to by a Choice item, Encore, Disable, or a
            move like Outrage; -1 none, and -1 for an opponent's unrevealed
            Choice lock

        The bench is the mean over living bench Pokemon, so it does not depend
        on which slot each one is in; fainted ones only count toward how many
        are left. Tera counts as used once any team member has Terastallized,
        as it stays for the rest of the battle. Everything else belongs to the
        active Pokemon and clears when it switches out. Volatiles show as
        present or absent, plus turns left for PUBLIC_VOLATILE_TURNS.

        Returns (player, pokemon): the player, (batch, output_dim), and each
        team slot's Pokemon, (batch, 6, pokemon_dim), for scoring switches."""
        pokemon = self.pokemon_embedding(**team)  # (batch, 6, pokemon_dim)
        slots = torch.arange(C.TEAM_SIZE, device=active_slot.device)
        is_active = slots == active_slot.unsqueeze(-1)  # (batch, 6)
        active = (pokemon * is_active.unsqueeze(-1)).sum(-2)

        on_bench = ~is_active & (team["hp"] > 0)
        bench_left = on_bench.sum(-1, keepdim=True)
        bench = (pokemon * on_bench.unsqueeze(-1)).sum(-2) / bench_left.clamp(min=1)

        tera_used = team["terastallized"].any(-1, keepdim=True)
        last_move = embed_or_zero(self.pokemon_embedding.move_embedding, last_move)
        held_slots = torch.cat([slot_one_hot(s) for s in
                                (choice_slot, encore_slot, disabled_slot, locked_slot)], dim=-1)
        dtype = active.dtype
        player = self.project(torch.cat([
            active, bench,
            (bench_left / (C.TEAM_SIZE - 1)).to(dtype), tera_used.to(dtype),
            (boosts / 6.0).to(dtype), (volatiles > 0).to(dtype),
            (volatiles[..., self.public_volatile_turns] / 5.0).to(dtype),
            last_move, (moves_since_switch == 0).unsqueeze(-1).to(dtype),
            held_slots.to(dtype),
        ], dim=-1))
        return player, pokemon


class FieldEmbeddingLayer(nn.Module):
    def __init__(self, output_dim: int = 32):
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
        self.project = nn.Sequential(nn.Linear(in_dim, output_dim), nn.ReLU())

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


# What each transformer token is. Every token has its type's embedding added,
# which is all the content the summary token has. The action tokens follow the
# engine's action encoding: 0-3 use a move, 4-7 Terastallize and use it, 8-13
# switch to that team slot.
(TOKEN_SUMMARY, TOKEN_ME, TOKEN_OPPONENT, TOKEN_FIELD,
 TOKEN_MOVE, TOKEN_TERA_MOVE, TOKEN_SWITCH) = range(7)
# Damage calcs on each action token, as (min, max) roll pairs; see
# `game_inputs._matchups`. A move is calculated against the opponent's active
# Pokemon and up to three of their others; a switch-in carries its best move
# into their active Pokemon and their best revealed move into it.
MOVE_MATCHUPS = 2 * 4
SWITCH_MATCHUPS = 2 * 2
TOKEN_TYPES = (
    [TOKEN_SUMMARY, TOKEN_ME, TOKEN_OPPONENT, TOKEN_FIELD]
    + [TOKEN_MOVE] * C.MOVES_PER_POKEMON
    + [TOKEN_TERA_MOVE] * C.MOVES_PER_POKEMON
    + [TOKEN_SWITCH] * C.TEAM_SIZE
)


class GameNetwork(nn.Module):
    """Both players, the field and one token per action, through a transformer.

    The action head scores each action token, so every move and switch is
    judged in the context of the whole game. The win head reads the summary
    token. Everything is from your point of view.
    """

    def __init__(self):
        super().__init__()
        self.type_embedding = TypeEmbeddingLayer()
        self.move_embedding = MoveEmbeddingLayer(self.type_embedding)
        pokemon = PokemonEmbeddingLayer(self.type_embedding, self.move_embedding)
        self.player_embedding = PlayerEmbeddingLayer(pokemon)  # shared by both players
        self.field_embedding = FieldEmbeddingLayer()

        d = config.TRANSFORMER_DIM
        move_dim = self.move_embedding.output_dim + 1  # with its PP left
        # Each kind of token has its own projection into the transformer's width.
        self.to_player = nn.Linear(self.player_embedding.output_dim, d)
        self.to_field = nn.Linear(self.field_embedding.output_dim, d)
        self.to_move = nn.Linear(move_dim + MOVE_MATCHUPS, d)
        self.to_tera_move = nn.Linear(
            move_dim + config.TYPE_EMBEDDING_OUTPUT_DIM + MOVE_MATCHUPS, d)
        self.to_switch = nn.Linear(pokemon.output_dim + SWITCH_MATCHUPS, d)
        self.token_type = nn.Embedding(len(set(TOKEN_TYPES)), d)
        self.register_buffer("token_types", torch.tensor(TOKEN_TYPES), persistent=False)
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
                legal_actions: torch.Tensor,
                matchups: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """me, opponent: PlayerEmbeddingLayer.forward's arguments by name, each
            with a leading batch dimension
        field: FieldEmbeddingLayer.forward's arguments by name
        legal_actions: bool (batch, C.NUM_ACTIONS), the engine's legal action mask
        matchups: damage calcs for the action tokens: moves and tera_moves
            (batch, 4, MOVE_MATCHUPS), switches (batch, 6, SWITCH_MATCHUPS)

        `game_inputs.game_inputs` builds all four from a battle, as one player
        is allowed to see it.

        Returns (action_logits, win_logit):
            action_logits: (batch, C.NUM_ACTIONS); illegal actions are set to
                the lowest float, so a softmax gives them probability 0
            win_logit: (batch,); its sigmoid is the probability that you win
        """
        my_team, my_active = me["team"], me["active_slot"]
        me, my_pokemon = self.player_embedding(**me)
        opponent, _ = self.player_embedding(**opponent)
        field = self.field_embedding(**field)

        # The active Pokemon's move slots, for the move and Tera-move actions
        rows = torch.arange(my_active.shape[0], device=my_active.device)
        active = {k: v[rows, my_active.long()] for k, v in my_team.items()}
        moves = embed_move_slots(self.move_embedding, active["move_ids"],
                                 active["pp"], active["maxpp"])
        tera_type = embed_or_zero(self.type_embedding, active["tera_type_ids"])
        tera_moves = torch.cat(
            [moves, tera_type.unsqueeze(-2).expand(-1, C.MOVES_PER_POKEMON, -1)], dim=-1)

        summary = me.new_zeros(me.shape[0], 1, self.token_type.embedding_dim)
        tokens = torch.cat([
            summary,
            self.to_player(me).unsqueeze(1),
            self.to_player(opponent).unsqueeze(1),
            self.to_field(field).unsqueeze(1),
            self.to_move(torch.cat([moves, matchups["moves"]], dim=-1)),
            self.to_tera_move(torch.cat([tera_moves, matchups["tera_moves"]], dim=-1)),
            self.to_switch(torch.cat([my_pokemon, matchups["switches"]], dim=-1)),
        ], dim=1) + self.token_type(self.token_types)  # (batch, 18, d)
        legality = self.action_legality(legal_actions.long())  # (batch, 14, d)
        tokens = torch.cat([tokens[:, :-C.NUM_ACTIONS],
                            tokens[:, -C.NUM_ACTIONS:] + legality], dim=1)
        out = self.transformer(tokens)

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
    print(out.shape)  # should be (32,)

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
    print(out.shape)  # should be (32,)

    player_layer = PlayerEmbeddingLayer(pokemon_layer)
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
    out, _ = player_layer(**player)
    print(out.shape)  # should be (64,)

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
    action_logits, win_logit = game(batch_of_one(player), batch_of_one(player), field, legal,
                                    matchups)
    print(action_logits.softmax(-1).shape)  # should be (1, 14)
    print(torch.sigmoid(win_logit))  # probability of winning
