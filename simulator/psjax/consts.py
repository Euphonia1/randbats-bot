"""Enumerations shared by the data builder and the JAX engine.

Every enum here is a plain Python int: it is *static* data baked into the
compiled arrays and into traced code as a constant, never a traced value.
`build.py` asserts these orderings against the Showdown dex so the compiled
arrays and the engine can never drift apart.
"""

# --- Types -------------------------------------------------------------------
# Canonical order. TYPE_NONE fills the unused slot of a mono-type species and
# means "no type" everywhere a type index is expected.
TYPE_NONE = -1
TYPES = [
    "normal", "fighting", "flying", "poison", "ground", "rock", "bug", "ghost",
    "steel", "fire", "water", "grass", "electric", "psychic", "ice", "dragon",
    "dark", "fairy", "stellar",
]
TYPE_IDX = {t: i for i, t in enumerate(TYPES)}
NUM_TYPES = len(TYPES)
(NORMAL, FIGHTING, FLYING, POISON, GROUND, ROCK, BUG, GHOST, STEEL, FIRE,
 WATER, GRASS, ELECTRIC, PSYCHIC, ICE, DRAGON, DARK, FAIRY, STELLAR) = range(NUM_TYPES)

# --- Stats -------------------------------------------------------------------
# HP is index 0 and is never boosted; BOOST_* index the 7-wide boost vector.
STAT_NAMES = ["hp", "atk", "def", "spa", "spd", "spe"]
HP, ATK, DEF, SPA, SPD, SPE = range(6)
NUM_STATS = 6

BOOST_NAMES = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
B_ATK, B_DEF, B_SPA, B_SPD, B_SPE, B_ACC, B_EVA = range(7)
NUM_BOOSTS = 7
# Maps a stat index (1..5) to its slot in the boost vector.
STAT_TO_BOOST = {ATK: B_ATK, DEF: B_DEF, SPA: B_SPA, SPD: B_SPD, SPE: B_SPE}

# --- Move categories ---------------------------------------------------------
CAT_PHYSICAL, CAT_SPECIAL, CAT_STATUS = 0, 1, 2
CATEGORIES = {"Physical": CAT_PHYSICAL, "Special": CAT_SPECIAL, "Status": CAT_STATUS}

# --- Major status ------------------------------------------------------------
STATUS_NONE, BRN, PAR, SLP, FRZ, PSN, TOX = range(7)
STATUS_IDX = {"brn": BRN, "par": PAR, "slp": SLP, "frz": FRZ, "psn": PSN, "tox": TOX}
NUM_STATUS = 7

# --- Weather -----------------------------------------------------------------
WEATHER_NONE, SUN, RAIN, SAND, SNOW, HARSH_SUN, HEAVY_RAIN, STRONG_WINDS = range(8)
WEATHER_IDX = {
    "sunnyday": SUN, "raindance": RAIN, "sandstorm": SAND, "snowscape": SNOW,
    "snow": SNOW, "hail": SNOW,
    "desolateland": HARSH_SUN, "primordialsea": HEAVY_RAIN, "deltastream": STRONG_WINDS,
}
NUM_WEATHER = 8

# --- Terrain -----------------------------------------------------------------
TERRAIN_NONE, ELECTRIC_TERRAIN, GRASSY_TERRAIN, MISTY_TERRAIN, PSYCHIC_TERRAIN = range(5)
TERRAIN_IDX = {
    "electricterrain": ELECTRIC_TERRAIN, "grassyterrain": GRASSY_TERRAIN,
    "mistyterrain": MISTY_TERRAIN, "psychicterrain": PSYCHIC_TERRAIN,
}
NUM_TERRAIN = 5

# --- Side conditions ---------------------------------------------------------
# Stored as an int counter per side: hazards hold a layer count, screens hold
# turns remaining, and 0 always means "not present".
(SC_STEALTHROCK, SC_SPIKES, SC_TOXICSPIKES, SC_STICKYWEB, SC_REFLECT,
 SC_LIGHTSCREEN, SC_AURORAVEIL, SC_TAILWIND, SC_SAFEGUARD, SC_MIST) = range(10)
NUM_SIDE_CONDITIONS = 10
SIDE_CONDITION_IDX = {
    "stealthrock": SC_STEALTHROCK, "spikes": SC_SPIKES, "toxicspikes": SC_TOXICSPIKES,
    "stickyweb": SC_STICKYWEB, "reflect": SC_REFLECT, "lightscreen": SC_LIGHTSCREEN,
    "auroraveil": SC_AURORAVEIL, "tailwind": SC_TAILWIND, "safeguard": SC_SAFEGUARD,
    "mist": SC_MIST,
}
# Hazards use a layer cap; everything else is a turn counter.
SIDE_CONDITION_MAX = {
    SC_STEALTHROCK: 1, SC_SPIKES: 3, SC_TOXICSPIKES: 2, SC_STICKYWEB: 1,
}

# --- Volatiles ---------------------------------------------------------------
# One int counter per active Pokemon; cleared on switch out. 0 means absent.
# Volatiles needing extra payload (substitute HP, the disabled move slot, the
# Protosynthesis stat) keep that payload in a dedicated BattleState field.
(V_CONFUSION, V_FLINCH, V_SUBSTITUTE, V_LEECHSEED, V_TAUNT, V_ENCORE, V_DISABLE,
 V_YAWN, V_PROTECT, V_ENDURE, V_DESTINYBOND, V_MAGNETRISE, V_ROOST, V_AQUARING,
 V_INGRAIN, V_CURSE, V_ATTRACT, V_PERISHSONG, V_SALTCURE, V_PARTIALLYTRAPPED,
 V_LOCKEDMOVE, V_TWOTURN, V_RECHARGE, V_FOCUSENERGY, V_HELPINGHAND, V_SLOWSTART,
 V_PROTOSYNTHESIS, V_QUARKDRIVE, V_GLAIVERUSH, V_THROATCHOP, V_TORMENT,
 V_MINIMIZE, V_DEFENSECURL, V_TARSHOT, V_FLASHFIRE, V_CHARGE) = range(36)
NUM_VOLATILES = 36
VOLATILE_IDX = {
    "confusion": V_CONFUSION, "flinch": V_FLINCH, "substitute": V_SUBSTITUTE,
    "leechseed": V_LEECHSEED, "taunt": V_TAUNT, "encore": V_ENCORE,
    "disable": V_DISABLE, "yawn": V_YAWN, "protect": V_PROTECT, "endure": V_ENDURE,
    "destinybond": V_DESTINYBOND, "magnetrise": V_MAGNETRISE, "roost": V_ROOST,
    "aquaring": V_AQUARING, "ingrain": V_INGRAIN, "curse": V_CURSE,
    "attract": V_ATTRACT, "perishsong": V_PERISHSONG, "saltcure": V_SALTCURE,
    "partiallytrapped": V_PARTIALLYTRAPPED, "lockedmove": V_LOCKEDMOVE,
    "twoturnmove": V_TWOTURN, "mustrecharge": V_RECHARGE, "focusenergy": V_FOCUSENERGY,
    "helpinghand": V_HELPINGHAND, "slowstart": V_SLOWSTART,
    "protosynthesis": V_PROTOSYNTHESIS, "quarkdrive": V_QUARKDRIVE,
    "glaiverush": V_GLAIVERUSH, "throatchop": V_THROATCHOP, "torment": V_TORMENT,
    "minimize": V_MINIMIZE, "defensecurl": V_DEFENSECURL, "tarshot": V_TARSHOT,
    "flashfire": V_FLASHFIRE, "charge": V_CHARGE,
}

# --- Random Battle set roles -------------------------------------------------
# Showdown tags every random-battle set with a role; the team builder uses it to
# pick a held item, which is otherwise decided by generator code we do not port.
ROLES = [
    "AV Pivot",
    "Bulky Attacker",
    "Bulky Setup",
    "Bulky Support",
    "Fast Attacker",
    "Fast Bulky Setup",
    "Fast Support",
    "Setup Sweeper",
    "Tera Blast user",
    "Wallbreaker",
]
ROLE_IDX = {r: i for i, r in enumerate(ROLES)}
NUM_ROLES = len(ROLES)

# --- Move targets ------------------------------------------------------------
(TGT_NORMAL, TGT_SELF, TGT_ADJACENT_ALLY, TGT_ADJACENT_ALLY_OR_SELF,
 TGT_ADJACENT_FOE, TGT_ALL_ADJACENT, TGT_ALL_ADJACENT_FOES, TGT_ALL,
 TGT_ALLY_SIDE, TGT_ALLY_TEAM, TGT_FOE_SIDE, TGT_ANY, TGT_RANDOM_NORMAL,
 TGT_SCRIPTED, TGT_ALLIES) = range(15)
TARGET_IDX = {
    "normal": TGT_NORMAL, "self": TGT_SELF, "adjacentAlly": TGT_ADJACENT_ALLY,
    "adjacentAllyOrSelf": TGT_ADJACENT_ALLY_OR_SELF, "adjacentFoe": TGT_ADJACENT_FOE,
    "allAdjacent": TGT_ALL_ADJACENT, "allAdjacentFoes": TGT_ALL_ADJACENT_FOES,
    "all": TGT_ALL, "allySide": TGT_ALLY_SIDE, "allyTeam": TGT_ALLY_TEAM,
    "foeSide": TGT_FOE_SIDE, "any": TGT_ANY, "randomNormal": TGT_RANDOM_NORMAL,
    "scripted": TGT_SCRIPTED, "allies": TGT_ALLIES,
}
# Targets that hit the opponent in a singles battle.
FOE_TARGETS = (TGT_NORMAL, TGT_ADJACENT_FOE, TGT_ALL_ADJACENT, TGT_ALL_ADJACENT_FOES,
               TGT_ANY, TGT_RANDOM_NORMAL, TGT_SCRIPTED)

# --- Move flags (bitmask) ----------------------------------------------------
FLAG_BITS = {
    "contact": 0, "protect": 1, "mirror": 2, "sound": 3, "punch": 4, "bite": 5,
    "bullet": 6, "powder": 7, "reflectable": 8, "bypasssub": 9, "wind": 10,
    "slicing": 11, "heal": 12, "recharge": 13, "charge": 14, "gravity": 15,
    "defrost": 16, "distance": 17, "nonsky": 18, "pledgecombo": 19, "snatch": 20,
    "dance": 21, "metronome": 22, "noparentalbond": 23, "failcopycat": 24,
    "failencore": 25, "failinstruct": 26, "failmefirst": 27, "failmimic": 28,
    "nosleeptalk": 29, "noassist": 30, "futuremove": 31,
}

# --- Battle phases -----------------------------------------------------------
# What the pending action from each player means.
PHASE_MOVE = 0       # both players pick a move or a switch
PHASE_SWITCH = 1     # one or both players must replace a fainted Pokemon
PHASE_END = 2        # battle over

# --- Action encoding ---------------------------------------------------------
# A single int per player per decision point.
#   0..3    use move in slot i
#   4..7    terastallize, then use move in slot i-4
#   8..13   switch to team slot i-8
ACTION_MOVE_BASE = 0
ACTION_TERA_BASE = 4
ACTION_SWITCH_BASE = 8
NUM_ACTIONS = 14

# --- Sizes -------------------------------------------------------------------
TEAM_SIZE = 6
NUM_PLAYERS = 2
MOVES_PER_POKEMON = 4
