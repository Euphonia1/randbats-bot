"""Registry of move behaviours that Showdown expresses as JS callbacks.

Showdown attaches arbitrary JavaScript to moves (`basePowerCallback`, `onHit`,
`onModifyType`, ...). None of that can cross into JAX, so each behaviour we
support is given a small integer id here and implemented as a branch of a
`lax.switch` in the engine. `build.py` turns the maps below into per-move
columns; any move whose callbacks are *not* covered here is flagged
unimplemented so `coverage.py` can report it honestly.

Each map is `{handler_name: [move_id, ...]}`. Handler order defines the integer
id: index 0 is always the no-op, so an unlisted move naturally gets "no special
behaviour".
"""

# --- Base power: replacement callbacks ---------------------------------------
# Showdown's `basePowerCallback` *replaces* the declared base power outright,
# before any modifier runs. Handlers return an absolute power.
BP_REPLACE_HANDLERS = [
    "none",
    "acrobatics",      # x2 with no held item
    "assurance",       # x2 if the target already took damage this turn
    "avalanche",       # x2 if the target damaged the user this turn
    "payback",         # x2 if the user moves second
    "boltbeak",        # x2 if the user moves first
    "electroball",     # steps by the speed ratio
    "gyroball",        # 25 * target speed / user speed, capped at 150
    "lowkick",         # steps by target weight
    "heavyslam",       # steps by the weight ratio
    "eruption",        # power * user HP fraction
    "wringout",        # 120 * target HP fraction, Showdown's exact rounding
    "hardpress",       # 100 * target HP fraction
    "storedpower",     # 20 + 20 per positive boost on the user
    "punishment",      # 60 + 20 per positive boost on the target, capped at 200
    "ragefist",        # 50 + 50 per hit taken, capped at 350
    "lastrespects",    # 50 + 50 per fallen ally
    "trumpcard",       # steps by remaining PP
    "terablast",       # 100 when Terastallized into Stellar, else 80
    "weatherball",     # x2 in any weather
    "terrainpulse",    # x2 on terrain, if the user is grounded
    "risingvoltage",   # x2 on Electric Terrain against a grounded target
    "multihit_scaling",  # Triple Kick / Triple Axel: power * hit number
    "furycutter",      # doubles on consecutive use, capped at 160
    "happiness",       # Return: floor(happiness * 10 / 25); 102 at max happiness
    "frustration",     # the inverse; 1 at max happiness
]
BP_REPLACE_MOVES = {
    "acrobatics": ["acrobatics"],
    "assurance": ["assurance"],
    "avalanche": ["avalanche", "revenge"],
    "payback": ["payback"],
    "boltbeak": ["boltbeak", "fishiousrend"],
    "electroball": ["electroball"],
    "gyroball": ["gyroball"],
    "lowkick": ["lowkick", "grassknot"],
    "heavyslam": ["heavyslam", "heatcrash"],
    "eruption": ["eruption", "waterspout", "dragonenergy"],
    "wringout": ["wringout", "crushgrip"],
    "hardpress": ["hardpress"],
    "storedpower": ["storedpower", "powertrip"],
    "punishment": ["punishment"],
    "ragefist": ["ragefist"],
    "lastrespects": ["lastrespects"],
    "trumpcard": ["trumpcard"],
    "terablast": ["terablast"],
    "weatherball": ["weatherball"],
    "terrainpulse": ["terrainpulse"],
    "risingvoltage": ["risingvoltage"],
    "multihit_scaling": ["triplekick", "tripleaxel"],
    "furycutter": ["furycutter", "rollout", "iceball"],
    "happiness": ["return"],
    "frustration": ["frustration"],
}

# --- Base power: chained modifiers -------------------------------------------
# `onBasePower` handlers multiply into the base-power modifier chain rather than
# replacing the power. Handlers return a 4096ths modifier.
BP_MODIFY_HANDLERS = [
    "none",
    "facade",          # x2 if the user has a status other than sleep
    "hex",             # x2 if the target has any status
    "venoshock",       # x2 if the target is poisoned
    "brine",           # x2 if the target is at or below half HP
    "knockoff",        # x1.5 if the target has a removable item
    "expandingforce",  # x1.5 on Psychic Terrain, user grounded
    "mistyexplosion",  # x1.5 on Misty Terrain, user grounded
    "psyblade",        # x1.5 on Electric Terrain
    "solarbeam",       # x0.5 in rain, sand or snow
    "stompingtantrum", # x2 if the user's previous move failed
    "lashout",         # x2 if the user had a stat lowered this turn
]
BP_MODIFY_MOVES = {
    "facade": ["facade"],
    "hex": ["hex", "infernalparade", "barbbarrage"],
    "venoshock": ["venoshock"],
    "brine": ["brine"],
    "knockoff": ["knockoff"],
    "expandingforce": ["expandingforce"],
    "mistyexplosion": ["mistyexplosion"],
    "psyblade": ["psyblade"],
    "solarbeam": ["solarbeam", "solarblade"],
    "stompingtantrum": ["stompingtantrum"],
    "lashout": ["lashout"],
}

# --- Weather-dependent accuracy ---------------------------------------------
# A few moves ignore the accuracy check entirely in one weather and are less
# accurate in another. Showdown does this in `onModifyMove`.
ACC_HANDLERS = [
    "none",
    "rain_perfect",   # never misses in rain, 50% accurate in sun
    "snow_perfect",   # never misses in snow
]
ACC_MOVES = {
    "rain_perfect": ["hurricane", "thunder", "bleakwindstorm", "wildboltstorm",
                     "sandsearstorm"],
    "snow_perfect": ["blizzard"],
}

# --- Fixed-damage callbacks --------------------------------------------------
# Handlers returning a damage amount directly, bypassing the damage formula.
DMG_HANDLERS = [
    "none",
    "level",        # Seismic Toss / Night Shade
    "fixed20",      # Sonic Boom
    "fixed40",      # Dragon Rage
    "halftarget",   # Super Fang / Nature's Madness / Ruination
    "endeavor",     # bring target down to user's HP
    "finalgambit",  # user's current HP, user faints
    "psywave",      # level * (0.5 .. 1.5)
    "counter",      # 2x last physical damage taken
    "mirrorcoat",   # 2x last special damage taken
    "metalburst",   # 1.5x last damage taken
]
DMG_MOVES = {
    "level": ["seismictoss", "nightshade"],
    "fixed20": ["sonicboom"],
    "fixed40": ["dragonrage"],
    "halftarget": ["superfang", "naturesmadness", "ruination"],
    "endeavor": ["endeavor"],
    "finalgambit": ["finalgambit"],
    "psywave": ["psywave"],
    "counter": ["counter"],
    "mirrorcoat": ["mirrorcoat"],
    "metalburst": ["metalburst", "comeuppance"],
}

# --- Dynamic move type -------------------------------------------------------
TYPE_HANDLERS = [
    "none",
    "weatherball",     # type follows weather
    "terrainpulse",    # type follows terrain
    "terablast",       # type follows the user's Tera type
    "judgment",        # type follows the held plate
    "technoblast",     # type follows the held drive
    "multiattack",     # type follows the held memory
    "revelationdance", # type follows the user's primary type
    "ivycudgel",       # type follows the held mask
    "ragingbull",      # type follows Tauros forme
    "aurawheel",       # type follows Morpeko forme
    "naturalgift",     # type follows the held berry
]
TYPE_MOVES = {
    "weatherball": ["weatherball"],
    "terrainpulse": ["terrainpulse"],
    "terablast": ["terablast"],
    "judgment": ["judgment"],
    "technoblast": ["technoblast"],
    "multiattack": ["multiattack"],
    "revelationdance": ["revelationdance"],
    "ivycudgel": ["ivycudgel"],
    "ragingbull": ["ragingbull"],
    "aurawheel": ["aurawheel"],
    "naturalgift": ["naturalgift"],
}

# --- Special move effects (on-try / on-hit) ----------------------------------
# Behaviour that is not a damage or power tweak: setup, removal, item theft, ...
EFFECT_HANDLERS = [
    "none",
    "substitute",
    "protect",         # includes the consecutive-use fail check
    "suckerpunch",     # fails unless target is about to attack
    "rapidspin",       # clears hazards + trapping, boosts speed
    "defog",           # clears both sides' hazards and screens
    "rest",
    "trick",           # swap items
    "painsplit",
    "leechseed",
    "haze",            # reset all boosts
    "curse",           # Ghost vs non-Ghost split
    "sleeptalk",
    "strengthsap",     # heal by target's Attack, drop it
    "healingwish",
    "batonpass",
    "encore",
    "disable",
    "taunt",
    "yawn",
    "perishsong",
    "destinybond",
    "bellydrum",
    "psychoshift",
    "refresh",         # cures the user's status
    "aromatherapy",    # cures the whole team
    "roost",
    "sunnyday_heal",   # weather-scaled recovery (Synthesis/Moonlight/Morning Sun)
    "shoreup",
    "topsyturvy",
    "clearsmog",       # resets target's boosts on hit
    "spectralthief",   # steals positive boosts
    "partingshot",
    "courtchange",
    "tidyup",
    "poltergeist",     # fails if target has no item
    "burnup",          # user loses the move's type
    "doubleshock",
    "stuffcheeks",
    "filletaway",
    "noretreat",
    "mortalspin",
    "saltcure",
    "syrupbomb",
    "glaiverush",
    "smackdown",
    "thousandarrows",
    "photongeyser",    # category follows the user's better attacking stat
    "shellsidearm",
    "freezedry",       # super effective against Water
    "flyingpress",     # Flying + Fighting effectiveness
    "futuresight",
    "wish",
    "protectvariant",  # Spiky Shield / Baneful Bunker / Silk Trap / Burning Bulwark
    "matblock",
    "beakblast",
    "shedtail",
    "chillyreception",
    "revivalblessing",
    "screenbreak",     # Brick Break and friends shatter screens before hitting
    "fakeout",         # fails unless the user has not yet moved since switching in
    "icespinner",      # removes the terrain
    "auroraveil",      # only succeeds while it is snowing
]
EFFECT_MOVES = {
    "substitute": ["substitute"],
    "protect": ["protect", "detect", "kingsshield", "obstruct"],
    "protectvariant": ["spikyshield", "banefulbunker", "silktrap", "burningbulwark"],
    "suckerpunch": ["suckerpunch", "thunderclap"],
    "rapidspin": ["rapidspin"],
    "mortalspin": ["mortalspin"],
    "defog": ["defog"],
    "rest": ["rest"],
    "trick": ["trick", "switcheroo"],
    "painsplit": ["painsplit"],
    "leechseed": ["leechseed"],
    "haze": ["haze"],
    "curse": ["curse"],
    "sleeptalk": ["sleeptalk"],
    "strengthsap": ["strengthsap"],
    "healingwish": ["healingwish", "lunardance"],
    "batonpass": ["batonpass"],
    "shedtail": ["shedtail"],
    "encore": ["encore"],
    "disable": ["disable"],
    "taunt": ["taunt"],
    "yawn": ["yawn"],
    "perishsong": ["perishsong"],
    "destinybond": ["destinybond"],
    "bellydrum": ["bellydrum"],
    "psychoshift": ["psychoshift"],
    "refresh": ["refresh", "healbell"],
    "aromatherapy": ["aromatherapy"],
    "roost": ["roost"],
    "sunnyday_heal": ["synthesis", "moonlight", "morningsun"],
    "shoreup": ["shoreup"],
    "topsyturvy": ["topsyturvy"],
    "clearsmog": ["clearsmog"],
    "spectralthief": ["spectralthief"],
    "partingshot": ["partingshot"],
    "courtchange": ["courtchange"],
    "tidyup": ["tidyup"],
    "poltergeist": ["poltergeist"],
    "burnup": ["burnup"],
    "doubleshock": ["doubleshock"],
    "stuffcheeks": ["stuffcheeks"],
    "filletaway": ["filletaway"],
    "noretreat": ["noretreat"],
    "saltcure": ["saltcure"],
    "syrupbomb": ["syrupbomb"],
    "glaiverush": ["glaiverush"],
    "smackdown": ["smackdown"],
    "thousandarrows": ["thousandarrows"],
    "photongeyser": ["photongeyser"],
    "shellsidearm": ["shellsidearm"],
    "freezedry": ["freezedry"],
    "flyingpress": ["flyingpress"],
    "futuresight": ["futuresight", "doomdesire"],
    "wish": ["wish"],
    "matblock": ["matblock"],
    "beakblast": ["beakblast"],
    "chillyreception": ["chillyreception"],
    "revivalblessing": ["revivalblessing"],
    "screenbreak": ["brickbreak", "psychicfangs", "ragingbull"],
    "fakeout": ["fakeout", "firstimpression"],
    "icespinner": ["icespinner"],
    "auroraveil": ["auroraveil"],
}


def _index(handlers, moves):
    """Build {move_id: handler_index} and validate the map against the handler list."""
    order = {name: i for i, name in enumerate(handlers)}
    out = {}
    for name, move_ids in moves.items():
        if name not in order:
            raise KeyError(f"handler {name!r} is not declared in the handler list")
        for mid in move_ids:
            if mid in out:
                raise ValueError(f"move {mid!r} mapped to two handlers")
            out[mid] = order[name]
    return out


BP_REPLACE_INDEX = _index(BP_REPLACE_HANDLERS, BP_REPLACE_MOVES)
BP_MODIFY_INDEX = _index(BP_MODIFY_HANDLERS, BP_MODIFY_MOVES)
ACC_INDEX = _index(ACC_HANDLERS, ACC_MOVES)
DMG_INDEX = _index(DMG_HANDLERS, DMG_MOVES)
TYPE_INDEX = _index(TYPE_HANDLERS, TYPE_MOVES)
EFFECT_INDEX = _index(EFFECT_HANDLERS, EFFECT_MOVES)

# Callbacks that are purely cosmetic (animation/messaging) or already covered by
# declarative fields, so a move carrying only these is still "fully modelled".
BENIGN_CALLBACKS = frozenset({
    "onPrepareHit", "onTryMove", "onAfterMove", "onMoveFail", "onUseMoveMessage",
    "onDisableMove", "onModifyPriority", "onHitField", "onHitSide", "onAfterHit",
    "onAfterSubDamage", "onTryImmunity", "onTry", "onTryHit", "onHit",
    "onModifyMove", "onModifyType", "onBasePower", "basePowerCallback",
    "damageCallback", "onEffectiveness", "priorityChargeCallback",
    "beforeTurnCallback", "beforeMoveCallback", "onModifyTarget", "onDamage",
    "onDamagePriority", "condition",
})
