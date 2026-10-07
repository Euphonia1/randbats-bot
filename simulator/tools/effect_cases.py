"""Scenario definitions for the effects differential test.

Each entry runs one turn in a real Showdown battle and in psjax, then compares
the fields named in `check`. Scenarios are deliberately deterministic: the
opposing move is usually Splash so HP changes come only from the effect under
test, and only RNG-independent fields are compared.

Two knobs make more of them deterministic. `turns` lists further turns' choices,
for effects that play out over several (Future Sight, Wish, charge moves).
`force` wins every chance roll in both engines and pins the damage roll to its
top value, for effects that trigger 30% of the time or checks that depend on a
damage number; forced scenarios avoid Speed ties and multi-hit moves, which it
does not pin.

    python tools/effect_cases.py > tools/effect_cases.json
"""
import json

# Inert stand-ins so the opponent never interferes.
DOLL = {"species": "Blissey", "ability": "Natural Cure", "moves": ["splash"]}
FROG = {"species": "Clefable", "ability": "Unaware", "moves": ["splash"]}


def case(name, p1, p2, check, **kw):
    return {"name": name, "p1": p1, "p2": p2, "check": check, **kw}


def mon(species, ability, moves=("splash",), **kw):
    return {"species": species, "ability": ability, "moves": list(moves), **kw}


CASES = []
A = CASES.append

# --- residual damage and healing --------------------------------------------
A(case("leftovers heals 1/16",
       mon("Clefable", "Unaware", item="Leftovers", hpPercent=0.5), DOLL, ["p1.hp"]))
A(case("burn chips 1/16",
       mon("Clefable", "Unaware", status="brn", hpPercent=0.9), DOLL, ["p1.hp"]))
A(case("poison chips 1/8",
       mon("Clefable", "Unaware", status="psn", hpPercent=0.9), DOLL, ["p1.hp"]))
A(case("toxic chips 1/16 on its first turn",
       mon("Clefable", "Unaware", status="tox", hpPercent=0.9), DOLL, ["p1.hp"]))
A(case("magic guard ignores poison",
       mon("Clefable", "Magic Guard", status="psn", hpPercent=0.9), DOLL, ["p1.hp"]))
A(case("poison heal turns poison into healing",
       mon("Gliscor", "Poison Heal", status="psn", hpPercent=0.5), DOLL, ["p1.hp"]))
A(case("sandstorm chips the non-immune",
       mon("Clefable", "Unaware", hpPercent=0.9), DOLL, ["p1.hp"], weather="sandstorm"))
A(case("sandstorm spares rock types",
       mon("Tyranitar", "Sand Stream", hpPercent=0.9), DOLL, ["p1.hp"], weather="sandstorm"))
A(case("sandstorm spares steel types",
       mon("Corviknight", "Pressure", hpPercent=0.9), DOLL, ["p1.hp"], weather="sandstorm"))
A(case("grassy terrain heals the grounded",
       mon("Clefable", "Unaware", hpPercent=0.5), DOLL, ["p1.hp"], terrain="grassyterrain"))
A(case("speed boost rises each turn",
       mon("Blaziken", "Speed Boost"), DOLL, ["p1.boosts"]))

# --- status infliction -------------------------------------------------------
A(case("thunder wave paralyses",
       mon("Clefable", "Unaware", ["thunderwave"]), DOLL, ["p2.status"]))
A(case("thunder wave misses ground types",
       mon("Clefable", "Unaware", ["thunderwave"]),
       mon("Great Tusk", "Protosynthesis"), ["p2.status"]))
A(case("will-o-wisp burns",
       mon("Clefable", "Unaware", ["willowisp"]), DOLL, ["p2.status"]))
A(case("will-o-wisp cannot burn a fire type",
       mon("Clefable", "Unaware", ["willowisp"]),
       mon("Volcarona", "Flame Body"), ["p2.status"]))
A(case("toxic badly poisons",
       mon("Clefable", "Unaware", ["toxic"]), DOLL, ["p2.status"]))
A(case("steel types resist toxic",
       mon("Clefable", "Unaware", ["toxic"]),
       mon("Corviknight", "Pressure"), ["p2.status"]))
A(case("corrosion poisons steel anyway",
       mon("Salazzle", "Corrosion", ["toxic"]),
       mon("Corviknight", "Pressure"), ["p2.status"]))
A(case("misty terrain blocks status",
       mon("Clefable", "Unaware", ["thunderwave"]), DOLL, ["p2.status"],
       terrain="mistyterrain"))
A(case("electric terrain blocks sleep",
       mon("Clefable", "Unaware", ["spore"]), DOLL, ["p2.status"],
       terrain="electricterrain"))
A(case("safeguard blocks status",
       mon("Clefable", "Unaware", ["thunderwave"]),
       dict(DOLL, sideConditions=["safeguard"]), ["p2.status"]))
A(case("immunity ability blocks poison",
       mon("Clefable", "Unaware", ["toxic"]),
       mon("Snorlax", "Immunity"), ["p2.status"]))

# --- boosts ------------------------------------------------------------------
A(case("swords dance raises attack two stages",
       mon("Clefable", "Unaware", ["swordsdance"]), DOLL, ["p1.boosts"]))
A(case("calm mind raises both special stats",
       mon("Clefable", "Unaware", ["calmmind"]), DOLL, ["p1.boosts"]))
A(case("dragon dance raises attack and speed",
       mon("Dragonite", "Multiscale", ["dragondance"]), DOLL, ["p1.boosts"]))
A(case("growl lowers the target's attack",
       mon("Clefable", "Unaware", ["growl"]), DOLL, ["p2.boosts"]))
A(case("clear body blocks the drop",
       mon("Clefable", "Unaware", ["growl"]),
       mon("Metagross", "Clear Body"), ["p2.boosts"]))
A(case("clear amulet blocks the drop",
       mon("Clefable", "Unaware", ["growl"]),
       dict(DOLL, item="Clear Amulet"), ["p2.boosts"]))
A(case("mist blocks the drop",
       mon("Clefable", "Unaware", ["growl"]),
       dict(DOLL, sideConditions=["mist"]), ["p2.boosts"]))
A(case("contrary inverts the drop",
       mon("Clefable", "Unaware", ["growl"]),
       mon("Serperior", "Contrary"), ["p2.boosts"]))
A(case("simple doubles the boost",
       mon("Numel", "Simple", ["swordsdance"]), DOLL, ["p1.boosts"]))
A(case("defiant answers an attack drop",
       mon("Clefable", "Unaware", ["growl"]),
       mon("Bisharp", "Defiant"), ["p2.boosts"]))
A(case("belly drum maxes attack for half the HP",
       mon("Azumarill", "Huge Power", ["bellydrum"]), DOLL, ["p1.boosts", "p1.hp"]))
A(case("haze wipes both sides' boosts",
       mon("Clefable", "Unaware", ["haze"], boosts={"atk": 2}),
       dict(DOLL, boosts={"spa": 3}), ["p1.boosts", "p2.boosts"]))

# --- healing -----------------------------------------------------------------
A(case("recover restores half",
       mon("Clefable", "Unaware", ["recover"], hpPercent=0.4), DOLL, ["p1.hp"]))
A(case("synthesis restores two thirds in sun",
       mon("Venusaur", "Overgrow", ["synthesis"], hpPercent=0.3), DOLL, ["p1.hp"],
       weather="sunnyday"))
A(case("synthesis restores a quarter in sand",
       mon("Venusaur", "Overgrow", ["synthesis"], hpPercent=0.3), DOLL, ["p1.hp"],
       weather="sandstorm"))
A(case("shore up restores two thirds in sand",
       mon("Sandaconda", "Shed Skin", ["shoreup"], hpPercent=0.3), DOLL, ["p1.hp"],
       weather="sandstorm"))
A(case("rest restores fully and sleeps",
       mon("Clefable", "Unaware", ["rest"], hpPercent=0.3), DOLL,
       ["p1.hp", "p1.status"]))
A(case("pain split averages the two HP totals",
       mon("Clefable", "Unaware", ["painsplit"], hpPercent=0.2), DOLL,
       ["p1.hp", "p2.hp"]))

# --- field and side conditions ----------------------------------------------
for move, field in (("sunnyday", "weather"), ("raindance", "weather"),
                    ("sandstorm", "weather"), ("snowscape", "weather"),
                    ("electricterrain", "terrain"), ("grassyterrain", "terrain"),
                    ("mistyterrain", "terrain"), ("psychicterrain", "terrain")):
    A(case(f"{move} sets the {field}",
           mon("Clefable", "Unaware", [move]), DOLL, [field]))

A(case("stealth rock lands on the foe's side",
       mon("Clefable", "Unaware", ["stealthrock"]), DOLL, ["p2.sideConditions"]))
A(case("spikes stack a layer",
       mon("Clefable", "Unaware", ["spikes"]), DOLL,
       ["p2.sideConditions", "p2.spikeLayers"]))
A(case("spikes stack onto an existing layer",
       mon("Clefable", "Unaware", ["spikes"]),
       dict(DOLL, sideConditions=["spikes"]), ["p2.spikeLayers"]))
A(case("toxic spikes stack a layer",
       mon("Clefable", "Unaware", ["toxicspikes"]), DOLL, ["p2.toxicSpikeLayers"]))
A(case("sticky web lands",
       mon("Clefable", "Unaware", ["stickyweb"]), DOLL, ["p2.sideConditions"]))
A(case("reflect goes up on your own side",
       mon("Clefable", "Unaware", ["reflect"]), DOLL, ["p1.sideConditions"]))
A(case("light screen goes up on your own side",
       mon("Clefable", "Unaware", ["lightscreen"]), DOLL, ["p1.sideConditions"]))
A(case("aurora veil needs snow",
       mon("Clefable", "Unaware", ["auroraveil"]), DOLL, ["p1.sideConditions"]))
A(case("aurora veil works in snow",
       mon("Clefable", "Unaware", ["auroraveil"]), DOLL, ["p1.sideConditions"],
       weather="snowscape"))
A(case("defog clears the foe's hazards",
       mon("Corviknight", "Pressure", ["defog"]),
       dict(DOLL, sideConditions=["stealthrock", "spikes"]),
       ["p2.sideConditions", "p1.sideConditions"]))
A(case("rapid spin clears your own hazards",
       dict(mon("Great Tusk", "Protosynthesis", ["rapidspin"]),
            sideConditions=["stealthrock", "spikes"]),
       DOLL, ["p1.sideConditions", "p1.boosts"]))

# --- volatiles ---------------------------------------------------------------
A(case("substitute costs a quarter and sets the volatile",
       mon("Clefable", "Unaware", ["substitute"]), DOLL,
       ["p1.hp", "p1.volatiles", "p1.subHP"]))
A(case("leech seed latches on",
       mon("Clefable", "Unaware", ["leechseed"]), DOLL, ["p2.volatiles"]))
A(case("grass types cannot be seeded",
       mon("Clefable", "Unaware", ["leechseed"]),
       mon("Venusaur", "Overgrow"), ["p2.volatiles"]))
A(case("leech seed drains at end of turn",
       mon("Clefable", "Unaware", hpPercent=0.5),
       dict(DOLL, hpPercent=0.9), ["p1.hp", "p2.hp"], seedP2=True))
A(case("taunt applies",
       mon("Clefable", "Unaware", ["taunt"]), DOLL, ["p2.volatiles"]))
A(case("protect guards the user",
       mon("Clefable", "Unaware", ["protect"]), DOLL, ["p1.volatiles"]))
A(case("salt cure sticks",
       mon("Garganacl", "Purifying Salt", ["saltcure"]), FROG, ["p2.volatiles"]))

# --- items and abilities on contact -----------------------------------------
A(case("rocky helmet chips the attacker",
       mon("Great Tusk", "Protosynthesis", ["closecombat"]),
       dict(FROG, item="Rocky Helmet"), ["p1.hp"]))
A(case("rough skin chips the attacker",
       mon("Great Tusk", "Protosynthesis", ["closecombat"]),
       mon("Garchomp", "Rough Skin"), ["p1.hp"]))
A(case("iron barbs chips the attacker",
       mon("Great Tusk", "Protosynthesis", ["closecombat"]),
       mon("Corviknight", "Iron Barbs"), ["p1.hp"]))
A(case("knock off removes the item",
       mon("Weavile", "Pressure", ["knockoff"]),
       dict(FROG, item="Leftovers"), ["p2.item"]))
A(case("trick swaps items",
       dict(mon("Clefable", "Unaware", ["trick"]), item="Choice Scarf"),
       dict(DOLL, item="Leftovers"), ["p1.item", "p2.item"]))

# --- ability immunities and absorption --------------------------------------
A(case("levitate ignores ground moves",
       mon("Great Tusk", "Protosynthesis", ["earthquake"]),
       mon("Rotom-Wash", "Levitate"), ["p2.hp"]))
A(case("water absorb heals instead",
       mon("Clefable", "Unaware", ["surf"]),
       dict(mon("Quagsire", "Water Absorb"), hpPercent=0.5), ["p2.hp"]))
A(case("volt absorb heals instead",
       mon("Clefable", "Unaware", ["thunderbolt"]),
       dict(mon("Jolteon", "Volt Absorb"), hpPercent=0.5), ["p2.hp"]))
A(case("sap sipper boosts attack instead",
       mon("Clefable", "Unaware", ["energyball"]),
       mon("Azumarill", "Sap Sipper"), ["p2.boosts", "p2.hp"]))
A(case("lightning rod boosts special attack instead",
       mon("Clefable", "Unaware", ["thunderbolt"]),
       mon("Pikachu", "Lightning Rod"), ["p2.boosts", "p2.hp"]))
A(case("soundproof ignores sound moves",
       mon("Clefable", "Unaware", ["boomburst"]),
       mon("Electrode", "Soundproof"), ["p2.hp"]))
A(case("overcoat ignores powder",
       mon("Venusaur", "Overgrow", ["sleeppowder"]),
       mon("Mandibuzz", "Overcoat"), ["p2.status"]))
A(case("grass types ignore powder",
       mon("Venusaur", "Overgrow", ["sleeppowder"]),
       mon("Rillaboom", "Grassy Surge"), ["p2.status"]))

# --- on-damage abilities -----------------------------------------------------
A(case("stamina raises defence when hit",
       mon("Clefable", "Unaware", ["tackle"]),
       mon("Mudsdale", "Stamina"), ["p2.boosts"]))
A(case("weak armor trades defence for speed",
       mon("Clefable", "Unaware", ["tackle"]),
       mon("Skarmory", "Weak Armor"), ["p2.boosts"]))
A(case("justified answers a dark move",
       mon("Weavile", "Pressure", ["knockoff"]),
       mon("Lucario", "Justified"), ["p2.boosts"]))
A(case("rattled answers a dark move",
       mon("Weavile", "Pressure", ["knockoff"]),
       mon("Sudowoodo", "Rattled"), ["p2.boosts"]))


# --- switching in: hazards and entry abilities -------------------------------
# These use a two-Pokemon team and a "switch 2" choice, so the incoming Pokemon
# takes hazards and fires its entry ability.

def sw(name, p1team, p2, check, p1=None, **kw):
    """A scenario whose p1 switches to team slot 2 on the turn.

    `p1` overrides the lead's spec (to put hazards on its own side); the team
    itself is what both engines are built from.
    """
    c = case(name, p1 or p1team[0], p2, check, **kw)
    c["p1team"] = [p1 or p1team[0]] + p1team[1:]
    c["p1move"] = "switch 2"
    return c


LEAD = mon("Clefable", "Unaware")

A(sw("stealth rock hits 1/8 at neutral",
     [LEAD, mon("Blissey", "Natural Cure")], DOLL, ["p1.hp"],
     p1={**LEAD, "sideConditions": ["stealthrock"]}))
A(sw("stealth rock hits 1/2 on a 4x weakness",
     [LEAD, mon("Talonflame", "Flame Body")], DOLL, ["p1.hp"],
     p1={**LEAD, "sideConditions": ["stealthrock"]}))
A(sw("stealth rock hits 1/32 on a 4x resist",
     [LEAD, mon("Great Tusk", "Protosynthesis")], DOLL, ["p1.hp"],
     p1={**LEAD, "sideConditions": ["stealthrock"]}))
A(sw("heavy-duty boots ignore stealth rock",
     [LEAD, dict(mon("Talonflame", "Flame Body"), item="Heavy-Duty Boots")],
     DOLL, ["p1.hp"], p1={**LEAD, "sideConditions": ["stealthrock"]}))
A(sw("spikes hit a grounded arrival",
     [LEAD, mon("Blissey", "Natural Cure")], DOLL, ["p1.hp"],
     p1={**LEAD, "sideConditions": ["spikes"]}))
A(sw("spikes spare a flyer",
     [LEAD, mon("Corviknight", "Pressure")], DOLL, ["p1.hp"],
     p1={**LEAD, "sideConditions": ["spikes"]}))
A(sw("toxic spikes poison a grounded arrival",
     [LEAD, mon("Blissey", "Natural Cure")], DOLL, ["p1.status"],
     p1={**LEAD, "sideConditions": ["toxicspikes"]}))
A(sw("a poison type absorbs toxic spikes",
     [LEAD, mon("Glimmora", "Toxic Debris")], DOLL,
     ["p1.status", "p1.sideConditions"],
     p1={**LEAD, "sideConditions": ["toxicspikes"]}))
A(sw("sticky web drops the arrival's speed",
     [LEAD, mon("Blissey", "Natural Cure")], DOLL, ["p1.boosts"],
     p1={**LEAD, "sideConditions": ["stickyweb"]}))
A(sw("intimidate drops the foe's attack",
     [LEAD, mon("Landorus-Therian", "Intimidate")], DOLL, ["p2.boosts"]))
A(sw("drought sets sun on entry",
     [LEAD, mon("Torkoal", "Drought")], DOLL, ["weather"]))
A(sw("drizzle sets rain on entry",
     [LEAD, mon("Pelipper", "Drizzle")], DOLL, ["weather"]))
A(sw("sand stream sets sand on entry",
     [LEAD, mon("Tyranitar", "Sand Stream")], DOLL, ["weather"]))
A(sw("snow warning sets snow on entry",
     [LEAD, mon("Abomasnow", "Snow Warning")], DOLL, ["weather"]))
A(sw("grassy surge sets terrain on entry",
     [LEAD, mon("Rillaboom", "Grassy Surge")], DOLL, ["terrain"]))
A(sw("electric surge sets terrain on entry",
     [LEAD, mon("Pincurchin", "Electric Surge")], DOLL, ["terrain"]))
A(sw("regenerator heals a third on the way out",
     [dict(mon("Slowbro", "Regenerator"), hpPercent=0.4),
      mon("Blissey", "Natural Cure")], DOLL, ["p1.hp"]))
A(sw("natural cure clears status on the way out",
     [dict(mon("Blissey", "Natural Cure"), status="brn"),
      mon("Clefable", "Unaware")], DOLL, ["p1.status"]))


# --- switching out mid-turn --------------------------------------------------
# A self-switch resolves before the opponent's already-locked move, so that move
# lands on the replacement. Phazing drags in a random Pokemon with no choice.

A({**case("u-turn: the opponent's move hits the replacement",
          mon("Weavile", "Pressure", ["uturn"]),
          mon("Snorlax", "Thick Fat", ["seismictoss"]), ["p1.hp", "p1.active"]),
   "p1team": [mon("Weavile", "Pressure", ["uturn"]),
              mon("Blissey", "Natural Cure", ["splash"])],
   "p1switchAfter": 2})
A({**case("parting shot also switches its user out",
          mon("Weavile", "Pressure", ["partingshot"]),
          mon("Snorlax", "Thick Fat", ["seismictoss"]), ["p2.boosts", "p1.active"]),
   "p1team": [mon("Weavile", "Pressure", ["partingshot"]),
              mon("Blissey", "Natural Cure", ["splash"])],
   "p1switchAfter": 2})


# =============================================================================
# Moves and abilities that used to be unmodelled.
# =============================================================================

def team(name, members, p2, check, choice="switch 2", **kw):
    """p1 fields a team; its first turn's choice defaults to switching to slot 2."""
    c = case(name, members[0], p2, check, **kw)
    c["p1team"] = members
    c["p1move"] = choice
    return c


TOSS = mon("Blissey", "Natural Cure", ["seismictoss"])     # 100 fixed damage
SLOW_TOSS = mon("Snorlax", "Thick Fat", ["seismictoss"])   # slower than most

# --- moves that call, charge, or land later ----------------------------------
A(case("sleep talk uses another of the user's moves",
       mon("Clefable", "Unaware", ["sleeptalk", "seismictoss"], status="slp", statusTurns=3),
       DOLL, ["p2.hp"]))
A(case("sleep talk does nothing awake",
       mon("Clefable", "Unaware", ["sleeptalk", "seismictoss"]), DOLL, ["p2.hp"]))
A(case("meteor beam spends its first turn charging",
       mon("Clefable", "Unaware", ["meteorbeam"]), DOLL,
       ["p1.boosts", "p2.hp", "p1.volatiles"]))
A(case("meteor beam fires on the second turn", mon("Clefable", "Unaware", ["meteorbeam"]),
       DOLL, ["p1.boosts", "p2.hp"], turns=[["move 1", "move 1"]], force=True))
A(case("power herb skips the charge",
       mon("Clefable", "Unaware", ["meteorbeam"], item="Power Herb"), DOLL,
       ["p1.boosts", "p1.item", "p2.hp"], force=True))
A(case("solar beam fires at once in sun", mon("Venusaur", "Overgrow", ["solarbeam"]),
       DOLL, ["p2.hp"], weather="sunnyday", force=True))
A(case("solar beam charges outside sun", mon("Venusaur", "Overgrow", ["solarbeam"]),
       DOLL, ["p2.hp", "p1.volatiles"]))
A(case("future sight does nothing on the turn it is used",
       mon("Clefable", "Unaware", ["futuresight", "splash"]), DOLL, ["p2.hp"]))
A(case("future sight lands two turns later",
       mon("Clefable", "Unaware", ["futuresight", "splash"]), DOLL, ["p2.hp"],
       turns=[["move 2", "move 1"], ["move 2", "move 1"]], force=True))
A(case("doom desire lands two turns later",
       mon("Clefable", "Unaware", ["doomdesire", "splash"]), DOLL, ["p2.hp"],
       turns=[["move 2", "move 1"], ["move 2", "move 1"]], force=True))
A(case("future sight cannot touch a dark type",
       mon("Clefable", "Unaware", ["futuresight", "splash"]),
       mon("Umbreon", "Synchronize"), ["p2.hp"],
       turns=[["move 2", "move 1"], ["move 2", "move 1"]], force=True))
A(case("wish heals at the end of the next turn",
       mon("Clefable", "Unaware", ["wish", "splash"], hpPercent=0.3), DOLL, ["p1.hp"],
       turns=[["move 2", "move 1"]]))
A(case("wish waits a turn", mon("Clefable", "Unaware", ["wish"], hpPercent=0.3), DOLL,
       ["p1.hp"]))
A(case("focus punch fails if the user is hit first",
       mon("Clefable", "Unaware", ["focuspunch"]), mon("Weavile", "Pressure", ["tackle"]),
       ["p2.hp"], force=True))
A(case("focus punch lands if nothing hits the user",
       mon("Clefable", "Unaware", ["focuspunch"]), mon("Weavile", "Pressure"),
       ["p2.hp"], force=True))
A(case("beak blast burns whatever touches it while it heats up",
       mon("Toucannon", "Keen Eye", ["beakblast"]), mon("Weavile", "Pressure", ["tackle"]),
       ["p2.status"], force=True))
A(case("transform copies species, stat stages and typing",
       mon("Ditto", "Limber", ["transform"]),
       dict(mon("Gyarados", "Moxie"), boosts={"atk": 2}),
       ["p1.species", "p1.boosts", "p1.types"]))
A(case("mat block stops attacks on its first turn",
       mon("Weavile", "Pressure", ["matblock"]), mon("Clefable", "Unaware", ["tackle"]),
       ["p1.hp"]))
A(case("beat up hits once per healthy party member",
       mon("Weavile", "Pressure", ["beatup"]), DOLL, ["p2.hp"], force=True,
       p1team=[mon("Weavile", "Pressure", ["beatup"]), mon("Blissey", "Natural Cure"),
               mon("Snorlax", "Thick Fat", status="brn")]))

# --- one-line effects ---------------------------------------------------------
A(case("sparkling aria cures the target's burn",
       mon("Primarina", "Torrent", ["sparklingaria"]), dict(DOLL, status="brn"),
       ["p2.status"]))
A(case("ceaseless edge lays spikes", mon("Samurott-Hisui", "Sharpness", ["ceaselessedge"]),
       DOLL, ["p2.sideConditions", "p2.spikeLayers"], force=True))
A(case("stone axe lays stealth rock", mon("Kleavor", "Sharpness", ["stoneaxe"]), DOLL,
       ["p2.sideConditions"], force=True))
A(case("sheer force swallows stone axe's rocks",
       mon("Kleavor", "Sheer Force", ["stoneaxe"]), DOLL, ["p2.sideConditions"], force=True))
A(case("grassy glide jumps the queue on grassy terrain",
       mon("Rillaboom", "Grassy Surge", ["grassyglide"]),
       mon("Weavile", "Pressure", ["seismictoss"], hpPercent=0.01), ["p1.hp"],
       terrain="grassyterrain", force=True))
A(case("hyperspace fury fails for all but hoopa-unbound",
       mon("Clefable", "Unaware", ["hyperspacefury"]), DOLL, ["p2.hp", "p1.boosts"]))
A(case("hyperspace fury from hoopa-unbound",
       mon("Hoopa-Unbound", "Magician", ["hyperspacefury"]), DOLL, ["p1.boosts"], force=True))
A(case("take heart cures and boosts",
       mon("Clefable", "Unaware", ["takeheart"], status="brn"), DOLL,
       ["p1.status", "p1.boosts"]))
A(case("fusion flare doubles after fusion bolt",
       mon("Clefable", "Unaware", ["fusionflare"]), mon("Zekrom", "Teravolt", ["fusionbolt"]),
       ["p2.hp"], force=True))
A(case("bug bite eats the target's sitrus berry",
       mon("Clefable", "Unaware", ["bugbite"], hpPercent=0.5),
       dict(DOLL, item="Sitrus Berry"), ["p1.hp", "p2.item"], force=True))
A(case("stuff cheeks eats the berry for +2 defense",
       mon("Clefable", "Unaware", ["stuffcheeks"], item="Sitrus Berry", hpPercent=0.9),
       DOLL, ["p1.hp", "p1.item", "p1.boosts"]))
A(case("clangorous soul trades a third of HP for +1 everything",
       mon("Kommo-o", "Bulletproof", ["clangoroussoul"]), DOLL, ["p1.hp", "p1.boosts"]))
A(case("clangorous soul fails at a third of HP",
       mon("Kommo-o", "Bulletproof", ["clangoroussoul"], hpPercent=0.3), DOLL,
       ["p1.hp", "p1.boosts"]))
A(case("relic song turns meloetta pirouette",
       mon("Meloetta", "Serene Grace", ["relicsong"]), DOLL, ["p1.species"]))
A(case("magnet rise lifts its user, not the target",
       mon("Clefable", "Unaware", ["magnetrise"]), DOLL, ["p1.volatiles", "p2.volatiles"]))
A(case("magnet rise fails under gravity",
       mon("Clefable", "Unaware", ["magnetrise"]), DOLL, ["p1.volatiles"], gravity=True))
A(case("syrup bomb drains speed at the end of the turn",
       mon("Dipplin", "Sticky Hold", ["syrupbomb"]), DOLL, ["p2.boosts"], force=True))
A(case("thousand arrows grounds a flyer",
       mon("Clefable", "Unaware", ["thousandarrows"]), mon("Corviknight", "Pressure"),
       ["p2.volatiles", "p2.hp"], force=True))
A(case("teleport switches its user out",
       mon("Clefable", "Unaware", ["teleport"]), DOLL, ["p1.active"],
       p1team=[mon("Clefable", "Unaware", ["teleport"]), mon("Blissey", "Natural Cure")],
       p1switchAfter=2))
A(team("baton pass hands over stat stages",
       [mon("Clefable", "Unaware", ["batonpass"], boosts={"atk": 2}),
        mon("Blissey", "Natural Cure")], DOLL, ["p1.boosts", "p1.active"],
       choice="move 1", p1switchAfter=2))
A(team("shed tail leaves its substitute behind",
       [mon("Cyclizar", "Regenerator", ["shedtail"]), mon("Blissey", "Natural Cure")], DOLL,
       ["p1.volatiles", "p1.subHP", "p1.active"], choice="move 1", p1switchAfter=2))
A(case("encore locks the target into its last move",
       mon("Weavile", "Pressure", ["splash", "encore"]),
       mon("Blissey", "Natural Cure", ["seismictoss", "splash"]),
       ["p1.hp", "p2.volatiles"], turns=[["move 2", "move 2"]]))
A(case("disable stops the target's last move",
       mon("Weavile", "Pressure", ["splash", "disable"]), TOSS,
       ["p1.hp", "p2.volatiles"], turns=[["move 2", "move 1"]]))
A(case("confuse ray's confusion makes its target hit itself",
       mon("Weavile", "Pressure", ["confuseray"]), TOSS, ["p1.hp", "p2.hp"], force=True))

# --- fixes found along the way ------------------------------------------------
A(case("status moves stop at a substitute",
       mon("Clefable", "Unaware", ["toxic"]), dict(DOLL, volatiles=["substitute"]),
       ["p2.status"]))
A(case("defiant answers a secondary's drop",
       mon("Weavile", "Pressure", ["icywind"]), mon("Kingambit", "Defiant"),
       ["p2.boosts"], force=True))
A(case("knock off takes nothing through protect",
       mon("Weavile", "Pressure", ["knockoff"]),
       mon("Blissey", "Natural Cure", ["protect"], item="Leftovers"), ["p2.item"]))
A(case("inner focus ignores flinching",
       mon("Weavile", "Pressure", ["fakeout"]), mon("Lucario", "Inner Focus", ["seismictoss"]),
       ["p1.hp"], force=True))
A(case("own tempo can still be made to flinch",
       mon("Weavile", "Pressure", ["fakeout"]), mon("Slowbro", "Own Tempo", ["seismictoss"]),
       ["p1.hp"], force=True))

# --- abilities -----------------------------------------------------------------
A(case("harvest regrows a sitrus berry in sun",
       mon("Exeggutor", "Harvest", item="Sitrus Berry", hpPercent=0.4), DOLL,
       ["p1.hp", "p1.item"], weather="sunnyday"))
A(case("cursed body disables the move that hit it",
       mon("Clefable", "Unaware", ["moonblast"]), mon("Gengar", "Cursed Body"),
       ["p1.volatiles"], force=True))
A(case("toxic chain badly poisons",
       mon("Okidogi", "Toxic Chain", ["tackle"]), DOLL, ["p2.status"], force=True))
A(case("dancer copies a swords dance",
       mon("Clefable", "Unaware", ["swordsdance"]), mon("Oricorio", "Dancer"),
       ["p2.boosts"]))
A(case("unnerve keeps the foe off its berries",
       mon("Pyroar", "Unnerve"), dict(DOLL, item="Sitrus Berry", hpPercent=0.4),
       ["p2.item", "p2.hp"]))
A(case("wind rider eats a wind move",
       mon("Clefable", "Unaware", ["hurricane"]), mon("Brambleghast", "Wind Rider"),
       ["p2.boosts", "p2.hp"]))
A(case("wind rider rides its own tailwind",
       mon("Brambleghast", "Wind Rider", ["tailwind"]), DOLL, ["p1.boosts"]))
A(case("magician steals the target's item",
       mon("Hoopa-Unbound", "Magician", ["darkpulse"]), dict(DOLL, item="Leftovers"),
       ["p1.item", "p2.item"], force=True))
A(case("libero turns into the move's type",
       mon("Cinderace", "Libero", ["tackle"]), DOLL, ["p1.types"], force=True))
A(case("protean turns into the move's type",
       mon("Greninja", "Protean", ["icebeam"]), DOLL, ["p1.types"], force=True))
A(case("magic bounce reflects stealth rock",
       mon("Clefable", "Unaware", ["stealthrock"]), mon("Espeon", "Magic Bounce"),
       ["p1.sideConditions", "p2.sideConditions"]))
A(case("magic bounce reflects toxic",
       mon("Clefable", "Unaware", ["toxic"]), mon("Espeon", "Magic Bounce"),
       ["p1.status", "p2.status"], force=True))
# The fainting side keeps a reserve: when the last Pokemon falls the battle
# ends there, before any of these abilities has a chance to fire.
A(case("soul-heart gains sp. atk when something faints",
       mon("Magearna", "Soul-Heart", ["seismictoss"]), dict(DOLL, hpPercent=0.01),
       ["p1.boosts"], p2team=[dict(DOLL, hpPercent=0.01), mon("Clefable", "Unaware")]))
A(team("download reads the foe's weaker defence",
       [LEAD, mon("Porygon2", "Download")], DOLL, ["p1.boosts"]))
A(case("aroma veil blocks taunt",
       mon("Clefable", "Unaware", ["taunt"]), mon("Alcremie", "Aroma Veil"),
       ["p2.volatiles"]))
A(case("ice body heals in snow",
       mon("Glaceon", "Ice Body", hpPercent=0.5), DOLL, ["p1.hp"], weather="snowscape"))
A(case("rain dish heals in rain",
       mon("Ludicolo", "Rain Dish", hpPercent=0.5), DOLL, ["p1.hp"], weather="raindance"))
A(case("dry skin heals an eighth in rain",
       mon("Toxicroak", "Dry Skin", hpPercent=0.5), DOLL, ["p1.hp"], weather="raindance"))
A(case("dry skin burns an eighth in sun",
       mon("Toxicroak", "Dry Skin", hpPercent=0.5), DOLL, ["p1.hp"], weather="sunnyday"))
A(case("solar power burns an eighth in sun",
       mon("Houndoom", "Solar Power", hpPercent=0.5), DOLL, ["p1.hp"], weather="sunnyday"))
A(case("truant loafs every other turn",
       mon("Slaking", "Truant", ["seismictoss"]), DOLL, ["p2.hp"],
       turns=[["move 1", "move 1"]]))
A(case("good as gold ignores status moves",
       mon("Clefable", "Unaware", ["thunderwave"]), mon("Gholdengo", "Good as Gold"),
       ["p2.status"]))
A(case("cud chew eats its berry twice",
       mon("Tauros-Paldea-Aqua", "Cud Chew", item="Sitrus Berry", hpPercent=0.4), DOLL,
       ["p1.hp", "p1.item"], turns=[["move 1", "move 1"]]))
A(case("bulletproof ignores ball moves",
       mon("Clefable", "Unaware", ["shadowball"]), mon("Chesnaught", "Bulletproof"),
       ["p2.hp"]))
A(case("leaf guard blocks status in sun",
       mon("Clefable", "Unaware", ["thunderwave"]), mon("Zarude", "Leaf Guard"),
       ["p2.status"], weather="sunnyday"))
A(case("cheek pouch heals on top of the berry",
       mon("Dedenne", "Cheek Pouch", item="Sitrus Berry", hpPercent=0.4), DOLL, ["p1.hp"]))
A(case("oblivious ignores taunt",
       mon("Clefable", "Unaware", ["taunt"]), mon("Whiscash", "Oblivious"), ["p2.volatiles"]))
A(team("oblivious ignores intimidate",
       [LEAD, mon("Landorus-Therian", "Intimidate")], mon("Whiscash", "Oblivious"),
       ["p2.boosts"]))
A(case("toxic debris scatters toxic spikes when hit physically",
       mon("Clefable", "Unaware", ["tackle"]), mon("Glimmora", "Toxic Debris"),
       ["p1.sideConditions", "p1.toxicSpikeLayers"], force=True))
A(case("as one gains attack from a knockout",
       mon("Calyrex-Ice", "As One (Glastrier)", ["seismictoss"]), dict(DOLL, hpPercent=0.01),
       ["p1.boosts"], p2team=[dict(DOLL, hpPercent=0.01), mon("Clefable", "Unaware")]))
A(case("hunger switch flips morpeko's mode each turn",
       mon("Morpeko", "Hunger Switch"), DOLL, ["p1.species"]))
A(case("ice face takes the first physical hit",
       mon("Clefable", "Unaware", ["tackle"]), mon("Eiscue", "Ice Face"),
       ["p2.hp", "p2.species"], force=True))
A(case("ice face lets special hits through",
       mon("Clefable", "Unaware", ["moonblast"]), mon("Eiscue", "Ice Face"),
       ["p2.species"], force=True))
A(case("tera shift turns terapagos terastal on entry",
       mon("Terapagos", "Tera Shift"), DOLL, ["p1.species", "p1.ability", "p1.maxhp", "p1.hp"]))
A(case("terapagos terastallizes into its stellar form",
       mon("Terapagos", "Tera Shift", ["terastarstorm"], tera="Stellar"), DOLL,
       ["p1.species", "p1.ability", "p2.hp"], p1move="move 1 terastallize", force=True))
A(team("zero to hero makes palafin a hero as it leaves",
       [mon("Palafin", "Zero to Hero"), mon("Blissey", "Natural Cure")], DOLL,
       ["p1.species"]))
A(case("unseen fist punches through protect",
       mon("Urshifu", "Unseen Fist", ["closecombat"]),
       mon("Blissey", "Natural Cure", ["protect"]), ["p2.hp"], force=True))
A(case("disguise takes the first hit, at a cost",
       mon("Clefable", "Unaware", ["shadowball"]), mon("Mimikyu", "Disguise"),
       ["p2.hp", "p2.species"], force=True))
A(case("damp stops explosion",
       mon("Clefable", "Unaware", ["explosion"]), mon("Swampert", "Damp"),
       ["p1.hp", "p2.hp"]))
A(case("cute charm infatuates on contact",
       mon("Clefable", "Unaware", ["tackle"], gender="M"),
       mon("Enamorus", "Cute Charm", gender="F"), ["p1.volatiles"], force=True))
A(case("cute charm needs opposite genders",
       mon("Clefable", "Unaware", ["tackle"], gender="F"),
       mon("Enamorus", "Cute Charm", gender="F"), ["p1.volatiles"], force=True))
A(case("poison puppeteer confuses what it poisons",
       mon("Pecharunt", "Poison Puppeteer", ["toxic"]), DOLL,
       ["p2.status", "p2.volatiles"]))
A(case("mirror armor reflects growl",
       mon("Clefable", "Unaware", ["growl"]), mon("Corviknight", "Mirror Armor"),
       ["p1.boosts", "p2.boosts"]))
A(case("gulp missile catches something on surf",
       mon("Cramorant", "Gulp Missile", ["surf"]), DOLL, ["p1.species"]))
A(case("gulp missile spits its catch",
       mon("Cramorant", "Gulp Missile", ["surf"]),
       mon("Blissey", "Natural Cure", ["splash", "tackle"]),
       ["p1.species", "p2.boosts"], turns=[["move 1", "move 2"]], force=True))
A(case("gooey slows a contact attacker",
       mon("Clefable", "Unaware", ["tackle"]), mon("Goodra", "Gooey"), ["p1.boosts"],
       force=True))
A(case("tangling hair slows a contact attacker",
       mon("Clefable", "Unaware", ["tackle"]), mon("Dugtrio-Alola", "Tangling Hair"),
       ["p1.boosts"], force=True))
A(case("seed sower raises grassy terrain when hit",
       mon("Clefable", "Unaware", ["tackle"]), mon("Arboliva", "Seed Sower"),
       ["terrain"], force=True))
A(case("electromorphosis charges when hit",
       mon("Clefable", "Unaware", ["tackle"]), mon("Bellibolt", "Electromorphosis"),
       ["p2.volatiles"], force=True))
A(case("sticky hold keeps its item",
       mon("Weavile", "Pressure", ["knockoff"]),
       mon("Dipplin", "Sticky Hold", item="Eviolite"), ["p2.item"], force=True))
A(case("queenly majesty stops priority moves",
       mon("Clefable", "Unaware", ["quickattack"]), mon("Tsareena", "Queenly Majesty"),
       ["p2.hp"]))
A(case("mycelium might's status moves go last",
       mon("Toedscruel", "Mycelium Might", ["spore"]), TOSS, ["p1.hp", "p2.status"]))
A(case("mycelium might's status moves ignore abilities",
       mon("Toedscruel", "Mycelium Might", ["spore"]), mon("Blissey", "Insomnia"),
       ["p2.status"]))
A(case("shields down arrives in its meteor shell",
       mon("Minior", "Shields Down", ["thunderwave"]),
       mon("Minior", "Shields Down", ["thunderwave"]),
       ["p1.species", "p2.species", "p1.status", "p2.status"]))
A(case("shields down drops its shell below half",
       mon("Minior", "Shields Down", hpPercent=0.4), DOLL, ["p1.species"]))
A(case("imposter transforms on arrival",
       mon("Ditto", "Imposter"), dict(mon("Gyarados", "Moxie"), boosts={"def": 1}),
       ["p1.species", "p1.ability", "p1.types"]))
A(case("trace copies intimidate and uses it",
       mon("Gardevoir", "Trace"), mon("Gyarados", "Intimidate"),
       ["p1.ability", "p2.boosts"]))
A(case("battle bond powers up after a knockout",
       mon("Greninja-Bond", "Battle Bond", ["seismictoss"]), dict(DOLL, hpPercent=0.01),
       ["p1.boosts"], p2team=[dict(DOLL, hpPercent=0.01), mon("Clefable", "Unaware")]))
A(case("anger shell snaps past half HP",
       mon("Clefable", "Unaware", ["seismictoss"]), mon("Klawf", "Anger Shell", hpPercent=0.6),
       ["p2.boosts"]))
A(case("effect spore can put an attacker to sleep",
       mon("Clefable", "Unaware", ["tackle"]), mon("Vileplume", "Effect Spore"),
       ["p1.status"], force=True))
A(case("pickpocket steals from a contact attacker",
       mon("Clefable", "Unaware", ["tackle"], item="Leftovers"),
       mon("Weavile", "Pickpocket"), ["p1.item", "p2.item"], force=True))
A(case("bad dreams hurt a sleeping foe",
       mon("Darkrai", "Bad Dreams"), dict(DOLL, status="slp", statusTurns=3), ["p2.hp"]))
A(case("a plate stays with arceus",
       mon("Weavile", "Pressure", ["knockoff"]),
       mon("Arceus-Fire", "Multitype", item="Flame Plate"), ["p2.item"], force=True))

print(json.dumps(CASES, indent=1))
