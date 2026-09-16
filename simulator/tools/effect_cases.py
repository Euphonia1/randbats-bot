"""Scenario definitions for the effects differential test.

Each entry runs one turn in a real Showdown battle and in psjax, then compares
the fields named in `check`. Scenarios are deliberately deterministic: the
opposing move is usually Splash so HP changes come only from the effect under
test, and only RNG-independent fields are compared.

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

print(json.dumps(CASES, indent=1))
