"""Abilities and items the engine models, with stable integer ids.

Engine code compares a traced ability/item index against the Python int
constants generated from these lists, e.g. `ab == A.GUTS`. Index 0 is always
`none`: any ability or item outside these lists compiles to 0 and therefore has
no effect, which `coverage.py` reports rather than hiding.

Keep the order append-only -- the ids are baked into the compiled `.npz`.
"""

# --- Abilities ---------------------------------------------------------------
ABILITY_NAMES = [
    "none",
    # stat / damage multipliers
    "adaptability", "analytic", "hugepower", "purepower", "guts", "hustle",
    "technician", "sheerforce", "toughclaws", "strongjaw", "megalauncher",
    "ironfist", "punkrock", "sharpness", "reckless", "rivalry", "steelworker",
    "transistor", "dragonsmaw", "rockypayload", "tintedlens", "neuroforce",
    "solidrock", "filter", "prismarmor", "multiscale", "shadowshield",
    "thickfat", "icescales", "fluffy", "furcoat", "marvelscale", "grasspelt",
    "heatproof", "waterbubble", "purifyingsalt", "overgrow", "blaze", "torrent",
    "swarm", "flashfire", "steamengine", "waterabsorb", "voltabsorb",
    "eartheater", "dryskin", "sapsipper", "lightningrod", "stormdrain",
    "motordrive", "windrider", "wellbakedbody",
    "levitate", "bulletproof", "soundproof", "overcoat", "magicguard",
    "unaware", "moldbreaker", "turboblaze", "teravolt", "scrappy", "infiltrator",
    "serenegrace", "sniper", "superluck", "moxie", "chillingneigh", "asoneglastrier",
    "grimneigh", "asonespectrier", "beastboost", "download", "intimidate",
    "defiant", "competitive", "speedboost", "moody", "simple", "contrary",
    "clearbody", "whitesmoke", "fullmetalbody", "hypercutter", "bigpecks",
    "keeneye", "mirrorarmor", "compoundeyes",
    # weather / terrain
    "drought", "drizzle", "sandstream", "snowwarning", "orichalcumpulse",
    "hadronengine", "desolateland", "primordialsea", "deltastream",
    "electricsurge", "grassysurge", "mistysurge", "psychicsurge",
    "chlorophyll", "swiftswim", "sandrush", "slushrush", "solarpower",
    "sandforce", "sandveil", "snowcloak", "icebody", "raindish",
    "protosynthesis", "quarkdrive", "flowergift", "forecast",
    # status / immunity
    "immunity", "limber", "insomnia", "vitalspirit", "waterveil", "magmaarmor",
    "sweetveil", "flowerveil", "leafguard", "naturalcure", "shedskin",
    "hydration", "poisonheal", "toxicboost", "flareboost", "quickfeet",
    "magicbounce", "goodasgold", "aromaveil", "innerfocus", "owntempo",
    "shielddust", "stickyhold", "sturdy", "battlearmor", "shellarmor",
    "damp", "wonderguard", "comatose",
    # contact / on-hit
    "roughskin", "ironbarbs", "aftermath", "cursedbody", "flamebody",
    "static", "poisonpoint", "effectspore", "gooey", "tanglinghair",
    "weakarmor", "stamina", "watercompaction", "justified", "rattled",
    "berserk", "angerpoint", "colorchange", "mummy", "wanderingspirit",
    "perishbody", "innardsout", "thermalexchange", "seedsower", "toxicdebris",
    "electromorphosis", "windpower", "sandspit", "cottondown", "gulpmissile",
    "angershell",
    # switch / field
    "regenerator", "intrepidsword", "dauntlessshield",
    "screencleaner", "neutralizinggas", "unnerve", "pressure", "arenatrap",
    "shadowtag", "magnetpull", "suctioncups", "runaway", "trace",
    "libero", "protean", "prankster", "galewings", "triage", "quickdraw",
    "stall", "myceliummight", "slowstart", "truant", "defeatist", "zenmode",
    "iceface", "disguise", "multitype", "rkssystem",
    "supremeoverlord", "costar", "opportunist", "guarddog",
    "embodyaspectteal", "embodyaspectwellspring", "embodyaspecthearthflame",
    "embodyaspectcornerstone", "toxicchain", "hospitality", "poisonpuppeteer",
    "tabletsofruin", "swordofruin", "vesselofruin", "beadsofruin", "mindseye", "supersweetsyrup",
    # Added for Random Battle coverage.
    "harvest", "poisontouch", "frisk", "synchronize", "liquidooze",
    "unburden", "illusion", "stakeout", "dancer", "rockhead",
    "skilllink", "airlock", "magician", "soulheart", "surgesurfer",
    "cloudnine", "galvanize", "earlybird", "cudchew", "oblivious",
    "noguard", "cheekpouch", "corrosion", "hungerswitch", "unseenfist",
    "zerotohero", "terashift", "imposter", "pickpocket", "baddreams",
    "battlebond", "pixilate", "liquidvoice", "queenlymajesty", "shieldsdown",
    "powerspot", "heavymetal", "lightmetal", "cutecharm", "refrigerate",
    "aerilate", "normalize", "steelyspirit",
]

# --- Items -------------------------------------------------------------------
ITEM_NAMES = [
    "none",
    # gen9 randbats pool
    "airballoon", "assaultvest", "blunderpolicy", "boosterenergy", "chestoberry",
    "choicescarf", "clearamulet", "covertcloak", "eviolite", "expertbelt",
    "focussash", "heavydutyboots", "leftovers", "leppaberry", "lifeorb",
    "lightball", "lightclay", "loadeddice", "lumberry", "lustrousorb",
    "magnet", "mysticwater", "powerherb", "rockyhelmet", "scopelens",
    "silkscarf", "sitrusberry", "souldew", "throatspray", "toxicorb",
    "weaknesspolicy", "whiteherb", "widelens",
    # common outside randbats, cheap to support
    "choiceband", "choicespecs", "flameorb", "blacksludge", "shellbell",
    "quickclaw", "brightpowder", "muscleband", "wiseglasses", "razorclaw",
    "kingsrock", "safetygoggles", "protectivepads", "utilityumbrella", "mentalherb", "roomservice", "adrenalineorb",
    "absorbbulb", "cellbattery", "luminousmoss", "snowball", "ejectbutton",
    "ejectpack", "redcard", "berryjuice", "oranberry", "figyberry",
    "chopleberry", "occaberry", "passhoberry", "wacanberry", "rindoberry",
    "yacheberry", "chartiberry", "kasibberry", "habanberry", "colburberry",
    "babiriberry", "roseliberry", "shucaberry", "payapaberry", "tangaberry",
    "chilanberry", "custapberry", "salacberry", "petayaberry", "liechiberry", "metronome", "punchingglove", "abilityshield",
    "mirrorherb", "leek", "thickclub", "stick",
]

ABILITY_IDX = {n: i for i, n in enumerate(ABILITY_NAMES)}
ITEM_IDX = {n: i for i, n in enumerate(ITEM_NAMES)}


class _Namespace:
    """Attribute access to the ids, e.g. `A.GUTS`, `I.LEFTOVERS`."""

    def __init__(self, mapping):
        for name, idx in mapping.items():
            setattr(self, name.upper().replace(" ", "").replace("-", ""), idx)


A = _Namespace(ABILITY_IDX)
I = _Namespace(ITEM_IDX)

# --- Derived membership sets -------------------------------------------------
# build.py turns these into boolean columns so the engine can test a whole class
# of ability with one lookup instead of a chain of comparisons.

MOLD_BREAKER = {"moldbreaker", "turboblaze", "teravolt"}
"""Abilities that ignore the target's ability during a move."""

UNSUPPRESSABLE = {"asoneglastrier", "asonespectrier", "battlebond", "comatose", "disguise",
                  "gulpmissile", "iceface", "multitype", "rkssystem", "schooling",
                  "shieldsdown", "stancechange", "zenmode", "zerotohero"}
"""Abilities Neutralizing Gas and friends cannot turn off."""

# Type-absorbing abilities: ability -> (absorbed type, heal fraction * 16, boosted
# stat or -1, boost amount). Heal 0 with a boost means "boost instead of heal".
ABSORB = {
    "waterabsorb":   ("water",    4, -1, 0),
    "dryskin":       ("water",    4, -1, 0),
    "voltabsorb":    ("electric", 4, -1, 0),
    "earthheater":   ("ground",   4, -1, 0),
    "eartheater":    ("ground",   4, -1, 0),
    "wellbakedbody": ("fire",     0, "def", 2),
    "sapsipper":     ("grass",    0, "atk", 1),
    "lightningrod":  ("electric", 0, "spa", 1),
    "stormdrain":    ("water",    0, "spa", 1),
    "motordrive":    ("electric", 0, "spe", 1),
    "flashfire":     ("fire",     0, "spa", 0),
}

# Abilities granting outright immunity to a type of damaging move.
TYPE_IMMUNE = {"levitate": "ground"}

# Ability -> weather set on switch-in.
WEATHER_SETTER = {
    "drought": "sunnyday", "orichalcumpulse": "sunnyday", "drizzle": "raindance",
    "sandstream": "sandstorm", "snowwarning": "snowscape",
    "desolateland": "desolateland", "primordialsea": "primordialsea",
    "deltastream": "deltastream",
}

# Ability -> terrain set on switch-in.
TERRAIN_SETTER = {
    "electricsurge": "electricterrain", "hadronengine": "electricterrain",
    "grassysurge": "grassyterrain", "mistysurge": "mistyterrain",
    "psychicsurge": "psychicterrain",
}

# Abilities that prevent the listed major status.
STATUS_IMMUNE = {
    "immunity": ("psn", "tox"), "limber": ("par",), "insomnia": ("slp",),
    "vitalspirit": ("slp",), "waterveil": ("brn",), "magmaarmor": ("frz",),
    "purifyingsalt": ("brn", "par", "slp", "frz", "psn", "tox"),
    "comatose": ("brn", "par", "slp", "frz", "psn", "tox"),
    "thermalexchange": ("brn",),
}

# Berry -> (type it resists, whether it also covers neutral hits).
RESIST_BERRY = {
    "chopleberry": "fighting", "occaberry": "fire", "passhoberry": "water",
    "wacanberry": "electric", "rindoberry": "grass", "yacheberry": "ice",
    "chartiberry": "rock", "kasibberry": "ghost", "habanberry": "dragon",
    "colburberry": "dark", "babiriberry": "steel", "roseliberry": "fairy",
    "shucaberry": "ground", "payapaberry": "psychic", "tangaberry": "bug",
    "chilanberry": "normal", "cobaberry": "flying",
}

# Type-boosting items: item -> (type, multiplier as a 4096-scaled modifier).
TYPE_BOOST_ITEM = {
    "magnet": ("electric", 4915), "mysticwater": ("water", 4915),
    "silkscarf": ("normal", 4915), "souldew": ("psychic", 4915),
}

CHOICE_ITEMS = {"choicescarf", "choiceband", "choicespecs"}

# Abilities that switch the weather off entirely while their holder is out.
# (`mechanics.weather_active` previously tested Neutralizing Gas here, which is
# a different ability with a different effect.)
WEATHER_SUPPRESS = {"airlock", "cloudnine"}

# "-ate" abilities: Normal-type moves become this type and gain 20% power.
ATE_ABILITIES = {
    "galvanize": "electric", "pixilate": "fairy", "refrigerate": "ice",
    "aerilate": "flying",
}
