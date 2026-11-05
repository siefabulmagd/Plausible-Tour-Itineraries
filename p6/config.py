"""Central configuration. Every assumption that is NOT in the dataset lives here."""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    data_dir: str = "tour-cikm16/data"
    city: str = "Melb"          # Edin / Melb / Toro / Glas / Osak
    min_visits: int = 20        # keep places with >= this many visits
    max_pool: int = 60          # ... and at most this many places
    # Timestamps in the raw files look shifted (see README, "Timezone assumption").
    tz_offset_h: int = 10       # local clock = epoch + tz_offset_h  (ASSUMPTION)
    min_dur_s: int = 300        # visits shorter than 5 min are unreliable -> not used as duration labels
    max_dur_s: int = 8 * 3600   # winsorise the long tail
    # ---- generated (not in dataset) travel model ----------------------------
    detour: float = 1.3         # road distance = detour * straight line
    walk_kmh: float = 4.5
    transit_kmh: float = 20.0
    transit_wait_min: float = 10.0
    tick: int = 5               # all times are multiples of 5 minutes
    # ---- evaluation --------------------------------------------------------
    k_folds: int = 5            # user-level cross-validation
    val_frac: float = 0.15      # of the training users, used for early stopping / tuning
    seed: int = 0
    # ---- search -------------------------------------------------------------
    lam: float = 0.05           # nats per minute of travel (tuned on validation queries)
    max_expansions: int = 200_000
    # ---- NN -----------------------------------------------------------------
    emb: int = 8
    hidden: int = 32
    dropout: float = 0.5
    lr: float = 3e-3
    weight_decay: float = 1e-3
    max_epochs: int = 300
    patience: int = 30
    hour_harm: int = 1          # harmonics of the hour encoding (sin/cos pairs)
    batch: int = 128
