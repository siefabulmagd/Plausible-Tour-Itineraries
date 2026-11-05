"""Loading, cleaning and splitting the Flickr tour data (Lim et al.).

Verified schema (see README):
  poi-<City>.csv : poiID, poiCat, poiLon, poiLat
  traj-<City>.csv: userID, trajID, poiID, startTime, endTime, #photo, trajLen, poiDuration
Row order inside a trajectory is NOT chronological, so we always sort ourselves.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def travel_minutes(dist_km: np.ndarray, cfg: Config) -> np.ndarray:
    """GENERATED travel model: the faster of walking and 'transit' (wait + ride), rounded up to ticks."""
    d = dist_km * cfg.detour
    walk = d / cfg.walk_kmh * 60
    transit = cfg.transit_wait_min + d / cfg.transit_kmh * 60
    t = np.minimum(walk, transit)
    return np.maximum(cfg.tick, cfg.tick * np.ceil(t / cfg.tick)).astype(int)


@dataclass
class Dataset:
    cfg: Config
    pois: pd.DataFrame        # row index == place index used everywhere (0..N-1)
    cats: list                # category names; cat index = position in this list
    cat_of: np.ndarray        # (N,) category index of every place
    visits: pd.DataFrame      # in-pool visits, chronologically sorted inside segments
    trans: pd.DataFrame       # consecutive visits (from, to, hour of departure)
    segments: pd.DataFrame    # one row per uninterrupted in-pool run of visits
    tt: np.ndarray            # (N,N) travel minutes (multiples of tick)
    dist_km: np.ndarray

    @property
    def N(self):
        return len(self.pois)

    @property
    def C(self):
        return len(self.cats)


def load_dataset(cfg: Config) -> Dataset:
    poi = pd.read_csv(f"{cfg.data_dir}/poi-{cfg.city}.csv")
    tr = pd.read_csv(f"{cfg.data_dir}/traj-{cfg.city}.csv").rename(columns={"#photo": "photos"})
    tr = tr.sort_values(["trajID", "startTime", "endTime", "poiID"], kind="mergesort").reset_index(drop=True)

    # --- place pool: >= min_visits visits, at most max_pool places (most visited first)
    counts = tr.poiID.value_counts()
    keep = counts[counts >= cfg.min_visits].head(cfg.max_pool).index
    pois = poi[poi.poiID.isin(keep)].copy()
    pois["visits"] = pois.poiID.map(counts)
    pois = pois.sort_values("poiID").reset_index(drop=True)
    id2idx = {pid: i for i, pid in enumerate(pois.poiID)}
    cats = sorted(pois.poiCat.unique())
    c2i = {c: i for i, c in enumerate(cats)}
    cat_of = pois.poiCat.map(c2i).values.astype(int)

    # --- segments: maximal runs of consecutive in-pool visits.  A segment is cut
    #     (a) at an out-of-pool visit  -> never create a fake transition across a removed place
    #     (b) at a tie in startTime    -> the true order is unknown
    off = cfg.tz_offset_h * 3600
    rows, seg = [], 0
    for _, g in tr.groupby("trajID", sort=True):
        prev_ok, prev_start, pos = False, None, 0
        for r in g.itertuples(index=False):
            idx = id2idx.get(r.poiID, -1)
            if idx < 0:
                prev_ok = False
                continue
            if (not prev_ok) or r.startTime == prev_start:
                seg += 1
                pos = 0
            else:
                pos += 1
            prev_ok, prev_start = True, r.startTime
            rows.append((r.userID, r.trajID, seg, pos, idx, r.startTime, r.endTime, r.poiDuration, r.photos))
    v = pd.DataFrame(rows, columns=["user", "traj", "seg", "pos", "poi", "start", "end", "dur", "photos"])
    v["tmin"] = ((v.start + off) % 86400) // 60                  # local minute of day (arrival)
    v["hour_arr"] = v.tmin // 60
    v["hour_dep"] = ((v.end + off) % 86400) // 3600              # local hour when leaving
    v["dur_clean"] = v.dur.clip(upper=cfg.max_dur_s)
    v["nxt"] = v.groupby("seg").poi.shift(-1)

    t = v.dropna(subset=["nxt"]).copy()
    t["nxt"] = t.nxt.astype(int)
    trans = t[["user", "seg", "poi", "nxt", "hour_dep"]].rename(columns={"poi": "frm", "nxt": "to"}).reset_index(drop=True)

    g = v.groupby("seg")
    segs = pd.DataFrame({
        "user": g.user.first(), "n": g.size(),
        "pois": g.poi.apply(list),
        "t0_min": g.tmin.first(),
        "span_min": (g.end.max() - g.start.min()) / 60.0,
    }).reset_index()

    lat, lon = pois.poiLat.values, pois.poiLon.values
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return Dataset(cfg, pois, cats, cat_of, v, trans, segs, travel_minutes(D, cfg), D)


# ---------------------------------------------------------------------------
# user-level splits (a user never appears in two different splits)
# ---------------------------------------------------------------------------
def user_folds(ds: Dataset, k: int, seed: int) -> dict:
    users = np.array(sorted(ds.visits.user.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    return {u: i % k for i, u in enumerate(users)}


def fold_split(ds: Dataset, fold_of: dict, fold: int, val_frac: float, seed: int):
    """Return (train_users, val_users, test_users) as sets."""
    users = sorted(fold_of)
    test = {u for u in users if fold_of[u] == fold}
    rest = np.array([u for u in users if u not in test])
    rng = np.random.default_rng(seed + 1000 + fold)
    rng.shuffle(rest)
    n_val = int(round(val_frac * len(rest)))
    return set(rest[n_val:]), set(rest[:n_val]), test
