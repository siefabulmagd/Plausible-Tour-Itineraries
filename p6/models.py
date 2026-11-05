"""Transition / duration models.

* MarkovCounts  - count-based first-order Markov baseline (hour-agnostic, additive smoothing)
* Popularity    - next place ~ global popularity (ignores the current place)
* TwoHeadNet    - neural network: shared trunk(place embedding + hour) -> {next-place head, log-duration head}

All transition models are exported as a 'cost table' nll[h, u, v] = -log P(next=v | current=u, hour=h)
with +inf on the diagonal (a tourist never revisits the place they are standing at; the data has no
repeated place inside any trajectory).  All models are therefore scored on the same support (N-1 places).
"""
import numpy as np

H = 24


# --------------------------------------------------------------------------- baselines
class MarkovCounts:
    def __init__(self, N, alpha=0.1):
        self.N, self.alpha = N, alpha
        self.counts = np.zeros((N, N))

    def fit(self, frm, to):
        self.counts = np.zeros((self.N, self.N))
        np.add.at(self.counts, (frm, to), 1)
        return self

    def nll_table(self):
        P = self.counts + self.alpha
        np.fill_diagonal(P, 0.0)
        P = P / P.sum(1, keepdims=True)
        with np.errstate(divide="ignore"):
            nll = -np.log(P)
        return np.broadcast_to(nll, (H, self.N, self.N)).copy()


class Popularity:
    def __init__(self, N, alpha=1.0):
        self.N, self.alpha = N, alpha

    def fit(self, visit_pois):
        self.pop = np.bincount(visit_pois, minlength=self.N).astype(float)
        return self

    def nll_table(self):
        P = np.tile(self.pop + self.alpha, (self.N, 1))
        np.fill_diagonal(P, 0.0)
        P = P / P.sum(1, keepdims=True)
        with np.errstate(divide="ignore"):
            nll = -np.log(P)
        return np.broadcast_to(nll, (H, self.N, self.N)).copy()


def tune_markov_alpha(N, tr, va, grid=(0.01, 0.03, 0.1, 0.3, 1.0)):
    best = None
    for a in grid:
        m = MarkovCounts(N, a).fit(tr.frm.values, tr.to.values)
        s = eval_next(m.nll_table(), va)["nll"]
        if best is None or s < best[0]:
            best = (s, a)
    return best[1]


def duration_baselines(tr_dur, N):
    """Global mean and shrunk per-place mean of log-duration (seconds) -> table [H,N] in minutes."""
    y = np.log(tr_dur.dur_clean.values)
    g = y.mean()
    per = np.full(N, g)
    for p in range(N):
        yy = y[tr_dur.poi.values == p]
        if len(yy):
            k = 5.0  # shrinkage strength
            per[p] = (yy.sum() + k * g) / (len(yy) + k)
    glob = np.full((H, N), np.exp(g) / 60)
    perp = np.tile(np.exp(per) / 60, (H, 1))
    return glob, perp


# --------------------------------------------------------------------------- metrics
def eval_next(nll, d):
    """d has columns frm, to, hour_dep.  nll is [H,N,N]."""
    rows = nll[d.hour_dep.values, d.frm.values]                  # (n, N)
    tgt = d.to.values
    n = len(tgt)
    ll = rows[np.arange(n), tgt]
    order = np.argsort(rows, axis=1)
    top1 = (order[:, 0] == tgt).mean()
    top3 = np.mean([t in o[:3] for t, o in zip(tgt, order)])
    return {"nll": float(ll.mean()), "top1": float(top1), "top3": float(top3), "n": n}


def eval_duration(dur_min, d):
    """dur_min is [H,N] (minutes); d has poi, hour_arr, dur_clean (>= min_dur only)."""
    pred = np.log(dur_min[d.hour_arr.values, d.poi.values] * 60)
    y = np.log(d.dur_clean.values)
    err = pred - y
    return {"mae_log": float(np.abs(err).mean()), "rmse_log": float(np.sqrt((err ** 2).mean())), "n": len(y),
            "medAE_min": float(np.median(np.abs(np.exp(pred) - np.exp(y))) / 60)}


# --------------------------------------------------------------------------- neural network
def hour_feats(h):
    ang = 2 * np.pi * np.asarray(h) / 24.0
    return np.stack([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)], 1).astype("float64")


class TwoHeadMLP:
    """Shared trunk + two heads, written in NumPy with manual back-propagation.

        x = [ place embedding | category embedding | hour features (sin/cos, 2 harmonics) ]
        h1 = dropout(relu(x W1 + b1));  h2 = relu(h1 W2 + b2)
        next head : logits = h2 Wn + bn  (the current place is masked: a tourist does not 'stay')
        dur head  : log-duration (standardised) = h2 Wd + bd
    (PyTorch could not be installed in the sandbox this was built in; the maths is identical.)
    """

    def __init__(self, N, C, cat_of, emb=16, hid=64, use_hour=True, seed=0, harm=2):
        rng = np.random.default_rng(seed)
        self.N, self.C, self.emb, self.use_hour, self.harm = N, C, emb, use_hour, harm
        self.cat_of = np.asarray(cat_of)
        D = emb + 4 + (2 * harm if use_hour else 0)
        self.p = {"Pe": rng.normal(0, .1, (N, emb)), "Ce": rng.normal(0, .1, (C, 4)),
                  "W1": rng.normal(0, np.sqrt(2 / D), (D, hid)), "b1": np.zeros(hid),
                  "W2": rng.normal(0, np.sqrt(2 / hid), (hid, hid)), "b2": np.zeros(hid),
                  "Wn": rng.normal(0, np.sqrt(1 / hid), (hid, N)), "bn": np.zeros(N),
                  "Wd": rng.normal(0, np.sqrt(1 / hid), (hid, 1)), "bd": np.zeros(1)}

    def forward(self, place, hf, drop=0.0, rng=None):
        p = self.p
        parts = [p["Pe"][place], p["Ce"][self.cat_of[place]]] + ([hf[:, :2 * self.harm]] if self.use_hour else [])
        x = np.concatenate(parts, 1)
        a1 = x @ p["W1"] + p["b1"]
        h1 = np.maximum(a1, 0)
        m = None
        if drop > 0:
            m = (rng.random(h1.shape) >= drop) / (1 - drop)
            h1 = h1 * m
        a2 = h1 @ p["W2"] + p["b2"]
        return np.maximum(a2, 0), (place, x, a1, h1, m, a2)

    def logits(self, h2, place):
        lg = h2 @ self.p["Wn"] + self.p["bn"]
        lg[np.arange(len(place)), place] = -np.inf
        return lg

    def dur(self, h2):
        return (h2 @ self.p["Wd"] + self.p["bd"])[:, 0]

    def backward(self, cache, dh2, g):
        place, x, a1, h1, m, a2 = cache
        p = self.p
        da2 = dh2 * (a2 > 0)
        g["W2"] += h1.T @ da2; g["b2"] += da2.sum(0)
        dh1 = da2 @ p["W2"].T
        if m is not None:
            dh1 = dh1 * m
        da1 = dh1 * (a1 > 0)
        g["W1"] += x.T @ da1; g["b1"] += da1.sum(0)
        dx = da1 @ p["W1"].T
        np.add.at(g["Pe"], place, dx[:, :self.emb])
        np.add.at(g["Ce"], self.cat_of[place], dx[:, self.emb:self.emb + 4])

    def loss_and_grads(self, nb, db, drop=0.0, rng=None):
        """nb=(place,hf,target) for the next-place head, db=(place,hf,y_std) for the duration head."""
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        place, hf, tgt = nb
        h2, c = self.forward(place, hf, drop, rng)
        lg = self.logits(h2, place)
        z = lg - lg.max(1, keepdims=True)
        e = np.exp(z)
        pr = e / e.sum(1, keepdims=True)
        n = len(tgt)
        l1 = -np.log(pr[np.arange(n), tgt] + 1e-300).mean()
        dl = pr.copy(); dl[np.arange(n), tgt] -= 1; dl /= n
        g["Wn"] += h2.T @ dl; g["bn"] += dl.sum(0)
        self.backward(c, dl @ self.p["Wn"].T, g)
        dplace, dhf, y = db
        h2, c = self.forward(dplace, dhf, drop, rng)
        err = self.dur(h2) - y
        l2 = float((err ** 2).mean())
        dp = (2 * err / len(y))[:, None]
        g["Wd"] += h2.T @ dp; g["bd"] += dp.sum(0)
        self.backward(c, dp @ self.p["Wd"].T, g)
        return l1 + l2, g


def train_two_head(N, C, cat_of, trans_tr, trans_va, dur_tr, dur_va, cfg, seed=0, use_hour=True, verbose=False):
    rng = np.random.default_rng(seed)
    net = TwoHeadMLP(N, C, cat_of, cfg.emb, cfg.hidden, use_hour, seed, getattr(cfg, 'hour_harm', 2))
    arr = lambda df, c: df[c].values.astype(int)
    tr_cur, tr_to, tr_hf = arr(trans_tr, "frm"), arr(trans_tr, "to"), hour_feats(trans_tr.hour_dep.values)
    va_cur, va_to, va_hf = arr(trans_va, "frm"), arr(trans_va, "to"), hour_feats(trans_va.hour_dep.values)
    ly = np.log(dur_tr.dur_clean.values)
    ymean, ystd = ly.mean(), ly.std()
    dp_, dhf_, dy_ = arr(dur_tr, "poi"), hour_feats(dur_tr.hour_arr.values), (ly - ymean) / ystd
    vp_, vhf_, vy_ = arr(dur_va, "poi"), hour_feats(dur_va.hour_arr.values), (np.log(dur_va.dur_clean.values) - ymean) / ystd

    mo = {k: np.zeros_like(v) for k, v in net.p.items()}
    ve = {k: np.zeros_like(v) for k, v in net.p.items()}
    step, best, best_p, bad, hist = 0, None, None, 0, []
    n_tr, n_d = len(tr_cur), len(dp_)
    decay = ("Pe", "Ce", "W1", "W2", "Wn", "Wd")
    for ep in range(cfg.max_epochs):
        perm, dperm = rng.permutation(n_tr), rng.permutation(n_d)
        tl, nb_ = 0.0, 0
        for b, i in enumerate(range(0, n_tr, cfg.batch)):
            idx = perm[i:i + cfg.batch]
            j = dperm[(b * cfg.batch) % n_d:(b * cfg.batch) % n_d + cfg.batch]
            if len(j) == 0:
                j = dperm[:cfg.batch]
            loss, g = net.loss_and_grads((tr_cur[idx], tr_hf[idx], tr_to[idx]), (dp_[j], dhf_[j], dy_[j]),
                                         cfg.dropout, rng)
            step += 1
            for k in net.p:                                   # Adam with L2 weight decay
                gk = g[k] + (cfg.weight_decay * net.p[k] if k in decay else 0)
                mo[k] = .9 * mo[k] + .1 * gk
                ve[k] = .999 * ve[k] + .001 * gk ** 2
                net.p[k] -= cfg.lr * (mo[k] / (1 - .9 ** step)) / (np.sqrt(ve[k] / (1 - .999 ** step)) + 1e-8)
            tl += loss; nb_ += 1
        h2, _ = net.forward(va_cur, va_hf)
        lg = net.logits(h2, va_cur)
        lg = lg - lg.max(1, keepdims=True)
        vn = float(-(lg[np.arange(len(va_to)), va_to] - np.log(np.exp(lg).sum(1))).mean())
        h2d, _ = net.forward(vp_, vhf_)
        vd = float(((net.dur(h2d) - vy_) ** 2).mean())
        hist.append({"epoch": ep, "train_loss": tl / nb_, "val_nll": vn, "val_dur_mse_std": vd})
        if best is None or vn + vd < best - 1e-4:
            best, bad = vn + vd, 0
            best_p = {k: v.copy() for k, v in net.p.items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    net.p = best_p

    place = np.tile(np.arange(N), H)
    hf = hour_feats(np.repeat(np.arange(H), N))
    h2, _ = net.forward(place, hf)
    lg = net.logits(h2, place)
    lg = lg - lg.max(1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
    nll = -logp.reshape(H, N, N)
    dur_min = np.clip(np.exp(net.dur(h2) * ystd + ymean).reshape(H, N) / 60.0, cfg.min_dur_s / 60.0, cfg.max_dur_s / 60.0)
    return {"nll": nll, "dur_min": dur_min, "history": hist,
            "best_epoch": int(np.argmin([h["val_nll"] + h["val_dur_mse_std"] for h in hist]))}
