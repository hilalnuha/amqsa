# ============================================================================
# A-M-Q-S-A
# ============================================================================
RUN_CGK       = True   # 6 solvers x 20 runs on the CGK instance   (~6 min)
RUN_ABLATION  = True   # 20-seed mechanism ablation                (~4 min)
RUN_IMPACT    = True   # quench-only curves + credit shares        (~3 min)
RUN_BENCHMARK = True   # 8 functions x 5 algorithms x 20 runs      (~8 min)
RUN_LANDSCAPE = True   # 4 pairwise decision-variable landscapes   (~1 min)
SAVE_FIGS     = True   # save each figure as a 300-dpi PNG with the paper filename

def _savefig(fig, name):
    if SAVE_FIGS:
        fig.savefig(name, dpi=300, bbox_inches="tight")
        print(f"    saved {name}")

import time as _time
_T0 = _time.time()
def _tick(msg):
    print(f"[{_time.time()-_T0:7.1f}s] {msg}", flush=True)

"""
SDS-FDR
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RNG_DATA = np.random.default_rng(7)

REOPEN = 180.0          # minutes after 18:00 -> 21:00
CURFEW = 260.0          # 22:20 night curfew: throughput-critical recovery window
CP = 6_000.0            # passenger delay cost, IDR per passenger-minute
COMP_CANCEL = 300_000.0     # airline compensation per passenger (PM 89/2015 tier)
REV_LOSS_FRAC = 0.7         # fraction of fare revenue forfeited on cancellation
PAX_CANCEL_LOSS = 1_300_000.0  # passenger trip-disruption value per cancelled pax
CANCEL_EQUIV_MIN = 600.0    # equivalent delay charged to f3 per cancelled flight
                            # (next-morning re-accommodation); prevents gaming f3

# type: (capacity, wake H/M, engine jet/prop, Cf+Cm IDR/min, repositioning IDR, fare band)
FLEET = {
    "B738":  (189, "M", "jet",  25_000, 60e6,  "dom"),
    "B739":  (215, "M", "jet",  27_000, 60e6,  "dom"),
    "A320":  (180, "M", "jet",  24_000, 60e6,  "dom"),
    "A333":  (290, "H", "jet",  65_000, 200e6, "intl"),
    "B77W":  (393, "H", "jet",  75_000, 250e6, "intl"),
    "ATR72": (72,  "M", "prop", 10_000, 30e6,  "reg"),
}
FARE = {"dom": (1.0e6, 1.5e6), "intl": (2.6e6, 3.8e6), "reg": (0.7e6, 1.1e6)}

# CGK operators: (callsign prefix, airline, fleet distribution)
OPERATORS = [
    ("GIA", "Garuda",        {"B738": 0.5, "A333": 0.3, "B77W": 0.2}),
    ("LNI", "Lion Air",      {"B738": 0.55, "B739": 0.45}),
    ("CTV", "Citilink",      {"A320": 1.0}),
    ("BTK", "Batik Air",     {"A320": 0.6, "B738": 0.4}),
    ("SJV", "Super Air Jet", {"A320": 1.0}),
    ("WON", "Wings Air",     {"ATR72": 1.0}),
    ("AWQ", "AirAsia Indonesia", {"A320": 1.0}),
    ("INTL", "Foreign carrier",  {"B77W": 0.6, "A333": 0.4}),
]
OP_WEIGHTS = [0.16, 0.28, 0.14, 0.12, 0.10, 0.06, 0.06, 0.08]
PREFIX2OP = {p: (name, fleet) for p, name, fleet in OPERATORS}
SIDS = ["CKG1A", "DOLTA1", "PURWO1", "NOKTA1"]  # 4 divergent SID groups


def _flight_row(i, prefix, airline, ac, sched):
    cap, wake, eng, cfm, repos, band = FLEET[ac]
    lo, hi = FARE[band]
    return {
        "flight": f"{prefix}{RNG_DATA.integers(100, 999)}",
        "airline": airline,
        "aircraft": ac,
        "capacity": cap,
        "wake": wake,
        "engine": eng,
        "sid": SIDS[int(RNG_DATA.integers(0, len(SIDS)))],
        "sched_min_from_1800": sched,
        "cfm_idr_per_min": cfm,
        "reposition_idr": repos,
        "fare_idr": float(np.round(RNG_DATA.uniform(lo, hi), -3)),
    }


def load_or_build_dataset():
    """Real OpenSky extract if available, otherwise a seeded realistic bank."""
    for path in ("cgk_opensky_raw.csv", "/mnt/user-data/uploads/cgk_opensky_raw.csv"):
        if os.path.exists(path):
            raw = pd.read_csv(path)
            rows, source = [], f"OpenSky extract: {path}"
            base = raw["firstSeen_utc"].min()
            for i, r in raw.iterrows():
                cs = str(r["callsign"]).strip()
                prefix = cs[:3].upper()
                airline, fleet = PREFIX2OP.get(prefix, PREFIX2OP["INTL"])
                ac = RNG_DATA.choice(list(fleet), p=list(fleet.values()))
                sched = (r["firstSeen_utc"] - base) / 60.0 + 1.0
                row = _flight_row(i, prefix if prefix in PREFIX2OP else "INT",
                                  airline, ac, float(np.round(sched)))
                row["flight"] = cs
                rows.append(row)
            df = pd.DataFrame(rows)
            df = df[df["sched_min_from_1800"] < REOPEN].reset_index(drop=True)
            return df, source
    # fallback: seeded realistic CGK evening bank, 48 departures in 18:00-21:00
    n = 48
    sched = np.sort(RNG_DATA.choice(np.arange(1, 180), size=n, replace=False))
    rows = []
    for i in range(n):
        k = RNG_DATA.choice(len(OPERATORS), p=OP_WEIGHTS)
        prefix, airline, fleet = OPERATORS[k]
        ac = RNG_DATA.choice(list(fleet), p=list(fleet.values()))
        rows.append(_flight_row(i, prefix, airline, ac, float(sched[i])))
    return pd.DataFrame(rows), "seeded synthetic CGK evening bank (no OpenSky file found)"


FLIGHTS, DATA_SOURCE = load_or_build_dataset()
N = len(FLIGHTS)
SCHED = FLIGHTS["sched_min_from_1800"].to_numpy(float)
CAP = FLIGHTS["capacity"].to_numpy(float)
NP_SEATS = 0.75 * CAP                     # 75% load factor, as in the paper
CFM = FLIGHTS["cfm_idr_per_min"].to_numpy(float)
FARE_V = FLIGHTS["fare_idr"].to_numpy(float)
REPOS = FLIGHTS["reposition_idr"].to_numpy(float)
WAKE = FLIGHTS["wake"].to_numpy()
ENG = FLIGHTS["engine"].to_numpy()
SID = FLIGHTS["sid"].to_numpy()


# ----------------------------------------------------------------------
# Sequence-dependent separation (seconds) between consecutive departures
# ----------------------------------------------------------------------
def separation(prev, nxt):
    s = 120.0 if SID[prev] == SID[nxt] else 90.0          # SID divergence
    if WAKE[prev] == "H" and WAKE[nxt] == "M":
        s = max(s, 120.0)                                  # wake: M behind H
    if ENG[prev] == "prop" and ENG[nxt] == "jet" and SID[prev] == SID[nxt]:
        s = max(s, 150.0)                                  # speed class overtake
    return s / 60.0                                        # minutes


SEP_M = np.array([[separation(u, v) for v in range(N)] for u in range(N)])


def separation(prev, nxt):  # noqa: F811  (fast lookup)
    return SEP_M[prev, nxt]


def mu_disappointment(t):
    return np.clip(np.cbrt((t / 29.0) ** 2 / 29.0), 0.0, 1.0)


def decode(perm, cancel):
    """Return (f1, f2, f3, n_cancelled, delays[N] with -1 for cancelled)."""
    f1 = f2 = f3 = 0.0
    delays = np.full(N, -1.0)
    prev = -1
    tcur = REOPEN
    ncan = 0
    for fl in perm:
        if cancel[fl]:
            f1 += (COMP_CANCEL + REV_LOSS_FRAC * FARE_V[fl]) * NP_SEATS[fl] + REPOS[fl]
            f2 += PAX_CANCEL_LOSS * NP_SEATS[fl]
            f3 += CANCEL_EQUIV_MIN
            ncan += 1
            continue
        dep = tcur if prev < 0 else tcur + separation(prev, fl)
        if dep > CURFEW:                                  # curfew: forced cancel
            f1 += (COMP_CANCEL + REV_LOSS_FRAC * FARE_V[fl]) * NP_SEATS[fl] + REPOS[fl]
            f2 += PAX_CANCEL_LOSS * NP_SEATS[fl]
            f3 += CANCEL_EQUIV_MIN
            ncan += 1
            continue
        t = dep - SCHED[fl]
        delays[fl] = t
        f1 += t * CFM[fl] + mu_disappointment(t) * NP_SEATS[fl] * FARE_V[fl]
        f2 += NP_SEATS[fl] * CP * t
        f3 += t
        prev, tcur = fl, dep
    return f1, f2, f3, ncan, delays


# normalization refs from random solutions
_r = np.random.default_rng(0)
_samples = []
for _ in range(300):
    p = _r.permutation(N)
    c = _r.random(N) < 0.05
    _samples.append(decode(p, c)[:3])
F_REF = np.array(_samples).mean(axis=0)
W = np.array([1 / 3, 1 / 3, 1 / 3])


def fitness(perm, cancel):
    f1, f2, f3, ncan, _ = decode(perm, cancel)
    f = np.array([f1, f2, f3])
    return float(np.sum(W * f / F_REF)), f, ncan


# ----------------------------------------------------------------------
# Degeneracy demonstration: original fixed-grid model vs SDS-FDR
# ----------------------------------------------------------------------
def degeneracy_check(n_random=1000):
    rng = np.random.default_rng(1)
    grid = REOPEN + 2.0 * np.arange(N)     # original model: bijection to 2-min grid
    f3_grid, f3_sds = [], []
    for _ in range(n_random):
        p = rng.permutation(N)
        pos = np.empty(N, int)
        pos[p] = np.arange(N)
        f3_grid.append(float(np.sum(grid[pos] - SCHED)))
        f3_sds.append(decode(p, np.zeros(N, bool))[2])
    return np.array(f3_grid), np.array(f3_sds)


# ----------------------------------------------------------------------
# Solvers (budget: POP x ITERS evaluations)
# ----------------------------------------------------------------------
POP, ITERS = 50, 100
P_CANCEL_INIT = 0.05


def _init_perm_pop(rng):
    return ([rng.permutation(N) for _ in range(POP)],
            [rng.random(N) < P_CANCEL_INIT for _ in range(POP)])


def _ox(rng, p1, p2):
    a, b = sorted(rng.choice(N, 2, replace=False))
    child = -np.ones(N, int)
    child[a:b] = p1[a:b]
    fill = [g for g in p2 if g not in set(p1[a:b])]
    child[child < 0] = fill
    return child


def run_ga(seed):
    rng = np.random.default_rng(seed)
    pc, pm = 1.0, 0.2
    perms, cans = _init_perm_pop(rng)
    fits = [fitness(p, c) for p, c in zip(perms, cans)]
    bi = int(np.argmin([f[0] for f in fits]))
    best = (perms[bi].copy(), cans[bi].copy(), fits[bi])
    curve = []
    for _ in range(ITERS):
        np_perms, np_cans = [], []
        while len(np_perms) < POP:
            i1 = min(rng.choice(POP, 3, replace=False), key=lambda i: fits[i][0])
            i2 = min(rng.choice(POP, 3, replace=False), key=lambda i: fits[i][0])
            if rng.random() < pc:
                child = _ox(rng, perms[i1], perms[i2])
                mask = rng.random(N) < 0.5
                cbit = np.where(mask, cans[i1], cans[i2])
            else:
                child, cbit = perms[i1].copy(), cans[i1].copy()
            if rng.random() < pm:
                a, b = rng.choice(N, 2, replace=False)
                child[a], child[b] = child[b], child[a]
            if rng.random() < 0.1:
                k = rng.integers(0, N)
                cbit = cbit.copy()
                cbit[k] = ~cbit[k]
            np_perms.append(child)
            np_cans.append(cbit)
        perms, cans = np_perms, np_cans
        fits = [fitness(p, c) for p, c in zip(perms, cans)]
        gi = int(np.argmin([f[0] for f in fits]))
        if fits[gi][0] < best[2][0]:
            best = (perms[gi].copy(), cans[gi].copy(), fits[gi])
        wi = int(np.argmax([f[0] for f in fits]))
        perms[wi], cans[wi], fits[wi] = best[0].copy(), best[1].copy(), best[2]
        curve.append(best[2][1][0])
    return best, np.array(curve)


def run_sa(seed):
    rng = np.random.default_rng(seed)
    cur_p = rng.permutation(N)
    cur_c = rng.random(N) < P_CANCEL_INIT
    cur_f = fitness(cur_p, cur_c)
    best = (cur_p.copy(), cur_c.copy(), cur_f)
    T, alpha = 0.02, 0.999
    curve = []
    for _ in range(ITERS):
        for _ in range(POP):
            cp, cc = cur_p.copy(), cur_c.copy()
            u = rng.random()
            if u < 0.45:
                a, b = rng.choice(N, 2, replace=False)
                cp[a], cp[b] = cp[b], cp[a]
            elif u < 0.90:
                a, b = rng.choice(N, 2, replace=False)
                g = cp[a]
                cp = np.delete(cp, a)
                cp = np.insert(cp, b, g)
            else:
                k = rng.integers(0, N)
                cc[k] = ~cc[k]
            cf = fitness(cp, cc)
            d = cf[0] - cur_f[0]
            if d < 0 or rng.random() < np.exp(-d / max(T, 1e-12)):
                cur_p, cur_c, cur_f = cp, cc, cf
                if cf[0] < best[2][0]:
                    best = (cp.copy(), cc.copy(), cf)
            T *= alpha
        curve.append(best[2][1][0])
    return best, np.array(curve)


def _keys_decode(keys):
    return np.argsort(keys[:N], kind="stable"), keys[N:] > 0.9


def run_pso(seed):
    rng = np.random.default_rng(seed)
    D = 2 * N
    w0, w1, c1, c2 = 0.9, 0.4, 1.8, 1.8
    x = rng.random((POP, D))
    v = np.zeros_like(x)
    pb = x.copy()
    pf = [fitness(*_keys_decode(xi)) for xi in x]
    gi = int(np.argmin([f[0] for f in pf]))
    gb, gf = pb[gi].copy(), pf[gi]
    curve = []
    for it in range(ITERS):
        w = w0 - (w0 - w1) * it / ITERS
        r1, r2 = rng.random((POP, D)), rng.random((POP, D))
        v = np.clip(w * v + c1 * r1 * (pb - x) + c2 * r2 * (gb - x), -0.5, 0.5)
        x = np.clip(x + v, 0, 1)
        for i in range(POP):
            fi = fitness(*_keys_decode(x[i]))
            if fi[0] < pf[i][0]:
                pb[i], pf[i] = x[i].copy(), fi
                if fi[0] < gf[0]:
                    gb, gf = x[i].copy(), fi
        curve.append(gf[1][0])
    p, c = _keys_decode(gb)
    return (p, c, gf), np.array(curve)


def run_zoa(seed):
    """Original (continuous) ZOA on random keys -- ablation baseline."""
    rng = np.random.default_rng(seed)
    D = 2 * N
    x = rng.random((POP, D))
    fits = [fitness(*_keys_decode(xi)) for xi in x]
    gi = int(np.argmin([f[0] for f in fits]))
    pz, pzf = x[gi].copy(), fits[gi]
    curve = []
    for t in range(1, ITERS + 1):
        for i in range(POP):
            r = rng.random(D)
            I = 1 + rng.integers(0, 2)
            xn = np.clip(x[i] + r * (pz - I * x[i]), 0, 1)
            fn = fitness(*_keys_decode(xn))
            if fn[0] < fits[i][0]:
                x[i], fits[i] = xn, fn
            if rng.random() < 0.5:
                xn = np.clip(x[i] + 0.01 * (2 * rng.random(D) - 1)
                             * (1 - t / ITERS) * x[i], 0, 1)
            else:
                k = rng.integers(0, POP)
                r = rng.random(D)
                I = 1 + rng.integers(0, 2)
                xn = np.clip(x[i] + r * (x[k] - I * x[i]), 0, 1)
            fn = fitness(*_keys_decode(xn))
            if fn[0] < fits[i][0]:
                x[i], fits[i] = xn, fn
            if fits[i][0] < pzf[0]:
                pz, pzf = x[i].copy(), fits[i]
        curve.append(pzf[1][0])
    p, c = _keys_decode(pz)
    return (p, c, pzf), np.array(curve)


# ---------------------------- dZOA ------------------------------------
def _order_impose(rng, perm, guide, frac):
    """Move perm toward guide: a random subset of flights is reordered within
    its own positions according to the guide's relative order."""
    s = max(2, int(np.ceil(frac * N)))
    subset = rng.choice(N, s, replace=False)          # flight ids
    mask = np.isin(perm, subset)
    pos_g = np.empty(N, int)
    pos_g[guide] = np.arange(N)
    ordered = sorted(perm[mask], key=lambda f: pos_g[f])
    out = perm.copy()
    out[mask] = ordered
    return out


def run_dzoa(seed):
    """Discrete Zebra Optimization Algorithm: ZOA's two phases with arithmetic
    updates replaced by order-imposition (foraging/attack) and shrinking swap
    perturbation (escape) on the native (sequence, cancellation) encoding."""
    rng = np.random.default_rng(seed)
    perms, cans = _init_perm_pop(rng)
    fits = [fitness(p, c) for p, c in zip(perms, cans)]
    gi = int(np.argmin([f[0] for f in fits]))
    pz_p, pz_c, pz_f = perms[gi].copy(), cans[gi].copy(), fits[gi]
    curve = []
    for t in range(1, ITERS + 1):
        for i in range(POP):
            # Phase 1: foraging toward the pioneer zebra
            r = rng.random()
            I = 1 + rng.integers(0, 2)
            xp = _order_impose(rng, perms[i], pz_p, frac=0.25 * r * I)
            xc = perms_c = cans[i].copy()
            m = rng.random(N) < 0.5 * r
            xc = np.where(m, pz_c, xc)
            fn = fitness(xp, xc)
            if fn[0] < fits[i][0]:
                perms[i], cans[i], fits[i] = xp, xc, fn
            # Phase 2: defense
            if rng.random() < 0.5:                    # S1 escape: shrinking swaps
                k = max(1, int(round(4 * (1 - t / ITERS))))
                xp = perms[i].copy()
                for _ in range(k):
                    a, b = rng.choice(N, 2, replace=False)
                    xp[a], xp[b] = xp[b], xp[a]
                xc = cans[i].copy()
                if rng.random() < 0.2:
                    j = rng.integers(0, N)
                    xc[j] = ~xc[j]
            else:                                     # S2 attack: toward random zebra
                k = rng.integers(0, POP)
                r = rng.random()
                I = 1 + rng.integers(0, 2)
                xp = _order_impose(rng, perms[i], perms[k], frac=0.25 * r * I)
                xc = cans[i].copy()
                m = rng.random(N) < 0.5 * r
                xc = np.where(m, cans[k], xc)
            fn = fitness(xp, xc)
            if fn[0] < fits[i][0]:
                perms[i], cans[i], fits[i] = xp, xc, fn
            if fits[i][0] < pz_f[0]:
                pz_p, pz_c, pz_f = perms[i].copy(), cans[i].copy(), fits[i]
        curve.append(pz_f[1][0])
    return (pz_p, pz_c, pz_f), np.array(curve)


# ----------------------------------------------------------------------
ALGOS = {"GA": run_ga, "SA": run_sa, "PSO": run_pso, "ZOA": run_zoa, "dZOA": run_dzoa}
N_RUNS = 20



# ---------------------- AMQSA (proposed) -------------------------------
# marginal cost-per-minute weight of each flight (who should depart first)
WGT_PRI = NP_SEATS * CP + CFM + NP_SEATS * FARE_V / 300.0


def run_amqsa(seed, T0=0.02, alpha=0.999, lam=0.9, quench_frac=0.6):
    """Adaptive Multi-Neighborhood Quenched Simulated Annealing (AMQSA).
    Three mechanisms on top of SA:
      M1 adaptive multi-neighborhood: {swap, insertion, block relocation
         (or-opt, length 2-3), priority-guided swap, cancel flip} selected by
         exponential credit assignment;
      M2 priority-guided swap: of a random position pair, the flight with the
         larger marginal delay cost per minute is moved to the earlier slot
         (problem-knowledge operator);
      M3 elite-restart quench: at 60% of the budget the search restarts from
         the incumbent best and the temperature drops to zero, turning the
         endgame into pure multi-neighborhood descent.
    """
    rng = np.random.default_rng(seed)
    cur_p = rng.permutation(N)
    cur_c = rng.random(N) < P_CANCEL_INIT
    cur_f = fitness(cur_p, cur_c)
    best = (cur_p.copy(), cur_c.copy(), cur_f)
    T = T0
    wts = np.array([1.5, 1.5, 1.0, 1.0, 0.3])
    quenched = False
    curve = []
    for it in range(ITERS):
        if not quenched and it >= quench_frac * ITERS:
            T = 1e-12
            quenched = True
            cur_p, cur_c, cur_f = best[0].copy(), best[1].copy(), best[2]
        for _ in range(POP):
            pr = np.maximum(wts, 0.05)
            pr = pr / pr.sum()
            oi = rng.choice(5, p=pr)
            cp, cc = cur_p.copy(), cur_c.copy()
            if oi == 0:                        # swap
                a, b = rng.choice(N, 2, replace=False)
                cp[a], cp[b] = cp[b], cp[a]
            elif oi == 1:                      # insertion
                a, b = rng.choice(N, 2, replace=False)
                g = cp[a]
                cp = np.delete(cp, a)
                cp = np.insert(cp, b, g)
            elif oi == 2:                      # block relocation (or-opt)
                L = int(rng.integers(2, 4))
                a = int(rng.integers(0, N - L))
                blk = cp[a:a + L].copy()
                rest = np.concatenate([cp[:a], cp[a + L:]])
                b = int(rng.integers(0, len(rest) + 1))
                cp = np.concatenate([rest[:b], blk, rest[b:]])
            elif oi == 3:                      # priority-guided swap
                i, j = sorted(rng.choice(N, 2, replace=False))
                if WGT_PRI[cp[j]] <= WGT_PRI[cp[i]]:
                    i, j = j, i
                cp[i], cp[j] = cp[j], cp[i]
            else:                              # cancel flip
                k2 = rng.integers(0, N)
                cc[k2] = ~cc[k2]
            cf = fitness(cp, cc)
            d = cf[0] - cur_f[0]
            acc = d < 0 or rng.random() < np.exp(-d / max(T, 1e-12))
            wts[oi] = lam * wts[oi] + (1 - lam) * (1.0 if d < 0 else (0.3 if acc else 0.0))
            if acc:
                cur_p, cur_c, cur_f = cp, cc, cf
                if cf[0] < best[2][0]:
                    best = (cp.copy(), cc.copy(), cf)
            T *= alpha
        curve.append(best[2][1][0])
    return best, np.array(curve)


ALGOS["AMQSA"] = run_amqsa



# ======================= CONTINUOUS BENCHMARK CORE =======================
BD = 30
BEVALS = 10_000
BN_RUNS = 20
BPOP = 30


def sphere(x): return float(np.sum(x * x))
def rosenbrock(x): return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
def rastrigin(x): return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))
def ackley(x):
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x * x) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)
def griewank(x):
    return float(np.sum(x * x) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1)
def schwefel(x): return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))
def levy(x):
    w = 1 + (x - 1) / 4
    return float(np.sin(np.pi * w[0]) ** 2
                 + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
                 + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2))
def zakharov(x):
    s = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return float(np.sum(x * x) + s ** 2 + s ** 4)


FUNCS = {
    "F1 Sphere": (sphere, -100, 100),
    "F2 Rosenbrock": (rosenbrock, -30, 30),
    "F3 Rastrigin": (rastrigin, -5.12, 5.12),
    "F4 Ackley": (ackley, -32, 32),
    "F5 Griewank": (griewank, -600, 600),
    "F6 Schwefel": (schwefel, -500, 500),
    "F7 Levy": (levy, -10, 10),
    "F8 Zakharov": (zakharov, -10, 10),
}


def _t0_estimate(f, lo, hi, rng, n=30):
    x = rng.uniform(lo, hi, BD)
    fs = [f(np.clip(x + rng.normal(0, 0.1 * (hi - lo), BD), lo, hi)) for _ in range(n)]
    return max(1e-12, np.std(fs))


def run_sa_c(f, lo, hi, seed, track=None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, BD)
    fx = f(x)
    bx, bf = x.copy(), fx
    T = _t0_estimate(f, lo, hi, rng)
    alpha = (1e-8) ** (1.0 / BEVALS)
    for _ in range(BEVALS):
        xn = np.clip(x + rng.normal(0, 0.1 * (hi - lo), BD), lo, hi)
        fn = f(xn)
        d = fn - fx
        if d < 0 or rng.random() < np.exp(-d / max(T, 1e-300)):
            x, fx = xn, fn
            if fn < bf:
                bx, bf = xn.copy(), fn
        if track is not None:
            track.append(bx.copy())
        T *= alpha
    return bf


def run_amqsa_c(f, lo, hi, seed, quench_frac=0.6, lam=0.9, track=None):
    rng = np.random.default_rng(seed)
    R = hi - lo
    x = rng.uniform(lo, hi, BD)
    fx = f(x)
    bx, bf = x.copy(), fx
    T = _t0_estimate(f, lo, hi, rng)
    alpha = (1e-8) ** (1.0 / BEVALS)
    wts = np.ones(4)
    quenched = False
    for e in range(BEVALS):
        if not quenched and e >= quench_frac * BEVALS:
            T = 0.0
            quenched = True
            x, fx = bx.copy(), bf
        pr = np.maximum(wts, 0.05)
        pr = pr / pr.sum()
        oi = rng.choice(4, p=pr)
        if oi == 0:
            xn = x + rng.normal(0, 0.10 * R, BD)
        elif oi == 1:
            xn = x + rng.normal(0, 0.01 * R, BD)
        elif oi == 2:
            xn = x.copy()
            k = rng.integers(0, BD)
            xn[k] += rng.normal(0, 0.10 * R)
        else:
            xn = x + 0.05 * R * rng.standard_cauchy(BD)
        xn = np.clip(xn, lo, hi)
        fn = f(xn)
        d = fn - fx
        acc = d < 0 or (T > 0 and rng.random() < np.exp(-d / max(T, 1e-300)))
        wts[oi] = lam * wts[oi] + (1 - lam) * (1.0 if d < 0 else (0.3 if acc else 0.0))
        if acc:
            x, fx = xn, fn
            if fn < bf:
                bx, bf = xn.copy(), fn
        if track is not None:
            track.append(bx.copy())
        T *= alpha
    return bf


def run_ga_c(f, lo, hi, seed):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(lo, hi, (BPOP, BD))
    fits = np.array([f(p) for p in pop])
    best, bf = pop[fits.argmin()].copy(), fits.min()
    gens = BEVALS // BPOP
    for _ in range(gens - 1):
        new = []
        for _ in range(BPOP):
            i1 = min(rng.choice(BPOP, 3, replace=False), key=lambda i: fits[i])
            i2 = min(rng.choice(BPOP, 3, replace=False), key=lambda i: fits[i])
            beta = rng.uniform(-0.25, 1.25, BD)          # BLX-alpha crossover
            child = beta * pop[i1] + (1 - beta) * pop[i2]
            mmask = rng.random(BD) < 0.1
            child = child + mmask * rng.normal(0, 0.05 * (hi - lo), BD)
            new.append(np.clip(child, lo, hi))
        pop = np.array(new)
        fits = np.array([f(p) for p in pop])
        if fits.min() < bf:
            best, bf = pop[fits.argmin()].copy(), fits.min()
        wi = fits.argmax()
        pop[wi], fits[wi] = best, bf                    # elitism
    return bf


def run_pso_c(f, lo, hi, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, (BPOP, BD))
    v = np.zeros_like(x)
    fits = np.array([f(p) for p in x])
    pb, pf = x.copy(), fits.copy()
    gb, gf = x[fits.argmin()].copy(), fits.min()
    iters = BEVALS // BPOP
    for it in range(iters - 1):
        w = 0.9 - 0.5 * it / iters
        r1, r2 = rng.random((BPOP, BD)), rng.random((BPOP, BD))
        v = np.clip(w * v + 1.8 * r1 * (pb - x) + 1.8 * r2 * (gb - x),
                    -0.2 * (hi - lo), 0.2 * (hi - lo))
        x = np.clip(x + v, lo, hi)
        fits = np.array([f(p) for p in x])
        imp = fits < pf
        pb[imp], pf[imp] = x[imp], fits[imp]
        if pf.min() < gf:
            gb, gf = pb[pf.argmin()].copy(), pf.min()
    return gf


def run_zoa_c(f, lo, hi, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, (BPOP, BD))
    fits = np.array([f(p) for p in x])
    pz, pzf = x[fits.argmin()].copy(), fits.min()
    iters = BEVALS // (2 * BPOP)
    for t in range(1, iters + 1):
        for i in range(BPOP):
            r = rng.random(BD)
            I = 1 + rng.integers(0, 2)
            xn = np.clip(x[i] + r * (pz - I * x[i]), lo, hi)
            fn = f(xn)
            if fn < fits[i]:
                x[i], fits[i] = xn, fn
            if rng.random() < 0.5:
                xn = np.clip(x[i] + 0.01 * (2 * rng.random(BD) - 1)
                             * (1 - t / iters) * x[i], lo, hi)
            else:
                k = rng.integers(0, BPOP)
                r = rng.random(BD)
                I = 1 + rng.integers(0, 2)
                xn = np.clip(x[i] + r * (x[k] - I * x[i]), lo, hi)
            fn = f(xn)
            if fn < fits[i]:
                x[i], fits[i] = xn, fn
            if fits[i] < pzf:
                pz, pzf = x[i].copy(), fits[i]
    return pzf





# ---- curve-tracked runners ----
GRID = np.linspace(BEVALS // 100, BEVALS, 100).astype(int)


class Rec:
    def __init__(self):
        self.e = 0
        self.best = np.inf
        self.pts = []
        self.gi = 0

    def __call__(self, f):
        self.e += 1
        if f < self.best:
            self.best = f
        while self.gi < len(GRID) and self.e >= GRID[self.gi]:
            self.pts.append(self.best)
            self.gi += 1

    def curve(self):
        while len(self.pts) < len(GRID):
            self.pts.append(self.best)
        return np.array(self.pts)


def sa_c(f, lo, hi, seed, rec):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, BD); fx = f(x); rec(fx)
    T = _t0_estimate(f, lo, hi, rng)
    alpha = (1e-8) ** (1.0 / BEVALS)
    while rec.e < BEVALS:
        xn = np.clip(x + rng.normal(0, 0.1 * (hi - lo), BD), lo, hi)
        fn = f(xn); rec(fn)
        d = fn - fx
        if d < 0 or rng.random() < np.exp(-d / max(T, 1e-300)):
            x, fx = xn, fn
        T *= alpha
    return rec.best


def amqsa_c(f, lo, hi, seed, rec, quench_frac=0.6, lam=0.9):
    rng = np.random.default_rng(seed)
    R = hi - lo
    x = rng.uniform(lo, hi, BD); fx = f(x); rec(fx)
    bx, bf = x.copy(), fx
    T = _t0_estimate(f, lo, hi, rng)
    alpha = (1e-8) ** (1.0 / BEVALS)
    wts = np.ones(4); quenched = False
    while rec.e < BEVALS:
        if not quenched and rec.e >= quench_frac * BEVALS:
            T = 0.0; quenched = True; x, fx = bx.copy(), bf
        pr = np.maximum(wts, 0.05); pr = pr / pr.sum()
        oi = rng.choice(4, p=pr)
        if oi == 0:
            xn = x + rng.normal(0, 0.10 * R, BD)
        elif oi == 1:
            xn = x + rng.normal(0, 0.01 * R, BD)
        elif oi == 2:
            xn = x.copy(); k = rng.integers(0, BD)
            xn[k] += rng.normal(0, 0.10 * R)
        else:
            xn = x + 0.05 * R * rng.standard_cauchy(BD)
        xn = np.clip(xn, lo, hi)
        fn = f(xn); rec(fn)
        d = fn - fx
        acc = d < 0 or (T > 0 and rng.random() < np.exp(-d / max(T, 1e-300)))
        wts[oi] = lam * wts[oi] + (1 - lam) * (1.0 if d < 0 else (0.3 if acc else 0.0))
        if acc:
            x, fx = xn, fn
            if fn < bf:
                bx, bf = xn.copy(), fn
    return rec.best


def ga_c(f, lo, hi, seed, rec):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(lo, hi, (BPOP, BD))
    fits = np.array([f(p) for p in pop])
    for v in fits: rec(v)
    best, bf = pop[fits.argmin()].copy(), fits.min()
    while rec.e < BEVALS:
        new = []
        for _ in range(BPOP):
            i1 = min(rng.choice(BPOP, 3, replace=False), key=lambda i: fits[i])
            i2 = min(rng.choice(BPOP, 3, replace=False), key=lambda i: fits[i])
            beta = rng.uniform(-0.25, 1.25, BD)
            child = beta * pop[i1] + (1 - beta) * pop[i2]
            mmask = rng.random(BD) < 0.1
            child = child + mmask * rng.normal(0, 0.05 * (hi - lo), BD)
            new.append(np.clip(child, lo, hi))
        pop = np.array(new)
        fits = np.array([f(p) for p in pop])
        for v in fits: rec(v)
        if fits.min() < bf:
            best, bf = pop[fits.argmin()].copy(), fits.min()
        wi = fits.argmax(); pop[wi], fits[wi] = best, bf
    return rec.best


def pso_c(f, lo, hi, seed, rec):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, (BPOP, BD)); v = np.zeros_like(x)
    fits = np.array([f(p) for p in x])
    for q in fits: rec(q)
    pb, pf = x.copy(), fits.copy()
    gb, gf = x[fits.argmin()].copy(), fits.min()
    iters = BEVALS // BPOP
    it = 0
    while rec.e < BEVALS:
        w = 0.9 - 0.5 * it / iters; it += 1
        r1, r2 = rng.random((BPOP, BD)), rng.random((BPOP, BD))
        v = np.clip(w * v + 1.8 * r1 * (pb - x) + 1.8 * r2 * (gb - x),
                    -0.2 * (hi - lo), 0.2 * (hi - lo))
        x = np.clip(x + v, lo, hi)
        fits = np.array([f(p) for p in x])
        for q in fits: rec(q)
        imp = fits < pf
        pb[imp], pf[imp] = x[imp], fits[imp]
        if pf.min() < gf:
            gb, gf = pb[pf.argmin()].copy(), pf.min()
    return rec.best


def zoa_c(f, lo, hi, seed, rec):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, (BPOP, BD))
    fits = np.array([f(p) for p in x])
    for q in fits: rec(q)
    pz, pzf = x[fits.argmin()].copy(), fits.min()
    iters = BEVALS // (2 * BPOP)
    t = 0
    while rec.e < BEVALS:
        t += 1
        for i in range(BPOP):
            r = rng.random(BD); I = 1 + rng.integers(0, 2)
            xn = np.clip(x[i] + r * (pz - I * x[i]), lo, hi)
            fn = f(xn); rec(fn)
            if fn < fits[i]:
                x[i], fits[i] = xn, fn
            if rng.random() < 0.5:
                xn = np.clip(x[i] + 0.01 * (2 * rng.random(BD) - 1)
                             * (1 - t / iters) * x[i], lo, hi)
            else:
                k = rng.integers(0, BPOP)
                r = rng.random(BD); I = 1 + rng.integers(0, 2)
                xn = np.clip(x[i] + r * (x[k] - I * x[i]), lo, hi)
            fn = f(xn); rec(fn)
            if fn < fits[i]:
                x[i], fits[i] = xn, fn
            if fits[i] < pzf:
                pz, pzf = x[i].copy(), fits[i]
            if rec.e >= BEVALS:
                break
    return rec.best


BENCH_ALGOS = {"SA": sa_c, "GA": ga_c, "PSO": pso_c, "ZOA": zoa_c, "AMQSA": amqsa_c}




# ============================ EXPERIMENT DRIVERS ============================
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata

WGT = WGT_PRI  # alias

EXPECTED = {  # paper values for verification (means over the fixed seeds)
    "cgk_f1": {"GA": 9.7876, "SA": 9.7911, "PSO": 10.0428, "ZOA": 9.9269,
               "dZOA": 9.7931, "AMQSA": 9.7291},   # Rp 1e9 (Colab/NumPy 2.0.2)
    "degeneracy": {"grid_std": 0.0, "sds_std": 326.4,
                   "sds_min": 5853, "sds_max": 7410},
    "ablation_f1": {"SA+reheat": 10.3175, "M3": 9.7433,
                    "M3+oropt": 9.7455, "M3+pri": 9.7477},
    "bench_examples": {"F6 AMQSA": 2218.5, "F3 ZOA": 0.0},
    "friedman_chi2": 404.9, "mae_amqsa": 416.7,
}


def exp_degeneracy():
    _tick("degeneracy check (1,000 random schedules)")
    g3, s3 = degeneracy_check()
    print(f"  fixed-grid model : f3 std = {g3.std():.6f} (expected 0.000000)")
    print(f"  SDS-FDR          : f3 std = {s3.std():.1f}  range {s3.min():.0f}-"
          f"{s3.max():.0f} (expected 326.4, 5853-7410)")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(s3, bins=40, color="tab:red", alpha=0.65, label="SDS-FDR")
    ax.axvline(g3[0], color="tab:blue", lw=2.5, ls="--",
               label=f"fixed grid: f3 = {g3[0]:,.0f} (constant)")
    ax.set_xlabel("total delay time f3 / min"); ax.set_ylabel("frequency")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("f3 degeneracy")
    plt.tight_layout(); _savefig(fig, "fig_sds_degeneracy.png"); plt.show()
    return g3, s3


def exp_cgk():
    _tick("CGK comparison: 6 solvers x 20 runs (seeds 2000-2019)")
    results, curves, bests = {}, {}, {}
    for name, fn in ALGOS.items():
        rs, cs, bb = [], [], None
        for r in range(N_RUNS):
            (p, c, (scal, f, ncan)), curve = fn(seed=2000 + r)
            rs.append(f); cs.append(curve)
            if bb is None or scal < bb[2][0]:
                bb = (p, c, (scal, f, ncan))
        results[name] = np.array(rs); curves[name] = np.vstack(cs); bests[name] = bb
        exp = EXPECTED["cgk_f1"][name]
        got = results[name][:, 0].mean() / 1e9
        flag = "OK" if abs(got - exp) < 5e-4 else "MISMATCH"
        _tick(f"  {name:5s} f1={got:.4f}e9 (expected {exp:.4f})  [{flag}]")
    print("\nWilcoxon on f1, AMQSA vs others:")
    for n in ["GA", "SA", "PSO", "ZOA", "dZOA"]:
        _, p = wilcoxon(results["AMQSA"][:, 0], results[n][:, 0])
        print(f"  AMQSA vs {n:5s}: p = {p:.3g}")
    summary = pd.DataFrame([{
        "Algorithm": n, "f1 (Rp 1e9)": results[n][:, 0].mean() / 1e9,
        "f1 std (1e7)": results[n][:, 0].std() / 1e7,
        "f2 (Rp 1e9)": results[n][:, 1].mean() / 1e9,
        "f3 (min)": results[n][:, 2].mean()} for n in ALGOS])
    print("\n", summary.round(4).to_string(index=False))
    # convergence + per-flight delay figures
    colors = {"GA": "tab:blue", "SA": "tab:green", "PSO": "tab:orange",
              "ZOA": "tab:purple", "dZOA": "tab:brown", "AMQSA": "tab:red"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    it = np.arange(1, ITERS + 1)
    for n in ALGOS:
        C = curves[n]
        ax.plot(it, C.mean(0) / 1e9, color=colors[n], lw=1.5, label=n)
        ax.fill_between(it, (C.mean(0) - C.std(0)) / 1e9,
                        (C.mean(0) + C.std(0)) / 1e9, color=colors[n], alpha=0.12)
    ax.set_xlabel("iterations"); ax.set_ylabel("airline delay cost f1 / 1e9 IDR")
    ax.legend(ncol=3); ax.grid(alpha=0.3); ax.set_title("CGK convergence")
    plt.tight_layout(); _savefig(fig, "fig_sds_convergence.png"); plt.show()
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    for n in ALGOS:
        d = decode(bests[n][0], bests[n][1])[4]
        ax.plot(np.arange(1, N + 1), np.where(d >= 0, d, np.nan),
                color=colors[n], lw=1.0, marker=".", ms=4, alpha=0.85, label=n)
    ax.set_xlabel("flight number"); ax.set_ylabel("delay / min")
    ax.legend(ncol=6, fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("delay of the 48 flights, best schedules")
    plt.tight_layout(); _savefig(fig, "fig_sds_flight_delays.png"); plt.show()
    return results, curves, bests


def _ablation_variant(seed, ops, w0, T0=0.02, alpha=0.999, lam=0.9,
                      quench=0.6, reheat=False, stag_limit=400, rf=25.0):
    rng = np.random.default_rng(seed)
    cur_p = rng.permutation(N); cur_c = rng.random(N) < P_CANCEL_INIT
    cur_f = fitness(cur_p, cur_c); best = (cur_p.copy(), cur_c.copy(), cur_f)
    T = T0; wts = np.array(w0, float); quenched = False; stag = 0
    for it in range(ITERS):
        if not reheat and not quenched and it >= quench * ITERS:
            T = 1e-12; quenched = True
            cur_p, cur_c, cur_f = best[0].copy(), best[1].copy(), best[2]
        for _ in range(POP):
            pr = np.maximum(wts, 0.05); pr = pr / pr.sum()
            oi = rng.choice(len(ops), p=pr); op = ops[oi]
            cp, cc = cur_p.copy(), cur_c.copy()
            if op == 'swap':
                a, b = rng.choice(N, 2, replace=False); cp[a], cp[b] = cp[b], cp[a]
            elif op == 'ins':
                a, b = rng.choice(N, 2, replace=False)
                g = cp[a]; cp = np.delete(cp, a); cp = np.insert(cp, b, g)
            elif op == 'blk':
                L = int(rng.integers(2, 4)); a = int(rng.integers(0, N - L))
                blk = cp[a:a + L].copy(); rest = np.concatenate([cp[:a], cp[a + L:]])
                b = int(rng.integers(0, len(rest) + 1))
                cp = np.concatenate([rest[:b], blk, rest[b:]])
            elif op == 'pri':
                i, j = sorted(rng.choice(N, 2, replace=False))
                if WGT[cp[j]] <= WGT[cp[i]]: i, j = j, i
                cp[i], cp[j] = cp[j], cp[i]
            else:
                k = rng.integers(0, N); cc[k] = ~cc[k]
            cf = fitness(cp, cc); d = cf[0] - cur_f[0]
            acc = d < 0 or rng.random() < np.exp(-d / max(T, 1e-12))
            wts[oi] = lam * wts[oi] + (1 - lam) * (1.0 if d < 0 else (0.3 if acc else 0.0))
            if acc:
                cur_p, cur_c, cur_f = cp, cc, cf
                if cf[0] < best[2][0]: best = (cp.copy(), cc.copy(), cf); stag = 0
                else: stag += 1
            else: stag += 1
            if reheat and stag >= stag_limit:
                T = min(T0, T * rf)
                cur_p, cur_c, cur_f = best[0].copy(), best[1].copy(), best[2]; stag = 0
            T *= alpha
    return best[2][1][0]


def _quench_only(seed, T0=0.02, alpha=0.999, quench_frac=0.6):
    rng = np.random.default_rng(seed)
    cur_p = rng.permutation(N); cur_c = rng.random(N) < P_CANCEL_INIT
    cur_f = fitness(cur_p, cur_c); best = (cur_p.copy(), cur_c.copy(), cur_f)
    T = T0; quenched = False; curve = []
    for it in range(ITERS):
        if not quenched and it >= quench_frac * ITERS:
            T = 1e-12; quenched = True
            cur_p, cur_c, cur_f = best[0].copy(), best[1].copy(), best[2]
        for _ in range(POP):
            cp, cc = cur_p.copy(), cur_c.copy()
            u = rng.random()
            if u < 0.45:
                a, b = rng.choice(N, 2, replace=False); cp[a], cp[b] = cp[b], cp[a]
            elif u < 0.90:
                a, b = rng.choice(N, 2, replace=False)
                g = cp[a]; cp = np.delete(cp, a); cp = np.insert(cp, b, g)
            else:
                k = rng.integers(0, N); cc[k] = ~cc[k]
            cf = fitness(cp, cc); d = cf[0] - cur_f[0]
            if d < 0 or rng.random() < np.exp(-d / max(T, 1e-12)):
                cur_p, cur_c, cur_f = cp, cc, cf
                if cf[0] < best[2][0]: best = (cp.copy(), cc.copy(), cf)
            T *= alpha
        curve.append(best[2][1][0])
    return np.array(curve)


def exp_ablation():
    _tick("mechanism ablation, 20 seeds per configuration")
    cfg = {"SA+reheat": dict(ops=('swap', 'ins', 'bit'), w0=(1.5, 1.5, 0.3), reheat=True),
           "M3+oropt": dict(ops=('swap', 'ins', 'blk', 'bit'), w0=(1.5, 1.5, 1.0, 0.3)),
           "M3+pri": dict(ops=('swap', 'ins', 'pri', 'bit'), w0=(1.5, 1.5, 1.0, 0.3))}
    out = {}
    for k, kw in cfg.items():
        out[k] = np.array([_ablation_variant(2000 + r, **kw) for r in range(20)])
        exp = EXPECTED["ablation_f1"][k]; got = out[k].mean() / 1e9
        flag = "OK" if abs(got - exp) < 5e-4 else "MISMATCH"
        _tick(f"  {k:10s} f1={got:.4f}e9 (expected {exp:.4f}) [{flag}]")
    q = np.vstack([_quench_only(2000 + r) for r in range(20)])
    got = q[:, -1].mean() / 1e9; exp = EXPECTED["ablation_f1"]["M3"]
    _tick(f"  M3 (quench) f1={got:.4f}e9 (expected {exp:.4f}) "
          f"[{'OK' if abs(got-exp)<5e-4 else 'MISMATCH'}]")
    return out, q


def _amqsa_credits(seed):
    rng = np.random.default_rng(seed)
    cur_p = rng.permutation(N); cur_c = rng.random(N) < P_CANCEL_INIT
    cur_f = fitness(cur_p, cur_c); best = (cur_p.copy(), cur_c.copy(), cur_f)
    T = 0.02; wts = np.array([1.5, 1.5, 1.0, 1.0, 0.3]); quenched = False
    lam = 0.9; shares = []
    for it in range(ITERS):
        if not quenched and it >= 0.6 * ITERS:
            T = 1e-12; quenched = True
            cur_p, cur_c, cur_f = best[0].copy(), best[1].copy(), best[2]
        for _ in range(POP):
            pr = np.maximum(wts, 0.05); pr = pr / pr.sum()
            oi = rng.choice(5, p=pr)
            cp, cc = cur_p.copy(), cur_c.copy()
            if oi == 0:
                a, b = rng.choice(N, 2, replace=False); cp[a], cp[b] = cp[b], cp[a]
            elif oi == 1:
                a, b = rng.choice(N, 2, replace=False)
                g = cp[a]; cp = np.delete(cp, a); cp = np.insert(cp, b, g)
            elif oi == 2:
                L = int(rng.integers(2, 4)); a = int(rng.integers(0, N - L))
                blk = cp[a:a + L].copy(); rest = np.concatenate([cp[:a], cp[a + L:]])
                b = int(rng.integers(0, len(rest) + 1))
                cp = np.concatenate([rest[:b], blk, rest[b:]])
            elif oi == 3:
                i, j = sorted(rng.choice(N, 2, replace=False))
                if WGT[cp[j]] <= WGT[cp[i]]: i, j = j, i
                cp[i], cp[j] = cp[j], cp[i]
            else:
                k = rng.integers(0, N); cc[k] = ~cc[k]
            cf = fitness(cp, cc); d = cf[0] - cur_f[0]
            acc = d < 0 or rng.random() < np.exp(-d / max(T, 1e-12))
            wts[oi] = lam * wts[oi] + (1 - lam) * (1.0 if d < 0 else (0.3 if acc else 0.0))
            if acc:
                cur_p, cur_c, cur_f = cp, cc, cf
                if cf[0] < best[2][0]: best = (cp.copy(), cc.copy(), cf)
            T *= 0.999
        pr = np.maximum(wts, 0.05); shares.append(pr / pr.sum())
    return np.array(shares)


def exp_impact(cgk_curves, quench_curves):
    _tick("impact analysis: credit shares (20 runs)")
    shares = np.stack([_amqsa_credits(2000 + r) for r in range(20)]).mean(axis=0)
    it = np.arange(1, ITERS + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for name, C, col in [("SA", cgk_curves["SA"], "tab:green"),
                         ("SA + M3 (quench)", quench_curves, "tab:olive"),
                         ("AMQSA (full)", cgk_curves["AMQSA"], "tab:red")]:
        a1.plot(it, C.mean(0) / 1e9, color=col, lw=1.6, label=name)
    a1.axvline(60, color="k", ls=":", lw=1); a1.legend(fontsize=9)
    a1.set_xlabel("iterations"); a1.set_ylabel("f1 / 1e9 IDR"); a1.grid(alpha=0.3)
    a1.set_title("(a) mechanism contribution")
    a2.stackplot(it, shares.T, labels=["swap", "insertion", "or-opt",
                 "priority (M2)", "cancel flip"], alpha=0.85)
    a2.axvline(60, color="k", ls=":", lw=1); a2.legend(fontsize=7)
    a2.set_xlabel("iterations"); a2.set_ylabel("selection share"); a2.set_ylim(0, 1)
    a2.set_title("(b) credit-based operator shares")
    plt.tight_layout(); _savefig(fig, "fig_mechanism.png"); plt.show()
    return shares


def exp_benchmark():
    _tick("standard benchmark: 8 functions x 5 algorithms x 20 runs (seeds 3000-3019)")
    finals, curves = {}, {}
    for fname, (f, lo, hi) in FUNCS.items():
        finals[fname], curves[fname] = {}, {}
        for aname, algo in BENCH_ALGOS.items():
            fs, cs = [], []
            for r in range(BN_RUNS):
                rec = Rec(); algo(f, lo, hi, 3000 + r, rec)
                fs.append(rec.best); cs.append(rec.curve())
            finals[fname][aname] = np.array(fs)
            curves[fname][aname] = np.vstack(cs)
        _tick(f"  {fname}: " + "  ".join(
            f"{a}={finals[fname][a].mean():.3e}" for a in BENCH_ALGOS))
    # verification spot checks
    got = finals["F6 Schwefel"]["AMQSA"].mean()
    print(f"  check F6 AMQSA mean = {got:.1f} (expected {EXPECTED['bench_examples']['F6 AMQSA']})")
    print(f"  check F3 ZOA mean   = {finals['F3 Rastrigin']['ZOA'].mean():.1f} (expected 0.0)")
    # stats: Friedman, MAE, OE
    ALG = list(BENCH_ALGOS)
    blocks = np.hstack([np.stack([finals[f][a] for a in ALG]) for f in FUNCS])
    chi2, pfr = friedmanchisquare(*[blocks[i] for i in range(len(ALG))])
    ranks = {a: np.mean([rankdata(np.stack([finals[f][x] for x in ALG]), axis=0)[i].mean()
                         for f in FUNCS]) for i, a in enumerate(ALG)}
    mae = {a: np.mean([finals[f][a].mean() for f in FUNCS]) for a in ALG}
    wtl = {a: [0, 0, 0] for a in ALG}
    for f in FUNCS:
        means = {a: finals[f][a].mean() for a in ALG}
        b = min(means, key=means.get)
        for a in ALG:
            wtl[a][0 if a == b else 2] += 1
    print(f"\n  Friedman chi2={chi2:.1f} (expected ~{EXPECTED['friedman_chi2']}), p={pfr:.2g}")
    print("  Alg    avg-rank   MAE        w/t/l   OE%")
    for a in ALG:
        oe = (len(FUNCS) - wtl[a][2]) / len(FUNCS) * 100
        print(f"  {a:6s} {ranks[a]:.2f}      {mae[a]:.3e}  "
              f"{wtl[a][0]}/{wtl[a][1]}/{wtl[a][2]}   {oe:.1f}")
    # convergence figure
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
    for ax, f in zip(axes.ravel(), FUNCS):
        for a in ALG:
            ax.plot(GRID, np.maximum(curves[f][a].mean(0), 1e-30), lw=1.2, label=a)
        ax.set_yscale("log"); ax.set_title(f, fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout(); _savefig(fig, "fig_bench_convergence.png"); plt.show()
    return finals, curves


def exp_landscapes(best_amqsa):
    _tick("pairwise decision-variable landscapes (4 pairs x 48x48 grid)")
    BASE, CANCEL = best_amqsa[0].astype(int), best_amqsa[1].astype(bool)

    def landscape(u, v):
        rest = np.array([f for f in BASE if f not in (u, v)])
        Z = np.full((N, N), np.nan)
        for a in range(N):
            for b in range(N):
                if a == b: continue
                seq = np.empty(N, int); mask = np.ones(N, bool)
                seq[a], seq[b] = u, v; mask[[a, b]] = False; seq[mask] = rest
                Z[b, a] = decode(seq, CANCEL)[0]
        return Z / 1e9

    F = FLIGHTS
    wide = int(F.index[F["aircraft"].isin(["B77W", "A333"])][0])
    atr = int(F.index[F["aircraft"] == "ATR72"][0])
    med = F.index[F["aircraft"].isin(["A320", "B738", "B739"])]
    same = None
    for s in F["sid"].unique():
        ms = [i for i in med if F.loc[i, "sid"] == s]
        if len(ms) >= 2: same = (int(ms[0]), int(ms[1])); break
    pairs = [(wide, atr, "P1 H vs turboprop"),
             (same[0], same[1], "P2 same-SID jets"),
             (int(np.argmax(WGT)), int(np.argmin(WGT)), "P3 max vs min priority"),
             (int(np.argmin(SCHED)), int(np.argmax(SCHED)), "P4 earliest vs latest")]
    pos = {fl: k for k, fl in enumerate(BASE)}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6))
    for ax, (u, v, t) in zip(axes.ravel(), pairs):
        Z = landscape(u, v)
        cs = ax.contourf(np.arange(N), np.arange(N), Z, levels=24, cmap="viridis")
        ax.scatter(pos[u], pos[v], marker="*", s=140, color="red",
                   edgecolor="w", zorder=5)
        ax.set_title(f"{t} (spread {np.nanmax(Z)-np.nanmin(Z):.3f}e9)", fontsize=9)
        fig.colorbar(cs, ax=ax, shrink=0.85)
    plt.tight_layout(); _savefig(fig, "fig_sds_pair_contours.png"); plt.show()




def exp_sequence(best_amqsa):
    """Recovered runway sequence timeline of the best AMQSA schedule
    (paper Figure: fig_sds_sequence.png)."""
    _tick("recovered runway sequence timeline (best AMQSA)")
    perm, canc = best_amqsa[0].astype(int), best_amqsa[1].astype(bool)
    _, _, _, _, delays = decode(perm, canc)
    dep = SCHED + delays                      # minutes after 18:00; -1+s if cancelled
    sid_list = sorted(FLIGHTS["sid"].unique())
    sid_col = dict(zip(sid_list, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    k = 0
    for fl in perm:                           # runway order
        if delays[fl] < 0:
            continue                          # cancelled: not on the runway
        mk = ("s" if FLIGHTS.loc[fl, "wake"] == "H"
              else ("^" if FLIGHTS.loc[fl, "engine"] == "prop" else "o"))
        ax.scatter(dep[fl], k, color=sid_col[FLIGHTS.loc[fl, "sid"]], marker=mk,
                   s=46, edgecolor="k", linewidth=0.4, zorder=3)
        k += 1
    ax.axvline(REOPEN, color="k", ls=":", lw=1.2)
    ax.axvline(CURFEW, color="k", ls="--", lw=1.2)
    ax.text(REOPEN + 1, 1, "reopen 21:00", fontsize=8, rotation=90)
    ax.text(CURFEW - 5, 1, "curfew 22:20", fontsize=8, rotation=90)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=s)
               for s, c in sid_col.items()]
    handles += [plt.Line2D([], [], color="gray", marker="s", ls="", label="Heavy"),
                plt.Line2D([], [], color="gray", marker="^", ls="", label="Turboprop"),
                plt.Line2D([], [], color="gray", marker="o", ls="", label="Medium jet")]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="lower right")
    ax.set_xlabel("Minutes after 18:00 WIB")
    ax.set_ylabel("Runway departure sequence position")
    ax.set_title("Recovered CGK departure bank (best AMQSA schedule)")
    ax.grid(alpha=0.3)
    plt.tight_layout(); _savefig(fig, "fig_sds_sequence.png"); plt.show()


# ================================= MAIN =====================================
if __name__ == "__main__":
    print(f"numpy {np.__version__} | instance: {N} flights | {DATA_SOURCE}")
    exp_degeneracy()
    cgk_results = cgk_curves = bests = None
    if RUN_CGK:
        cgk_results, cgk_curves, bests = exp_cgk()
    q = None
    if RUN_ABLATION:
        _, q = exp_ablation()
    if RUN_IMPACT and cgk_curves is not None:
        if q is None:
            q = np.vstack([_quench_only(2000 + r) for r in range(20)])
        exp_impact(cgk_curves, q)
    if RUN_BENCHMARK:
        exp_benchmark()
    if RUN_LANDSCAPE and bests is not None:
        exp_landscapes(bests["AMQSA"])
    if bests is not None:
        exp_sequence(bests["AMQSA"])
    _tick("ALL EXPERIMENTS COMPLETE")
    if SAVE_FIGS:
        print("Figure PNGs are in the Colab file panel (folder icon) - "
              "download and drop them into the latex folder, then recompile.")
