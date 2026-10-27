"""E13 time-resolved surrogate fidelity (post hoc, motivated by the drift seen in paired differences).

1. Conditional twin test with a single context, operating day (standardised), for every channel;
   999 sign-flip permutations; BH across channels. Unconditional twin test on the same pairs for
   comparison. The context was chosen after inspecting trends, so results are labelled post hoc.
2. Trend of the standardised paired difference against operating day: Spearman rho with a
   moving-block bootstrap CI over dates (block = 4 consecutive pairs).
3. Rolling paired difference (28-day window, 7-day step, >= 4 pairs) in SD units with t-based 95% CI.
4. Surrogate fidelity scorecard per channel, combining E11 (twin AUC, q), S7 (robust to removing
   post-reinoculation dates), item 1 (time-varying fidelity) and E7 (synthetic-to-real offset).
5. Standardised paired difference by co-substrate phase (descriptive matrix for Fig. 4).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from twinc2st import twin_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
D = pd.read_csv(ROOT / "data_derived" / "features_e2.csv", parse_dates=["date"])
S = D[D.phase == "semicontinuous"].copy()
S["meth"] = S.biogas * S.ch4 / 100
rng = np.random.default_rng(53)
CH = ["biogas", "meth", "ch4", "co2", "h2s", "ph", "cond", "cod_s", "vfa", "acetic", "propionic", "tc", "tn"]


def pairs(c):
    p = S.pivot_table(index="day", columns="reactor", values=c).dropna()
    d = ((p.R2 - p.R1) / np.nanstd(S[c])).values
    return p.index.values.astype(float), d


# 1 + 2
rows = []
for c in CH:
    day, d = pairs(c)
    wk = (day // 7).astype(int)
    z = ((day - day.mean()) / day.std())[:, None]
    a_u, p_u = twin_test(d[:, None], z, wk, False, rng, n_perm=999)
    a_c, p_c = twin_test(d[:, None], z, wk, True, rng, n_perm=999)
    rho = stats.spearmanr(day, d).statistic
    nb = max(1, len(d) // 4)
    boot = []
    for _ in range(3000):
        idx = np.concatenate([np.arange(s, min(s + 4, len(d))) for s in rng.integers(0, len(d), nb)])
        if len(np.unique(day[idx])) > 3:
            boot.append(stats.spearmanr(day[idx], d[idx]).statistic)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    rows.append(dict(channel=c, n_pairs=len(d), auc_uncond=a_u, p_uncond=p_u, auc_time=a_c, p_time=p_c,
                     rho_day=rho, rho_lo=lo, rho_hi=hi))
    print(rows[-1], flush=True)
T = pd.DataFrame(rows)
T["q_time_BH"] = multipletests(T.p_time, method="fdr_bh")[1]
T.to_csv(OUT / "E13_time_conditional.csv", index=False)
print(T.round(3).to_string())

# 3
roll = []
for c in ["ch4", "tc", "cod_s", "cond", "ph", "co2", "meth"]:
    day, d = pairs(c)
    for start in range(0, 126 - 27, 7):
        m = (day >= start) & (day < start + 28)
        if m.sum() >= 4:
            x = d[m]
            se = x.std(ddof=1) / np.sqrt(len(x))
            tq = stats.t.ppf(.975, len(x) - 1)
            roll.append(dict(channel=c, centre=start + 14, n=int(m.sum()), mean=x.mean(), lo=x.mean() - tq * se,
                             hi=x.mean() + tq * se))
pd.DataFrame(roll).to_csv(OUT / "E13_rolling_difference.csv", index=False)

# 4 scorecard
F = pd.read_csv(OUT / "E11_fidelity_map.csv").set_index("channel")
S7 = pd.read_csv(OUT / "S7_sensitivity.csv")
rob = S7[S7.variant == "Without post-reinoculation weeks"].set_index("channel")
E7 = pd.read_csv(OUT / "E7_surrogate_transfer.csv").set_index("channel")
T = T.set_index("channel")
card = pd.DataFrame(index=CH)
card["std_diff"] = F.std_diff
card["twin_auc"] = F.twin_auc
card["q"] = F.q_BH
card["auc_without_reinoc"] = rob.twin_auc
card["p_without_reinoc"] = rob.p_value
card["auc_time"] = T.auc_time
card["q_time"] = T.q_time_BH
card["rho_day"] = T.rho_day
card["transfer_bias_sd"] = E7.bias_transfer_std


def verdict(r):
    """Persistent difference: BH q < 0.05 and still P < 0.05 without post-reinoculation dates.
    Unresolved: some evidence of a difference (q < 0.10) that does not survive that removal.
    Drifts with time: no overall difference but time-conditional q < 0.05.
    No difference detected: none of the above (absence of evidence, not proof of equivalence)."""
    robust = r.p_without_reinoc < .05
    if r.q < .05 and robust:
        return "Persistent difference"
    if r.q < .10 and not robust:
        return "Unresolved"
    if r.q_time < .05:
        return "Drifts with time"
    return "No difference detected"


card["verdict"] = card.apply(verdict, axis=1)
card.reset_index(names="channel").to_csv(OUT / "E13_fidelity_scorecard.csv", index=False)
print(card.round(3).to_string())

# 5. standardised paired difference by co-substrate phase (descriptive; cells with < 3 pairs left empty)
ph_rows = []
for c in CH:
    p = S.pivot_table(index="day", columns="reactor", values=c).dropna()
    co = S[S.reactor == "R1"].drop_duplicates("day").set_index("day").loc[p.index, "co_type"]
    d = (p.R2 - p.R1) / np.nanstd(S[c])
    for k in ["Etanol", "Glicerol", "Suero"]:
        x = d[co.values == k]
        ph_rows.append(dict(channel=c, co_substrate=k, n=len(x), std_diff=x.mean() if len(x) >= 3 else np.nan))
    ph_rows.append(dict(channel=c, co_substrate="All", n=len(d), std_diff=d.mean()))
pd.DataFrame(ph_rows).to_csv(OUT / "E13_phase_matrix.csv", index=False)
