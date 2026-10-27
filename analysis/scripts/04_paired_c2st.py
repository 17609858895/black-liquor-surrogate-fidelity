"""E5: paired, schedule-conditioned classifier two-sample test (twin C2ST) of surrogate fidelity.

For each date where both twins have the channel measured, form the within-date difference
d = x_real - x_syn. A sample is (s*d, label s) with random sign s in {+1,-1}; a classifier must
recover which twin is the real-BL reactor. Because both members of a pair share the date and
nominal schedule, the test is conditional on operating conditions by design.
Evaluation: leave-one-week-out (pairs in the same week never split); null distribution by
re-running the whole pipeline after random within-pair label swaps (sign-flip of d).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
D = pd.read_csv(ROOT / "data_derived" / "features_e2.csv", parse_dates=["date"])
D = D[D.phase == "semicontinuous"]
D["meth"] = D.biogas * D.ch4 / 100

GROUPS = {
    "gas_volume": ["biogas"],
    "gas_composition": ["ch4", "co2"],
    "methane": ["meth"],
    "H2S": ["h2s"],
    "pH": ["ph"],
    "conductivity": ["cond"],
    "soluble_COD": ["cod_s"],
    "VFA_total": ["vfa"],
    "acetate": ["acetic"],
    "propionate": ["propionic"],
    "TC_TN": ["tc", "tn"],
}


def pair_diffs(cols):
    piv = {c: D.pivot_table(index="date", columns="reactor", values=c) for c in cols}
    idx = None
    for c in cols:
        ok = piv[c].dropna().index
        idx = ok if idx is None else idx.intersection(ok)
    dd = np.column_stack([(piv[c].loc[idx, "R2"] - piv[c].loc[idx, "R1"]).values for c in cols])
    week = ((pd.Series(idx) - D.date.min()).dt.days // 7).values
    co = D.drop_duplicates("date").set_index("date").loc[idx, "co_type"].values
    return dd, week, co


def twin_auc(dd, week, rng):
    s = rng.choice([-1, 1], len(dd))
    X, y = dd * s[:, None], (s > 0).astype(int)
    scores = np.full(len(y), np.nan)
    for w in np.unique(week):
        te, tr = week == w, week != w
        if len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        # no intercept: a pure location shift must be learned from the sign of d
        m = LogisticRegression(fit_intercept=False, C=1.0).fit((X[tr]) / sd, y[tr])
        scores[te] = m.decision_function(X[te] / sd)
    ok = ~np.isnan(scores)
    return roc_auc_score(y[ok], scores[ok]) if len(np.unique(y[ok])) == 2 else np.nan


rng = np.random.default_rng(1)
rows = []
for g, cols in GROUPS.items():
    dd, week, co = pair_diffs(cols)
    if len(dd) < 8:
        continue
    obs = np.nanmean([twin_auc(dd, week, rng) for _ in range(50)])
    null = []
    for _ in range(300):
        flip = rng.choice([-1, 1], len(dd))[:, None]
        null.append(np.nanmean([twin_auc(dd * flip, week, rng) for _ in range(3)]))
    null = np.array(null)
    rows.append(dict(channel=g, n_pairs=len(dd), n_weeks=len(np.unique(week)), twin_auc=obs,
                     null_mean=null.mean(), p_value=((null >= obs).sum() + 1) / (len(null) + 1)))
    print(rows[-1])
R = pd.DataFrame(rows)
R.to_csv(OUT / "E5_twin_c2st.csv", index=False)
print(R.round(3).to_string())
