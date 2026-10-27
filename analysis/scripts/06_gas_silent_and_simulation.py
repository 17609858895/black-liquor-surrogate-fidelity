"""E8 gas-silent stress and E9 semi-synthetic null/power study of the twin-C2ST.

E8: can same-date gas (biogas, CH4, CO2) or pH/conductivity channels flag high-acetate days
    (acetate above the pooled median)? Leave-one-week-out logistic regression, both reactors.
E9: a pseudo-real twin = the synthetic twin's trajectory (interpolated in time) + pair noise +
    delta * SD, observed only on the real twin's actual measurement dates, so both twins inherit
    the schedule-driven nonstationarity and the real sampling pattern. Tests compared:
    twin-C2ST (paired, leave-one-week-out, sign-flip null), Wilcoxon signed-rank (paired),
    Mann-Whitney and Welch (unpaired, all available readings), and an unpaired random-5-fold
    kNN C2ST on readings (label-permutation null). `--resume` skips settings already saved.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
D = pd.read_csv(ROOT / "data_derived" / "features_e2.csv", parse_dates=["date"])
S = D[D.phase == "semicontinuous"].copy()
S["week"] = (S.date - S.date.min()).dt.days // 7
T0 = S.date.min()
rng = np.random.default_rng(11)

# ---------- E8 ----------
lab = S.dropna(subset=["acetic"]).copy()
lab["high_ac"] = (lab.acetic > lab.acetic.median()).astype(int)
rows = []
for name, feats in {"gas (biogas, CH4, CO2)": ["biogas", "ch4", "co2"], "CH4 + CO2": ["ch4", "co2"],
                    "pH": ["ph"], "pH + conductivity": ["ph", "cond"],
                    "gas + pH + conductivity": ["ch4", "co2", "ph", "cond"]}.items():
    z = lab.dropna(subset=feats).reset_index(drop=True)
    sc = np.full(len(z), np.nan)
    for w in z.week.unique():
        te = (z.week == w).values
        if z.high_ac[~te].nunique() < 2:
            continue
        mu, sd = z.loc[~te, feats].mean(), z.loc[~te, feats].std() + 1e-9
        m = LogisticRegression(C=1.0).fit((z.loc[~te, feats] - mu) / sd, z.high_ac[~te])
        sc[te] = m.decision_function((z.loc[te, feats] - mu) / sd)
    ok = ~np.isnan(sc)
    auc = roc_auc_score(z.high_ac[ok], sc[ok])
    null = []
    for _ in range(500):
        yb = rng.permutation(z.high_ac[ok].values)
        null.append(roc_auc_score(yb, sc[ok]))
    rows.append(dict(features=name, n=int(ok.sum()), auc=auc, p_perm=(np.sum(np.array(null) >= auc) + 1) / 501))
E8 = pd.DataFrame(rows)
E8.to_csv(OUT / "E8_gas_silent_detection.csv", index=False)
print(E8.round(3).to_string())


# ---------- E9 ----------
def twin_auc(d, wk, rng, reps=5):
    """Shrunken mean-direction scorer, leave-one-week-out, averaged over random sign draws."""
    out = []
    uw = np.unique(wk)
    for _ in range(reps):
        s = rng.choice([-1, 1], len(d))
        X, y = d * s, (s > 0).astype(int)
        sc = np.zeros(len(y))
        for w in uw:
            te = wk == w
            dtr = d[~te]
            z = dtr.mean() / (dtr.std(ddof=1) / np.sqrt(len(dtr)) + 1e-12)
            sc[te] = np.sign(z) * max(abs(z) - 1.0, 0) * X[te]
        out.append(roc_auc_score(y, sc) if np.std(sc) > 0 and 0 < y.mean() < 1 else 0.5)
    return float(np.mean(out))


def unpaired_c2st(y1, y2, rng, n_perm=40):
    X = np.r_[y1, y2][:, None]
    lab = np.r_[np.zeros(len(y1)), np.ones(len(y2))]

    def cv_auc(lb):
        pr = np.zeros(len(lb))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1e6))).split(X, lb):
            pr[te] = KNeighborsClassifier(7).fit(X[tr], lb[tr]).predict_proba(X[te])[:, 1]
        return roc_auc_score(lb, pr)

    obs = cv_auc(lab)
    null = [cv_auc(rng.permutation(lab)) for _ in range(n_perm)]
    return (np.sum(np.array(null) >= obs) + 1) / (n_perm + 1)


def sim_once(col, delta, rng):
    r1 = S[S.reactor == "R1"].set_index("date")[col].dropna()
    r2 = S[S.reactor == "R2"].set_index("date")[col].dropna()
    common0 = r1.index.intersection(r2.index)
    resid_sd = (r2[common0] - r1[common0]).std()
    sd = np.nanstd(S[col])
    t1 = (r1.index - T0).days.values
    t2 = (r2.index - T0).days.values
    y1 = r1.values + rng.normal(0, resid_sd / np.sqrt(2), len(t1))
    y2 = np.interp(t2, t1, r1.values) + rng.normal(0, resid_sd / np.sqrt(2), len(t2)) + delta * sd
    s1, s2 = pd.Series(y1, index=r1.index), pd.Series(y2, index=r2.index)
    common = s1.index.intersection(s2.index)
    d = (s2[common] - s1[common]).values
    wk = ((common - T0).days // 7).values
    a = twin_auc(d, wk, rng)
    null = [twin_auc(d * rng.choice([-1, 1], len(d)), wk, rng, reps=1) for _ in range(99)]
    p_twin = (np.sum(np.array(null) >= a) + 1) / 100
    return dict(p_twin=p_twin, p_wilcoxon=stats.wilcoxon(d).pvalue,
                p_mannwhitney=stats.mannwhitneyu(y1, y2).pvalue,
                p_welch=stats.ttest_ind(y1, y2, equal_var=False).pvalue,
                p_unpaired_c2st=unpaired_c2st(y1, y2, rng))


SETTINGS = [(c, dl) for c in ["ch4", "ph", "biogas", "acetic"] for dl in [0.0, 0.5, 1.0]]
done = set()
f9 = OUT / "E9_simulation_null_power.csv"
if "--resume" in sys.argv and f9.exists():
    prev = pd.read_csv(f9)
    done = {(r.channel, float(r.delta_sd)) for r in prev.itertuples()}
rows = [] if not done else prev.drop(columns=[c for c in ["note"] if c in prev]).to_dict("records")
for col, delta in SETTINGS:
    if (col, delta) in done:
        continue
    sims = pd.DataFrame([sim_once(col, delta, rng) for _ in range(200)])
    row = dict(channel=col, delta_sd=delta, n_sim=len(sims))
    row.update({f"reject_{k[2:]}": (sims[k] < 0.05).mean() for k in sims.columns})
    rows.append(row)
    print(row, flush=True)
    pd.DataFrame(rows).to_csv(f9, index=False)  # checkpoint after every setting
E9 = pd.DataFrame(rows)
E9.to_csv(OUT / "E9_simulation_null_power.csv", index=False)
print(E9.round(3).to_string())
