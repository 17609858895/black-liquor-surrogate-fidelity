"""E6 robustness of twin-C2ST and E7 surrogate-to-real transfer.

E6: BH q-values; alternative paired scorer (shrunken mean-direction rule); drop carry-forward
    rows; per co-substrate phase paired effects.
E7: per channel, a schedule-only model trained on the synthetic twin (R1) is applied to the real
    twin (R2); its error is compared with an in-domain leave-one-week-out model on R2.
E8: can gas/pH channels flag high-acetate days (acetate above the pooled median)?
E9: semi-synthetic null/power study. A pseudo-real twin = R1 trajectory + pair noise + delta*SD,
    sampled on R2's real measurement dates. Compare the paired twin-C2ST with common unpaired
    alternatives (random-CV C2ST on values, Welch t-test).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
D = pd.read_csv(ROOT / "data_derived" / "features_e2.csv", parse_dates=["date"])
S = D[D.phase == "semicontinuous"].copy()
S["meth"] = S.biogas * S.ch4 / 100
S["week"] = (S.date - S.date.min()).dt.days // 7
CH = ["biogas", "meth", "ch4", "co2", "h2s", "ph", "cond", "cod_s", "vfa", "acetic", "propionic", "tc", "tn"]
rng = np.random.default_rng(7)


def paired(col, frame=S):
    p = frame.pivot_table(index="date", columns="reactor", values=col).dropna()
    wk = ((p.index - S.date.min()).days // 7).values
    return p, wk


def twin_auc(d, wk, rng, scorer="logit", reps=30):
    out = []
    for _ in range(reps):
        s = rng.choice([-1, 1], len(d))
        X, y = (d * s)[:, None], (s > 0).astype(int)
        sc = np.full(len(y), np.nan)
        for w in np.unique(wk):
            te, tr = wk == w, wk != w
            sd = X[tr].std() + 1e-9
            if scorer == "logit":
                if len(np.unique(y[tr])) < 2:
                    continue
                m = LogisticRegression(fit_intercept=False).fit(X[tr] / sd, y[tr])
                sc[te] = m.decision_function(X[te] / sd)
            else:  # shrunken mean-direction rule: direction = mean(d_train) / (sd/sqrt(n)) shrunk toward 0
                dtr = d[tr]
                z = dtr.mean() / (dtr.std(ddof=1) / np.sqrt(len(dtr)) + 1e-9)
                wgt = np.sign(z) * max(abs(z) - 1.0, 0)
                sc[te] = wgt * X[te, 0] / sd
        ok = ~np.isnan(sc)
        if len(np.unique(y[ok])) == 2 and np.nanstd(sc[ok]) > 0:
            out.append(roc_auc_score(y[ok], sc[ok]))
        else:
            out.append(0.5)
    return float(np.mean(out))


def perm_p(d, wk, obs, scorer, n=200):
    null = [twin_auc(d * rng.choice([-1, 1], len(d)), wk, rng, scorer, reps=3) for _ in range(n)]
    return (np.sum(np.array(null) >= obs) + 1) / (n + 1)


# ---------- E6 ----------
carry = set()
for r, g in D.sort_values("date").groupby("reactor"):
    g = g.reset_index(drop=True)
    for i in range(1, len(g)):
        a, b = g.loc[i - 1, ["ch4", "co2", "ph", "cond"]], g.loc[i, ["ch4", "co2", "ph", "cond"]]
        both = a.notna() & b.notna()
        if both.sum() >= 2 and np.allclose(a[both].astype(float), b[both].astype(float)):
            carry.add(g.loc[i, "date"])
rows = []
for c in CH:
    p, wk = paired(c)
    d = (p.R2 - p.R1).values
    a1 = twin_auc(d, wk, rng, "logit"); a2 = twin_auc(d, wk, rng, "shrunk")
    keep = ~p.index.isin(list(carry))
    a3 = twin_auc(d[keep], wk[keep], rng, "logit") if keep.sum() >= 8 else np.nan
    rows.append(dict(channel=c, n=len(d), auc_logit=a1, p_logit=perm_p(d, wk, a1, "logit"),
                     auc_shrunk=a2, p_shrunk=perm_p(d, wk, a2, "shrunk"),
                     n_no_carry=int(keep.sum()), auc_no_carry=a3,
                     std_diff=d.mean() / np.nanstd(S[c])))
E6 = pd.DataFrame(rows)
from statsmodels.stats.multitest import multipletests
E6["q_logit_BH"] = multipletests(E6.p_logit, method="fdr_bh")[1]
E6["q_shrunk_BH"] = multipletests(E6.p_shrunk, method="fdr_bh")[1]
E6.to_csv(OUT / "E6_twin_c2st_robustness.csv", index=False)
print(E6.round(3).to_string())

ph_rows = []
for c in ["meth", "ch4", "biogas", "ph", "cond", "cod_s", "vfa", "acetic", "propionic"]:
    p, wk = paired(c)
    co = S.drop_duplicates("date").set_index("date").loc[p.index, "co_type"]
    for k in ["Etanol", "Glicerol", "Suero"]:
        dd = (p.R2 - p.R1)[co == k]
        if len(dd) >= 3:
            ph_rows.append(dict(channel=c, co_substrate=k, n=len(dd), diff=dd.mean(),
                                std_diff=dd.mean() / np.nanstd(S[c]),
                                p_signflip=(np.abs([(dd.values * rng.choice([-1, 1], len(dd))).mean() for _ in range(5000)]) >= abs(dd.mean())).mean()))
PH = pd.DataFrame(ph_rows)
PH.to_csv(OUT / "E6_phase_heterogeneity.csv", index=False)
print(PH.round(3).to_string())

# ---------- E7 ----------
SCHED = ["co_code", "olr", "hrt", "bl_pct", "days_in_regime", "t", "feed_last3"]
rows = []
for c in CH:
    r1 = S[(S.reactor == "R1") & S[c].notna()]
    r2 = S[(S.reactor == "R2") & S[c].notna()].reset_index(drop=True)
    if len(r1) < 10 or len(r2) < 10:
        continue
    m = ExtraTreesRegressor(400, min_samples_leaf=3, random_state=0).fit(r1[SCHED], r1[c])
    pt = m.predict(r2[SCHED])
    pin = np.full(len(r2), np.nan)
    for w in r2.week.unique():
        te = r2.week == w
        mi = ExtraTreesRegressor(400, min_samples_leaf=3, random_state=0).fit(r2.loc[~te, SCHED], r2.loc[~te, c])
        pin[te] = mi.predict(r2.loc[te, SCHED])
    rows.append(dict(channel=c, n_r2=len(r2), mae_transfer=mean_absolute_error(r2[c], pt),
                     mae_in_domain=mean_absolute_error(r2[c], pin),
                     bias_transfer_std=(r2[c] - pt).mean() / np.nanstd(S[c])))
E7 = pd.DataFrame(rows)
E7["transfer_penalty"] = E7.mae_transfer / E7.mae_in_domain
E7 = E7.merge(E6[["channel", "auc_logit", "auc_shrunk"]], on="channel")
E7.to_csv(OUT / "E7_surrogate_transfer.csv", index=False)
print(E7.round(3).to_string())
print("Spearman(twin AUC, transfer penalty):", stats.spearmanr(E7.auc_shrunk, E7.transfer_penalty))
print("Spearman(twin AUC, |transfer bias|):", stats.spearmanr(E7.auc_shrunk, E7.bias_transfer_std.abs()))
# E8 and E9 moved to 06_gas_silent_and_simulation.py (faster implementation).
