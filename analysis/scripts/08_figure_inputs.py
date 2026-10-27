"""E11 figure inputs from frozen data (no new modelling choices).

1. Single-channel fidelity map with the closed-form twin-C2ST (twinc2st.py): twin AUC, 999
   sign-flip permutations, null 2.5/97.5% band, p and BH q.
2. Standardised paired effects (real - synthetic) / pooled SD with moving-block bootstrap CI
   (block = 4 consecutive paired dates).
3. Synthetic-to-real transfer predictions (schedule-only ExtraTrees trained on R1, applied to R2)
   for the example channels shown in Fig. 6.
4. Spearman correlation between twin AUC and the E7 synthetic-to-real transfer bias.
5. Biogas readings versus days since the previous reading (does a reading scale with its interval?).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from statsmodels.stats.multitest import multipletests

from twinc2st import twin_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
D = pd.read_csv(ROOT / "data_derived" / "features_e2.csv", parse_dates=["date"])
S = D[D.phase == "semicontinuous"].copy()
S["meth"] = S.biogas * S.ch4 / 100
T0 = S.date.min()
rng = np.random.default_rng(31)
CH = ["biogas", "meth", "ch4", "co2", "h2s", "ph", "cond", "cod_s", "vfa", "acetic", "propionic", "tc", "tn"]
GROUP = {c: "gas" for c in ["biogas", "meth", "ch4", "co2", "h2s"]} | {c: "liquid" for c in CH[5:]}

rows = []
for c in CH:
    p = S.pivot_table(index="date", columns="reactor", values=c).dropna()
    d = (p.R2 - p.R1).values
    wk = ((p.index - T0).days // 7).values
    z = np.zeros((len(d), 1))
    auc, pv, null = twin_test(d[:, None], z, wk, False, rng, n_perm=999, return_null=True)
    sd = np.nanstd(S[c])
    nb = max(1, len(d) // 4)
    boots = [d[np.concatenate([np.arange(s, min(s + 4, len(d))) for s in rng.integers(0, len(d), nb)])].mean()
             for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append(dict(channel=c, group=GROUP[c], n_pairs=len(d), twin_auc=auc, null_lo=np.percentile(null, 2.5),
                     null_hi=np.percentile(null, 97.5), null_median=np.median(null), p_value=pv,
                     diff=d.mean(), diff_lo=lo, diff_hi=hi, std_diff=d.mean() / sd, std_lo=lo / sd, std_hi=hi / sd))
    print(rows[-1], flush=True)
F = pd.DataFrame(rows)
F["q_BH"] = multipletests(F.p_value, method="fdr_bh")[1]
F.to_csv(OUT / "E11_fidelity_map.csv", index=False)

SCHED = ["co_code", "olr", "hrt", "bl_pct", "days_in_regime", "t", "feed_last3"]
tr_rows = []
for c in ["cond", "tc", "acetic", "ph", "meth", "ch4"]:
    r1 = S[(S.reactor == "R1") & S[c].notna()]
    r2 = S[(S.reactor == "R2") & S[c].notna()]
    m = ExtraTreesRegressor(400, min_samples_leaf=3, random_state=0).fit(r1[SCHED], r1[c])
    tr_rows.append(pd.DataFrame(dict(channel=c, date=r2.date.values, observed_real=r2[c].values,
                                     predicted_from_synthetic=m.predict(r2[SCHED]))))
pd.concat(tr_rows).to_csv(OUT / "E11_transfer_predictions.csv", index=False)
print(F.round(3).to_string())

# 4. fidelity (twin AUC, this estimator) vs synthetic-to-real transfer bias (E7)
E7 = pd.read_csv(OUT / "E7_surrogate_transfer.csv")[["channel", "bias_transfer_std", "transfer_penalty"]]
J = F.merge(E7, on="channel")
J["abs_bias"] = J.bias_transfer_std.abs()
rho_b = stats.spearmanr(J.twin_auc, J.abs_bias)
rho_p = stats.spearmanr(J.twin_auc, J.transfer_penalty)
J.to_csv(OUT / "E11_fidelity_vs_transfer.csv", index=False)
pd.DataFrame([dict(pair="twin_auc~abs_bias", rho=rho_b.statistic, p=rho_b.pvalue, n=len(J)),
              dict(pair="twin_auc~mae_penalty", rho=rho_p.statistic, p=rho_p.pvalue, n=len(J))]).to_csv(
    OUT / "E11_fidelity_transfer_spearman.csv", index=False)
print(rho_b, rho_p)

# 5. biogas readings vs days since the previous biogas reading (same reactor), all dated rows
iv = []
for r, g in D.dropna(subset=["biogas"]).sort_values("date").groupby("reactor"):
    g = g.assign(days_since_prev_reading=g.t.diff())
    iv.append(g[["reactor", "phase", "date", "day", "biogas", "days_since_prev_reading"]])
IV = pd.concat(iv).dropna(subset=["days_since_prev_reading"])
IV.to_csv(OUT / "E11_biogas_intervals.csv", index=False)
rho_i = stats.spearmanr(IV.days_since_prev_reading, IV.biogas)
print("biogas ~ days since previous reading:", rho_i, len(IV))
pd.DataFrame([dict(pair="biogas~days_since_prev_reading", rho=rho_i.statistic, p=rho_i.pvalue, n=len(IV))]).to_csv(
    OUT / "E11_biogas_interval_spearman.csv", index=False)
