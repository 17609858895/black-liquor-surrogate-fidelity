"""E3 twin-reactor paired contrast (synthetic vs real BL) and E4 weekly-scale forecasting.

E3: same-date paired differences R2(real) - R1(synthetic) in the semicontinuous phase, where both
reactors followed the same nominal schedule (Cond. Operations sheet). Inference by date-level
sign-flip permutation and by moving-block bootstrap over dates (block = 7 days).
E4: weekly totals; forward-chaining by week; mechanistic COD baseline vs ML vs hybrid residual,
and cross-reactor transfer (train on one BL type, test on the other).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
rng = np.random.default_rng(0)
D = pd.read_csv(ROOT / "data_derived" / "features_e2.csv", parse_dates=["date"])
D["cod_fed"] = D.v_feed.fillna(0) / 1000 * D.olr * D.hrt
D["meth"] = D.biogas * D.ch4 / 100

# ---------------- E3 ----------------
semi = D[D.phase == "semicontinuous"]
rows = []
for v in ["ch4", "biogas", "meth", "ph", "cond", "cod_s", "vfa", "acetic", "propionic", "co2", "h2s"]:
    p = semi.pivot_table(index="date", columns="reactor", values=v).dropna()
    if len(p) < 6:
        continue
    diff = (p.R2 - p.R1).values
    obs = diff.mean()
    flips = np.array([(diff * rng.choice([-1, 1], len(diff))).mean() for _ in range(20000)])
    p_perm = (np.abs(flips) >= abs(obs)).mean()
    # moving-block bootstrap over date order
    nb = max(1, len(diff) // 4)
    boots = []
    for _ in range(5000):
        idx = np.concatenate([np.arange(s, min(s + 4, len(diff))) for s in rng.integers(0, len(diff), nb)])
        boots.append(diff[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    scale = np.nanstd(semi[v])
    rows.append(dict(variable=v, n_pairs=len(diff), mean_R1_synthetic=p.R1.mean(), mean_R2_real=p.R2.mean(),
                     diff_real_minus_syn=obs, ci_lo=lo, ci_hi=hi, p_signflip=p_perm, std_effect=obs / scale))
E3 = pd.DataFrame(rows)
E3.to_csv(OUT / "E3_twin_contrast.csv", index=False)
print(E3.round(3).to_string())

# ---------------- E4 weekly ----------------
D["week"] = D.t // 7
W = D.groupby(["reactor", "week"]).agg(
    biogas=("biogas", "sum"), n_read=("biogas", "count"), meth=("meth", "sum"), ch4=("ch4", "mean"),
    cod_fed=("cod_fed", "sum"), bl_pct=("bl_pct", "mean"), olr=("olr", "mean"), hrt=("hrt", "mean"),
    co=("co_code", "first"), ph=("ph", "mean"), phase=("phase", "first")).reset_index()
W = W[W.n_read > 0].copy()
W["is_real"] = (W.reactor == "R2").astype(int)
W["cod_co"] = W.cod_fed * (1 - W.bl_pct / 100)  # assumes BL % is a COD share (to verify)
W["cod_bl"] = W.cod_fed * W.bl_pct / 100
for c in ["biogas", "cod_fed", "cod_co", "cod_bl", "ph", "ch4"]:
    W[f"lag_{c}"] = W.groupby("reactor")[c].shift()
W["lag2_biogas"] = W.groupby("reactor").biogas.shift(2)
W = W.dropna(subset=["lag_biogas"]).reset_index(drop=True)
FE = ["cod_co", "cod_bl", "lag_cod_co", "lag_cod_bl", "olr", "hrt", "co", "is_real", "lag_biogas", "n_read"]
print("weekly rows", len(W), W.groupby("reactor").size().to_dict())


def fit_pred(train, test, kind):
    y = train.biogas
    if kind == "Mean":
        return np.full(len(test), y.mean())
    if kind == "Persistence":
        return test.lag_biogas.values
    if kind == "COD-linear":  # mechanistic-style: gas ~ substrate COD loads (non-negative yields)
        m = LinearRegression(positive=True).fit(train[["cod_co", "cod_bl", "lag_cod_co", "lag_cod_bl", "n_read"]], y)
        return m.predict(test[["cod_co", "cod_bl", "lag_cod_co", "lag_cod_bl", "n_read"]])
    if kind == "Ridge":
        m = RidgeCV(np.logspace(-2, 4, 30)).fit(train[FE].fillna(0), y)
        return m.predict(test[FE].fillna(0))
    if kind == "ET":
        m = ExtraTreesRegressor(500, min_samples_leaf=3, random_state=0).fit(train[FE].fillna(0), y)
        return m.predict(test[FE].fillna(0))
    if kind == "Hybrid":  # COD-linear backbone + ET on residuals
        cols = ["cod_co", "cod_bl", "lag_cod_co", "lag_cod_bl", "n_read"]
        m = LinearRegression(positive=True).fit(train[cols], y)
        r = y - m.predict(train[cols])
        e = ExtraTreesRegressor(500, min_samples_leaf=4, random_state=0).fit(train[FE].fillna(0), r)
        return m.predict(test[cols]) + 0.5 * e.predict(test[FE].fillna(0))


KINDS = ["Mean", "Persistence", "COD-linear", "Ridge", "ET", "Hybrid"]
res = []
# forward chaining: start predicting at week index 8 per reactor; train on all earlier weeks (both reactors)
pred = {k: np.full(len(W), np.nan) for k in KINDS}
for i in range(len(W)):
    wk = W.week[i]
    tr = W[W.week < wk]
    if tr.week.nunique() < 8:
        continue
    for k in KINDS:
        pred[k][i] = fit_pred(tr, W.iloc[[i]], k)[0]
m = ~np.isnan(pred["Mean"])
for k in KINDS:
    res.append(dict(scheme="forward_weekly", model=k, n=int(m.sum()), r2=r2_score(W.biogas[m], pred[k][m]),
                    mae=mean_absolute_error(W.biogas[m], pred[k][m])))
# cross-reactor transfer
for k in KINDS:
    ys, ps = [], []
    for r in [0, 1]:
        tr, te = W[W.is_real != r], W[W.is_real == r]
        ps.append(fit_pred(tr, te, k)); ys.append(te.biogas.values)
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    res.append(dict(scheme="cross_reactor_weekly", model=k, n=len(ys), r2=r2_score(ys, ps), mae=mean_absolute_error(ys, ps)))
R = pd.DataFrame(res)
R.to_csv(OUT / "E4_weekly_benchmark.csv", index=False)
print(R.round(3).to_string())
lin = LinearRegression(positive=True).fit(W[["cod_co", "cod_bl", "lag_cod_co", "lag_cod_bl", "n_read"]], W.biogas)
print("COD-linear coefficients (mL biogas per g COD):", dict(zip(["cod_co", "cod_bl", "lag_cod_co", "lag_cod_bl", "n_read"], lin.coef_.round(1))))
