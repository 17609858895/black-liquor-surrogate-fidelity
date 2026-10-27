"""E2: what a conventional ML study would do, under three validation designs.

Targets: CH4 (%), biogas reading (mL), pH. Features use only information available at the
reading date: nominal operating set-points, feed actually added, phase clocks, and the most
recent PAST observation of each monitored variable (strictly earlier date).
Validation: (a) random 5-fold on rows, (b) forward-chaining time blocks within reactor,
(c) cross-reactor transfer (train synthetic BL reactor, test real BL reactor and vice versa).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
d = pd.read_csv(ROOT / "data_derived" / "series_merged.csv", parse_dates=["date"])
d = d.sort_values(["reactor", "date"]).reset_index(drop=True)

MON = ["biogas", "ch4", "co2", "ph", "cond", "cod_s", "vfa"]
CO = {"Etanol": 0, "Glicerol": 1, "Suero": 2}


def build(g):
    g = g.copy().reset_index(drop=True)
    g["co_code"] = g.co_type.map(CO)
    g["dt_prev"] = g.t.diff().fillna(0)
    # clock since last change in nominal recipe
    key = g.bl_pct.astype(str) + g.co_type + g.olr.round(3).astype(str) + g.hrt.round(2).astype(str)
    blk = (key != key.shift()).cumsum()
    g["days_in_regime"] = g.t - g.groupby(blk).t.transform("min")
    g["cum_cod"] = (g.v_feed.fillna(0) / 1000 * g.olr * g.hrt).cumsum().shift().fillna(0)
    g["feed_last3"] = [g.v_feed.fillna(0)[(g.t >= t - 3) & (g.t < t)].sum() for t in g.t]
    for v in MON:
        last, age = [], []
        for i in range(len(g)):
            past = g.loc[: i - 1, [v, "t"]].dropna()
            last.append(past[v].iloc[-1] if len(past) else np.nan)
            age.append(g.t[i] - past.t.iloc[-1] if len(past) else np.nan)
        g[f"lag_{v}"] = last
        g[f"age_{v}"] = age
    return g


D = pd.concat([build(g) for _, g in d.groupby("reactor")], ignore_index=True)
D["is_real"] = (D.bl_type == "real").astype(int)
FEAT = ["bl_pct", "co_code", "olr", "hrt", "v_feed", "dt_prev", "days_in_regime", "cum_cod", "feed_last3",
        "t", "is_real"] + [f"lag_{v}" for v in MON] + [f"age_{v}" for v in MON]
D.to_csv(ROOT / "data_derived" / "features_e2.csv", index=False, encoding="utf-8-sig")


def models():
    return {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(np.logspace(-2, 3, 20))),
        "RF": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(300, min_samples_leaf=3, random_state=0)),
        "ET": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(300, min_samples_leaf=3, random_state=0)),
        "LGBM": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=5, verbose=-1),
    }


def evaluate(df, target, scheme):
    df = df[df[target].notna()].reset_index(drop=True)
    preds = {m: np.full(len(df), np.nan) for m in list(models()) + ["Persistence", "TrainMean"]}
    splits = []
    if scheme == "random":
        splits = list(KFold(5, shuffle=True, random_state=0).split(df))
    elif scheme == "forward":
        # within each reactor: 5 contiguous time blocks; train on all data strictly before block start
        for k in range(1, 5):
            te = []
            for r, g in df.groupby("reactor"):
                q = np.quantile(g.t, [k / 5, (k + 1) / 5])
                te += list(g.index[(g.t >= q[0]) & ((g.t < q[1]) if k < 4 else (g.t <= q[1]))])
            te = np.array(te)
            t_cut = df.loc[te].groupby("reactor").t.min()
            tr = np.array([i for i in df.index if df.t[i] < t_cut[df.reactor[i]]])
            splits.append((tr, te))
    elif scheme == "cross_reactor":
        for r in ["R1", "R2"]:
            splits.append((np.where(df.reactor != r)[0], np.where(df.reactor == r)[0]))
    for tr, te in splits:
        X, y = df[FEAT], df[target]
        for name, m in models().items():
            m.fit(X.iloc[tr], y.iloc[tr])
            preds[name][te] = m.predict(X.iloc[te])
        lag = df[f"lag_{target}"].iloc[te].fillna(y.iloc[tr].mean())
        preds["Persistence"][te] = lag
        preds["TrainMean"][te] = y.iloc[tr].mean()
    mask = ~np.isnan(preds["Ridge"])
    rows = []
    for name, p in preds.items():
        rows.append(dict(target=target, scheme=scheme, model=name, n=int(mask.sum()),
                         r2=r2_score(df[target][mask], p[mask]), mae=mean_absolute_error(df[target][mask], p[mask])))
    return rows


rows = []
for target in ["ch4", "biogas", "ph"]:
    for scheme in ["random", "forward", "cross_reactor"]:
        rows += evaluate(D, target, scheme)
res = pd.DataFrame(rows)
res.to_csv(OUT / "E2_naive_benchmark.csv", index=False)
print(res.pivot_table(index=["target", "model"], columns="scheme", values="r2").round(3).to_string())
print(res.groupby(["target", "scheme"]).n.first())
