"""SI inputs (frozen data; no new modelling choices beyond the stated sensitivity rules).

S1 overlap between the acclimation and semicontinuous sheets (same reactor, same date).
S2 exact repeats of the previous value per channel (possible carried-forward entries).
S3 gas ledger: apparent methane yield (mL CH4 per g COD fed) per twin and phase under three
   readings of the biogas column: (A) volume since the previous reading, (B) daily rate times
   days since the previous reading, (C) as A with 700 mL readings treated as a cap and raised to
   1000 or 1500 mL. Cumulative COD fed and methane by day.
S4 time from each regime start to the first CH4 >= 30% or >= 40% reading (Kaplan-Meier, log-rank).
S5 pH carbonate soft sensor: pH ~ log10(CO2) and extensions, cross-twin transfer R2.
S7 twin-test sensitivity: (i) drop dates within 7 days after any reinoculation of either twin,
   (ii) drop dates within 7 days after a reinoculation that differed between the twins (date or
   volume) or a feeding mismatch.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from twinc2st import twin_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
DER = ROOT / "data_derived"
rng = np.random.default_rng(41)
tidy = pd.read_csv(DER / "tidy_long.csv", parse_dates=["date"])
D = pd.read_csv(DER / "features_e2.csv", parse_dates=["date"])
S = D[D.phase == "semicontinuous"].copy()
S["meth"] = S.biogas * S.ch4 / 100
T0 = S.date.min()

# ---------- S1 ----------
acc, semi = tidy[tidy.phase == "acclimation"], tidy[tidy.phase == "semicontinuous"]
ov = acc.merge(semi, on=["date", "reactor"], suffixes=("_acc", "_semi"))
cols = ["biogas", "ch4", "co2", "ph", "cond"]
ov[["date", "reactor"] + [f"{c}_{s}" for c in cols for s in ("acc", "semi")]].to_csv(OUT / "S1_sheet_overlap.csv", index=False)
cov = tidy.groupby(["phase", "reactor"]).date.agg(["min", "max", "count"]).reset_index()
cov.to_csv(OUT / "S1_sheet_coverage.csv", index=False)
print("S1 overlap rows", len(ov), "identical biogas/CH4:",
      int((np.isclose(ov.biogas_acc, ov.biogas_semi, equal_nan=True) & np.isclose(ov.ch4_acc, ov.ch4_semi, equal_nan=True)).sum()))

# ---------- S2 ----------
CH = ["biogas", "ch4", "co2", "h2s", "ph", "cond", "cod_s", "vfa", "acetic", "tc"]
rep = []
for r, g in D.sort_values("date").groupby("reactor"):
    for c in CH:
        v = g[["date", "day", "phase", c]].dropna()
        same = v[c].values[1:] == v[c].values[:-1]
        rep.append(dict(reactor=r, channel=c, n_values=len(v), n_repeat=int(same.sum())))
REP = pd.DataFrame(rep)
REP.to_csv(OUT / "S2_repeat_counts.csv", index=False)
ser = []
for r, g in D.sort_values("date").groupby("reactor"):
    v = g[["date", "t", "phase", "ch4"]].dropna().copy()
    v["repeat"] = np.r_[False, v.ch4.values[1:] == v.ch4.values[:-1]]
    v["reactor"] = r
    ser.append(v)
pd.concat(ser).to_csv(OUT / "S2_ch4_series_repeats.csv", index=False)
print(REP.pivot_table(index="channel", columns="reactor", values="n_repeat"))

# ---------- S3 ----------
M = pd.read_csv(DER / "series_merged.csv", parse_dates=["date"]).sort_values(["reactor", "date"])
M["cod_fed"] = M.v_feed.fillna(0) / 1000 * M.olr * M.hrt
rows, cum = [], []
for r, g in M.groupby("reactor"):
    g = g.copy()
    g["ch4i"] = g.ch4.interpolate(limit_area="inside").bfill().ffill()
    gb = g.dropna(subset=["biogas"]).copy()
    gb["gap"] = gb.t.diff().fillna(1).clip(lower=1)
    scen = {"A interval total": gb.biogas,
            "B daily rate x days": gb.biogas * gb.gap,
            "C cap 700 -> 1000": gb.biogas.where(gb.biogas != 700, 1000),
            "C cap 700 -> 1500": gb.biogas.where(gb.biogas != 700, 1500)}
    for ph, gp in g.groupby("phase"):
        cod = gp.cod_fed.sum()
        for name, vol in scen.items():
            sel = gb.phase == ph
            meth = (vol[sel] * gb.ch4i[sel] / 100).sum()
            rows.append(dict(reactor=r, phase=ph, scenario=name, cod_fed_g=cod, methane_ml=meth, yield_ml_per_g=meth / cod))
    g["meth_A"] = (g.biogas.fillna(0) * g.ch4i / 100)
    cum.append(pd.DataFrame(dict(reactor=r, t=g.t, date=g.date, cum_cod_g=g.cod_fed.cumsum(), cum_meth_ml=g.meth_A.cumsum())))
Y = pd.DataFrame(rows)
Y.to_csv(OUT / "S3_methane_yield_scenarios.csv", index=False)
pd.concat(cum).to_csv(OUT / "S3_cumulative_ledger.csv", index=False)
print(Y.pivot_table(index=["reactor", "phase"], columns="scenario", values="yield_ml_per_g").round(1))

# ---------- S4 ----------
def km(times, events):
    times, events = np.asarray(times, float), np.asarray(events, int)
    ts = np.unique(times[events == 1])
    s, out, at = 1.0, [(0.0, 1.0)], []
    for t in ts:
        n = (times >= t).sum(); d = ((times == t) & (events == 1)).sum()
        s *= 1 - d / n
        out.append((t, s))
    return out


def logrank(t1, e1, t2, e2):
    t = np.r_[t1, t2]; e = np.r_[e1, e2]; g = np.r_[np.zeros(len(t1)), np.ones(len(t2))]
    O = E = V = 0.0
    for u in np.unique(t[e == 1]):
        at = t >= u; n = at.sum(); n1 = (at & (g == 0)).sum()
        d = ((t == u) & (e == 1)).sum(); d1 = ((t == u) & (e == 1) & (g == 0)).sum()
        O += d1; E += d * n1 / n
        V += d * (n1 / n) * (1 - n1 / n) * (n - d) / max(n - 1, 1)
    chi = (O - E) ** 2 / V if V > 0 else 0
    return chi, stats.chi2.sf(chi, 1)


ep, curves, tests = [], [], []
for thr in [30, 40]:
    per = {}
    for r, g in D.sort_values("date").groupby("reactor"):
        g = g.reset_index(drop=True)
        key = g.bl_pct.astype(str) + g.co_type + g.olr.round(3).astype(str) + g.hrt.round(2).astype(str)
        blk = (key != key.shift()).cumsum()
        tt, ee = [], []
        for b, gb in g.groupby(blk):
            if gb.ch4.notna().sum() == 0:
                continue
            t0 = gb.t.min(); hit = gb[gb.ch4 >= thr]
            ev = len(hit) > 0
            tt.append((hit.t.min() if ev else gb.t.max()) - t0); ee.append(int(ev))
            ep.append(dict(threshold=thr, reactor=r, regime=int(b), phase=gb.phase.iloc[0], co_type=gb.co_type.iloc[0],
                           duration=gb.t.max() - t0, time=tt[-1], event=ee[-1]))
        per[r] = (np.array(tt), np.array(ee))
        for t, s in km(tt, ee):
            curves.append(dict(threshold=thr, reactor=r, time=t, survival=s))
    chi, p = logrank(*per["R1"], *per["R2"])
    tests.append(dict(threshold=thr, chi2=chi, p=p, n_R1=len(per["R1"][0]), n_R2=len(per["R2"][0]),
                      events_R1=int(per["R1"][1].sum()), events_R2=int(per["R2"][1].sum())))
pd.DataFrame(ep).to_csv(OUT / "S4_regime_episodes.csv", index=False)
pd.DataFrame(curves).to_csv(OUT / "S4_km_curves.csv", index=False)
pd.DataFrame(tests).to_csv(OUT / "S4_logrank.csv", index=False)
print(pd.DataFrame(tests))

# ---------- S5 ----------
x = D.dropna(subset=["ph", "co2", "cond"]).copy()
x = x[x.co2 > 0]
x["lco2"], x["lcond"] = np.log10(x.co2), np.log10(x.cond)
x[["reactor", "phase", "date", "ph", "co2", "cond", "lco2"]].to_csv(OUT / "S5_ph_co2_points.csv", index=False)
res = []
for fname, feats in {"log CO2": ["lco2"], "+ log conductivity": ["lco2", "lcond"],
                     "+ CH4": ["lco2", "lcond", "ch4"],
                     "+ schedule": ["lco2", "lcond", "ch4", "olr", "bl_pct", "co_code"]}.items():
    for lname, mk in [("Linear", LinearRegression), ("Extra trees", lambda: ExtraTreesRegressor(500, min_samples_leaf=3, random_state=0))]:
        ys, ps = [], []
        for r in ["R1", "R2"]:
            tr, te = x[x.reactor != r], x[x.reactor == r]
            m = mk().fit(tr[feats], tr.ph); ys += list(te.ph); ps += list(m.predict(te[feats]))
        res.append(dict(features=fname, learner=lname, n=len(ys), cross_twin_r2=r2_score(ys, ps)))
S5 = pd.DataFrame(res)
S5.to_csv(OUT / "S5_carbonate_transfer.csv", index=False)
rho5 = stats.spearmanr(x.ph, x.lco2)
pd.DataFrame([dict(n=len(x), rho=rho5.statistic, p=rho5.pvalue)]).to_csv(OUT / "S5_ph_co2_spearman.csv", index=False)
print(S5.round(3), rho5)

# ---------- S7 ----------
re = S[S.reinoc.notna()][["reactor", "day", "reinoc"]]
r1 = dict(zip(re[re.reactor == "R1"].day, re[re.reactor == "R1"].reinoc))
r2 = dict(zip(re[re.reactor == "R2"].day, re[re.reactor == "R2"].reinoc))
any_days = sorted(set(r1) | set(r2))
mismatch_days = sorted({d for d in any_days if r1.get(d) != r2.get(d)})
fv = S.pivot_table(index="day", columns="reactor", values="v_feed")
feed_mis = sorted(fv.index[(fv.R1.notna() & fv.R2.notna()) & (fv.R1 != fv.R2)])
pd.DataFrame(dict(day=any_days, reinoc_R1=[r1.get(d) for d in any_days], reinoc_R2=[r2.get(d) for d in any_days],
                  mismatched=[d in mismatch_days for d in any_days])).to_csv(OUT / "S7_interventions.csv", index=False)
fv.reset_index().to_csv(OUT / "S7_feed_volumes.csv", index=False)


def near(days, events, w=7):
    days = np.asarray(days)
    return np.any([(days >= e) & (days < e + w) for e in events], axis=0) if len(events) else np.zeros(len(days), bool)


rows = []
for c in ["biogas", "meth", "ch4", "co2", "h2s", "ph", "cond", "cod_s", "vfa", "acetic", "propionic", "tc", "tn"]:
    p = S.pivot_table(index="day", columns="reactor", values=c).dropna()
    for name, drop in [("All pairs", np.zeros(len(p), bool)),
                       ("Without post-reinoculation weeks", near(p.index, any_days)),
                       ("Without mismatched interventions", near(p.index, mismatch_days + feed_mis))]:
        q = p[~drop]
        if len(q) < 8:
            rows.append(dict(channel=c, variant=name, n_pairs=len(q), twin_auc=np.nan, p_value=np.nan)); continue
        d = (q.R2 - q.R1).values[:, None]
        wk = (q.index.values // 7).astype(int)
        auc, pv = twin_test(d, np.zeros((len(d), 1)), wk, False, rng, n_perm=499)
        rows.append(dict(channel=c, variant=name, n_pairs=len(q), twin_auc=auc, p_value=pv))
S7 = pd.DataFrame(rows)
S7.to_csv(OUT / "S7_sensitivity.csv", index=False)
print("mismatched intervention days", mismatch_days, "feed mismatch", feed_mis)
print(S7.pivot_table(index="channel", columns="variant", values="twin_auc").round(3))
print(S7.pivot_table(index="channel", columns="variant", values="n_pairs"))
