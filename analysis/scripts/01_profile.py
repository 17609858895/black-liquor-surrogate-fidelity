"""Profile the observation process: overlap, carry-forward repeats, gas-meter cap, COD balance."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

d = pd.read_csv(ROOT / "data_derived" / "tidy_long.csv", parse_dates=["date"])

# 1. acclimation / semicontinuous overlap (same date, same reactor)
acc = d[d.phase == "acclimation"]
semi = d[d.phase == "semicontinuous"]
ov = acc.merge(semi, on=["date", "reactor"], suffixes=("_a", "_s"))
same = (np.isclose(ov.biogas_a, ov.biogas_s, equal_nan=True) & np.isclose(ov.ch4_a, ov.ch4_s, equal_nan=True))
print(f"overlap rows: {len(ov)}, identical biogas+CH4: {same.sum()}")

# merged continuous series: acclimation before first semicontinuous date + semicontinuous
t0 = semi.date.min()
full = pd.concat([acc[acc.date < t0], semi]).sort_values(["reactor", "date"]).reset_index(drop=True)
full["t"] = (full.date - full.date.min()).dt.days
full.to_csv(ROOT / "data_derived" / "series_merged.csv", index=False, encoding="utf-8-sig")
print(full.groupby("reactor").size())

# 2. carry-forward repeats: consecutive rows with identical gas-composition / pH / conductivity
lab = ["ch4", "co2", "ph", "cond"]
rep_rows = []
for r, g in full.groupby("reactor"):
    g = g.reset_index(drop=True)
    for i in range(1, len(g)):
        prev, cur = g.loc[i - 1, lab], g.loc[i, lab]
        both = prev.notna() & cur.notna()
        if both.sum() >= 2 and np.allclose(prev[both].astype(float), cur[both].astype(float)):
            rep_rows.append((r, g.loc[i, "date"].date(), int(both.sum())))
print("rows whose >=2 lab values exactly repeat the previous row:", len(rep_rows))
for x in rep_rows:
    print("  ", x)
ch4_rep = sum(1 for r, g in full.groupby("reactor") for a, b in zip(g.ch4.dropna().values[:-1], g.ch4.dropna().values[1:]) if a == b)
print("consecutive identical CH4 readings:", ch4_rep, "of", full.ch4.notna().sum())

# 3. gas readings: heaping and cap
bg = full.biogas.dropna()
print("biogas readings:", len(bg), " zeros:", (bg == 0).sum(), " >=700:", (bg >= 700).sum(),
      " multiples of 50:", (bg % 50 == 0).mean().round(3))
print(full.groupby("phase").biogas.apply(lambda s: s.value_counts().head(5).to_dict()))

# 4. COD balance: COD fed per feeding event = v_feed * (OLR*HRT) g/L; methane = biogas*CH4%
full["cod_feed_conc"] = full.olr * full.hrt  # g COD / L of feed
full["cod_fed_g"] = full.v_feed.fillna(0) / 1000 * full.cod_feed_conc
for (r, ph), g in full.groupby(["reactor", "phase"]):
    ch4f = g.ch4.interpolate(limit_area="inside")
    meth = (g.biogas * ch4f / 100).sum()
    cod = g.cod_fed_g.sum()
    print(f"{r} {ph}: COD fed {cod:.1f} g, measured CH4 {meth:.0f} mL, "
          f"apparent yield {meth / cod:.0f} mL CH4/g COD (theory ~380 at 35 C)")
