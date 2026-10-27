"""Build a tidy long table (one row = date x reactor) from the official workbook.

Source: Dataset_paper_Scientific_Data_Raquel_Nunez.xlsb (Mendeley Data jpnjmzp6bj v1, CC BY 4.0).
The .xlsb was converted to .xlsx with LibreOffice (headless) into data_derived/; the original
file is untouched (SHA-256 verified before and after conversion).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data_derived" / "Dataset_paper_Scientific_Data_Raquel_Nunez.xlsx"
OUT = ROOT / "data_derived"


def col_idx(c):
    n = 0
    for ch in c:
        n = n * 26 + ord(ch) - 64
    return n - 1


SEMI_LEFT = dict(bl_pct="H", co_type="I", co_pct="J", v_feed="K", feed_rate="L", olr="M", hrt="N",
                 biogas="O", ch4="P", co2="Q", h2s="R", o2="S", ts="T", vs="U", ph="V", ph_corr="W",
                 cond="X", alk="Y", cod_t="Z", cod_s="AA", tc="AB", tn="AC", vfa="AD", acetic="AE",
                 propionic="AF", isobutyric="AG", butyric="AH", reinoc="AI")
SEMI_RIGHT = dict(bl_pct="AK", co_type="AL", co_pct="AM", v_feed="AN", feed_rate="AO", olr="AP", hrt="AQ",
                  biogas="AR", ch4="AS", co2="AT", h2s="AU", o2="AV", ts="AW", vs="AX", ph="AY", ph_corr="AZ",
                  cond="BA", alk="BB", cod_t="BC", cod_s="BD", tc="BE", tn="BF", vfa="BG", acetic="BH",
                  propionic="BI", isobutyric="BJ", butyric="BK", reinoc="BL")
ACC_LEFT = dict(bl_pct="H", co_type="I", co_pct="J", v_feed="K", feed_rate="L", olr="M", hrt="N",
                biogas="O", ch4="P", co2="Q", ph="R", ph_corr="S", cond="T")
ACC_RIGHT = dict(bl_pct="V", co_type="W", co_pct="X", v_feed="Y", feed_rate="Z", olr="AA", hrt="AB",
                 biogas="AC", ch4="AD", co2="AE", h2s="AF", o2="AG", ph="AH", ph_corr="AI", cond="AJ")


def to_num(s):
    return pd.to_numeric(s.replace({"?": np.nan, " ": np.nan}), errors="coerce")


def block(sheet, rows, mapping, reactor, bl_type, phase):
    d = pd.DataFrame({
        "date": pd.to_datetime(sheet.iloc[rows, 0], errors="coerce"),
        "day": to_num(sheet.iloc[rows, 1]),
        "temp": to_num(sheet.iloc[rows, 2]),
    })
    for k, c in mapping.items():
        v = sheet.iloc[rows, col_idx(c)]
        d[k] = v.astype(str).str.strip() if k == "co_type" else to_num(v)
    d["reactor"] = reactor
    d["bl_type"] = bl_type
    d["phase"] = phase
    return d


def main():
    x = pd.read_excel(SRC, sheet_name=None, header=None)
    semi, acc = x["Semicontinuo"], x["Aclimatacion"]
    semi_rows = list(range(1, 70))   # Excel rows 2-70
    acc_rows = list(range(1, 42))    # Excel rows 2-42
    parts = [
        block(semi, semi_rows, SEMI_LEFT, "R1", "synthetic", "semicontinuous"),
        block(semi, semi_rows, SEMI_RIGHT, "R2", "real", "semicontinuous"),
        block(acc, acc_rows, ACC_LEFT, "R1", "synthetic", "acclimation"),
        block(acc, acc_rows, ACC_RIGHT, "R2", "real", "acclimation"),
    ]
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["date"]).sort_values(["phase", "reactor", "date"]).reset_index(drop=True)
    df.to_csv(OUT / "tidy_long.csv", index=False, encoding="utf-8-sig")
    print(df.groupby(["phase", "reactor"]).size())
    print(df.groupby(["phase", "reactor"])[["biogas", "ch4", "ph", "vfa", "cod_s"]].count())


if __name__ == "__main__":
    main()
