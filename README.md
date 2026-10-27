# Black-liquor surrogate fidelity: core analysis code

Core code for **Channel-dependent and time-varying fidelity of synthetic black liquor in paired anaerobic digesters**.

This repository contains the analysis implementation: data extraction, prediction benchmarks, paired discrimination, simulations, intervention sensitivity and time-resolved comparisons. It is a code-only release. Manuscripts, raw or derived data, figure assets, rendered figures and local writing records are excluded.

## Scientific comparison

The study reanalyses paired measurements from one synthetic-liquor digester and one real-liquor digester. The main question is how their correspondence varies across monitored channels and operating stages. Same-date differences are real minus synthetic. Persistent liquid-state contrasts and evolving methane-content differences are assessed separately.

`analysis/scripts/twinc2st.py` implements the main ridge-discriminant paired score. Scaling and fitting use the training weeks. Scores average the off-diagonal ordered comparisons, with denominator n(n−1) and half credit for ties. Sign-flip inference requires the relevant joint sign invariance; week-held-out scoring does not establish this assumption for a serial record. The operating-day analysis is exploratory. One reactor per liquor does not isolate a causal liquor-composition effect.

## Public source data

Raquel Núñez, *A longitudinal dataset of reactor adaptation and operation during semicontinuous anaerobic digestion of black liquor*, Mendeley Data, version 1: https://doi.org/10.17632/jpnjmzp6bj.1.

The original dataset is distributed under CC BY 4.0. Retain its attribution and licence when downloading or reusing it. That dataset licence is separate from the code in this repository; a separate software licence has not been assigned here.

Download the source workbook from Mendeley Data. Convert `Dataset_paper_Scientific_Data_Raquel_Nunez.xlsb` to XLSX using Excel or LibreOffice, retaining sheet names and cell layout. Place the converted file at:

```text
analysis/data_derived/Dataset_paper_Scientific_Data_Raquel_Nunez.xlsx
```

Create `analysis/results/` before running the analysis. The supplied scripts resolve paths relative to their own locations. Use Python 3.12 and install the pinned environment:

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

## Run the analysis

Run `00_build_tidy.py` through `06_gas_silent_and_simulation.py` in numerical order, then `07_multichannel_conditional.py`, `08_figure_inputs.py`, `09_si_inputs.py` and `10_time_resolved_fidelity.py`:

```text
python analysis/scripts/00_build_tidy.py
python analysis/scripts/01_profile.py
python analysis/scripts/02_naive_benchmark.py
python analysis/scripts/03_twin_and_weekly.py
python analysis/scripts/04_paired_c2st.py
python analysis/scripts/05_robustness_sim_transfer.py
python analysis/scripts/06_gas_silent_and_simulation.py
python analysis/scripts/07_multichannel_conditional.py
python analysis/scripts/08_figure_inputs.py
python analysis/scripts/09_si_inputs.py
python analysis/scripts/10_time_resolved_fidelity.py
```

These commands can be computationally expensive. Several analysis scripts run their computations at module load; inspect their text instead of importing them for discovery. Use a fresh working directory to preserve existing results.

The main pooled results are produced by `08_figure_inputs.py` (E11), intervention sensitivity by `09_si_inputs.py` (S7), and time-conditional results by `10_time_resolved_fidelity.py` (E13). Earlier exploratory estimators remain separate. The null/power checks in `06_gas_silent_and_simulation.py` use a shrunken scorer; the structured scenarios in `07_multichannel_conditional.py` use the main ridge-discriminant scorer.

### Reported structured-simulation budgets

The reported E10 simulations used 200 null replications with 99 permutations and 100 replications with 49 permutations per alternative at delta = 1. These differ from the script's generic defaults. In a fresh copy with no existing E10 simulation CSV, use these two commands in place of the generic step 07 above:

```text
python analysis/scripts/07_multichannel_conditional.py --n-sim 200 --n-perm 99 --deltas
python analysis/scripts/07_multichannel_conditional.py --sim-only --resume --n-sim 100 --n-perm 49 --deltas 1
```

Resuming or batching changes the random-number stream; fresh simulation rates can differ by Monte Carlo error. The article's numerical supplement supplies the reported results. Prediction designs, gas-volume assumptions and supplementary analyses are defined in the article and Supporting Information. No frozen fitted model is needed to execute this code.

## Checks and contact

The small test suite checks the off-diagonal score denominator and tie handling. It does not replace the experimental and inferential assumptions described above or claim that every stochastic analysis has been rerun during publication of this repository.

Contact: Chong Liu, Department of Chemical and Materials Engineering, University of Auckland, Auckland 1010, New Zealand. chong.liu@auckland.ac.nz.
