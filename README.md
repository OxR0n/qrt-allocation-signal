# QRT Asset Allocation Sign Forecasting

Machine Learning Project 2026 — ESILV. A supervised framework to predict the
sign of the next-day return of QRT asset allocations, from their 20-day
historical footprint (returns, signed volumes, turnover, anonymous group).

Repository also submitted on Moodle as `Group_XX_DS_Project.zip`.

---

## 1. Business scope

Systematic trading desks evaluate thousands of candidate portfolio
constructions every day. Each allocation is a set of instrument weights with
sum of absolute weights equal to 1, held for one trading session and
rebalanced daily. The decision at hand is binary and symmetrical:

- predicted positive next-day return -> follow the allocation (long),
- predicted negative next-day return -> short the allocation.

Only the sign matters: magnitude prediction is out of scope and is not scored.
This framing matches the [official Challenge Data #167 problem]
(https://challengedata.ens.fr/challenges/167), hosted by ENS and provided by
Qube Research & Technologies.

## 2. Problem formalisation

Each observation is a (timestamp, allocation) pair. For allocation S at day t
over M instruments with weights w_{S,t,i} and next-day asset returns
r_{i,t+1}:

    r_{S,t+1} = sum_{i=1..M} w_{S,t,i} * r_{i,t+1}

The target is the sign of r_{S,t+1}, and the evaluation metric is plain
accuracy over all rows:

    Accuracy = (1/N) * sum_i 1[ sign(r_hat_i) = sign(r_i) ]

Two properties drive the whole methodology:

1. Rows are NOT independent: they are organised by date and by allocation.
   Any random split leaks date-level information and inflates scores.
2. Dates are anonymised and shuffled (DATE_0001, DATE_0002, ...), so there
   is no continuity guarantee; the validation protocol must be date-blocked,
   not time-ordered.

## 3. Data

From the challenge data section (registration required):

| File | Content |
|---|---|
| `X_train.csv` | 527,073 rows, features |
| `y_train.csv` | training targets under the shared ROW_ID |
| `X_test.csv` | 31,870 rows, features |
| `sample_submission.csv` | submission format |
| `benchmark_submission.ipynb` | official benchmark notebook |

Feature groups per row (20-day history, RET_1 = most recent confirmed against
the benchmark notebook):

- `RET_1..RET_20`: allocation daily returns over the last 20 sessions,
- `SIGNED_VOLUME_1..20`: allocation signed volume, rolling-rescaled,
- `MEDIAN_DAILY_TURNOVER`: median daily allocation turnover,
- `GROUP`: anonymised allocation group,
- `TARGET`: next-day return (train only; we predict and score its sign).

## 4. Methodology

### 4.1 Pipeline

    raw data -> quality checks -> date-blocked split -> feature engineering
    -> baseline -> gradient boosting -> ensembles -> evaluation -> submission

### 4.2 Validation protocol

- Single split definition, owned by `src/data/make_splits.py`; every script
  imports it, nothing else is allowed to call train_test_split.
- Date-blocked grouped validation: whole dates held out as validation folds,
  never random KFold over rows.
- All scalers, imputers and encoders are fitted inside sklearn Pipelines on
  training folds only.
- `random_state = 42` everywhere, fixed and unique entry point.

### 4.3 Models

| Stage | Model | Role |
|---|---|---|
| Baseline | sign-agreement of lagged mean return | sanity floor |
| Benchmark | Ridge and LightGBM on extended features | public score 0.5079 |
| v0.1 | AutoML-style LGBM with CV selection | first competitive model |
| v0.2+ | ensembles, calibration, out-of-scope model | project step 3 |

Every model beyond the baseline cites its reference paper in
`report/Report_Group_XX.pdf` (in English, LaTeX).

### 4.4 Known risks and failure modes

- Tiny signal-to-noise ratio: benchmark accuracy is 0.5079, barely above 0.5;
  reported metrics include confidence intervals and a random-guess reference.
- Class balance and date heterogeneity: balance is checked per fold.
- Leakage: the highest-scoring mistake; see `ai_governance/AI_AUDIT.md` for
  the caught-and-fixed instances (a leaked score caps the technical grade at
  10/20 per the course policy).

## 5. Repository layout

    src/data/         download, quality checks, split definitions
    src/features/     rolling statistics and feature pipelines
    src/models/       ridge baseline, LGBM, tuning scripts
    src/evaluation/   accuracy + confidence intervals, submission writer
    notebooks/        EDA and experiments (numbered, readable)
    report/           final report PDF
    ai_governance/    CONTRIBUTIONS.md, PROMPT_LOG.md, AI_AUDIT.md

## 6. Setup and reproduction

    # Python 3.11
    pip install -r requirements.txt

    # 1. download the challenge files into data/raw/ (see challenge data
    #    section, registration required)
    # 2. run quality checks
    python src/data/quality_check.py
    # 3. train the baseline and the main model
    python src/models/baseline_ridge.py
    python src/models/model_lgbm.py
    # 4. generate the submission file
    python src/evaluation/make_submission.py

All experiments are reproducible with the pinned dependencies and the fixed
random state. Runtime and hardware used are logged next to each result table.

## 7. Results

Results are updated at every project milestone (Weeks 1 to 4).

| Model | Validation accuracy (95% CI) | Public LB | Status |
|---|---|---|---|
| Random guess | 0.5000 (reference) | 0.5 | reference |
| Lagged-mean baseline | TBD | - | Week 2 |
| Ridge (benchmark setup) | TBD | 0.5079 | Week 2 |
| LightGBM (benchmark setup) | TBD | 0.5079 | Week 2 |
| Ours v0.1 | TBD | - | Week 3 |
| Ours v0.2 | TBD | - | Week 4 |

 CI obtained by block bootstrap over dates, 1000 resamples.

## 8. Team and ownership

See `ai_governance/CONTRIBUTIONS.md` for the full ownership matrix. Each
student owns one module, documents it, and defends it orally.

## 9. AI governance

Generative AI is used as an engineering accelerator only, under the course
policy: prompts for high-stakes decisions are logged in
`ai_governance/PROMPT_LOG.md`, and every caught AI mistake is documented with
its fix in `ai_governance/AI_AUDIT.md`. Every statistical claim was verified
by the team.

## 10. References

- Challenge QRT 2026, Asset Allocation Performance forecasting,
  https://challengedata.ens.fr/challenges/167
- Ruta et al. (2025), ChatGPT for Univariate Statistics: Validation of
  AI-Assisted Data Analysis in Healthcare Research, PMC11845875.
- Ke et al. (2017), LightGBM: A Highly Efficient Gradient Boosting Decision
  Tree, NeurIPS 2017.
