# FPL Optimizer

Fantasy Premier League points predictor and squad optimiser.

## Structure

- `data/` — raw, processed, and external data (xG/xA, fixture difficulty, etc.)
- `src/data/` — data fetching (FPL API, historical archives, external sources)
- `src/features/` — feature engineering, including a rotation/minutes-risk sub-model
- `src/models/` — points prediction model (training, inference, evaluation)
- `src/optimization/` — ILP-based squad selection, transfer planning, captain choice
- `src/utils/` — shared config and FPL rule constants
- `notebooks/` — exploration and experimentation
- `models/` — saved model artifacts
- `outputs/gameweek_recommendations/` — weekly generated recommendations
- `tests/` — unit tests
- `app/` — optional Streamlit dashboard

## Setup

```bash
pip install -r requirements.txt
```

## Status

Scaffolding only — no implementation yet.
