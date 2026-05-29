# Kumbh Monitor — Intelligence Dashboard

Production-grade Streamlit dashboard for the Kumbh Monitor news intelligence project.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place the CSV at:

```
data_pipeline/articles_export_clean.csv
```

Required base columns: `id, source, publish_date, headline, extracted_topic, clean_body`.

## Run

```bash
streamlit run dashboard.py
```

## ML columns

The dashboard works immediately with only the base columns. When your ML engineer
adds any of the following columns, the corresponding visualizations
auto-enable — no code change needed:

| Column | Unlocks |
|---|---|
| `ml_themes` (JSON array) | Theme badges in article cards |
| `ml_event_type` | Content-type breakdown chart |
| `ml_temporal_phase` | 7-phase Kumbh timeline chart |
| `ml_cluster_id` | AI cluster summary cards + cluster coloring on map |
| `risk_score` / `risk_band` | Risk filter, KPI, risk-distribution chart, source ranking |
| `viz_x`, `viz_y` | Article similarity scatter map |

Missing columns are filled with safe defaults so nothing breaks.

## Features

- 6 sidebar filters (topic, phase, source, risk band, search, date range)
- 5 KPI metric cards with hover animations
- Auto-generated Key Insights (most-covered topic, peak phase, misinfo alert,
  source diversity, emerging topic, volume trend)
- AI Similarity Map (scatter)
- Topic distribution, phase donut, monthly volume timeline with event windows
- ML-enhanced charts (conditional)
- Searchable / paginated data explorer + article-card view
- CSV export of filtered data
# Kumbh-Monitor
