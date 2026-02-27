# 🏈 NFL Combine Career Predictor

A Streamlit web app that uses NFL combine data and Gemini AI to compare prospects against thousands of historical players and predict their career trajectory.

---

## What It Does

Enter any NFL combine participant's name — or manually enter stats for current combine players whose data hasn't been uploaded yet — and the app will:

1. **Find the most similar historical players** based on combine metrics (40-yard dash, vertical jump, bench press, shuttle, cone, etc.), matched by position
2. **Display a comparison table** of the closest historical comps with their draft results
3. **Generate an AI-powered career prediction** using Gemini 2.0 Flash, analyzing the player's combine profile, grading each metric, reviewing how the comps' careers actually played out, and delivering a final verdict

---

## Features

- **Search by name** — looks up any player from the nflverse combine database (2000–2025)
- **Manual stats entry** — designed for current combine players not yet in the database; enter only the stats you have (partial stats are fully supported)
- **Position-aware comparisons** — different metrics are weighted by position (e.g. shuttle/cone for DBs and LBs, bench for OL)
- **8,300+ historical records** — covers every NFL combine participant from 2000 to 2025
- **Streaming AI analysis** — Gemini streams the career prediction live as it's generated
- **Jump stats in ft-in format** — vertical and broad jump entered as feet-inches (e.g. `3-9`)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/mangesh80/Claude-Code.git
cd Claude-Code
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install nfl_data_py --no-deps
```

### 3. Get a Google API key

Go to [aistudio.google.com](https://aistudio.google.com), click **Get API key**, and create a new key.

### 4. Run the app

```bash
export GOOGLE_API_KEY="AIza..."
./run.sh
```

Or paste your key directly into the sidebar after launching — no environment variable needed.

---

## Usage

### Search mode
Type a player name (e.g. *Caleb Williams*, *Ja'Marr Chase*, *Patrick Mahomes*) and click **Analyze**.

### Manual entry mode
Switch to **Enter stats manually** for current combine players. Enter as many or as few stats as you have — only the stats you fill in are used for the historical comparison. Even a single metric like the 40-yard dash is enough to generate comps.

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Combine data | [nflverse](https://github.com/nflverse/nflverse-data) via `nfl_data_py` |
| Similarity matching | Normalized Euclidean distance |
| AI analysis | [Gemini 2.0 Flash](https://aistudio.google.com) |

---

## Project Structure

```
├── app.py                  # Streamlit UI
├── nfl_combine_analyzer.py # Data loading, similarity search, core logic
├── requirements.txt        # Python dependencies
└── run.sh                  # Launch script
```

---

*By Maxintech*
