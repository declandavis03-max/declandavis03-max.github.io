# NBA Analytics RAG Agent

A retrieval-augmented generation (RAG) system that lets you ask natural language questions about the 2025-26 NBA regular season - powered by Claude AI and Basketball Reference data.

**[→ Live Demo](https://nba-ppg-predictor-aash2dexwqcmfme7dshg77.streamlit.app)** &nbsp;·&nbsp; **[Portfolio](https://declandavis03-max.github.io/)**

---

## What It Does

Instead of manually digging through stat tables, you ask plain English questions and get answers grounded in real data:

> *"Who is the most efficient scorer under 25?"*

> *"Compare OKC and Denver - offense vs defense?"*

> *"Which team shoots the most 3s and how efficiently?"*

The agent searches across 9 Basketball Reference datasets, retrieves the most relevant chunks, and uses Claude to generate a cited, structured answer.

---

## How It Works

```
User Question
     ↓
TF-IDF Search across 74 chunks
     ↓
Top 5 most relevant passages retrieved
     ↓
Claude reads passages + writes grounded answer
     ↓
Cited response with source document + date
```

This is a **RAG (Retrieval-Augmented Generation)** architecture — the model never answers from memory. Every stat it cites comes directly from the documents.

---

## Knowledge Base

| # | Document | Coverage |
|---|---|---|
| 1 | Player Per-Game Stats | PPG, RPG, APG, FG%, 3P%, FT% — Top 75 |
| 2 | Player Advanced Stats | PER, VORP, BPM, WS, TS% — Top 75 |
| 3 | Player Adjusted Shooting | FG+, TS+, eFG+ (league-adjusted) — Top 75 |
| 4 | Player Shooting by Distance | Zone-by-zone FG% breakdown — Top 75 |
| 5 | Player Per-36 Stats | Pace-neutral production — Top 75 |
| 6 | Player Play-Type & On-Off | Positional role, OnCourt +/- — Top 75 |
| 7 | Team Advanced Metrics | ORtg, DRtg, NRtg, Pace, SRS — All 30 |
| 8 | Team Per-Game Stats | PTS, FG%, 3P%, REB, AST, STL — All 30 |
| 9 | Team Shooting by Distance | Shot zone % and FG% — All 30 |

**Source:** Basketball Reference · 2025-26 NBA Regular Season

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web interface |
| Anthropic Claude API | Language model (claude-sonnet-4-5) |
| scikit-learn TF-IDF | Document retrieval |
| NumPy | Recency scoring |

---

## Run It Locally

**1. Clone the repo**
```bash
git clone https://github.com/declandavis03-max/nba-rag-agent.git
cd nba-rag-agent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Get a free key at [console.anthropic.com](https://console.anthropic.com), then either:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Or paste it directly into the sidebar when the app launches.

**4. Run the app**
```bash
streamlit run streamlit_app.py
```

---

## Project Structure

```
nba-rag-agent/
├── streamlit_app.py   # Full app — UI, RAG engine, documents
├── requirements.txt   # Python dependencies
└── README.md
```

All 9 stat tables are embedded directly in `streamlit_app.py` as structured text — no external database or file reads required.

---

## Built For

ISOM 260: AI for Business · Suffolk University


---

*Declan Davis · [declandavis03-max.github.io](https://declandavis03-max.github.io) · [@declandavis17](https://twitter.com/declandavis17)*
