import streamlit as st
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
from anthropic.types import ToolUseBlock, TextBlock

st.set_page_config(
    page_title="NBA RAG Agent · Declan Davis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0D0D0D;
    --surface:   #141414;
    --border:    #222222;
    --accent:    #4ADE80;
    --muted:     #F0F0F0;
    --white:     #F0F0F0;
    --serif:     'DM Serif Display', Georgia, serif;
    --sans:      'DM Sans', sans-serif;
    --mono:      'JetBrains Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--white) !important;
    font-family: var(--sans);
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Header ── */
.dd-header {
    padding: 3rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.dd-eyebrow {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.dd-title {
    font-family: var(--serif);
    font-size: 3rem;
    font-weight: 400;
    color: var(--white);
    line-height: 1.1;
    margin: 0 0 0.5rem;
}
.dd-title em {
    color: var(--accent);
    font-style: italic;
}
.dd-desc {
    color: #F0F0F0;
    font-size: 0.9rem;
    font-weight: 400;
    max-width: 600px;
    line-height: 1.6;
}

/* ── Sidebar ── */
.dd-sidebar-name {
    font-family: var(--serif);
    font-size: 1.3rem;
    color: var(--white);
    margin-bottom: 0.1rem;
}
.dd-sidebar-sub {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.dd-stat {
    border-top: 1px solid var(--border);
    padding: 0.8rem 0;
}
.dd-stat-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: #F0F0F0;
    text-transform: uppercase;
}
.dd-stat-val {
    font-family: var(--serif);
    font-size: 1.8rem;
    color: var(--accent);
    line-height: 1;
    margin-top: 0.1rem;
}
.dd-tag {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 1px;
    color: #F0F0F0;
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.15rem 0.5rem;
    margin: 0.15rem 0.15rem 0 0;
    text-transform: uppercase;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: #F0F0F0 !important;
    border-radius: 4px !important;
    font-family: var(--sans) !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input::placeholder {
    color: #666666 !important;
    opacity: 1 !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.12) !important;
}

/* ── Button ── */
.stButton > button {
    background-color: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background-color: var(--accent) !important;
    color: var(--bg) !important;
}

/* ── Quick question buttons ── */
.qq-btn > button {
    background-color: var(--surface) !important;
    color: #F0F0F0 !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: var(--sans) !important;
    font-size: 0.72rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 0.45rem 0.7rem !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    transition: all 0.2s !important;
    line-height: 1.3 !important;
}
.qq-btn > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background-color: var(--surface) !important;
}

/* ── Answer ── */
.dd-answer {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    border-radius: 4px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.2rem;
    font-size: 0.9rem;
    line-height: 1.8;
    color: var(--white);
}
.dd-search-log {
    margin-bottom: 1rem;
}
.dd-search-item {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    margin-bottom: 0.2rem;
}
.dd-search-item::before { content: "→ "; color: var(--accent); }

/* ── Section label ── */
.dd-section {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: #F0F0F0;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* ── History ── */
.dd-history-q {
    font-family: var(--serif);
    font-size: 1rem;
    font-style: italic;
    color: var(--muted);
    margin-bottom: 0.3rem;
}

/* ── Tables ── */
div[data-testid="stMarkdownContainer"] table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.85rem;
    margin: 1rem 0;
}
div[data-testid="stMarkdownContainer"] th {
    background: #1A1A1A;
    color: var(--accent);
    padding: 0.5rem 0.8rem;
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
div[data-testid="stMarkdownContainer"] td {
    padding: 0.45rem 0.8rem;
    border-bottom: 1px solid var(--border);
    color: var(--white);
}
div[data-testid="stMarkdownContainer"] strong { color: var(--accent); }
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 {
    font-family: var(--serif) !important;
    font-weight: 400 !important;
    color: var(--white) !important;
}

/* ── Footer ── */
.dd-footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 1px;
    color: var(--muted);
    text-align: center;
}
.dd-footer a { color: var(--accent); text-decoration: none; }

hr { border-color: var(--border) !important; }
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    background: var(--surface) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Documents ────────────────────────────────────────────────────────────────
my_documents = [
    {
        "title": 'Player Per-Game Stats — 2024-25 NBA Regular Season',
        "source": 'players_per_game_stats.txt',
        "last_updated": "2025",
        "content": "NBA 2024-25 Regular Season — Player Per-Game Stats (Top 75 by PPG, min qual.)\nPlayer | Age | Team | Pos | G | MP | PTS | AST | TRB | STL | BLK | FG% | 3P% | FT% | eFG% | TOV\n------------------------------------------------------------------------------------------------------------------------\nLuka Dončić | 26 | LAL | PG | 64 | 35.8 | 33.5 | 8.3 | 7.7 | 1.6 | 0.5 | .476 | .366 | .780 | .563 | 4.0\nShai Gilgeous-Alexander | 27 | OKC | PG | 68 | 33.2 | 31.1 | 6.6 | 4.3 | 1.4 | 0.8 | .553 | .386 | .879 | .597 | 2.2\nAnthony Edwards | 24 | MIN | SG | 61 | 35.0 | 28.8 | 3.7 | 5.0 | 1.4 | 0.8 | .489 | .399 | .796 | .572 | 2.9\nJaylen Brown | 29 | BOS | SF | 71 | 34.4 | 28.7 | 5.1 | 6.9 | 1.0 | 0.4 | .477 | .347 | .795 | .522 | 3.6\nTyrese Maxey | 25 | PHI | PG | 70 | 38.0 | 28.3 | 6.6 | 4.1 | 1.9 | 0.8 | .462 | .367 | .892 | .536 | 2.4\nKawhi Leonard | 34 | LAC | SF | 65 | 32.1 | 27.9 | 3.6 | 6.4 | 1.9 | 0.4 | .505 | .387 | .892 | .573 | 2.0\nDonovan Mitchell | 29 | CLE | SG | 70 | 33.5 | 27.9 | 5.7 | 4.5 | 1.5 | 0.3 | .483 | .364 | .865 | .563 | 2.8\nNikola Jokić | 30 | DEN | C | 65 | 34.8 | 27.7 | 10.7 | 12.9 | 1.4 | 0.8 | .569 | .380 | .831 | .618 | 3.7\nGiannis Antetokounmpo | 31 | MIL | PF | 36 | 28.9 | 27.6 | 5.4 | 9.8 | 0.9 | 0.7 | .624 | .333 | .650 | .636 | 3.2\nJoel Embiid | 31 | PHI | C | 38 | 31.6 | 26.9 | 3.9 | 7.7 | 0.6 | 1.2 | .489 | .333 | .854 | .527 | 2.9\nLauri Markkanen | 28 | UTA | PF | 42 | 34.4 | 26.7 | 2.1 | 6.9 | 1.0 | 0.5 | .477 | .355 | .896 | .548 | 1.5\nStephen Curry | 37 | GSW | PG | 43 | 30.9 | 26.6 | 4.7 | 3.6 | 1.1 | 0.4 | .468 | .393 | .923 | .587 | 2.8\nDevin Booker | 29 | PHO | SG | 64 | 33.5 | 26.1 | 6.0 | 3.9 | 0.8 | 0.3 | .456 | .330 | .873 | .507 | 3.1\nJalen Brunson | 29 | NYK | PG | 74 | 35.0 | 26.0 | 6.8 | 3.3 | 0.8 | 0.1 | .467 | .369 | .841 | .533 | 2.4\nKevin Durant | 37 | HOU | SF | 78 | 36.4 | 26.0 | 4.8 | 5.5 | 0.8 | 0.9 | .520 | .413 | .874 | .588 | 3.2\nJamal Murray | 28 | DEN | PG | 75 | 35.4 | 25.4 | 7.1 | 4.4 | 0.9 | 0.4 | .483 | .435 | .887 | .573 | 2.3\nVictor Wembanyama | 22 | SAS | C | 64 | 29.2 | 25.0 | 3.1 | 11.5 | 1.0 | 3.1 | .512 | .349 | .827 | .569 | 2.4\nDeni Avdija | 25 | POR | SF | 66 | 33.3 | 24.2 | 6.7 | 6.9 | 0.8 | 0.6 | .462 | .318 | .802 | .521 | 3.8\nMichael Porter Jr. | 27 | BRK | SF | 52 | 32.5 | 24.2 | 3.0 | 7.1 | 1.1 | 0.3 | .463 | .363 | .859 | .555 | 2.3\nPascal Siakam | 31 | IND | PF | 62 | 33.2 | 24.0 | 3.8 | 6.6 | 1.1 | 0.4 | .484 | .358 | .693 | .529 | 2.2\nCade Cunningham | 24 | DET | PG | 64 | 33.9 | 23.9 | 9.9 | 5.5 | 1.4 | 0.8 | .461 | .342 | .812 | .513 | 3.7\nJames Harden | 36 | 2TM | PG | 70 | 34.8 | 23.6 | 8.0 | 4.8 | 1.1 | 0.4 | .434 | .375 | .884 | .529 | 3.5\nKeyonte George | 22 | UTA | PG | 54 | 33.1 | 23.6 | 6.1 | 3.7 | 1.1 | 0.3 | .456 | .371 | .892 | .532 | 3.1\nAustin Reaves | 27 | LAL | SG | 51 | 34.5 | 23.3 | 5.5 | 4.7 | 1.1 | 0.4 | .490 | .360 | .871 | .567 | 3.0\nJalen Johnson | 24 | ATL | SF | 72 | 35.2 | 22.5 | 7.9 | 10.3 | 1.2 | 0.4 | .489 | .352 | .788 | .537 | 3.4\nPaolo Banchero | 23 | ORL | PF | 72 | 34.8 | 22.2 | 5.2 | 8.4 | 0.7 | 0.6 | .459 | .305 | .775 | .496 | 3.1\nJayson Tatum | 27 | BOS | PF | 16 | 32.6 | 21.8 | 5.3 | 10.0 | 1.4 | 0.2 | .411 | .329 | .823 | .493 | 2.4\nNorman Powell | 32 | MIA | SG | 58 | 29.6 | 21.7 | 2.5 | 3.5 | 1.1 | 0.2 | .470 | .380 | .827 | .558 | 1.9\nBrandon Ingram | 28 | TOR | SF | 77 | 33.8 | 21.5 | 3.7 | 5.6 | 0.8 | 0.7 | .477 | .382 | .820 | .529 | 2.4\nTrey Murphy III | 25 | NOP | SF | 66 | 35.5 | 21.5 | 3.8 | 5.7 | 1.5 | 0.4 | .470 | .379 | .886 | .572 | 1.8\nJulius Randle | 31 | MIN | PF | 79 | 33.0 | 21.1 | 5.0 | 6.7 | 1.1 | 0.2 | .481 | .315 | .802 | .526 | 2.7\nCooper Flagg | 19 | DAL | SF | 70 | 33.5 | 21.0 | 4.5 | 6.7 | 1.2 | 0.9 | .468 | .295 | .827 | .498 | 2.3\nZion Williamson | 25 | NOP | PF | 62 | 29.7 | 21.0 | 3.2 | 5.7 | 1.0 | 0.5 | .600 | .250 | .716 | .600 | 2.0\nLeBron James | 41 | LAL | SF | 60 | 33.2 | 20.9 | 7.2 | 6.1 | 1.2 | 0.6 | .515 | .317 | .737 | .557 | 3.0\nNickeil Alexander-Walker | 27 | ATL | SG | 78 | 33.4 | 20.8 | 3.7 | 3.4 | 1.3 | 0.5 | .459 | .399 | .902 | .564 | 2.1\nShaedon Sharpe | 22 | POR | SG | 50 | 29.4 | 20.8 | 2.6 | 4.3 | 1.4 | 0.1 | .452 | .337 | .787 | .511 | 2.9\nFranz Wagner | 24 | ORL | SF | 34 | 30.0 | 20.6 | 3.3 | 5.2 | 0.9 | 0.3 | .481 | .345 | .823 | .529 | 1.7\nTyler Herro | 26 | MIA | SG | 33 | 31.3 | 20.5 | 4.1 | 4.8 | 0.7 | 0.4 | .480 | .378 | .917 | .561 | 1.9\nAnthony Davis | 32 | DAL | PF | 20 | 31.3 | 20.4 | 2.8 | 11.1 | 1.1 | 1.7 | .506 | .270 | .728 | .521 | 2.1\nAlperen Şengün | 23 | HOU | C | 72 | 33.3 | 20.4 | 6.2 | 8.9 | 1.2 | 1.1 | .519 | .305 | .691 | .537 | 3.2\nDillon Brooks | 30 | PHO | SF | 56 | 30.4 | 20.2 | 1.8 | 3.6 | 1.0 | 0.2 | .435 | .344 | .842 | .501 | 1.8\nBrandon Miller | 23 | CHO | SF | 65 | 30.3 | 20.2 | 3.3 | 4.9 | 1.0 | 0.7 | .435 | .383 | .892 | .533 | 2.5\nBam Adebayo | 28 | MIA | C | 73 | 32.4 | 20.1 | 3.2 | 10.0 | 1.2 | 0.7 | .442 | .318 | .778 | .497 | 1.6\nLaMelo Ball | 24 | CHO | PG | 72 | 28.0 | 20.1 | 7.1 | 4.8 | 1.2 | 0.2 | .407 | .368 | .899 | .516 | 2.8\nDesmond Bane | 27 | ORL | SG | 82 | 33.6 | 20.1 | 4.1 | 4.1 | 1.0 | 0.5 | .484 | .391 | .908 | .553 | 2.0\nKarl-Anthony Towns | 30 | NYK | C | 75 | 31.0 | 20.1 | 3.0 | 11.9 | 0.9 | 0.5 | .501 | .368 | .858 | .556 | 2.5\nJimmy Butler | 36 | GSW | SF | 38 | 31.1 | 20.0 | 4.9 | 5.6 | 1.4 | 0.2 | .519 | .376 | .864 | .554 | 1.6\nTy Jerome | 28 | MEM | SG | 15 | 22.6 | 19.7 | 5.7 | 2.8 | 1.1 | 0.3 | .474 | .420 | .875 | .572 | 1.8\nJalen Duren | 22 | DET | C | 70 | 28.2 | 19.5 | 2.0 | 10.5 | 0.8 | 0.8 | .650 |  | .747 | .650 | 1.9\nJa Morant | 26 | MEM | PG | 20 | 28.5 | 19.5 | 8.1 | 3.3 | 1.0 | 0.3 | .410 | .235 | .897 | .441 | 3.6\nJaren Jackson Jr. | 26 | 2TM | C | 48 | 30.3 | 19.4 | 2.0 | 5.7 | 1.1 | 1.4 | .476 | .357 | .803 | .534 | 2.2\nRJ Barrett | 25 | TOR | SF | 57 | 30.3 | 19.3 | 3.3 | 5.3 | 0.7 | 0.3 | .491 | .339 | .717 | .550 | 1.7\nZach LaVine | 30 | SAC | SG | 39 | 31.4 | 19.2 | 2.3 | 2.8 | 0.7 | 0.3 | .479 | .390 | .880 | .569 | 1.9\nDarius Garland | 26 | 2TM | PG | 45 | 29.9 | 18.8 | 6.7 | 2.4 | 1.0 | 0.2 | .460 | .396 | .861 | .551 | 2.9\nCJ McCollum | 34 | 2TM | PG | 76 | 29.8 | 18.7 | 3.9 | 3.3 | 0.8 | 0.5 | .455 | .375 | .772 | .538 | 1.8\nDe'Aaron Fox | 28 | SAS | PG | 72 | 31.0 | 18.6 | 6.2 | 3.8 | 1.2 | 0.3 | .486 | .332 | .760 | .549 | 2.3\nJerami Grant | 31 | POR | PF | 57 | 29.7 | 18.6 | 2.1 | 3.5 | 0.7 | 0.6 | .453 | .389 | .814 | .547 | 2.1\nKon Knueppel | 20 | CHO | SF | 81 | 31.5 | 18.5 | 3.4 | 5.3 | 0.7 | 0.2 | .475 | .425 | .863 | .601 | 2.0\nDeMar DeRozan | 36 | SAC | PF | 77 | 31.2 | 18.4 | 4.1 | 2.9 | 1.0 | 0.3 | .497 | .320 | .868 | .520 | 1.2\nAmen Thompson | 23 | HOU | PG | 79 | 37.4 | 18.3 | 5.3 | 7.8 | 1.5 | 0.6 | .534 | .216 | .779 | .545 | 2.4\nEvan Mobley | 24 | CLE | PF | 65 | 31.9 | 18.2 | 3.6 | 9.0 | 0.7 | 1.7 | .546 | .297 | .606 | .582 | 1.9\nScottie Barnes | 24 | TOR | PF | 80 | 33.5 | 18.1 | 5.9 | 7.5 | 1.4 | 1.5 | .507 | .304 | .815 | .537 | 2.6\nTrae Young | 27 | 2TM | PG | 15 | 25.6 | 17.9 | 8.0 | 2.0 | 0.9 | 0.1 | .458 | .338 | .825 | .534 | 2.6\nJalen Green | 23 | PHO | SG | 32 | 25.9 | 17.8 | 2.8 | 3.6 | 1.1 | 0.3 | .422 | .313 | .747 | .491 | 2.3\nSaddiq Bey | 26 | NOP | SF | 72 | 31.2 | 17.7 | 2.5 | 5.6 | 0.9 | 0.1 | .451 | .367 | .841 | .529 | 0.9\nBennedict Mathurin | 23 | 2TM | SF | 54 | 30.0 | 17.6 | 2.4 | 5.4 | 0.8 | 0.2 | .430 | .315 | .869 | .485 | 2.2\nCoby White | 25 | 2TM | SG | 50 | 25.0 | 17.4 | 4.0 | 3.4 | 0.5 | 0.1 | .446 | .362 | .817 | .539 | 2.6\nKevin Porter Jr. | 25 | MIL | PG | 38 | 33.2 | 17.4 | 7.4 | 5.2 | 2.2 | 0.5 | .465 | .322 | .878 | .510 | 2.9\nPaul George | 35 | PHI | PF | 37 | 30.7 | 17.3 | 3.6 | 5.3 | 1.7 | 0.4 | .439 | .392 | .820 | .536 | 1.7\nRyan Rollins | 23 | MIL | PG | 74 | 32.1 | 17.3 | 5.6 | 4.6 | 1.5 | 0.4 | .472 | .406 | .796 | .561 | 2.7\nMiles Bridges | 27 | CHO | PF | 77 | 31.0 | 17.1 | 3.2 | 5.8 | 0.6 | 0.4 | .460 | .333 | .822 | .531 | 1.4\nChet Holmgren | 23 | OKC | PF | 69 | 28.9 | 17.1 | 1.7 | 8.9 | 0.6 | 1.9 | .557 | .362 | .792 | .614 | 1.6\nJalen Williams | 24 | OKC | SG | 33 | 28.4 | 17.1 | 5.5 | 4.6 | 1.2 | 0.3 | .484 | .299 | .837 | .510 | 1.9\nJosh Giddey | 23 | CHI | PG | 54 | 32.1 | 17.0 | 9.1 | 8.3 | 1.0 | 0.5 | .448 | .364 | .763 | .520 | 3.6\nPayton Pritchard | 28 | BOS | PG | 79 | 32.4 | 17.0 | 5.2 | 3.9 | 0.7 | 0.1 | .464 | .377 | .890 | .561 | 1.4\n"
    },
    {
        "title": 'Player Advanced Stats — 2024-25 NBA Regular Season',
        "source": 'players_advanced_stats.txt',
        "last_updated": "2025",
        "content": "NBA 2024-25 Regular Season — Player Advanced Stats (Top 75 by VORP)\nPlayer | Age | Team | Pos | G | MP | PER | TS% | WS | WS/48 | BPM | VORP | USG% | OWS | DWS\n------------------------------------------------------------------------------------------------------------------------\nNikola Jokić | 30 | DEN | C | 65 | 2265 | 32.3 | .670 | 14.9 | .316 | 14.2 | 9.2 | 30.4 | 11.5 | 3.4\nShai Gilgeous-Alexander | 27 | OKC | PG | 68 | 2259 | 30.8 | .665 | 15.2 | .323 | 11.7 | 7.8 | 33.4 | 11.1 | 4.1\nLuka Dončić | 26 | LAL | PG | 64 | 2289 | 27.9 | .616 | 9.5 | .199 | 9.3 | 6.6 | 38.1 | 6.6 | 2.9\nVictor Wembanyama | 22 | SAS | C | 64 | 1866 | 29.9 | .626 | 10.0 | .257 | 10.7 | 6.0 | 32.4 | 5.0 | 5.0\nKawhi Leonard | 34 | LAC | SF | 65 | 2085 | 27.9 | .629 | 9.2 | .212 | 8.0 | 5.3 | 33.5 | 6.5 | 2.7\nTyrese Maxey | 25 | PHI | PG | 70 | 2661 | 21.9 | .588 | 8.7 | .156 | 5.4 | 4.9 | 29.4 | 5.9 | 2.8\nKevin Durant | 37 | HOU | SF | 78 | 2840 | 21.1 | .641 | 10.7 | .180 | 4.5 | 4.7 | 27.1 | 7.5 | 3.2\nCade Cunningham | 24 | DET | PG | 64 | 2172 | 21.6 | .564 | 7.9 | .174 | 6.3 | 4.6 | 30.5 | 4.2 | 3.6\nScottie Barnes | 24 | TOR | PF | 80 | 2681 | 19.2 | .577 | 7.8 | .140 | 4.2 | 4.2 | 23.4 | 3.3 | 4.5\nDonovan Mitchell | 29 | CLE | SG | 70 | 2342 | 22.9 | .613 | 8.2 | .169 | 5.1 | 4.2 | 32.2 | 5.8 | 2.4\nJamal Murray | 28 | DEN | PG | 75 | 2652 | 21.8 | .622 | 9.5 | .173 | 4.1 | 4.1 | 27.9 | 7.8 | 1.7\nJalen Johnson | 24 | ATL | SF | 72 | 2532 | 20.4 | .581 | 7.5 | .141 | 4.2 | 4.0 | 27.0 | 3.6 | 3.9\nAlperen Şengün | 23 | HOU | C | 72 | 2398 | 21.4 | .569 | 8.0 | .161 | 4.2 | 3.7 | 26.6 | 4.4 | 3.6\nJames Harden | 36 | 2TM | PG | 70 | 2438 | 20.7 | .610 | 7.7 | .151 | 3.9 | 3.6 | 28.6 | 5.6 | 2.1\nDerrick White | 31 | BOS | SG | 77 | 2625 | 16.0 | .529 | 7.0 | .129 | 3.2 | 3.5 | 22.0 | 3.5 | 3.5\nDeni Avdija | 25 | POR | SF | 66 | 2199 | 20.3 | .600 | 7.0 | .152 | 4.3 | 3.5 | 29.3 | 4.5 | 2.4\nAnthony Edwards | 24 | MIN | SG | 61 | 2137 | 21.8 | .617 | 6.5 | .147 | 4.5 | 3.5 | 31.5 | 3.9 | 2.7\nJalen Duren | 22 | DET | C | 70 | 1976 | 26.1 | .688 | 11.0 | .266 | 5.0 | 3.5 | 23.6 | 7.2 | 3.8\nAmen Thompson | 23 | HOU | PG | 79 | 2953 | 18.9 | .594 | 10.3 | .167 | 2.6 | 3.4 | 20.0 | 6.5 | 3.8\nLaMelo Ball | 24 | CHO | PG | 72 | 2017 | 19.7 | .546 | 6.2 | .147 | 4.7 | 3.4 | 32.0 | 3.6 | 2.6\nJalen Brunson | 29 | NYK | PG | 74 | 2590 | 20.1 | .580 | 8.8 | .163 | 3.1 | 3.3 | 30.4 | 6.5 | 2.2\nJaylen Brown | 29 | BOS | SF | 71 | 2443 | 22.0 | .573 | 6.9 | .135 | 3.3 | 3.3 | 36.2 | 3.5 | 3.4\nMikal Bridges | 29 | NYK | SF | 82 | 2692 | 15.3 | .587 | 8.0 | .143 | 2.7 | 3.2 | 17.3 | 4.7 | 3.3\nKon Knueppel | 20 | CHO | SF | 81 | 2551 | 17.1 | .633 | 8.0 | .151 | 2.8 | 3.1 | 22.3 | 5.6 | 2.5\nKarl-Anthony Towns | 30 | NYK | C | 75 | 2322 | 22.0 | .619 | 9.5 | .197 | 3.3 | 3.1 | 25.9 | 5.5 | 4.0\nChet Holmgren | 23 | OKC | PF | 69 | 1997 | 21.6 | .653 | 9.1 | .219 | 4.2 | 3.1 | 21.9 | 4.4 | 4.7\nGiannis Antetokounmpo | 31 | MIL | PF | 36 | 1039 | 32.6 | .658 | 5.0 | .231 | 9.5 | 3.0 | 37.0 | 3.9 | 1.1\nPayton Pritchard | 28 | BOS | PG | 79 | 2556 | 17.1 | .584 | 8.5 | .160 | 2.3 | 2.8 | 21.4 | 6.1 | 2.4\nLeBron James | 41 | LAL | SF | 60 | 1989 | 20.8 | .594 | 5.4 | .131 | 3.5 | 2.8 | 27.2 | 3.4 | 2.1\nTrey Murphy III | 25 | NOP | SF | 66 | 2341 | 18.3 | .613 | 6.0 | .123 | 2.4 | 2.6 | 22.7 | 4.5 | 1.6\nCollin Gillespie | 26 | PHO | PG | 80 | 2282 | 15.3 | .575 | 6.2 | .130 | 2.5 | 2.6 | 18.7 | 3.6 | 2.5\nDonovan Clingan | 21 | POR | C | 77 | 2094 | 20.9 | .604 | 8.6 | .197 | 3.0 | 2.6 | 16.9 | 5.1 | 3.5\nEvan Mobley | 24 | CLE | PF | 65 | 2074 | 20.0 | .596 | 6.5 | .150 | 3.0 | 2.6 | 22.6 | 3.6 | 2.9\nImmanuel Quickley | 26 | TOR | PG | 70 | 2231 | 16.6 | .577 | 6.5 | .140 | 2.4 | 2.5 | 21.1 | 3.9 | 2.6\nReed Sheppard | 21 | HOU | SG | 82 | 2147 | 15.7 | .564 | 5.9 | .131 | 2.7 | 2.5 | 21.7 | 2.7 | 3.1\nJosh Hart | 30 | NYK | SF | 66 | 1994 | 15.7 | .612 | 6.1 | .147 | 3.0 | 2.5 | 16.7 | 3.1 | 3.0\nStephen Curry | 37 | GSW | PG | 43 | 1329 | 22.6 | .637 | 4.1 | .147 | 5.4 | 2.5 | 32.5 | 2.9 | 1.2\nBam Adebayo | 28 | MIA | C | 73 | 2365 | 18.7 | .551 | 6.4 | .129 | 2.0 | 2.4 | 25.0 | 2.8 | 3.6\nDe'Aaron Fox | 28 | SAS | PG | 72 | 2231 | 17.6 | .578 | 6.4 | .138 | 2.2 | 2.4 | 24.9 | 3.4 | 3.0\nNeemias Queta | 26 | BOS | C | 76 | 1926 | 20.3 | .674 | 9.0 | .224 | 2.8 | 2.4 | 14.7 | 5.4 | 3.6\nAusar Thompson | 23 | DET | SF | 73 | 1896 | 16.2 | .545 | 5.7 | .144 | 3.1 | 2.4 | 16.9 | 1.5 | 4.1\nDesmond Bane | 27 | ORL | SG | 82 | 2756 | 16.9 | .607 | 7.4 | .129 | 1.4 | 2.3 | 23.3 | 4.8 | 2.6\nJulius Randle | 31 | MIN | PF | 79 | 2610 | 18.4 | .585 | 7.4 | .135 | 1.5 | 2.3 | 26.5 | 4.3 | 3.1\nOG Anunoby | 28 | NYK | PF | 67 | 2224 | 15.8 | .620 | 6.3 | .135 | 2.0 | 2.3 | 19.7 | 3.1 | 3.2\nDevin Booker | 29 | PHO | SG | 64 | 2146 | 20.4 | .585 | 6.5 | .145 | 2.2 | 2.3 | 32.2 | 4.7 | 1.8\nZion Williamson | 25 | NOP | PF | 62 | 1841 | 22.6 | .644 | 6.1 | .159 | 2.9 | 2.3 | 25.7 | 5.0 | 1.0\nJimmy Butler | 36 | GSW | SF | 38 | 1182 | 23.5 | .646 | 5.7 | .233 | 5.5 | 2.3 | 23.3 | 4.6 | 1.1\nNickeil Alexander-Walker | 27 | ATL | SG | 78 | 2603 | 16.5 | .610 | 6.3 | .116 | 1.2 | 2.1 | 23.9 | 3.6 | 2.7\nDyson Daniels | 22 | ATL | SG | 76 | 2520 | 15.5 | .542 | 6.6 | .126 | 1.3 | 2.1 | 16.0 | 3.1 | 3.6\nPaolo Banchero | 23 | ORL | PF | 72 | 2502 | 17.6 | .566 | 5.0 | .095 | 1.4 | 2.1 | 27.6 | 2.0 | 3.0\nAustin Reaves | 27 | LAL | SG | 51 | 1762 | 20.2 | .641 | 5.6 | .153 | 2.8 | 2.1 | 26.7 | 4.2 | 1.4\nSandro Mamukelashvili | 26 | TOR | C | 80 | 1751 | 18.1 | .637 | 6.2 | .169 | 2.8 | 2.1 | 18.8 | 3.6 | 2.5\nMichael Porter Jr. | 27 | BRK | SF | 52 | 1689 | 19.9 | .595 | 2.9 | .083 | 3.0 | 2.1 | 30.5 | 1.7 | 1.3\nJoel Embiid | 31 | PHI | C | 38 | 1201 | 24.7 | .605 | 4.1 | .163 | 4.9 | 2.1 | 33.5 | 2.8 | 1.3\nCooper Flagg | 19 | DAL | SF | 70 | 2344 | 17.9 | .548 | 3.8 | .078 | 1.4 | 2.0 | 26.9 | 1.3 | 2.5\nStephon Castle | 21 | SAS | PG | 68 | 2038 | 17.4 | .575 | 5.8 | .136 | 1.9 | 2.0 | 24.9 | 2.8 | 2.9\nJosh Giddey | 23 | CHI | PG | 54 | 1731 | 18.0 | .562 | 3.2 | .090 | 2.7 | 2.0 | 24.4 | 1.6 | 1.7\nLuke Kornet | 30 | SAS | C | 68 | 1430 | 17.9 | .689 | 6.9 | .231 | 3.6 | 2.0 | 10.3 | 4.6 | 2.3\nMiles Bridges | 27 | CHO | PF | 77 | 2387 | 16.0 | .570 | 6.5 | .130 | 1.1 | 1.9 | 22.3 | 4.0 | 2.5\nJulian Champagnie | 24 | SAS | SF | 82 | 2266 | 13.1 | .609 | 6.2 | .132 | 1.4 | 1.9 | 15.2 | 2.8 | 3.4\nRudy Gobert | 33 | MIN | C | 76 | 2380 | 17.5 | .664 | 8.7 | .176 | 1.0 | 1.8 | 12.9 | 4.8 | 4.0\nBrandon Miller | 23 | CHO | SF | 65 | 1968 | 16.8 | .574 | 4.5 | .109 | 1.6 | 1.8 | 28.1 | 2.1 | 2.4\nMoussa Diabaté | 24 | CHO | C | 73 | 1899 | 17.2 | .653 | 7.4 | .188 | 1.7 | 1.8 | 11.4 | 4.7 | 2.8\nCam Spencer | 25 | MEM | SG | 72 | 1714 | 17.2 | .649 | 5.3 | .149 | 2.2 | 1.8 | 17.2 | 4.7 | 0.6\nKel'el Ware | 21 | MIA | C | 77 | 1704 | 20.0 | .616 | 6.2 | .175 | 2.1 | 1.8 | 18.0 | 3.2 | 3.0\nJalen Suggs | 24 | ORL | PG | 57 | 1574 | 15.6 | .561 | 2.7 | .083 | 2.5 | 1.8 | 22.9 | 0.3 | 2.4\nJarrett Allen | 27 | CLE | C | 56 | 1519 | 22.1 | .669 | 6.4 | .202 | 2.6 | 1.8 | 19.9 | 4.2 | 2.2\nIsaiah Hartenstein | 27 | OKC | C | 47 | 1137 | 19.8 | .631 | 5.3 | .223 | 4.2 | 1.8 | 16.1 | 2.5 | 2.8\nBrandon Ingram | 28 | TOR | SF | 77 | 2604 | 16.3 | .573 | 4.6 | .084 | 0.5 | 1.7 | 26.9 | 1.5 | 3.1\nDonte DiVincenzo | 29 | MIN | SG | 82 | 2494 | 12.5 | .567 | 5.2 | .100 | 0.8 | 1.7 | 16.9 | 2.3 | 2.9\nAndrew Wiggins | 30 | MIA | SF | 68 | 2063 | 15.1 | .586 | 4.6 | .106 | 1.2 | 1.7 | 19.7 | 2.2 | 2.3\nCason Wallace | 22 | OKC | SG | 77 | 2046 | 12.1 | .538 | 5.2 | .122 | 1.2 | 1.7 | 14.5 | 1.0 | 4.2\nIsaiah Joe | 26 | OKC | SG | 71 | 1507 | 15.5 | .657 | 5.5 | .176 | 2.5 | 1.7 | 18.4 | 3.2 | 2.4\nRobert Williams | 28 | POR | C | 59 | 1008 | 22.1 | .722 | 4.5 | .213 | 4.5 | 1.7 | 13.1 | 2.4 | 2.0\nOnyeka Okongwu | 25 | ATL | C | 74 | 2297 | 15.9 | .592 | 6.0 | .125 | 0.7 | 1.6 | 19.5 | 2.6 | 3.3\n"
    },
    {
        "title": 'Player Adjusted Shooting — 2024-25 NBA Regular Season',
        "source": 'players_adjusted_shooting.txt',
        "last_updated": "2025",
        "content": 'NBA 2024-25 Regular Season — Player Adjusted Shooting (Top 75 by TS+)\nPlayer | Age | Team | Pos | G | MP | FG% | TS% | eFG% | FT% | FG+ | TS+ | eFG+ | FT+ | 3P+ | 3PAr\n------------------------------------------------------------------------------------------------------------------------\nHarrison Ingram | 23 | SAS | SF | 7 | 26 | .833 | .917 | .917 |  | 177 | 158 | 168 |  | 278 | .167\nVladislav Goldin | 24 | MIA | C | 9 | 24 | .750 | .788 | .750 | 1.000 | 159 | 136 | 137 | 128 |  | .000\nWalker Kessler | 24 | UTA | C | 5 | 154 | .703 | .786 | .784 | .700 | 149 | 135 | 144 | 89 | 209 | .216\nZyon Pullin | 24 | MIN | SG | 5 | 43 | .692 | .779 | .731 | 1.000 | 147 | 134 | 134 | 128 | 93 | .231\nJericho Sims | 27 | MIL | C | 67 | 1318 | .784 | .772 | .784 | .620 | 166 | 133 | 144 | 79 |  | .000\nHayden Gray | 22 | UTA |  | 1 | 25 | .667 | .773 | .667 | 1.000 | 142 | 133 | 122 | 128 | 0 | .333\nMason Plumlee | 35 | 2TM | C | 20 | 172 | .786 | .769 | .786 | .643 | 167 | 132 | 144 | 82 |  | .000\nCaleb Houstan | 23 | ATL | SF | 18 | 75 | .538 | .769 | .750 | 1.000 | 114 | 132 | 137 | 128 | 146 | .808\nRyan Kalkbrenner | 24 | CHO | C | 69 | 1479 | .753 | .762 | .753 | .716 | 160 | 131 | 138 | 91 | 0 | .017\nOlivier Sarr | 26 | CLE | C | 4 | 39 | .714 | .761 | .857 | .400 | 152 | 131 | 157 | 51 | 185 | .429\nJaxson Hayes | 25 | LAL | C | 66 | 1207 | .756 | .758 | .761 | .653 | 160 | 130 | 140 | 83 | 278 | .011\nEric Gordon | 37 | PHI | SG | 6 | 74 | .571 | .754 | .762 | .500 | 121 | 130 | 140 | 64 | 159 | .667\nNorchad Omier | 24 | LAC | PF | 6 | 24 | .700 | .751 | .700 | 1.000 | 149 | 129 | 128 | 128 | 0 | .100\nAlondes Williams | 26 | WAS | SG | 4 | 101 | .615 | .745 | .712 | .875 | 131 | 128 | 130 | 112 | 99 | .538\nJoe Ingles | 38 | MIN | SF | 27 | 153 | .593 | .735 | .722 | 1.000 | 126 | 126 | 132 | 128 | 122 | .593\nDaQuan Jeffries | 28 | SAC | SG | 3 | 60 | .579 | .731 | .684 | 1.000 | 123 | 126 | 125 | 128 | 124 | .474\nHaywood Highsmith | 29 | PHO | SF | 7 | 91 | .522 | .729 | .696 | .857 | 111 | 125 | 127 | 109 | 159 | .609\nRobert Williams | 28 | POR | C | 59 | 1008 | .708 | .722 | .726 | .596 | 150 | 124 | 133 | 76 | 109 | .092\nKadary Richmond | 24 | WAS | SG | 3 | 67 | .625 | .722 | .688 | 1.000 | 133 | 124 | 126 | 128 | 139 | .250\nRiley Minix | 25 | 2TM | SF | 9 | 58 | .609 | .720 | .696 | 1.000 | 129 | 124 | 127 | 128 | 93 | .522\nRocco Zikarsky | 19 | MIN | C | 5 | 36 | .625 | .717 | .625 | 1.000 | 133 | 123 | 115 | 128 |  | .000\nJakob Poeltl | 30 | TOR | C | 46 | 1149 | .700 | .698 | .700 | .602 | 149 | 120 | 128 | 77 |  | .000\nGoga Bitadze | 26 | ORL | C | 64 | 972 | .676 | .700 | .680 | .711 | 143 | 120 | 125 | 91 | 51 | .049\nKarlo Matković | 24 | NOP | PF | 62 | 912 | .604 | .696 | .685 | .732 | 128 | 120 | 126 | 93 | 117 | .383\nDwight Powell | 34 | DAL | C | 63 | 904 | .644 | .699 | .653 | .696 | 137 | 120 | 120 | 89 | 79 | .069\nOrlando Robinson | 25 | ORL | C | 4 | 25 | .600 | .700 | .700 |  | 127 | 120 | 128 |  | 139 | .400\nLuke Kornet | 30 | SAS | C | 68 | 1430 | .643 | .689 | .643 | .825 | 137 | 119 | 118 | 105 |  | .000\nYanic Konan Niederhäuser | 22 | LAC | C | 41 | 424 | .640 | .691 | .645 | .758 | 136 | 119 | 118 | 97 | 56 | .050\nCormac Ryan | 27 | MIL | SG | 11 | 271 | .520 | .692 | .652 | .923 | 110 | 119 | 119 | 118 | 127 | .578\nDaRon Holmes | 23 | DEN | PF | 25 | 210 | .508 | .692 | .672 | .786 | 108 | 119 | 123 | 100 | 124 | .738\nJalen Duren | 22 | DET | C | 70 | 1976 | .650 | .688 | .650 | .747 | 138 | 118 | 119 | 95 |  | .000\nLuke Kennard | 29 | 2TM | SG | 78 | 1681 | .533 | .689 | .665 | .913 | 113 | 118 | 122 | 117 | 133 | .553\nBlake Hinson | 26 | UTA | SF | 14 | 285 | .513 | .684 | .674 | .750 | 109 | 118 | 124 | 96 | 130 | .687\nMark Williams | 24 | PHO | C | 60 | 1416 | .644 | .680 | .645 | .771 | 137 | 117 | 118 | 99 | 278 | .002\nLuka Garza | 27 | BOS | C | 69 | 1118 | .577 | .682 | .654 | .769 | 123 | 117 | 120 | 98 | 120 | .356\nAnthony Gill | 33 | WAS | PF | 55 | 951 | .628 | .682 | .667 | .739 | 133 | 117 | 122 | 94 | 98 | .223\nMicah Potter | 27 | IND | C | 47 | 908 | .515 | .679 | .637 | .842 | 109 | 117 | 117 | 108 | 118 | .573\nJoan Beringer | 19 | MIN | PF | 40 | 314 | .663 | .683 | .663 | .703 | 141 | 117 | 122 | 90 |  | .000\nCharles Bassey | 25 | 4TM | C | 13 | 153 | .660 | .681 | .660 | .700 | 140 | 117 | 121 | 89 |  | .000\nDeandre Ayton | 27 | LAL | C | 72 | 1958 | .671 | .676 | .671 | .645 | 142 | 116 | 123 | 82 |  | .000\nNeemias Queta | 26 | BOS | C | 76 | 1926 | .653 | .674 | .654 | .703 | 139 | 116 | 120 | 90 | 35 | .016\nDaniel Gafford | 27 | DAL | C | 55 | 1195 | .655 | .677 | .655 | .683 | 139 | 116 | 120 | 87 |  | .000\nMitchell Robinson | 27 | NYK | C | 60 | 1175 | .723 | .676 | .723 | .408 | 154 | 116 | 133 | 52 |  | .000\nAntonio Reeves | 25 | CHO | SG | 10 | 68 | .500 | .675 | .675 |  | 106 | 116 | 124 |  | 150 | .650\nJayson Kent | 23 | POR | SF | 5 | 22 | .571 | .672 | .643 | 1.000 | 121 | 116 | 118 | 128 | 70 | .571\nNikola Jokić | 30 | DEN | C | 65 | 2265 | .569 | .670 | .618 | .831 | 121 | 115 | 113 | 106 | 106 | .261\nJarrett Allen | 27 | CLE | C | 56 | 1519 | .638 | .669 | .639 | .709 | 136 | 115 | 117 | 91 | 28 | .019\nZach Edey | 23 | MEM | C | 11 | 284 | .633 | .669 | .638 | .781 | 134 | 115 | 117 | 100 | 56 | .051\nDeAndre Jordan | 37 | NOP | C | 12 | 199 | .656 | .671 | .656 | .647 | 139 | 115 | 120 | 83 |  | .000\nZach Collins | 28 | CHI | C | 10 | 184 | .578 | .666 | .648 | .700 | 123 | 115 | 119 | 89 | 119 | .328\nDereck Lively II | 21 | DAL | C | 7 | 115 | .611 | .670 | .611 | .800 | 130 | 115 | 112 | 102 |  | .000\nEnrique Freeman | 25 | MIN | PF | 4 | 37 | .500 | .666 | .625 | .750 | 106 | 115 | 115 | 96 | 139 | .500\nMark Sears | 23 | MIL | PG | 7 | 26 | .462 | .666 | .615 | .750 | 98 | 115 | 113 | 96 | 139 | .615\nJacob Toppin | 25 | ATL | SF | 5 | 17 | .667 | .667 | .667 |  | 142 | 115 | 122 |  |  | .000\nTrentyn Flowers | 20 | CHI | SF | 2 | 6 | .667 | .667 | .667 |  | 142 | 115 | 122 |  | 0 | .333\nRudy Gobert | 33 | MIN | C | 76 | 2380 | .682 | .664 | .682 | .526 | 145 | 114 | 125 | 67 | 0 | .010\nShai Gilgeous-Alexander | 27 | OKC | PG | 68 | 2259 | .553 | .665 | .597 | .879 | 117 | 114 | 109 | 112 | 107 | .226\nMarvin Bagley III | 26 | 2TM | PF | 60 | 1201 | .618 | .661 | .647 | .660 | 131 | 114 | 119 | 84 | 128 | .127\nOlivier-Maxence Prosper | 23 | MEM | PF | 53 | 984 | .549 | .662 | .634 | .754 | 116 | 114 | 116 | 96 | 113 | .423\nIsaiah Joe | 26 | OKC | SG | 71 | 1507 | .455 | .657 | .622 | .894 | 97 | 113 | 114 | 114 | 118 | .788\nGiannis Antetokounmpo | 31 | MIL | PF | 36 | 1039 | .624 | .658 | .636 | .650 | 132 | 113 | 117 | 83 | 93 | .075\nDrew Timme | 25 | LAL | PF | 27 | 235 | .576 | .658 | .659 | .556 | 122 | 113 | 121 | 71 | 122 | .379\nChet Holmgren | 23 | OKC | PF | 69 | 1997 | .557 | .653 | .614 | .792 | 118 | 112 | 112 | 101 | 101 | .312\nMoussa Diabaté | 24 | CHO | C | 73 | 1899 | .631 | .653 | .632 | .659 | 134 | 112 | 116 | 84 | 139 | .006\nCam Spencer | 25 | MEM | SG | 72 | 1714 | .473 | .649 | .604 | .940 | 100 | 112 | 111 | 120 | 125 | .585\nIsaiah Jackson | 24 | 2TM | C | 55 | 908 | .637 | .653 | .637 | .648 | 135 | 112 | 117 | 83 | 0 | .013\nPaul Reed | 26 | DET | C | 65 | 901 | .617 | .650 | .636 | .664 | 131 | 112 | 117 | 85 | 90 | .118\nJosh Minott | 23 | 2TM | PF | 49 | 834 | .500 | .653 | .627 | .787 | 106 | 112 | 115 | 101 | 116 | .607\nSeth Curry | 35 | GSW | SG | 10 | 133 | .490 | .654 | .612 | .917 | 104 | 112 | 112 | 117 | 133 | .510\nSkal Labissière | 29 | WAS | PF | 3 | 38 | .600 | .650 | .650 |  | 127 | 112 | 119 |  | 56 | .500\nAlex Antetokounmpo | 24 | MIL | SF | 6 | 21 | .375 | .651 | .500 | .733 | 80 | 112 | 92 | 94 | 79 | .875\nJohn Collins | 28 | LAC | PF | 69 | 1872 | .552 | .643 | .620 | .766 | 117 | 111 | 114 | 98 | 113 | .332\nZion Williamson | 25 | NOP | PF | 62 | 1841 | .600 | .644 | .600 | .716 | 127 | 111 | 110 | 91 | 70 | .005\nSam Merrill | 29 | CLE | SG | 52 | 1377 | .461 | .644 | .624 | .855 | 98 | 111 | 114 | 109 | 117 | .775\nJimmy Butler | 36 | GSW | SF | 38 | 1182 | .519 | .646 | .554 | .864 | 110 | 111 | 102 | 110 | 105 | .184\n'
    },
    {
        "title": 'Player Shooting by Distance — 2024-25 NBA Regular Season',
        "source": 'players_shooting_stats.txt',
        "last_updated": "2025",
        "content": "NBA 2024-25 Regular Season — Player Shooting by Distance (Top 75 by FG%)\nPlayer | Age | Team | Pos | G | MP | FG% | Dist. | 0-3 | 3-10 | 10-16 | 16-3P | 3P | 3PAr | %3PA | 3P%\n------------------------------------------------------------------------------------------------------------------------\nHarrison Ingram | 23 | SAS | SF | 7 | 26 | .833 | 6.9 | 1.000 | .500 |  |  | 1.000 |  | .000 | \nMason Plumlee | 35 | 2TM | C | 20 | 172 | .786 | 1.2 | .917 | .000 |  |  |  |  |  | \nJericho Sims | 27 | MIL | C | 67 | 1318 | .784 | 2.2 | .861 | .444 | .429 |  |  |  |  | \nJaxson Hayes | 25 | LAL | C | 66 | 1207 | .756 | 2.8 | .797 | .662 | .333 |  | 1.000 |  | .000 | \nRyan Kalkbrenner | 24 | CHO | C | 69 | 1479 | .753 | 2.5 | .820 | .590 | .000 | .500 |  |  | .200 | .000\nVladislav Goldin | 24 | MIA | C | 9 | 24 | .750 | 1.5 | .667 | 1.000 |  |  |  |  |  | \nMitchell Robinson | 27 | NYK | C | 60 | 1175 | .723 | 1.0 | .732 | .600 | 1.000 |  |  |  |  | \nOlivier Sarr | 26 | CLE | C | 4 | 39 | .714 | 11.6 | .750 |  |  |  | 1.000 |  | .000 | \nRobert Williams | 28 | POR | C | 59 | 1008 | .708 | 4.5 | .807 | .589 | .375 | 1.000 | 1.000 |  | .826 | .421\nWalker Kessler | 24 | UTA | C | 5 | 154 | .703 | 7.5 | .647 | .750 |  |  | 1.000 |  | .250 | 1.000\nJakob Poeltl | 30 | TOR | C | 46 | 1149 | .700 | 3.7 | .809 | .576 | .457 |  |  |  |  | \nNorchad Omier | 24 | LAC | PF | 6 | 24 | .700 | 6.2 | .667 | 1.000 | 1.000 |  |  |  | .000 | \nZyon Pullin | 24 | MIN | SG | 5 | 43 | .692 | 10.2 | .750 | .833 |  |  | 1.000 |  | .000 | \nRudy Gobert | 33 | MIN | C | 76 | 2380 | .682 | 2.3 | .773 | .379 | .500 | .000 |  |  | .400 | .000\nGoga Bitadze | 26 | ORL | C | 64 | 972 | .676 | 3.3 | .756 | .558 | .333 | .000 | 1.000 |  | .364 | .250\nDeandre Ayton | 27 | LAL | C | 72 | 1958 | .671 | 5.3 | .842 | .590 | .420 | .563 |  |  |  | \nHayden Gray | 22 | UTA |  | 1 | 25 | .667 | 15.8 |  | 1.000 | 1.000 |  |  |  | .000 | \nJacob Toppin | 25 | ATL | SF | 5 | 17 | .667 | 4.8 | 1.000 | .000 | .500 |  |  |  |  | \nTrentyn Flowers | 20 | CHI | SF | 2 | 6 | .667 | 8.3 | 1.000 |  |  |  |  |  | 1.000 | .000\nJoan Beringer | 19 | MIN | PF | 40 | 314 | .663 | 1.9 | .734 | .368 |  |  |  |  |  | \nCharles Bassey | 25 | 4TM | C | 13 | 153 | .660 | 3.1 | .739 | .647 | .429 |  |  |  |  | \nDeAndre Jordan | 37 | NOP | C | 12 | 199 | .656 | 1.9 | .720 | .500 | .000 |  |  |  |  | \nDaniel Gafford | 27 | DAL | C | 55 | 1195 | .655 | 2.7 | .756 | .465 | .333 | .500 |  |  |  | \nNeemias Queta | 26 | BOS | C | 76 | 1926 | .653 | 3.9 | .768 | .567 | .273 | .000 | 1.000 |  | .125 | .000\nOso Ighodaro | 23 | PHO | PF | 82 | 1808 | .653 | 4.1 | .796 | .504 | .486 |  |  |  | .000 | \nJalen Duren | 22 | DET | C | 70 | 1976 | .650 | 3.7 | .749 | .570 | .160 | .529 |  |  |  | \nJosh Oduro | 25 | NOP | PF | 3 | 82 | .647 | 6.7 | .889 | .500 | .500 |  |  |  | .500 | .000\nMark Williams | 24 | PHO | C | 60 | 1416 | .644 | 3.3 | .730 | .496 | .545 | .000 | 1.000 |  | 1.000 | 1.000\nDwight Powell | 34 | DAL | C | 63 | 904 | .644 | 3.8 | .739 | .458 |  | 1.000 | 1.000 |  | .714 | .400\nLuke Kornet | 30 | SAS | C | 68 | 1430 | .643 | 2.0 | .693 | .385 | .571 | .000 |  |  |  | \nYanic Konan Niederhäuser | 22 | LAC | C | 41 | 424 | .640 | 4.1 | .702 | .629 | .333 |  | 1.000 |  | .200 | .000\nJarrett Allen | 27 | CLE | C | 56 | 1519 | .638 | 5.1 | .813 | .518 | .500 | .333 | 1.000 |  | .600 | .000\nIsaiah Jackson | 24 | 2TM | C | 55 | 908 | .637 | 3.2 | .738 | .526 | .167 | .000 |  |  | .667 | .000\nOscar Tshiebwe | 26 | UTA | C | 27 | 448 | .634 | 3.9 | .701 | .569 | .500 |  |  |  |  | \nZach Edey | 23 | MEM | C | 11 | 284 | .633 | 5.2 | .732 | .630 | .333 |  | 1.000 |  | .000 | \nMoussa Diabaté | 24 | CHO | C | 73 | 1899 | .631 | 3.1 | .749 | .465 | .077 | .333 | 1.000 |  | .500 | 1.000\nAnthony Gill | 33 | WAS | PF | 55 | 951 | .628 | 8.6 | .824 | .582 | .444 |  | 1.000 |  | .354 | .412\nKadary Richmond | 24 | WAS | SG | 3 | 67 | .625 | 9.1 | .857 | .250 | 1.000 |  | 1.000 |  | .250 | .000\nRocco Zikarsky | 19 | MIN | C | 5 | 36 | .625 | 5.7 | .750 | .667 |  | .000 |  |  |  | \nGiannis Antetokounmpo | 31 | MIL | PF | 36 | 1039 | .624 | 6.1 | .800 | .427 | .289 | .259 | .867 |  | .022 | 1.000\nIsaiah Hartenstein | 27 | OKC | C | 47 | 1137 | .622 | 4.8 | .768 | .484 | .488 |  |  |  | .286 | .000\nMarvin Bagley III | 26 | 2TM | PF | 60 | 1201 | .618 | 6.2 | .733 | .485 | .714 | 1.000 | 1.000 |  | .192 | .500\nPaul Reed | 26 | DET | C | 65 | 901 | .617 | 6.9 | .793 | .552 | .542 | .300 | .923 |  | .300 | .333\nAlondes Williams | 26 | WAS | SG | 4 | 101 | .615 | 15.6 | 1.000 | 1.000 |  | .000 | 1.000 |  | .286 | .250\nDereck Lively II | 21 | DAL | C | 7 | 115 | .611 | 2.0 | .643 | .500 |  |  |  |  |  | \nRiley Minix | 25 | 2TM | SF | 9 | 58 | .609 | 14.2 | 1.000 | .667 |  | 1.000 | 1.000 |  | .583 | .571\nKarlo Matković | 24 | NOP | PF | 62 | 912 | .604 | 10.8 | .848 | .405 | .000 |  | .974 |  | .522 | .489\nDay'Ron Sharpe | 24 | BRK | C | 62 | 1160 | .601 | 5.2 | .751 | .462 | .000 | .000 | 1.000 |  | .333 | .385\nZion Williamson | 25 | NOP | PF | 62 | 1841 | .600 | 3.6 | .744 | .407 | .452 | .000 | 1.000 |  | .000 | \nBismack Biyombo | 33 | SAS | C | 25 | 141 | .600 | 3.0 | .800 | .200 |  |  |  |  |  | \nTrey Jemison | 26 | NYK | C | 13 | 82 | .600 | 7.0 | 1.000 | .500 | .667 |  |  |  |  | \nJames Wiseman | 24 | IND | C | 4 | 58 | .600 | 4.6 | .800 | .500 |  | .000 |  |  |  | \nLawson Lovering | 22 | MEM | C | 2 | 49 | .600 | 2.6 | 1.000 | .200 |  |  |  |  |  | \nSkal Labissière | 29 | WAS | PF | 3 | 38 | .600 | 16.5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | .200 | .000\nHunter Sallis | 22 | PHI | SG | 7 | 26 | .600 | 5.4 | .500 | 1.000 | .000 |  |  |  |  | \nOrlando Robinson | 25 | ORL | C | 4 | 25 | .600 | 13.0 | 1.000 | .000 |  |  | 1.000 |  | .000 | \nIvica Zubac | 28 | 2TM | C | 48 | 1446 | .598 | 4.6 | .694 | .516 | .513 |  |  |  |  | \nDrew Eubanks | 28 | SAC | C | 42 | 552 | .596 | 6.4 | .837 | .519 | .524 | .000 | 1.000 |  | .714 | .200\nAdem Bona | 22 | PHI | C | 71 | 1234 | .595 | 2.7 | .669 | .391 | .000 |  | 1.000 |  | .833 | .200\nJoe Ingles | 38 | MIN | SF | 27 | 153 | .593 | 17.8 | .833 | 1.000 |  | .667 | .714 |  | .188 | .333\nJonathan Mogbo | 24 | TOR | PF | 40 | 249 | .591 | 3.6 | .625 | .563 | .667 |  |  |  | 1.000 | .000\nDylan Cardwell | 24 | SAC | C | 44 | 907 | .587 | 4.8 | .810 | .444 | .214 | .000 | 1.000 |  | .750 | .333\nGary Payton II | 33 | GSW | SG | 73 | 1142 | .583 | 9.6 | .812 | .541 | .250 |  | .946 |  | .480 | .328\nJonas Valančiūnas | 33 | DEN | C | 65 | 871 | .582 | 7.3 | .677 | .616 | .515 | .235 | .875 |  | .038 | 1.000\nCollin Murray-Boyles | 20 | TOR | PF | 57 | 1246 | .579 | 6.4 | .697 | .475 | .333 | .182 | 1.000 |  | .300 | .200\nDaQuan Jeffries | 28 | SAC | SG | 3 | 60 | .579 | 12.8 | .625 | 1.000 |  |  | 1.000 |  | .444 | .750\nZach Collins | 28 | CHI | C | 10 | 184 | .578 | 10.8 | .778 | .400 | .400 | 1.000 | 1.000 |  | .429 | .556\nLuka Garza | 27 | BOS | C | 69 | 1118 | .577 | 11.7 | .722 | .578 | .588 | .500 | .982 |  | .134 | .235\nDrew Timme | 25 | LAL | PF | 27 | 235 | .576 | 14.0 | .833 | .538 | .667 | .000 | 1.000 |  | .360 | .556\nMoussa Cisse | 23 | DAL | C | 38 | 530 | .574 | 1.8 | .648 | .292 |  |  |  |  |  | \nTrayce Jackson-Davis | 25 | 2TM | C | 53 | 496 | .573 | 2.3 | .682 | .286 |  |  | 1.000 |  | .000 | \nMaxime Raynaud | 22 | SAC | C | 74 | 1964 | .571 | 8.3 | .774 | .587 | .433 | .444 | .958 |  | .324 | .292\nNic Claxton | 26 | BRK | C | 69 | 1920 | .571 | 4.6 | .717 | .470 | .235 | .600 | 1.000 |  | .053 | .000\nEric Gordon | 37 | PHI | SG | 6 | 74 | .571 | 18.9 | .800 | .000 |  |  | 1.000 |  | .143 | 1.000\nWendell Moore Jr. | 24 | DET | SG | 6 | 60 | .571 | 5.8 | 1.000 | .500 |  |  |  |  | 1.000 | .000\n"
    },
    {
        "title": 'Player Per-36 Stats — 2024-25 NBA Regular Season',
        "source": 'players_per_thirtysix_stats.txt',
        "last_updated": "2025",
        "content": "NBA 2024-25 Regular Season — Player Per-36-Min Stats (Top 75 by PTS/36)\nPlayer | Age | Team | Pos | G | MP | PTS | AST | TRB | STL | BLK | FG% | 3P% | FT% | TOV | PF\n------------------------------------------------------------------------------------------------------------------------\nTristen Newton | 24 | HOU | SG | 1 | 12 | 36.0 | 0.0 | 9.0 | 3.0 | 0.0 | .444 | .400 | .667 | 0.0 | 0.0\nGiannis Antetokounmpo | 31 | MIL | PF | 36 | 1039 | 34.4 | 6.8 | 12.2 | 1.2 | 0.8 | .624 | .333 | .650 | 4.0 | 3.0\nLuka Dončić | 26 | LAL | PG | 64 | 2289 | 33.7 | 8.3 | 7.8 | 1.7 | 0.5 | .476 | .366 | .780 | 4.0 | 2.4\nShai Gilgeous-Alexander | 27 | OKC | PG | 68 | 2259 | 33.7 | 7.1 | 4.7 | 1.5 | 0.8 | .553 | .386 | .879 | 2.4 | 2.2\nAlex Antetokounmpo | 24 | MIL | SF | 6 | 21 | 32.6 | 1.7 | 10.3 | 0.0 | 1.7 | .375 | .286 | .733 | 1.7 | 1.7\nKawhi Leonard | 34 | LAC | SF | 65 | 2085 | 31.3 | 4.0 | 7.1 | 2.1 | 0.5 | .505 | .387 | .892 | 2.3 | 1.3\nTy Jerome | 28 | MEM | SG | 15 | 339 | 31.3 | 9.0 | 4.5 | 1.7 | 0.4 | .474 | .420 | .875 | 2.9 | 3.3\nVictor Wembanyama | 22 | SAS | C | 64 | 1866 | 30.9 | 3.8 | 14.2 | 1.3 | 3.8 | .512 | .349 | .827 | 3.0 | 3.0\nStephen Curry | 37 | GSW | PG | 43 | 1329 | 30.9 | 5.5 | 4.2 | 1.3 | 0.5 | .468 | .393 | .923 | 3.3 | 2.0\nJoel Embiid | 31 | PHI | C | 38 | 1201 | 30.6 | 4.5 | 8.8 | 0.6 | 1.4 | .489 | .333 | .854 | 3.3 | 2.5\nMark Sears | 23 | MIL | PG | 7 | 26 | 30.5 | 2.8 | 2.8 | 0.0 | 0.0 | .462 | .500 | .750 | 5.5 | 5.5\nJaylen Brown | 29 | BOS | SF | 71 | 2443 | 30.0 | 5.4 | 7.3 | 1.1 | 0.4 | .477 | .347 | .795 | 3.8 | 2.8\nDonovan Mitchell | 29 | CLE | SG | 70 | 2342 | 30.0 | 6.1 | 4.9 | 1.6 | 0.3 | .483 | .364 | .865 | 3.0 | 2.5\nAnthony Edwards | 24 | MIN | SG | 61 | 2137 | 29.6 | 3.8 | 5.1 | 1.4 | 0.8 | .489 | .399 | .796 | 2.9 | 2.0\nNikola Jokić | 30 | DEN | C | 65 | 2265 | 28.6 | 11.1 | 13.3 | 1.5 | 0.8 | .569 | .380 | .831 | 3.9 | 2.7\nDevin Booker | 29 | PHO | SG | 64 | 2146 | 28.0 | 6.5 | 4.2 | 0.9 | 0.3 | .456 | .330 | .873 | 3.4 | 2.8\nLauri Markkanen | 28 | UTA | PF | 42 | 1443 | 28.0 | 2.2 | 7.2 | 1.0 | 0.5 | .477 | .355 | .896 | 1.6 | 1.9\nTyrese Maxey | 25 | PHI | PG | 70 | 2661 | 26.8 | 6.2 | 3.9 | 1.8 | 0.7 | .462 | .367 | .892 | 2.3 | 2.0\nJalen Brunson | 29 | NYK | PG | 74 | 2590 | 26.8 | 7.0 | 3.4 | 0.8 | 0.1 | .467 | .369 | .841 | 2.4 | 2.4\nMichael Porter Jr. | 27 | BRK | SF | 52 | 1689 | 26.8 | 3.4 | 7.8 | 1.2 | 0.3 | .463 | .363 | .859 | 2.6 | 2.5\nNorman Powell | 32 | MIA | SG | 58 | 1717 | 26.4 | 3.0 | 4.3 | 1.3 | 0.3 | .470 | .380 | .827 | 2.3 | 2.7\nDeni Avdija | 25 | POR | SF | 66 | 2199 | 26.2 | 7.2 | 7.4 | 0.8 | 0.7 | .462 | .318 | .802 | 4.1 | 2.8\nPascal Siakam | 31 | IND | PF | 62 | 2057 | 26.0 | 4.1 | 7.2 | 1.2 | 0.4 | .484 | .358 | .693 | 2.4 | 2.6\nJamal Murray | 28 | DEN | PG | 75 | 2652 | 25.9 | 7.3 | 4.5 | 0.9 | 0.4 | .483 | .435 | .887 | 2.3 | 1.8\nLaMelo Ball | 24 | CHO | PG | 72 | 2017 | 25.8 | 9.2 | 6.2 | 1.6 | 0.3 | .407 | .368 | .899 | 3.6 | 3.5\nKevin Durant | 37 | HOU | SF | 78 | 2840 | 25.7 | 4.7 | 5.4 | 0.8 | 0.9 | .520 | .413 | .874 | 3.1 | 1.8\nKeyonte George | 22 | UTA | PG | 54 | 1786 | 25.7 | 6.7 | 4.0 | 1.2 | 0.3 | .456 | .371 | .892 | 3.3 | 2.4\nZion Williamson | 25 | NOP | PF | 62 | 1841 | 25.5 | 3.9 | 6.9 | 1.2 | 0.7 | .600 | .250 | .716 | 2.5 | 2.8\nShaedon Sharpe | 22 | POR | SG | 50 | 1471 | 25.5 | 3.2 | 5.3 | 1.7 | 0.1 | .452 | .337 | .787 | 3.6 | 2.4\nNorchad Omier | 24 | LAC | PF | 6 | 24 | 25.5 | 3.0 | 10.5 | 1.5 | 0.0 | .700 | .000 | 1.000 | 0.0 | 6.0\nCade Cunningham | 24 | DET | PG | 64 | 2172 | 25.4 | 10.5 | 5.9 | 1.5 | 0.9 | .461 | .342 | .812 | 3.9 | 3.2\nTrae Young | 27 | 2TM | PG | 15 | 384 | 25.2 | 11.3 | 2.8 | 1.2 | 0.2 | .458 | .338 | .825 | 3.7 | 3.1\nCoby White | 25 | 2TM | SG | 50 | 1249 | 25.0 | 5.8 | 4.9 | 0.7 | 0.1 | .446 | .362 | .817 | 3.7 | 3.2\nKristaps Porziņģis | 30 | 2TM | C | 32 | 769 | 25.0 | 3.8 | 7.8 | 0.8 | 1.8 | .446 | .338 | .842 | 1.9 | 4.3\nJalen Duren | 22 | DET | C | 70 | 1976 | 24.9 | 2.5 | 13.4 | 1.0 | 1.1 | .650 |  | .747 | 2.4 | 3.6\nFranz Wagner | 24 | ORL | SF | 34 | 1019 | 24.7 | 4.0 | 6.3 | 1.1 | 0.4 | .481 | .345 | .823 | 2.0 | 2.5\nJalen Green | 23 | PHO | SG | 32 | 828 | 24.7 | 4.0 | 5.0 | 1.5 | 0.3 | .422 | .313 | .747 | 3.1 | 1.8\nJa Morant | 26 | MEM | PG | 20 | 569 | 24.6 | 10.2 | 4.1 | 1.3 | 0.4 | .410 | .235 | .897 | 4.5 | 2.5\nJames Harden | 36 | 2TM | PG | 70 | 2438 | 24.4 | 8.2 | 5.0 | 1.2 | 0.4 | .434 | .375 | .884 | 3.6 | 2.0\nAustin Reaves | 27 | LAL | SG | 51 | 1762 | 24.3 | 5.7 | 4.9 | 1.1 | 0.4 | .490 | .360 | .871 | 3.1 | 2.2\nBrandon Miller | 23 | CHO | SF | 65 | 1968 | 24.0 | 4.0 | 5.8 | 1.2 | 0.8 | .435 | .383 | .892 | 3.0 | 3.3\nJayson Tatum | 27 | BOS | PF | 16 | 522 | 24.0 | 5.9 | 11.0 | 1.5 | 0.2 | .411 | .329 | .823 | 2.7 | 1.8\nTrentyn Flowers | 20 | CHI | SF | 2 | 6 | 24.0 | 6.0 | 6.0 | 0.0 | 0.0 | .667 | .000 |  | 6.0 | 6.0\nDillon Brooks | 30 | PHO | SF | 56 | 1702 | 23.9 | 2.1 | 4.3 | 1.2 | 0.2 | .435 | .344 | .842 | 2.1 | 3.9\nTristan Vukcevic | 22 | WAS | C | 49 | 671 | 23.7 | 3.0 | 7.9 | 1.2 | 1.8 | .479 | .347 | .784 | 3.3 | 4.4\nObi Toppin | 27 | IND | PF | 24 | 424 | 23.7 | 4.7 | 9.0 | 1.0 | 0.1 | .503 | .352 | .913 | 2.4 | 2.4\nTyler Herro | 26 | MIA | SG | 33 | 1033 | 23.6 | 4.7 | 5.5 | 0.8 | 0.4 | .480 | .378 | .917 | 2.2 | 2.2\nCollin Sexton | 27 | 2TM | SG | 68 | 1614 | 23.4 | 5.0 | 3.5 | 1.7 | 0.2 | .485 | .401 | .855 | 3.1 | 3.2\nJonas Valančiūnas | 33 | DEN | C | 65 | 871 | 23.4 | 3.2 | 13.6 | 0.6 | 1.5 | .582 | .308 | .772 | 3.0 | 5.5\nAnthony Davis | 32 | DAL | PF | 20 | 626 | 23.4 | 3.2 | 12.7 | 1.3 | 1.9 | .506 | .270 | .728 | 2.4 | 2.4\nKarl-Anthony Towns | 30 | NYK | C | 75 | 2322 | 23.3 | 3.5 | 13.8 | 1.0 | 0.6 | .501 | .368 | .858 | 2.9 | 4.0\nCam Thomas | 24 | 2TM | SG | 42 | 881 | 23.2 | 4.4 | 2.9 | 0.4 | 0.2 | .410 | .310 | .811 | 3.0 | 1.8\nJaren Jackson Jr. | 26 | 2TM | C | 48 | 1454 | 23.1 | 2.3 | 6.7 | 1.3 | 1.7 | .476 | .357 | .803 | 2.6 | 4.4\nJimmy Butler | 36 | GSW | SF | 38 | 1182 | 23.1 | 5.6 | 6.4 | 1.7 | 0.2 | .519 | .376 | .864 | 1.8 | 1.4\nJulius Randle | 31 | MIN | PF | 79 | 2610 | 23.0 | 5.5 | 7.3 | 1.2 | 0.2 | .481 | .315 | .802 | 3.0 | 3.1\nJalen Johnson | 24 | ATL | SF | 72 | 2532 | 23.0 | 8.0 | 10.5 | 1.3 | 0.4 | .489 | .352 | .788 | 3.5 | 2.1\nPaolo Banchero | 23 | ORL | PF | 72 | 2502 | 23.0 | 5.4 | 8.7 | 0.7 | 0.6 | .459 | .305 | .775 | 3.2 | 2.1\nBrandon Ingram | 28 | TOR | SF | 77 | 2604 | 22.9 | 3.9 | 5.9 | 0.8 | 0.8 | .477 | .382 | .820 | 2.6 | 2.0\nRJ Barrett | 25 | TOR | SF | 57 | 1726 | 22.9 | 3.9 | 6.3 | 0.9 | 0.4 | .491 | .339 | .717 | 2.1 | 3.1\nBrice Sensabaugh | 22 | UTA | SF | 75 | 1762 | 22.8 | 2.9 | 4.7 | 1.1 | 0.2 | .460 | .367 | .826 | 2.5 | 2.9\nLeBron James | 41 | LAL | SF | 60 | 1989 | 22.7 | 7.8 | 6.6 | 1.3 | 0.6 | .515 | .317 | .737 | 3.2 | 1.5\nCooper Flagg | 19 | DAL | SF | 70 | 2344 | 22.6 | 4.9 | 7.2 | 1.3 | 1.0 | .468 | .295 | .827 | 2.5 | 2.2\nCJ McCollum | 34 | 2TM | PG | 76 | 2264 | 22.6 | 4.7 | 4.0 | 1.0 | 0.6 | .455 | .375 | .772 | 2.1 | 2.3\nDarius Garland | 26 | 2TM | PG | 45 | 1344 | 22.6 | 8.1 | 2.9 | 1.2 | 0.2 | .460 | .396 | .861 | 3.5 | 2.3\nNickeil Alexander-Walker | 27 | ATL | SG | 78 | 2603 | 22.5 | 4.0 | 3.7 | 1.4 | 0.6 | .459 | .399 | .902 | 2.2 | 2.4\nJerami Grant | 31 | POR | PF | 57 | 1695 | 22.5 | 2.6 | 4.2 | 0.9 | 0.8 | .453 | .389 | .814 | 2.5 | 3.0\nBam Adebayo | 28 | MIA | C | 73 | 2365 | 22.3 | 3.5 | 11.1 | 1.3 | 0.7 | .442 | .318 | .778 | 1.8 | 1.9\nAlperen Şengün | 23 | HOU | C | 72 | 2398 | 22.0 | 6.7 | 9.6 | 1.3 | 1.2 | .519 | .305 | .691 | 3.4 | 3.6\nZach LaVine | 30 | SAC | SG | 39 | 1224 | 22.0 | 2.6 | 3.2 | 0.8 | 0.3 | .479 | .390 | .880 | 2.2 | 2.4\nTrey Murphy III | 25 | NOP | SF | 66 | 2341 | 21.8 | 3.8 | 5.8 | 1.5 | 0.4 | .470 | .379 | .886 | 1.8 | 2.1\nJalen Williams | 24 | OKC | SG | 33 | 936 | 21.7 | 7.0 | 5.8 | 1.5 | 0.3 | .484 | .299 | .837 | 2.4 | 2.5\nDejounte Murray | 29 | NOP | PG | 14 | 389 | 21.7 | 8.2 | 6.9 | 2.1 | 0.3 | .484 | .306 | .867 | 4.3 | 3.1\nRiley Minix | 25 | 2TM | SF | 9 | 58 | 21.7 | 3.1 | 3.7 | 1.2 | 0.6 | .609 | .333 | 1.000 | 1.9 | 3.1\nDe'Aaron Fox | 28 | SAS | PG | 72 | 2231 | 21.6 | 7.2 | 4.4 | 1.4 | 0.3 | .486 | .332 | .760 | 2.6 | 2.6\nAlex Sarr | 20 | WAS | C | 48 | 1305 | 21.6 | 3.6 | 9.8 | 1.1 | 2.6 | .482 | .333 | .692 | 2.3 | 2.9\n"
    },
    {
        "title": 'Player Play-Type and On-Off Stats — 2024-25 NBA Regular Season',
        "source": 'players_play_type.txt',
        "last_updated": "2025",
        "content": "NBA 2024-25 Regular Season — Player Play-Type & On/Off Stats (Top 75 by OnCourt +/-)\nPlayer | Age | Team | Pos | G | MP | PG% | SG% | SF% | PF% | C% | OnCourt | On-Off | BadPass | LostBall\n------------------------------------------------------------------------------------------------------------------------\nColby Jones | 23 | DET | SG | 1 | 7 | 0 | 0 | 100 | 0 | 0 | 89.5 | 81.5 | 0 | 0\nJacob Toppin | 25 | ATL | SF | 5 | 17 | 0 | 0 | 92 | 8 | 0 | 45.2 | 42.9 | 0 | 0\nWendell Moore Jr. | 24 | DET | SG | 6 | 60 | 1 | 25 | 66 | 8 | 0 | 36.1 | 28.3 | 1 | 0\nOlivier Sarr | 26 | CLE | C | 4 | 39 | 0 | 0 | 0 | 0 | 100 | 33.1 | 29.3 | 0 | 1\nEnrique Freeman | 25 | MIN | PF | 4 | 37 | 0 | 0 | 6 | 94 | 0 | 32.1 | 28.9 | 0 | 1\nN'Faly Dante | 24 | ATL | C | 4 | 15 | 0 | 0 | 0 | 0 | 100 | 30.3 | 28.0 | 1 | 0\nRiley Minix | 25 | 2TM | SF | 9 | 58 | 0 | 17 | 50 | 33 | 0 | 28.9 | 24.5 | 2 | 0\nTristan Enaruna | 24 | CLE | SF | 9 | 85 | 7 | 27 | 40 | 27 | 0 | 27.1 | 23.4 | 1 | 1\nMark Sears | 23 | MIL | PG | 7 | 26 | 100 | 0 | 0 | 0 | 0 | 23.5 | 30.2 | 4 | 0\nDavid Jones García | 24 | SAS | SF | 11 | 68 | 6 | 82 | 13 | 0 | 0 | 22.5 | 14.8 | 2 | 3\nChris Livingston | 22 | CLE | SF | 3 | 17 | 0 | 0 | 100 | 0 | 0 | 19.6 | 15.5 | 0 | 0\nAlondes Williams | 26 | WAS | SG | 4 | 101 | 0 | 53 | 43 | 4 | 0 | 18.9 | 31.3 | 4 | 0\nAlex Caruso | 31 | OKC | SG | 56 | 1020 | 1 | 24 | 52 | 24 | 0 | 18.5 | 9.7 | 32 | 12\nZach Edey | 23 | MEM | C | 11 | 284 | 0 | 0 | 0 | 0 | 100 | 17.7 | 25.4 | 6 | 8\nChet Holmgren | 23 | OKC | PF | 69 | 1997 | 0 | 0 | 1 | 37 | 62 | 16.7 | 10.9 | 25 | 48\nShai Gilgeous-Alexander | 27 | OKC | PG | 68 | 2259 | 100 | 0 | 0 | 0 | 0 | 16.5 | 12.3 | 81 | 48\nVictor Wembanyama | 22 | SAS | C | 64 | 1866 | 0 | 0 | 0 | 0 | 100 | 16.4 | 16.0 | 50 | 66\nIsaiah Hartenstein | 27 | OKC | C | 47 | 1137 | 0 | 0 | 0 | 0 | 100 | 15.7 | 6.2 | 42 | 14\nAjay Mitchell | 23 | OKC | SG | 57 | 1473 | 13 | 40 | 43 | 4 | 0 | 15.4 | 6.5 | 47 | 24\nCJ Huntley | 24 | PHO | SF | 4 | 40 | 0 | 0 | 69 | 31 | 0 | 15.3 | 13.8 | 0 | 0\nAntonio Reeves | 25 | CHO | SG | 10 | 68 | 12 | 66 | 22 | 0 | 0 | 15.2 | 10.7 | 1 | 0\nKeshad Johnson | 24 | MIA | SF | 32 | 280 | 0 | 5 | 39 | 54 | 2 | 14.2 | 12.6 | 8 | 2\nGrant Williams | 27 | CHO | PF | 36 | 711 | 0 | 0 | 0 | 88 | 12 | 13.3 | 10.5 | 17 | 3\nTristen Newton | 24 | HOU | SG | 1 | 12 | 0 | 53 | 48 | 0 | 0 | 13.3 | 8.2 | 0 | 0\nTrevor Keels | 22 | MIA | SG | 8 | 15 | 0 | 79 | 19 | 3 | 0 | 13.2 | 10.7 | 0 | 0\nCason Wallace | 22 | OKC | SG | 77 | 2046 | 19 | 67 | 13 | 0 | 0 | 13.0 | 3.7 | 44 | 21\nNeemias Queta | 26 | BOS | C | 76 | 1926 | 0 | 0 | 0 | 0 | 100 | 13.0 | 9.8 | 23 | 22\nAaron Gordon | 30 | DEN | PF | 36 | 1005 | 0 | 0 | 0 | 79 | 21 | 12.8 | 10.2 | 22 | 9\nJosh Green | 25 | CHO | SG | 58 | 908 | 20 | 66 | 12 | 1 | 0 | 12.6 | 10.2 | 20 | 7\nHugo González | 19 | BOS | SF | 74 | 1084 | 2 | 21 | 67 | 11 | 0 | 12.4 | 6.2 | 16 | 13\nJalen Williams | 24 | OKC | SG | 33 | 936 | 0 | 18 | 51 | 31 | 0 | 12.3 | 1.3 | 24 | 23\nAusar Thompson | 23 | DET | SF | 73 | 1896 | 1 | 65 | 27 | 7 | 0 | 12.2 | 7.5 | 52 | 38\nBuddy Boeheim | 26 | OKC | SF | 4 | 15 | 0 | 0 | 54 | 14 | 32 | 12.2 | 0.9 | 0 | 0\nJulian Champagnie | 24 | SAS | SF | 82 | 2266 | 0 | 5 | 52 | 43 | 1 | 12.1 | 9.7 | 43 | 14\nCurtis Jones | 24 | DEN | SG | 10 | 88 | 15 | 77 | 8 | 0 | 0 | 12.1 | 7.2 | 0 | 1\nSteven Adams | 32 | HOU | C | 32 | 730 | 0 | 0 | 0 | 0 | 100 | 11.3 | 7.4 | 14 | 8\nDerrick White | 31 | BOS | SG | 77 | 2625 | 22 | 75 | 4 | 0 | 0 | 11.2 | 9.7 | 77 | 43\nJalen Duren | 22 | DET | C | 70 | 1976 | 0 | 0 | 0 | 0 | 100 | 11.2 | 6.0 | 47 | 44\nJaylin Williams | 23 | OKC | PF | 65 | 1277 | 0 | 0 | 4 | 35 | 61 | 11.2 | -0.1 | 32 | 12\nJayson Tatum | 27 | BOS | PF | 16 | 522 | 0 | 0 | 0 | 95 | 5 | 11.2 | 3.8 | 17 | 14\nDevin Vassell | 25 | SAS | SG | 67 | 2044 | 16 | 54 | 31 | 0 | 0 | 11.1 | 6.4 | 34 | 17\nCade Cunningham | 24 | DET | PG | 64 | 2172 | 83 | 17 | 0 | 0 | 0 | 10.9 | 5.9 | 147 | 67\nIsaiah Joe | 26 | OKC | SG | 71 | 1507 | 55 | 46 | 0 | 0 | 0 | 10.8 | -0.7 | 23 | 14\nNikola Jokić | 30 | DEN | C | 65 | 2265 | 0 | 0 | 0 | 0 | 100 | 10.7 | 13.5 | 151 | 67\nMiles McBride | 25 | NYK | SG | 41 | 1080 | 33 | 67 | 0 | 0 | 0 | 10.6 | 5.2 | 13 | 12\nMoussa Diabaté | 24 | CHO | C | 73 | 1899 | 0 | 0 | 0 | 1 | 99 | 10.4 | 10.7 | 26 | 10\nChristian Braun | 24 | DEN | SG | 44 | 1399 | 0 | 42 | 42 | 16 | 0 | 10.4 | 8.2 | 21 | 11\nMarcus Sasser | 25 | DET | PG | 38 | 457 | 100 | 0 | 0 | 0 | 0 | 10.4 | 2.4 | 21 | 7\nDuncan Robinson | 31 | DET | SF | 77 | 2113 | 0 | 6 | 78 | 16 | 0 | 10.1 | 4.2 | 32 | 13\nCameron Johnson | 29 | DEN | SF | 54 | 1647 | 0 | 0 | 36 | 62 | 2 | 9.6 | 7.7 | 25 | 17\nDylan Harper | 19 | SAS | SG | 69 | 1558 | 9 | 60 | 31 | 0 | 0 | 9.6 | 2.7 | 43 | 34\nAlex Morales | 28 | ORL | PG | 4 | 24 | 45 | 55 | 0 | 0 | 0 | 9.6 | 9.1 | 1 | 2\nTobias Harris | 33 | DET | PF | 63 | 1746 | 0 | 0 | 0 | 99 | 1 | 9.4 | 2.1 | 23 | 25\nDe'Aaron Fox | 28 | SAS | PG | 72 | 2231 | 99 | 1 | 0 | 0 | 0 | 9.3 | 3.2 | 92 | 58\nStephon Castle | 21 | SAS | PG | 68 | 2038 | 48 | 52 | 0 | 0 | 0 | 9.3 | 2.8 | 115 | 71\nClint Capela | 31 | HOU | C | 75 | 926 | 0 | 0 | 0 | 13 | 87 | 9.2 | 5.2 | 21 | 5\nMyron Gardner | 24 | MIA | SF | 45 | 411 | 0 | 25 | 49 | 26 | 0 | 9.2 | 7.5 | 8 | 4\nPaul Reed | 26 | DET | C | 65 | 901 | 0 | 0 | 0 | 7 | 93 | 9.1 | 1.2 | 22 | 20\nMatisse Thybulle | 28 | POR | SG | 30 | 480 | 12 | 51 | 33 | 4 | 0 | 9.1 | 10.9 | 17 | 8\nOG Anunoby | 28 | NYK | PF | 67 | 2224 | 0 | 0 | 8 | 90 | 2 | 9.0 | 5.1 | 59 | 35\nSam Hauser | 28 | BOS | PF | 78 | 1934 | 0 | 0 | 20 | 77 | 3 | 9.0 | 2.2 | 17 | 11\nLuguentz Dort | 26 | OKC | SF | 69 | 1849 | 0 | 1 | 41 | 56 | 1 | 9.0 | -4.2 | 27 | 12\nAaron Holiday | 29 | HOU | PG | 57 | 781 | 100 | 0 | 0 | 0 | 0 | 8.9 | 4.7 | 20 | 9\nMitchell Robinson | 27 | NYK | C | 60 | 1175 | 0 | 0 | 0 | 24 | 76 | 8.7 | 2.6 | 13 | 11\nDillon Jones | 24 | NYK | SF | 7 | 39 | 0 | 19 | 73 | 8 | 0 | 8.6 | 1.8 | 0 | 0\nJavonte Green | 32 | DET | SG | 82 | 1446 | 8 | 60 | 31 | 2 | 0 | 8.4 | 0.3 | 35 | 7\nLandry Shamet | 28 | NYK | SG | 51 | 1171 | 3 | 85 | 12 | 0 | 0 | 8.4 | 2.3 | 20 | 6\nBrandon Miller | 23 | CHO | SF | 65 | 1968 | 0 | 3 | 87 | 10 | 0 | 8.2 | 7.1 | 81 | 47\nKawhi Leonard | 34 | LAC | SF | 65 | 2085 | 0 | 3 | 67 | 30 | 1 | 8.1 | 14.5 | 52 | 60\nLaMelo Ball | 24 | CHO | PG | 72 | 2017 | 94 | 6 | 0 | 0 | 0 | 7.7 | 6.3 | 114 | 49\nJimmy Butler | 36 | GSW | SF | 38 | 1182 | 0 | 6 | 39 | 54 | 0 | 7.6 | 11.6 | 34 | 16\nRasheer Fleming | 21 | PHO | PF | 55 | 673 | 0 | 0 | 3 | 86 | 11 | 7.6 | 7.1 | 14 | 10\nJamison Battle | 24 | TOR | SF | 61 | 517 | 0 | 0 | 56 | 43 | 1 | 7.6 | 5.3 | 7 | 7\nPayton Pritchard | 28 | BOS | PG | 79 | 2556 | 100 | 0 | 0 | 0 | 0 | 7.4 | -1.4 | 47 | 30\nMarcus Smart | 31 | LAL | SG | 62 | 1769 | 25 | 72 | 3 | 0 | 0 | 7.4 | 9.9 | 62 | 17\n"
    },
    {
        "title": 'Team Advanced Metrics — 2024-25 NBA Regular Season',
        "source": 'team_advanced_metrics.txt',
        "last_updated": "2025",
        "content": 'NBA 2024-25 Regular Season — Team Advanced Metrics (All 30 Teams)\nTeam | W | L | MOV | SRS | ORtg | DRtg | NRtg | Pace | TS% | FTr | 3PAr\n------------------------------------------------------------------------------------------------------------------------\nOklahoma City Thunder* | 64 | 18 | 11.15 | 11.04 | 118.9 | 107.7 | +11.2 | 99.3 | .599 | .261 | .426\nSan Antonio Spurs* | 62 | 20 | 8.30 | 8.28 | 119.6 | 111.3 | +8.3 | 99.9 | .595 | .274 | .422\nDetroit Pistons* | 60 | 22 | 8.16 | 7.53 | 117.9 | 109.7 | +8.2 | 99.3 | .583 | .292 | .345\nBoston Celtics* | 56 | 26 | 7.70 | 7.37 | 120.8 | 112.7 | +8.1 | 94.8 | .583 | .207 | .467\nNew York Knicks* | 53 | 29 | 6.33 | 6.05 | 119.8 | 113.3 | +6.5 | 96.8 | .590 | .238 | .428\nHouston Rockets* | 52 | 30 | 5.22 | 4.87 | 118.6 | 113.2 | +5.4 | 96.1 | .576 | .260 | .350\nDenver Nuggets* | 54 | 28 | 5.15 | 4.97 | 122.6 | 117.4 | +5.2 | 98.4 | .616 | .294 | .408\nCharlotte Hornets* | 44 | 38 | 4.83 | 4.48 | 119.4 | 114.4 | +5.0 | 96.8 | .589 | .244 | .487\nCleveland Cavaliers* | 52 | 30 | 4.11 | 3.78 | 119.2 | 115.1 | +4.1 | 99.9 | .595 | .265 | .442\nMinnesota Timberwolves* | 49 | 33 | 3.35 | 3.07 | 116.8 | 113.5 | +3.3 | 100.5 | .592 | .285 | .420\nToronto Raptors* | 46 | 36 | 2.83 | 2.75 | 115.9 | 113.0 | +2.9 | 98.4 | .581 | .265 | .363\nAtlanta Hawks* | 46 | 36 | 2.41 | 2.38 | 116.1 | 113.7 | +2.4 | 101.7 | .584 | .234 | .429\nMiami Heat* | 43 | 39 | 2.33 | 2.35 | 116.7 | 114.5 | +2.2 | 103.4 | .580 | .268 | .406\nLos Angeles Lakers* | 53 | 29 | 1.76 | 1.68 | 118.2 | 116.4 | +1.8 | 98.3 | .609 | .320 | .394\nPhoenix Suns* | 45 | 37 | 1.46 | 1.75 | 115.4 | 113.9 | +1.5 | 97.2 | .568 | .225 | .453\nLos Angeles Clippers* | 42 | 40 | 1.13 | 0.96 | 117.3 | 116.1 | +1.2 | 96.5 | .602 | .295 | .404\nOrlando Magic* | 45 | 37 | 0.63 | 0.81 | 114.9 | 114.3 | +0.6 | 100.0 | .576 | .311 | .386\nPhiladelphia 76ers* | 45 | 37 | -0.18 | -0.27 | 115.4 | 115.5 | -0.1 | 99.4 | .572 | .275 | .391\nPortland Trail Blazers* | 42 | 40 | -0.29 | -0.29 | 114.4 | 114.7 | -0.3 | 100.5 | .570 | .280 | .469\nGolden State Warriors* | 37 | 45 | -0.56 | -0.34 | 115.0 | 115.6 | -0.6 | 99.0 | .584 | .238 | .497\nNew Orleans Pelicans | 26 | 56 | -4.50 | -4.26 | 114.4 | 118.9 | -4.5 | 100.3 | .568 | .284 | .355\nChicago Bulls | 31 | 51 | -5.21 | -5.09 | 113.0 | 118.1 | -5.1 | 102.5 | .580 | .246 | .443\nDallas Mavericks | 26 | 56 | -5.51 | -5.33 | 111.2 | 116.5 | -5.3 | 101.7 | .564 | .287 | .355\nMemphis Grizzlies | 25 | 57 | -6.01 | -5.83 | 112.9 | 118.8 | -5.9 | 101.3 | .570 | .251 | .436\nMilwaukee Bucks | 32 | 50 | -6.21 | -6.12 | 112.9 | 119.3 | -6.4 | 97.6 | .589 | .223 | .457\nIndiana Pacers | 19 | 63 | -7.99 | -7.64 | 110.9 | 118.8 | -7.9 | 101.0 | .568 | .252 | .422\nUtah Jazz | 22 | 60 | -8.43 | -8.05 | 114.1 | 122.3 | -8.2 | 102.3 | .575 | .277 | .402\nSacramento Kings | 22 | 60 | -10.00 | -9.58 | 111.4 | 121.5 | -10.1 | 99.2 | .560 | .256 | .339\nBrooklyn Nets | 20 | 62 | -9.99 | -9.70 | 108.7 | 119.0 | -10.3 | 97.1 | .559 | .272 | .455\nWashington Wizards | 17 | 65 | -11.98 | -11.62 | 111.0 | 122.7 | -11.7 | 101.4 | .566 | .235 | .403\n'
    },
    {
        "title": 'Team Per-Game Stats — 2024-25 NBA Regular Season',
        "source": 'team_per_game_stats.txt',
        "last_updated": "2025",
        "content": 'NBA 2024-25 Regular Season — Team Per-Game Stats (All 30 Teams)\nTeam | G | PTS | FG | FGA | FG% | 3P | 3PA | 3P% | FT | FTA | FT% | TRB | AST | STL | BLK | TOV | PF\n------------------------------------------------------------------------------------------------------------------------\nDenver Nuggets* | 82 | 122.1 | 43.5 | 87.8 | .496 | 14.2 | 35.8 | .396 | 20.8 | 25.8 | .808 | 44.0 | 29.0 | 6.8 | 4.0 | 12.9 | 19.1\nMiami Heat* | 82 | 120.9 | 43.7 | 93.3 | .468 | 13.7 | 37.9 | .361 | 19.8 | 25.0 | .792 | 46.3 | 29.0 | 8.6 | 4.3 | 13.7 | 18.9\nSan Antonio Spurs* | 82 | 119.8 | 43.4 | 89.9 | .483 | 13.6 | 37.9 | .359 | 19.3 | 24.6 | .787 | 47.0 | 28.1 | 7.5 | 5.5 | 13.5 | 18.9\nCleveland Cavaliers* | 82 | 119.5 | 43.3 | 90.0 | .482 | 14.3 | 39.8 | .360 | 18.5 | 23.9 | .776 | 44.4 | 28.3 | 8.5 | 5.0 | 14.0 | 19.6\nOklahoma City Thunder* | 82 | 119.0 | 43.1 | 89.1 | .484 | 13.8 | 37.9 | .365 | 19.0 | 23.2 | .817 | 44.1 | 25.8 | 9.7 | 5.5 | 12.6 | 19.0\nAtlanta Hawks* | 82 | 118.5 | 43.6 | 92.0 | .474 | 14.6 | 39.5 | .371 | 16.6 | 21.5 | .774 | 43.5 | 30.1 | 9.4 | 4.7 | 14.2 | 19.7\nMinnesota Timberwolves* | 82 | 118.0 | 42.6 | 88.5 | .481 | 13.8 | 37.2 | .370 | 19.0 | 25.3 | .752 | 44.1 | 26.1 | 8.7 | 5.8 | 14.8 | 21.2\nDetroit Pistons* | 82 | 117.8 | 43.4 | 89.5 | .485 | 11.0 | 30.9 | .356 | 19.9 | 26.1 | .763 | 45.6 | 27.8 | 10.4 | 6.4 | 15.1 | 22.0\nUtah Jazz | 82 | 117.6 | 42.5 | 91.1 | .467 | 12.6 | 36.7 | .345 | 19.9 | 25.3 | .786 | 43.8 | 29.6 | 8.9 | 3.8 | 15.5 | 21.1\nNew York Knicks* | 82 | 116.5 | 42.7 | 89.4 | .478 | 14.2 | 38.2 | .373 | 16.8 | 21.3 | .792 | 45.6 | 27.4 | 8.1 | 3.9 | 13.6 | 20.0\nChicago Bulls | 82 | 116.3 | 42.4 | 90.5 | .469 | 14.3 | 40.1 | .356 | 17.3 | 22.2 | .776 | 45.0 | 28.5 | 7.6 | 5.0 | 15.3 | 19.0\nLos Angeles Lakers* | 82 | 116.3 | 42.0 | 83.7 | .502 | 11.8 | 32.9 | .359 | 20.4 | 26.8 | .763 | 41.0 | 25.9 | 8.5 | 4.3 | 14.5 | 18.5\nCharlotte Hornets* | 82 | 116.0 | 40.9 | 88.9 | .460 | 16.4 | 43.3 | .378 | 17.8 | 21.7 | .818 | 46.1 | 26.3 | 7.0 | 4.5 | 15.4 | 19.1\nPhiladelphia 76ers* | 82 | 115.9 | 41.7 | 90.3 | .462 | 12.3 | 35.3 | .349 | 20.1 | 24.8 | .809 | 43.6 | 24.6 | 9.1 | 5.7 | 13.6 | 20.3\nOrlando Magic* | 82 | 115.7 | 41.0 | 88.3 | .464 | 11.7 | 34.1 | .343 | 22.0 | 27.5 | .801 | 43.4 | 26.5 | 8.5 | 4.7 | 14.2 | 21.0\nNew Orleans Pelicans | 82 | 115.5 | 42.1 | 90.5 | .466 | 11.1 | 32.1 | .347 | 20.1 | 25.7 | .784 | 43.9 | 25.1 | 8.9 | 5.2 | 14.2 | 19.9\nPortland Trail Blazers* | 82 | 115.5 | 40.9 | 90.1 | .453 | 14.5 | 42.2 | .343 | 19.3 | 25.2 | .765 | 46.0 | 25.1 | 8.3 | 5.5 | 17.3 | 19.9\nHouston Rockets* | 82 | 115.2 | 43.0 | 89.8 | .479 | 11.5 | 31.5 | .364 | 17.8 | 23.4 | .763 | 48.1 | 25.4 | 8.5 | 5.8 | 15.4 | 18.3\nBoston Celtics* | 82 | 114.9 | 42.1 | 90.2 | .467 | 15.5 | 42.1 | .367 | 15.1 | 18.7 | .807 | 46.4 | 24.6 | 7.1 | 5.0 | 12.4 | 19.0\nMemphis Grizzlies | 82 | 114.7 | 41.3 | 90.6 | .456 | 13.9 | 39.5 | .353 | 18.0 | 22.8 | .793 | 42.1 | 27.8 | 8.8 | 4.8 | 15.1 | 20.5\nGolden State Warriors* | 82 | 114.6 | 40.9 | 88.8 | .461 | 15.7 | 44.1 | .356 | 17.0 | 21.1 | .807 | 42.3 | 28.9 | 9.7 | 4.2 | 15.7 | 19.6\nToronto Raptors* | 82 | 114.6 | 42.6 | 88.4 | .482 | 11.4 | 32.1 | .354 | 18.0 | 23.4 | .769 | 42.1 | 29.5 | 8.8 | 4.8 | 13.7 | 20.9\nDallas Mavericks | 82 | 114.1 | 41.9 | 89.8 | .466 | 11.0 | 31.9 | .344 | 19.5 | 25.8 | .755 | 44.7 | 25.3 | 7.5 | 5.2 | 14.5 | 18.4\nLos Angeles Clippers* | 82 | 113.8 | 40.5 | 83.6 | .484 | 12.4 | 33.8 | .368 | 20.3 | 24.7 | .823 | 40.6 | 23.7 | 9.0 | 4.9 | 14.3 | 18.9\nWashington Wizards | 82 | 112.9 | 41.8 | 90.4 | .463 | 13.0 | 36.4 | .359 | 16.2 | 21.2 | .766 | 41.9 | 25.0 | 8.0 | 5.5 | 15.7 | 21.3\nPhoenix Suns* | 82 | 112.6 | 41.0 | 90.1 | .455 | 14.8 | 40.8 | .361 | 15.8 | 20.3 | .781 | 43.1 | 24.6 | 9.5 | 4.2 | 14.5 | 20.6\nIndiana Pacers | 82 | 112.4 | 40.8 | 89.1 | .458 | 13.4 | 37.6 | .356 | 17.4 | 22.5 | .776 | 41.9 | 27.7 | 7.4 | 4.6 | 14.5 | 20.9\nSacramento Kings | 82 | 111.0 | 41.6 | 89.0 | .467 | 10.3 | 30.2 | .340 | 17.6 | 22.8 | .772 | 42.3 | 25.5 | 8.0 | 4.5 | 14.3 | 19.7\nMilwaukee Bucks | 82 | 110.6 | 40.8 | 85.5 | .477 | 15.1 | 39.1 | .387 | 13.9 | 19.1 | .730 | 40.7 | 25.8 | 7.4 | 4.0 | 15.2 | 19.7\nBrooklyn Nets | 82 | 105.9 | 37.5 | 84.5 | .443 | 13.1 | 38.4 | .340 | 17.9 | 23.0 | .780 | 39.3 | 25.0 | 8.0 | 4.3 | 15.8 | 21.1\n'
    },
    {
        "title": 'Team Shooting by Distance — 2024-25 NBA Regular Season',
        "source": 'team_shooting_distances_stats.txt',
        "last_updated": "2025",
        "content": 'NBA 2024-25 Regular Season — Team Shooting by Distance (All 30 Teams)\nTeam | FG% | Dist. | 0-3 | 3-10 | 10-16 | 16-3P | 3P | 2P | 3P\n------------------------------------------------------------------------------------------------------------------------\nAtlanta Hawks* | .474 | 14.2 | .687 | .450 | .474 | .405 | .909 | .581 | .909\nBoston Celtics* | .467 | 16.2 | .754 | .490 | .481 | .382 | .794 | .463 | .794\nBrooklyn Nets | .443 | 14.5 | .686 | .419 | .410 | .347 | .891 | .548 | .891\nChicago Bulls | .469 | 13.7 | .659 | .429 | .486 | .359 | .906 | .553 | .906\nCharlotte Hornets* | .460 | 15.5 | .706 | .430 | .408 | .363 | .832 | .518 | .832\nCleveland Cavaliers* | .482 | 14.4 | .708 | .484 | .480 | .375 | .814 | .574 | .814\nDallas Mavericks | .466 | 13.4 | .715 | .455 | .428 | .368 | .888 | .503 | .888\nDenver Nuggets* | .496 | 14.4 | .720 | .491 | .486 | .424 | .829 | .586 | .829\nDetroit Pistons* | .485 | 12.4 | .696 | .487 | .413 | .398 | .863 | .565 | .863\nGolden State Warriors* | .461 | 15.7 | .736 | .483 | .418 | .365 | .886 | .593 | .886\nHouston Rockets* | .479 | 13.7 | .700 | .492 | .439 | .422 | .859 | .494 | .859\nIndiana Pacers | .458 | 14.8 | .677 | .466 | .453 | .376 | .907 | .566 | .907\nLos Angeles Clippers* | .484 | 14.3 | .732 | .495 | .470 | .379 | .789 | .496 | .789\nLos Angeles Lakers* | .502 | 14.4 | .778 | .531 | .471 | .433 | .735 | .569 | .735\nMemphis Grizzlies | .456 | 14.7 | .673 | .482 | .446 | .354 | .851 | .583 | .851\nMiami Heat* | .468 | 13.7 | .683 | .463 | .432 | .369 | .923 | .544 | .923\nMilwaukee Bucks | .477 | 15.5 | .746 | .425 | .440 | .401 | .868 | .494 | .868\nMinnesota Timberwolves* | .481 | 14.2 | .715 | .454 | .464 | .386 | .824 | .511 | .824\nNew Orleans Pelicans | .466 | 12.4 | .671 | .405 | .386 | .338 | .836 | .510 | .836\nNew York Knicks* | .478 | 14.4 | .689 | .464 | .449 | .439 | .860 | .534 | .860\nOklahoma City Thunder* | .484 | 14.6 | .720 | .441 | .524 | .447 | .860 | .474 | .860\nOrlando Magic* | .464 | 13.5 | .705 | .438 | .415 | .416 | .856 | .564 | .856\nPhiladelphia 76ers* | .462 | 14.0 | .670 | .434 | .413 | .444 | .805 | .499 | .805\nPhoenix Suns* | .455 | 15.8 | .690 | .445 | .449 | .425 | .808 | .485 | .808\nPortland Trail Blazers* | .453 | 14.7 | .699 | .424 | .380 | .405 | .829 | .499 | .829\nSacramento Kings | .467 | 13.9 | .729 | .503 | .423 | .402 | .849 | .538 | .849\nSan Antonio Spurs* | .483 | 13.9 | .704 | .468 | .422 | .467 | .920 | .521 | .920\nToronto Raptors* | .482 | 13.4 | .708 | .444 | .458 | .426 | .942 | .600 | .942\nUtah Jazz | .467 | 14.0 | .687 | .521 | .430 | .402 | .882 | .618 | .882\nWashington Wizards | .463 | 14.6 | .732 | .402 | .460 | .419 | .817 | .500 | .817\n'
    }
]



# ── RAG Engine ───────────────────────────────────────────────────────────────
ALIASES = {
    "defensive rating": "DRtg", "offensive rating": "ORtg",
    "net rating": "NRtg", "true shooting": "TS%",
    "player efficiency": "PER", "value over replacement": "VORP",
    "box plus minus": "BPM", "win shares": "WS",
    "usage rate": "USG%", "three point": "3P%",
    "field goal": "FG%", "points per game": "PPG",
    "rebounds per game": "RPG", "assists per game": "APG",
    "two way": "BPM DRtg STL BLK", "two-way": "BPM DRtg STL BLK",
}

def expand_query(query):
    q = query.lower()
    expansions = [abbr for phrase, abbr in ALIASES.items() if phrase in q]
    return query + (" " + " ".join(expansions) if expansions else "")

def chunk_documents(documents, chunk_size=300, chunk_overlap=50):
    chunks = []
    for doc in documents:
        lines = doc["content"].splitlines()
        prose_words, table_rows, in_table = [], [], False
        def flush_prose():
            if not prose_words: return
            for i in range(0, len(prose_words), chunk_size - chunk_overlap):
                text = " ".join(prose_words[i: i + chunk_size])
                if text.strip(): chunks.append({**doc, "content": text})
            prose_words.clear()
        def flush_table():
            if not table_rows: return
            for i in range(0, len(table_rows), 10):
                text = "\n".join(table_rows[i: i + 10])
                chunks.append({**doc, "content": text})
            table_rows.clear()
        for line in lines:
            if " | " in line or line.startswith("---"):
                if not in_table: flush_prose(); in_table = True
                table_rows.append(line)
            else:
                if in_table: flush_table(); in_table = False
                prose_words.extend(line.split())
        flush_prose(); flush_table()
    return chunks

def search_documents(query, chunks, top_n=5):
    if not chunks: return []
    expanded = expand_query(query)
    corpus = [c["content"] for c in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([expanded])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    years = []
    for c in chunks:
        try: years.append(int(c.get("last_updated", "0")))
        except: years.append(0)
    min_y, max_y = min(years), max(years)
    year_range = max(max_y - min_y, 1)
    recency_boost = np.array([(y - min_y) / year_range * 0.10 for y in years])
    adjusted = similarities + recency_boost
    top_indices = adjusted.argsort()[-top_n:][::-1]
    results = []
    for i in top_indices:
        result = dict(chunks[i])
        result["relevance_score"] = float(adjusted[i])
        results.append(result)
    return results

SYSTEM_PROMPT = """You are an NBA basketball analytics assistant powered by a RAG knowledge base.
Rules:
1. ALWAYS call search_my_docs before answering — never answer from memory.
2. Call search_my_docs MULTIPLE TIMES if the question spans multiple documents.
3. Prefer documents with the most recent last_updated date.
4. Always cite your source document and last_updated date.
5. If the answer is not in any document say: "I don't have that information in my knowledge base." Do NOT hallucinate.
6. Format comparisons as clear tables or structured lists.
"""

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = chunk_documents(my_documents)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('''
    <div class="dd-sidebar-name">Declan Davis</div>
    <div class="dd-sidebar-sub">2025-26 Season · RAG</div>
    ''', unsafe_allow_html=True)

    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")

    st.markdown('''<div class="dd-stat"><div class="dd-stat-label">Documents</div><div class="dd-stat-val">9</div></div>''', unsafe_allow_html=True)
    st.markdown('''<div class="dd-stat"><div class="dd-stat-label">Chunks</div><div class="dd-stat-val">{}</div></div>'''.format(len(st.session_state.chunks)), unsafe_allow_html=True)
    st.markdown('''<div class="dd-stat"><div class="dd-stat-label">Players</div><div class="dd-stat-val">75</div></div>''', unsafe_allow_html=True)
    st.markdown('''<div class="dd-stat"><div class="dd-stat-label">Teams</div><div class="dd-stat-val">30</div></div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''
    <div class="dd-stat-label" style="margin-bottom:0.5rem;">Stack</div>
    <span class="dd-tag">Python</span>
    <span class="dd-tag">RAG</span>
    <span class="dd-tag">Claude AI</span>
    <span class="dd-tag">TF-IDF</span>
    <span class="dd-tag">Streamlit</span>
    ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''
    <div class="dd-stat-label" style="margin-bottom:0.4rem;">Data Source</div>
    <div style="font-size:0.75rem;color:#F0F0F0;">Basketball Reference<br>2025-26 Regular Season</div>
    ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

    st.markdown('''
    <div style="margin-top:1.5rem;border-top:1px solid #222;padding-top:1rem;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#F0F0F0;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.5rem;">Links</div>
    <a href="https://declandavis03-max.github.io/" style="display:block;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4ADE80;text-decoration:none;margin-bottom:0.3rem;">← Portfolio</a>
    <a href="https://nba-ppg-predictor-aash2dexwqcmfme7dshg77.streamlit.app" style="display:block;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4ADE80;text-decoration:none;margin-bottom:0.3rem;">↗ Live App</a>
    <a href="https://twitter.com/declandavis17" style="display:block;font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#4ADE80;text-decoration:none;">𝕏 @declandavis17</a>
    </div>
    ''', unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="dd-header">
    <div class="dd-eyebrow">04 // AI Project · RAG · Python</div>
    <div class="dd-title">NBA Analytics <em>Agent</em></div>
    <div class="dd-desc">
        A retrieval-augmented generation system built on 9 Basketball Reference datasets —
        75 players, 30 teams, 2025-26 regular season. Ask any natural language question
        and get answers grounded in real statistics.
    </div>
</div>
''', unsafe_allow_html=True)

EXAMPLES = [
    "Which team shoots the most 3s and how efficiently?",
    "Who is the most efficient scorer under 25?",
    "Which team has the best net rating?",
]

col_main, col_key = st.columns([3, 2], gap="large")

with col_main:
    st.markdown('<div class="dd-section">Ask a Question</div>', unsafe_allow_html=True)
    user_input = st.text_input(
        label="q", label_visibility="collapsed",
        placeholder="e.g. Who leads the NBA in assists per game?",
        key="user_input",
    )
    ask_btn = st.button("Run Agent →")

with col_key:
    st.markdown('<div class="dd-section">API Key</div>', unsafe_allow_html=True)
    st.markdown('''<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:#F0F0F0;margin-bottom:0.7rem;padding:0.5rem 0.8rem;border:1px solid #222222;border-left:2px solid #4ADE80;border-radius:4px;line-height:1.5;">
    🔑 Get a free key at <a href="https://console.anthropic.com" target="_blank" style="color:#4ADE80;text-decoration:none;">console.anthropic.com</a><br>then paste it in the sidebar →
    </div>''', unsafe_allow_html=True)

# ── Agent ─────────────────────────────────────────────────────────────────────
def run_agent(question, api_key, chunks):
    client = anthropic.Anthropic(api_key=api_key)
    tools = [{
        "name": "search_my_docs",
        "description": "Search the NBA stats knowledge base. Call multiple times with different queries if needed.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    }]

    def search_my_docs(query):
        results = search_documents(query, chunks, top_n=5)
        if not results: return "No relevant documents found."
        return "\n".join(
            f"[Source: {r['title']} | Updated: {r['last_updated']}]\n{r['content']}\n---"
            for r in results
        )

    messages = [{"role": "user", "content": question}]
    search_log = []

    for _ in range(6):
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        )
        text_parts = [b.text for b in response.content if isinstance(b, TextBlock)]
        tool_calls = [b for b in response.content if isinstance(b, ToolUseBlock)]

        if response.stop_reason == "end_turn" or not tool_calls:
            return " ".join(text_parts).strip(), search_log

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tc in tool_calls:
            q = tc.input.get("query", "")
            search_log.append(q)
            output = search_my_docs(q)
            tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})
        messages.append({"role": "user", "content": tool_results})

    return "Agent reached max search rounds.", search_log

if ask_btn and user_input:
    if not api_key:
        st.error("Add your Anthropic API key in the sidebar. Get one free at console.anthropic.com")
    else:
        with st.spinner("Searching knowledge base..."):
            try:
                answer, searches = run_agent(user_input, api_key, st.session_state.chunks)
                st.session_state.history.insert(0, {
                    "question": user_input,
                    "answer": answer,
                    "searches": searches,
                })
            except Exception as e:
                st.error(f"Error: {e}")

# ── Answer + History ─────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Most recent answer — always shown prominently ──
    latest = st.session_state.history[0]
    st.markdown('''
    <div style="
        background:#141414;
        border:1px solid #222222;
        border-top:3px solid #4ADE80;
        border-radius:6px;
        padding:0.5rem 1.5rem 0.4rem;
        margin-bottom:0.5rem;
    ">
        <div style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;letter-spacing:3px;color:#4ADE80;text-transform:uppercase;margin-bottom:0.3rem;">
            ▶ Agent Answer
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Question label
    st.markdown(f'''<div style="background:#141414;border:1px solid #222;border-top:none;border-bottom:none;padding:0.6rem 1.5rem;">
        <div style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:#F0F0F0;opacity:0.6;">Q: {latest["question"]}</div>
    </div>''', unsafe_allow_html=True)

    # Search log
    if latest["searches"]:
        log_items = "".join(f'<div class="dd-search-item">{s}</div>' for s in latest["searches"])
        st.markdown(f'''<div style="background:#141414;border:1px solid #222;border-top:none;border-bottom:none;padding:0.5rem 1.5rem;">
            <div class="dd-search-log">{log_items}</div>
        </div>''', unsafe_allow_html=True)

    # Answer body
    st.markdown('''<div style="background:#141414;border:1px solid #222;border-top:none;border-radius:0 0 6px 6px;padding:1rem 1.5rem 1.5rem;">''', unsafe_allow_html=True)
    st.markdown(latest["answer"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Previous questions (collapsed) ──
    if len(st.session_state.history) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="dd-section">Previous Questions</div>', unsafe_allow_html=True)
        for item in st.session_state.history[1:]:
            with st.expander(f"↳ {item['question']}", expanded=False):
                if item["searches"]:
                    log_html = "".join(f'<div class="dd-search-item">{s}</div>' for s in item["searches"])
                    st.markdown(f'<div class="dd-search-log">{log_html}</div>', unsafe_allow_html=True)
                st.markdown(item["answer"])
else:
    # Empty state — shown before any question is asked
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''
    <div style="
        background:#141414;
        border:1px solid #222222;
        border-top:3px solid #4ADE80;
        border-radius:6px;
        padding:2.5rem 1.5rem;
        text-align:center;
    ">
        <div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;letter-spacing:3px;color:#4ADE80;text-transform:uppercase;margin-bottom:0.8rem;">▶ Agent Answer</div>
        <div style="color:#F0F0F0;opacity:0.4;font-size:0.85rem;">Your answer will appear here after you ask a question.</div>
    </div>
    ''', unsafe_allow_html=True)

# ── Quick Questions ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="dd-section">Sample Questions</div>', unsafe_allow_html=True)
qq_cols = st.columns(3, gap="small")
for i, ex in enumerate(EXAMPLES):
    with qq_cols[i]:
        st.markdown('<div class="qq-btn">', unsafe_allow_html=True)
        if st.button(ex, key=f"qq_{ex[:25]}"):
            user_input = ex
            ask_btn = True
        st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="dd-footer">
    ISOM 260 · AI for Business · Suffolk University &nbsp;·&nbsp;
    Data: Basketball Reference 2025-26 &nbsp;·&nbsp;
    <a href="https://declandavis03-max.github.io/">declandavis03-max.github.io</a>
</div>
''', unsafe_allow_html=True)
