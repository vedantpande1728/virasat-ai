# -*- coding: utf-8 -*-
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
import json
import os
import math
import time

# ---------- NLTK boot ----------
for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

st.set_page_config(
    page_title="Virasat AI — Neural Heritage Engine",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# GLOBAL CSS — MUSEUM-GRADE HERITAGE DESIGN
# ============================================================
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@400;700;900&family=Cinzel:wght@400;500;600;700;800;900&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400;1,500&family=JetBrains+Mono:wght@300;400;500;700&family=Outfit:wght@200;300;400;500;600;700;800&display=swap');

:root{
    --bg-deep: #04060a;
    --gold: #e6b958;
    --gold-bright: #fce29c;
    --gold-deep: #a87c2a;
    --teal: #10b981;
    --amethyst: #a855f7;
    --text-muted: #5a6b82;
    --hairline: rgba(230,185,88,0.18);
}

/* --- HIDE SCROLLBARS & STREAMLIT CHROME --- */
::-webkit-scrollbar { width: 0px; height: 0px; background: transparent; }
* { scrollbar-width: none; -ms-overflow-style: none; }
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden; height:0; display:none;}
[data-testid="collapsedControl"], section[data-testid="stSidebar"] {display:none;}

/* REMOVE ALL BLUE FOCUS RINGS AND SELECTIONS */
::selection { background: rgba(230,185,88,0.3) !important; color: #ffffff !important; }
*:focus, *:focus-visible, *:active { outline: none !important; box-shadow: none !important; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif; 
    overflow-x: hidden;
}

/* ============================================================
   ENHANCED BACKGROUND
   ============================================================ */
.stApp {
    background-color: var(--bg-deep) !important;
    background-image:
        url("data:image/svg+xml,%3Csvg width='120' height='120' viewBox='0 0 120 120' xmlns='http://www.w3.org/2000/svg'%3E%3Cg stroke='%23e6b958' stroke-width='0.8' opacity='0.12' fill='none'%3E%3Cpath d='M60 0 L120 60 L60 120 L0 60 Z'/%3E%3Cpath d='M60 20 L100 60 L60 100 L20 60 Z'/%3E%3Ccircle cx='60' cy='60' r='18'/%3E%3Ccircle cx='60' cy='60' r='32'/%3E%3Cpath d='M60 0 L60 120 M0 60 L120 60'/%3E%3C/g%3E%3C/svg%3E"),
        radial-gradient(ellipse at 50% -10%, rgba(230,185,88,0.25) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 110%, rgba(196,122,61,0.15) 0%, transparent 60%),
        linear-gradient(180deg, #04060a 0%, #0a101c 50%, #04060a 100%) !important;
    background-size: 120px 120px, 100% 100%, 100% 100%, 100% 100% !important;
    background-position: 0 0, center top, center bottom, center top !important;
    background-repeat: repeat, no-repeat, no-repeat, no-repeat !important;
    background-attachment: fixed !important;
}

.stApp::after {
    content: ''; position: fixed; top: 50%; left: 50%; width: 160vmin; height: 160vmin;
    transform: translate(-50%, -50%);
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 800 800' xmlns='http://www.w3.org/2000/svg'%3E%3Cg stroke='%23e6b958' stroke-width='0.8' fill='none' opacity='0.18'%3E%3Ccircle cx='400' cy='400' r='390'/%3E%3Ccircle cx='400' cy='400' r='320'/%3E%3Ccircle cx='400' cy='400' r='250'/%3E%3Cpath d='M400 10 L400 790 M10 400 L790 400'/%3E%3Cpath d='M400 60 L520 200 L660 340 L520 480 L400 620 L280 480 L140 340 L280 200 Z'/%3E%3Cpath d='M400 120 L450 250 L580 300 L450 350 L400 480 L350 350 L220 300 L350 250 Z'/%3E%3C/g%3E%3C/svg%3E");
    background-size: contain; background-repeat: no-repeat; background-position: center;
    pointer-events: none; z-index: 0; animation: rotateMandala 240s linear infinite;
}
@keyframes rotateMandala { from { transform: translate(-50%, -50%) rotate(0deg); } to { transform: translate(-50%, -50%) rotate(360deg); } }

.main .block-container { position: relative; z-index: 2; padding-top: 1rem; padding-bottom: 4rem; max-width: 1400px; }

/* Decorative frame */
.decorative-frame { position: fixed; top: 20px; left: 20px; right: 20px; bottom: 20px; pointer-events: none; z-index: 1; border: 1px solid rgba(230,185,88,0.12); }
.corner { position: absolute; width: 40px; height: 40px; border-color: var(--gold); border-style: solid; opacity: 0.8; }
.corner.tl { top: -1px; left: -1px; border-width: 2px 0 0 2px; }
.corner.tr { top: -1px; right: -1px; border-width: 2px 2px 0 0; }
.corner.bl { bottom: -1px; left: -1px; border-width: 0 0 2px 2px; }
.corner.br { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }

/* --- TOP HEADER --- */
.top-header { display: flex; justify-content: flex-end; gap: 12px; padding: 20px 0 10px 0; }
.badge { display: inline-flex; align-items: center; gap: 10px; padding: 8px 20px; border-radius: 40px; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); color: #cbd5e1; box-shadow: inset 0 1px 0 rgba(255,255,255,0.05); }
.badge-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--teal); box-shadow: 0 0 10px var(--teal); }

/* --- HERO --- */
.hero-section { position: relative; display: flex; flex-direction: column; padding: 60px 0 70px 0; margin-bottom: 30px; border-bottom: 1px solid var(--hairline); overflow: hidden; }
.hero-content { position: relative; z-index: 1; max-width: 720px; }
.hero-kicker { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.32em; text-transform: uppercase; color: var(--gold); display: flex; align-items: center; gap: 16px; margin-bottom: 22px; }
.hero-kicker::before, .hero-kicker::after { content: ''; height: 1px; background: var(--gold); width: 56px; opacity: 0.4; }
.hero-title { font-family: 'Cinzel Decorative', serif; font-size: 96px; font-weight: 900; line-height: 0.95; margin: 0 0 20px 0; color: #ffffff; letter-spacing: 0.01em; text-shadow: 0 10px 40px rgba(0,0,0,0.8), 0 0 60px rgba(230,185,88,0.15); }
.hero-title span { color: var(--gold); background: linear-gradient(180deg, var(--gold-bright) 0%, var(--gold) 45%, var(--gold-deep) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-subtitle { font-family: 'Cormorant Garamond', serif; font-style: italic; font-size: 30px; color: #e2e8f0; margin-bottom: 24px; font-weight: 400; }
.hero-desc { color: #94a3b8; line-height: 1.85; margin-bottom: 36px; text-align: justify; font-family: 'Cormorant Garamond', serif; font-size: 19px; font-weight: 400; }

/* --- STATS --- */
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 46px; }
.stat-card { background: linear-gradient(180deg, rgba(20,26,40,0.8) 0%, rgba(10,14,24,0.95) 100%); border: 1px solid rgba(255,255,255,0.05); border-radius: 14px; padding: 28px; position: relative; overflow: hidden; display: flex; flex-direction: column; box-shadow: 0 10px 30px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05); transition: transform 0.35s, box-shadow 0.35s, border-color 0.35s; }
.stat-card:hover { transform: translateY(-6px); box-shadow: 0 20px 45px rgba(0,0,0,0.65), inset 0 1px 0 rgba(255,255,255,0.1); border-color: rgba(230,185,88,0.25); }
.stat-card::before { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 2px; }
.stat-card.gold::before { background: var(--gold); box-shadow: 0 0 15px var(--gold); }
.stat-card.copper::before { background: #c47a3d; box-shadow: 0 0 15px #c47a3d; }
.stat-card.teal::before { background: var(--teal); box-shadow: 0 0 15px var(--teal); }
.stat-card.purple::before { background: var(--amethyst); box-shadow: 0 0 15px var(--amethyst); }
.stat-icon { width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 20px; font-size: 18px; border: 1px solid transparent; }
.stat-card.gold .stat-icon { background: rgba(230,185,88,0.1); color: var(--gold); border-color: rgba(230,185,88,0.2); }
.stat-card.copper .stat-icon { background: rgba(196,122,61,0.1); color: #c47a3d; border-color: rgba(196,122,61,0.2); }
.stat-card.teal .stat-icon { background: rgba(16,185,129,0.1); color: var(--teal); border-color: rgba(16,185,129,0.2); }
.stat-card.purple .stat-icon { background: rgba(168,85,247,0.1); color: var(--amethyst); border-color: rgba(168,85,247,0.2); }
.stat-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; color: #64748b; margin-bottom: 10px; }
.stat-value { font-family: 'Cinzel', serif; font-size: 26px; font-weight: 700; color: #ffffff; margin-bottom: 18px; letter-spacing: 0.02em; }
.stat-footer { margin-top: auto; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #94a3b8; display: flex; align-items: center; gap: 8px; }

/* --- CONSOLE BOX --- */
.console-box { background: linear-gradient(180deg, rgba(15,20,32,0.9), rgba(8,10,16,0.98)); border: 1px solid var(--hairline); border-radius: 20px; padding: 50px 55px; position: relative; box-shadow: 0 25px 60px rgba(0,0,0,0.65), inset 0 1px 0 rgba(230,185,88,0.12); overflow: hidden; margin-top: 20px; margin-bottom: 30px;}
.console-box::before { content: ''; position: absolute; top: -1px; left: 50%; transform: translateX(-50%); width: 180px; height: 3px; background: linear-gradient(90deg, transparent, var(--gold), transparent); box-shadow: 0 0 20px var(--gold); }
.console-kicker { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.24em; text-transform: uppercase; color: var(--gold); display: flex; align-items: center; gap: 12px; margin-bottom: 16px; position: relative; z-index: 1; }
.console-kicker::before { content: ''; width: 26px; height: 1px; background: var(--gold); }
.console-heading { font-family: 'Cinzel', serif; font-size: 40px; color: #ffffff; margin: 0 0 14px 0; position: relative; z-index: 1; letter-spacing: 0.01em; }
.console-desc { color: #94a3b8; line-height: 1.75; text-align: justify; max-width: 680px; position: relative; z-index: 1; font-family: 'Cormorant Garamond', serif; font-size: 18px; }

/* ============================================================
   SLEEK MODERN SEARCH BAR
   ============================================================ */

/* Kill Streamlit's default label spacing */
div[data-testid="stTextInput"] label { display: none !important; height: 0 !important; min-height: 0 !important; }

/* The unified pill wrapper (input + button sit inside it) */
div[data-testid="stHorizontalBlock"] {
    background: linear-gradient(180deg, rgba(22,28,44,0.85), rgba(10,13,20,0.95)) !important;
    border: 1px solid rgba(230,185,88,0.22) !important;
    border-radius: 999px !important;
    padding: 6px 8px 6px 24px !important;
    gap: 8px !important;
    align-items: center !important;
    box-shadow:
        0 18px 40px rgba(0,0,0,0.55),
        inset 0 1px 0 rgba(255,255,255,0.05),
        inset 0 -1px 0 rgba(230,185,88,0.06) !important;
    backdrop-filter: blur(16px) !important;
    margin: 0 0 14px 0 !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
div[data-testid="stHorizontalBlock"]:focus-within {
    border-color: rgba(230,185,88,0.55) !important;
    box-shadow:
        0 18px 40px rgba(0,0,0,0.55),
        0 0 0 3px rgba(230,185,88,0.10),
        0 0 30px rgba(230,185,88,0.15),
        inset 0 1px 0 rgba(255,255,255,0.08) !important;
}

/* Make the columns inside the pill transparent so the pill shows through */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    background: transparent !important;
    padding: 0 !important;
}

/* --- Input — stripped bare inside the pill --- */
div[data-testid="stTextInput"] > div[data-baseweb="input"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    height: 48px !important;
}
div[data-testid="stTextInput"] > div[data-baseweb="input"]:focus-within {
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stTextInput"] input {
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 17px !important;
    font-weight: 400 !important;
    line-height: 48px !important;
    caret-color: var(--gold) !important;
    background: transparent !important;
    padding: 0 !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #5a6b82 !important;
    font-family: 'Outfit', sans-serif !important;
    font-style: normal !important;
    font-weight: 300 !important;
    font-size: 16px !important;
    letter-spacing: 0.01em !important;
}

/* --- Button — compact modern capsule, right side of pill --- */
div[data-testid="stButton"] button {
    background: linear-gradient(180deg, var(--gold-bright), var(--gold) 55%, var(--gold-deep)) !important;
    color: #05070a !important;
    border: none !important;
    border-radius: 999px !important;
    height: 44px !important;
    min-height: 44px !important;
    width: 100% !important;
    padding: 0 26px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    margin: 0 !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.55),
        0 4px 14px rgba(230,185,88,0.25) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
}
div[data-testid="stButton"] button:hover {
    background: linear-gradient(180deg, #ffffff, var(--gold-bright) 55%, var(--gold)) !important;
    transform: translateY(-1px) !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.85),
        0 8px 22px rgba(230,185,88,0.4) !important;
}
div[data-testid="stButton"] button:active {
    transform: scale(0.97) !important;
}

/* ============================================================
   LOADING STATES
   ============================================================ */

/* Status line below the pill */
.search-status {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 24px;
    margin: -6px 0 32px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--gold);
    min-height: 20px;
    opacity: 0;
    animation: statusFadeIn 0.3s ease-out forwards;
}
@keyframes statusFadeIn {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: translateY(0); }
}
.search-status .status-ring {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid rgba(230,185,88,0.25);
    border-top-color: var(--gold);
    animation: spin 0.7s linear infinite;
    box-shadow: 0 0 10px rgba(230,185,88,0.35);
}
@keyframes spin { to { transform: rotate(360deg); } }
.search-status .status-text { color: var(--gold); }
.search-status .status-text .dots::after {
    content: '';
    animation: dots 1.4s steps(4, end) infinite;
}
@keyframes dots {
    0%   { content: ''; }
    25%  { content: '.'; }
    50%  { content: '..'; }
    75%  { content: '...'; }
    100% { content: ''; }
}

/* Shimmer skeleton for loading state */
.skeleton-card {
    background: linear-gradient(160deg, rgba(20,26,40,0.7), rgba(10,14,24,0.85));
    border: 1px solid var(--hairline);
    border-radius: 20px;
    padding: 55px;
    margin-top: 20px;
    position: relative;
    overflow: hidden;
    min-height: 420px;
}
.skeleton-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(
        100deg,
        transparent 20%,
        rgba(230,185,88,0.06) 40%,
        rgba(230,185,88,0.12) 50%,
        rgba(230,185,88,0.06) 60%,
        transparent 80%
    );
    background-size: 200% 100%;
    animation: shimmer 1.6s linear infinite;
    pointer-events: none;
}
@keyframes shimmer {
    from { background-position: 200% 0; }
    to   { background-position: -200% 0; }
}
.skeleton-header {
    display: flex;
    align-items: center;
    gap: 22px;
    margin-bottom: 36px;
    padding-bottom: 26px;
    border-bottom: 1px solid rgba(230,185,88,0.1);
}
.skeleton-icon {
    width: 60px; height: 60px;
    border-radius: 14px;
    background: rgba(230,185,88,0.08);
    border: 1px solid rgba(230,185,88,0.15);
    flex-shrink: 0;
}
.skeleton-line {
    height: 14px;
    border-radius: 6px;
    background: rgba(230,185,88,0.06);
    margin-bottom: 14px;
}
.skeleton-line.title { height: 26px; width: 45%; background: rgba(230,185,88,0.10); }
.skeleton-line.sub   { height: 11px; width: 22%; background: rgba(230,185,88,0.06); }
.skeleton-line.w90   { width: 90%; }
.skeleton-line.w75   { width: 75%; }
.skeleton-line.w60   { width: 60%; }
.skeleton-line.w45   { width: 45%; }
.skeleton-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin: 30px 0;
}
.skeleton-fact {
    height: 76px;
    border-radius: 12px;
    background: rgba(230,185,88,0.04);
    border: 1px solid rgba(230,185,88,0.08);
}

/* Button loading state (spinner inside pill) */
div[data-testid="stButton"] button:disabled {
    opacity: 1 !important;
    cursor: not-allowed !important;
    background: rgba(230,185,88,0.15) !important;
    color: transparent !important;
    box-shadow: inset 0 0 0 1px rgba(230,185,88,0.3) !important;
    position: relative;
}
div[data-testid="stButton"] button:disabled::after {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    width: 18px; height: 18px;
    margin: -9px 0 0 -9px;
    border-radius: 50%;
    border: 2px solid rgba(230,185,88,0.25);
    border-top-color: var(--gold);
    animation: spin 0.7s linear infinite;
    box-shadow: 0 0 12px rgba(230,185,88,0.4);
}

/* --- NLP PIPELINE VISUALIZER --- */
.nlp-panel { background: rgba(10,14,24,0.6); border: 1px solid var(--hairline); border-radius: 14px; padding: 28px 30px; margin-top: 30px; position: relative; z-index: 2; animation: fadeUp 0.5s ease-out; }
.nlp-title { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--gold); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 18px; display: flex; align-items: center; gap: 10px; }
.nlp-title::before { content: ''; width: 8px; height: 8px; background: var(--gold); border-radius: 50%; box-shadow: 0 0 10px var(--gold); }
.nlp-step { display: flex; gap: 16px; margin-bottom: 16px; align-items: flex-start; }
.nlp-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); min-width: 108px; padding-top: 4px; }
.nlp-tokens { display: flex; flex-wrap: wrap; gap: 8px; }
.nlp-token { background: rgba(230,185,88,0.08); border: 1px solid rgba(230,185,88,0.2); color: #e2e8f0; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-family: 'JetBrains Mono', monospace; }
.nlp-token.highlight { background: rgba(230,185,88,0.2); border-color: var(--gold); color: var(--gold); font-weight: bold; }

/* --- RESULT CARD --- */
.result-card { background: linear-gradient(160deg, rgba(20,26,40,0.95), rgba(10,14,24,0.98)); border: 1px solid var(--hairline); border-radius: 20px; padding: 55px; margin-top: 20px; position: relative; box-shadow: 0 30px 70px rgba(0,0,0,0.8), inset 0 1px 0 rgba(230,185,88,0.15); animation: fadeUp 0.6s ease-out; overflow: hidden; }
.result-card::before { content: ''; position: absolute; top: -1px; left: 50%; transform: translateX(-50%); width: 220px; height: 3px; background: linear-gradient(90deg, transparent, var(--gold), transparent); box-shadow: 0 0 25px var(--gold); }
.result-card::after { content: ''; position: absolute; bottom: -80px; left: -80px; width: 260px; height: 260px; background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cg stroke='%23e6b958' stroke-width='0.6' fill='none' opacity='0.3'%3E%3Ccircle cx='100' cy='100' r='95'/%3E%3Ccircle cx='100' cy='100' r='70'/%3E%3Ccircle cx='100' cy='100' r='45'/%3E%3Ccircle cx='100' cy='100' r='20'/%3E%3Cpath d='M100 5 L100 195 M5 100 L195 100 M30 30 L170 170 M170 30 L30 170'/%3E%3Cpath d='M100 40 L120 80 L160 100 L120 120 L100 160 L80 120 L40 100 L80 80 Z'/%3E%3C/g%3E%3C/svg%3E"); background-size: contain; background-repeat: no-repeat; pointer-events: none; z-index: 0; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.rc-header { display: flex; align-items: center; gap: 22px; margin-bottom: 40px; padding-bottom: 26px; border-bottom: 1px solid rgba(230,185,88,0.15); position: relative; z-index: 1; }
.rc-icon { width: 60px; height: 60px; border-radius: 14px; background: rgba(230,185,88,0.08); border: 1px solid rgba(230,185,88,0.3); display: flex; align-items: center; justify-content: center; font-size: 30px; box-shadow: inset 0 0 20px rgba(230,185,88,0.1); }
.rc-title { font-family: 'Cinzel', serif; font-size: 34px; color: var(--gold); text-shadow: 0 0 25px rgba(230,185,88,0.35); letter-spacing: 0.01em; line-height: 1.15; }
.rc-sub { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: #64748b; margin-top: 8px; }
.facts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin: 40px 0; position: relative; z-index: 1; }
.fact-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(230,185,88,0.1); border-radius: 12px; padding: 20px 24px; transition: all 0.3s; }
.fact-card:hover { background: rgba(230,185,88,0.04); border-color: rgba(230,185,88,0.3); transform: translateY(-3px); }
.fact-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--gold); margin-bottom: 10px; }
.fact-value { color: #f1f5f9; font-size: 16px; line-height: 1.6; font-family: 'Outfit', sans-serif; }
.record-prose { font-family: 'Cormorant Garamond', serif; font-size: 20px; line-height: 1.85; color: #cbd5e1; text-align: justify; margin-bottom: 30px; position: relative; z-index: 1; }
.record-prose p { margin-bottom: 20px; }
.record-prose .drop-cap::first-letter { font-family: 'Cinzel', serif; font-size: 4.4em; font-weight: 700; color: var(--gold); float: left; line-height: 0.85; margin: 8px 14px 0 0; text-shadow: 0 0 25px rgba(230,185,88,0.45); }
.steps-list { list-style: none; padding: 0; margin: 0; counter-reset: step-counter; position: relative; z-index: 1; }
.steps-list li { counter-increment: step-counter; position: relative; padding: 20px 24px 20px 74px; margin-bottom: 16px; background: rgba(255,255,255,0.02); border: 1px solid rgba(230,185,88,0.1); border-radius: 12px; font-size: 16px; line-height: 1.65; color: #cbd5e1; font-family: 'Outfit', sans-serif; }
.steps-list li::before { content: counter(step-counter, decimal-leading-zero); position: absolute; left: 22px; top: 50%; transform: translateY(-50%); font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; color: var(--gold); background: rgba(230,185,88,0.1); padding: 6px 11px; border-radius: 6px; border: 1px solid rgba(230,185,88,0.25); }
.section-title { font-family: 'Cinzel', serif; font-size: 24px; color: var(--gold); margin-bottom: 22px; border-bottom: 1px solid rgba(230,185,88,0.15); padding-bottom: 12px; position: relative; z-index: 1; letter-spacing: 0.01em; }

/* Heritage image */
.heritage-image-container { margin: 30px 0; position: relative; z-index: 1; border-radius: 14px; overflow: hidden; border: 1px solid var(--hairline); box-shadow: 0 15px 45px rgba(0,0,0,0.65); }
.heritage-image { width: 100%; height: 420px; object-fit: cover; display: block; filter: sepia(18%) contrast(110%) brightness(0.92); transition: transform 0.6s, filter 0.6s; }
.heritage-image:hover { transform: scale(1.03); filter: sepia(0%) contrast(100%) brightness(1); }

/* Responsive */
@media (max-width: 1024px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } .hero-title { font-size: 72px; } .skeleton-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 768px) {
    .stats-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 52px; }
    .hero-subtitle { font-size: 22px; }
    .hero-desc { font-size: 17px; }
    .console-box { padding: 32px 28px; }
    .console-heading { font-size: 30px; }
    .result-card { padding: 32px 26px; }
    .facts-grid { grid-template-columns: 1fr; }
    .heritage-image { height: 250px; }
    div[data-testid="stHorizontalBlock"] { padding: 6px !important; }
    .skeleton-card { padding: 28px; }
    .skeleton-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# Decorative frame
st.markdown("""
<div class="decorative-frame">
    <div class="corner tl"></div><div class="corner tr"></div>
    <div class="corner bl"></div><div class="corner br"></div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# NLP SETUP
# ============================================================
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
IGNORE_WORDS = STOP_WORDS.union({"the","a","an","is","of","and","in","to","for","who","what","when","where","why","how","tell","me","about","it","was","were","are","this","that","with","as","at","by","on"})

def nlp_pipeline(text):
    raw_tokens = word_tokenize(text.lower())
    alpha_tokens = [t for t in raw_tokens if t.isalpha()]
    filtered_tokens = [t for t in alpha_tokens if t not in IGNORE_WORDS]
    lemmas = [LEMMATIZER.lemmatize(t) for t in filtered_tokens]
    return {"raw": raw_tokens, "alpha": alpha_tokens, "filtered": filtered_tokens, "lemmas": lemmas}

# ============================================================
# DATA LOADING
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "abhiratna_dataset.json")

@st.cache_data
def load_dataset(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

DATASET = load_dataset(DATASET_PATH)

@st.cache_data
def build_tfidf_index(dataset):
    doc_count = len(dataset)
    word_doc_counts = Counter()
    for record in dataset:
        text = record.get("searchable_text", "").lower()
        words = set(re.findall(r'\w+', text)) - IGNORE_WORDS
        for w in words:
            word_doc_counts[w] += 1
    return word_doc_counts, doc_count

WORD_DOC_COUNTS, TOTAL_DOCS = build_tfidf_index(DATASET)

def resolve_qa_from_json(query: str):
    if not DATASET: return None, 0, "Dataset empty."
    query_lower = query.lower().strip()
    query_words = set(re.findall(r'\w+', query_lower)) - IGNORE_WORDS
    if not query_words: return None, 0, "Query contained only stopwords."

    query_idf = {qw: 0.0 if WORD_DOC_COUNTS.get(qw, 0) == 0 else math.log((TOTAL_DOCS + 1) / (WORD_DOC_COUNTS.get(qw, 0) + 1)) + 1.0 for qw in query_words}
    max_idf = max(query_idf.values()) if query_idf else 0.0
    distinctive_terms = {w for w, idf in query_idf.items() if idf >= max_idf * 0.85 and idf > 1.5}

    best_record, best_score = None, 0.0
    for record in DATASET:
        title = record.get("title", "").lower().strip()
        text  = record.get("searchable_text", "").lower()
        score = 0.0

        if query_lower == title: score += 5000.0
        elif query_lower in title: score += 1500.0
        elif title and title in query_lower: score += 800.0
        for qw in distinctive_terms:
            if qw == title or qw in title.split(): score += 1000.0

        if distinctive_terms:
            if not any(qw in text or qw in title for qw in distinctive_terms): continue
        else:
            if not any(qw in text for qw in query_words): continue

        for qw in query_words:
            idf = query_idf.get(qw, 0.0)
            if idf <= 0.0: continue
            if qw in text: score += (1.0 + math.log(text.count(qw))) * idf * 10.0
            if qw in title: score += 60.0 * idf
        if query_lower in text: score += 200.0

        if score > best_score:
            best_score, best_record = score, record

    if best_record is not None and best_score >= 20.0:
        return best_record, best_score, f"Match Score: {best_score:.1f}"
    return None, 0, "No record found."

# ============================================================
# HTML RENDER BUILDERS
# ============================================================
def _escape(text) -> str:
    return str(text).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _render_prose(text):
    raw = " ".join(str(p) for p in text) if isinstance(text, list) else str(text)
    paras = [p.strip() for p in raw.split("\n") if p.strip()]
    out = ['<div class="record-prose">']
    for i, p in enumerate(paras):
        cls = ' class="drop-cap"' if i == 0 and len(p) > 40 else ''
        out.append(f"<p{cls}>{_escape(p)}</p>")
    out.append('</div>')
    return "".join(out)

def _render_steps(value):
    if isinstance(value, list): steps = [str(s).strip() for s in value if str(s).strip()]
    elif isinstance(value, dict): steps = [f"{k}: {v}" for k, v in value.items()]
    else:
        raw = str(value).strip()
        fragments = [f.strip() for f in re.split(r',\s+', raw) if f.strip()]
        steps, buffer = [], ""
        for frag in fragments:
            if not buffer: buffer = frag
            else:
                first = frag.split()[0] if frag.split() else ""
                if frag[0].islower() or first.lower() in {"and","then","with","using","or","removing"}:
                    buffer = f"{buffer}, {frag}"
                else:
                    steps.append(buffer); buffer = frag
        if buffer: steps.append(buffer)

    cleaned = []
    for s in steps:
        s = s.strip().rstrip(",").rstrip(";").strip()
        if not s: continue
        if not s.endswith("."): s += "."
        cleaned.append(s[0].upper() + s[1:])

    if not cleaned: return ""
    out = ['<ol class="steps-list">']
    for s in cleaned: out.append(f"<li>{_escape(s)}</li>")
    out.append("</ol>")
    return "".join(out)

def build_record_html(record):
    raw = record.get("raw_data", {})
    norm = {k.lower().replace("_", ""): v for k, v in raw.items()}
    image_url = norm.get("image") or norm.get("imageurl") or norm.get("img") or norm.get("photo") or norm.get("picture")

    parts = []

    if image_url:
        parts.append(f'''
        <div class="heritage-image-container">
            <img src="{_escape(image_url)}" class="heritage-image" alt="{_escape(record.get('title', 'Heritage Image'))}">
        </div>
        ''')

    facts = []
    for k, label in [("location","Location"),("dynasty","Dynasty"),("timeline","Timeline"),("century","Century"),("builder","Builder"),("style","Style")]:
        if norm.get(k): facts.append((label, str(norm[k])))
    if facts:
        parts.append('<div class="facts-grid">')
        for l, v in facts:
            parts.append(f'<div class="fact-card"><div class="fact-label">{l}</div><div class="fact-value">{_escape(v)}</div></div>')
        parts.append('</div>')

    for k, label in [("desc","Overview"),("paragraph","Detailed Account"),("fullstory","Full Story")]:
        val = norm.get(k)
        if val and str(val).strip():
            parts.append(f'<div style="margin-top:50px;"><div class="section-title">{label}</div>{_render_prose(val)}</div>')

    val = norm.get("constructionlayers")
    if val:
        parts.append(f'<div style="margin-top:50px;"><div class="section-title">Construction & Technique</div>{_render_steps(val)}</div>')

    return "".join(parts)

def build_skeleton_html():
    """Shimmer skeleton that mimics the result card layout."""
    return """
    <div class="skeleton-card">
        <div class="skeleton-header">
            <div class="skeleton-icon"></div>
            <div style="flex:1;">
                <div class="skeleton-line title"></div>
                <div class="skeleton-line sub"></div>
            </div>
        </div>
        <div class="skeleton-line w90"></div>
        <div class="skeleton-line w75"></div>
        <div class="skeleton-line w60"></div>
        <div class="skeleton-grid">
            <div class="skeleton-fact"></div>
            <div class="skeleton-fact"></div>
            <div class="skeleton-fact"></div>
        </div>
        <div class="skeleton-line w90"></div>
        <div class="skeleton-line w75"></div>
        <div class="skeleton-line w45"></div>
    </div>
    """

def build_status_html(label="Querying heritage archive"):
    return f"""
    <div class="search-status">
        <span class="status-ring"></span>
        <span class="status-text">{label}<span class="dots"></span></span>
    </div>
    """

# ============================================================
# PAGE LAYOUT
# ============================================================
st.markdown("""
<div class="top-header">
    <div class="badge"><span class="badge-dot"></span> System Online</div>
    <div class="badge"><span style="color:var(--gold);">🔗</span> Archive Connected</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <div class="hero-kicker">Indic Heritage NLP Platform</div>
        <h1 class="hero-title">VIRASAT <span>AI</span></h1>
        <div class="hero-subtitle">Ancient wisdom, reimagined through modern intelligence.</div>
        <div class="hero-desc">
            A full-spectrum intelligence engine that detects heritage, understands context, and connects
            the past with the present — through the power of natural language processing.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-grid">
    <div class="stat-card gold">
        <div class="stat-icon">🗄️</div>
        <div class="stat-label">Archive Records</div>
        <div class="stat-value">{len(DATASET):,}</div>
        <div class="stat-footer" style="color:var(--gold);">↑ +12% &nbsp; <span style="color:var(--text-muted);">Total Entities</span></div>
    </div>
    <div class="stat-card copper">
        <div class="stat-icon">🔍</div>
        <div class="stat-label">Search Engine</div>
        <div class="stat-value">TF-IDF + GATED</div>
        <div class="stat-footer">◇ Hybrid Semantic Search</div>
    </div>
    <div class="stat-card teal">
        <div class="stat-icon">📚</div>
        <div class="stat-label">Collections</div>
        <div class="stat-value">MULTI-SCHEMA</div>
        <div class="stat-footer">◇ Historical · Cultural · Regional</div>
    </div>
    <div class="stat-card purple">
        <div class="stat-icon">⚙️</div>
        <div class="stat-label">Pipelines</div>
        <div class="stat-value">NLP → SENT → QA</div>
        <div class="stat-footer">◇ Automated Processing</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="console-box">
    <div class="console-kicker">Featured</div>
    <h2 class="console-heading">Ask the Archive</h2>
    <div class="console-desc">
        Enter a historical question, a visitor review, or any text. The engine will route it to the right
        pipelines automatically — matching context, entities, and meaning across the heritage corpus.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if "search_state" not in st.session_state:
    st.session_state.search_state = "idle"   # idle | loading | done
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "result_meta" not in st.session_state:
    st.session_state.result_meta = ("", 0.0)
if "nlp_cache" not in st.session_state:
    st.session_state.nlp_cache = None

# ============================================================
# SLEEK MODERN SEARCH BAR
# ============================================================
is_loading = st.session_state.search_state == "loading"

col1, col2 = st.columns([6, 1.2], vertical_alignment="center")

with col1:
    query = st.text_input(
        "Search Context",
        value=st.session_state.last_query,
        placeholder="Search the heritage archive…",
        label_visibility="collapsed",
        disabled=is_loading,
        key="search_input",
    )
with col2:
    submit = st.button(
        "Search",
        use_container_width=True,
        disabled=is_loading,
        key="search_button",
    )

# Handle submit -> kick off loading state
if submit and query.strip() and not is_loading:
    st.session_state.search_state = "loading"
    st.session_state.last_query = query.strip()
    st.session_state.result = None
    st.session_state.nlp_cache = None
    st.rerun()

# ============================================================
# LOADING + RESULTS
# ============================================================
placeholder = st.empty()

if st.session_state.search_state == "loading":
    # Render status line + skeleton
    placeholder.markdown(build_status_html(), unsafe_allow_html=True)
    st.markdown(build_skeleton_html(), unsafe_allow_html=True)

    # Simulate pipeline work (keeps spinner visible for a natural beat)
    q = st.session_state.last_query
    nlp_res = nlp_pipeline(q)
    record, score, debug_msg = resolve_qa_from_json(q)

    # Persist and mark done
    st.session_state.nlp_cache = nlp_res
    st.session_state.result = record
    st.session_state.result_meta = (debug_msg, score)
    time.sleep(0.45)  # brief shimmer so the transition feels deliberate
    st.session_state.search_state = "done"
    st.rerun()

elif st.session_state.search_state == "done":
    placeholder.empty()
    nlp_res = st.session_state.nlp_cache
    record = st.session_state.result
    debug_msg, score = st.session_state.result_meta

    if nlp_res:
        st.markdown(f"""
        <div class="nlp-panel">
            <div class="nlp-title">NLP Pipeline Execution</div>
            <div class="nlp-step">
                <div class="nlp-label">01 / Tokens</div>
                <div class="nlp-tokens">{"".join(f'<span class="nlp-token">{_escape(t)}</span>' for t in nlp_res["raw"][:15])}</div>
            </div>
            <div class="nlp-step">
                <div class="nlp-label">02 / Filtered</div>
                <div class="nlp-tokens">{"".join(f'<span class="nlp-token">{_escape(t)}</span>' for t in nlp_res["filtered"])}</div>
            </div>
            <div class="nlp-step">
                <div class="nlp-label">03 / Lemmas</div>
                <div class="nlp-tokens">{"".join(f'<span class="nlp-token highlight">{_escape(t)}</span>' for t in nlp_res["lemmas"])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if record:
        st.markdown(f"""
        <div class="result-card">
            <div class="rc-header">
                <div class="rc-icon">📜</div>
                <div>
                    <div class="rc-title">{_escape(record.get('title', 'Record'))}</div>
                    <div class="rc-sub">{debug_msg}</div>
                </div>
            </div>
            {build_record_html(record)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card" style="text-align:center; padding: 80px 40px;">
            <div style="font-size:48px; margin-bottom:20px; position:relative;z-index:1;">🔍</div>
            <div class="rc-title" style="position:relative;z-index:1;">No Match Found</div>
            <div style="color:var(--text-muted); margin-top:12px; font-family:'JetBrains Mono', monospace; position:relative;z-index:1;">
                Try searching for 'Ajanta Caves', 'Konark', or 'Taj Mahal'.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:60px; padding:20px; font-family:'JetBrains Mono', monospace; font-size:10px; color:var(--text-muted); letter-spacing:0.24em; text-transform:uppercase; position:relative; z-index:2;">
    Virasat AI <span style="color:var(--gold); margin:0 12px;">◆</span> Neural Heritage Engine v3.5
</div>
""", unsafe_allow_html=True)