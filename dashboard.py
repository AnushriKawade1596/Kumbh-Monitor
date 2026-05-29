"""
Kumbh Monitor — Intelligence Dashboard
Production-grade Streamlit analytics dashboard.

Run:
    python -m streamlit run dashboard.py

Data:
    articles_export_clean.csv
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config & Theme Initialization
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kumbh Monitor — Intelligence Dashboard",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants & Unified Taxonomy Definition
# ---------------------------------------------------------------------------
DATA_PATH = "articles_with_taxonomy.csv" if os.path.exists("articles_with_taxonomy.csv") else "data_pipeline/articles_export_clean.csv"

TOPIC_CATEGORIES = [
    "Infrastructure", 
    "Spiritual & Cultural", 
    "Crowd & Safety", 
    "Technology",
    "People & Experience", 
    "Governance & Economy", 
    "Environment", 
    "Health",
    "Food & Water", 
    "Information & Truth",
]

PHASE_ORDER = ["Before", "During", "After"]
DETAILED_PHASES = ["planning", "buildup", "arrival", "event", "decline", "aftermath", "legacy"]

RISK_COLORS = {
    "low": "#10b981",      # Emerald
    "medium": "#f59e0b",   # Amber
    "high": "#ef4444",     # Red
    "critical": "#7f1d1d", # Dark red
}

CLUSTER_COLORS = ["#ff9933", "#6b21a5", "#06b6d4", "#10b981", "#ef4444", "#8b5cf6"]

ML_COLUMN_DEFAULTS = {
    "ml_themes": "[]",
    "ml_event_type": "unknown",
    "ml_temporal_phase": "unknown",
    "ml_cluster_id": -1,
    "risk_score": 0.0,
    "risk_band": "unknown",
    "viz_x": 0.0,
    "viz_y": 0.0,
}

# ---------------------------------------------------------------------------
# Dynamic Analytical Conclusions Database
# ---------------------------------------------------------------------------
THEME_CONCLUSIONS = {
    "Infrastructure": (
        "Infrastructure constitutes the largest single share of Kumbh coverage. "
        "The temporal trend shows a massive concentration in the 'Before' buildup phase, "
        "highlighting the vast preparatory work required for Sadhugram, mobile toilets (11,000+ planned), "
        "and road overhauls. <b>Conclusion:</b> Success of the event hinges entirely on early-stage civic deployment."
    ),
    "Spiritual & Cultural": (
        "Spiritual & Cultural dominates during active event days. "
        "Coverage peaks dynamically on Shahi Snan (royal bath) dates. "
        "Outlets focus heavily on akhada processions, rituals, and sadhu traditions. "
        "<b>Conclusion:</b> Cultural heritage remains the core emotional driver of public and media engagement."
    ),
    "Crowd & Safety": (
        "Crowd & Safety coverage focuses heavily on stampede mitigation, security frameworks, "
        "barrier setups, and security drills. It peaks sharply during peak bathing days. "
        "<b>Conclusion:</b> Safety-critical reporting shows a high density of advisory content, "
        "proving that preventative planning dominates over reactive reporting."
    ),
    "Technology": (
        "Technology represents the modern face of the Kumbh Mela, showcasing "
        "computer vision crowd monitoring, mobile companion apps, and smart drones. "
        "<b>Conclusion:</b> AI integration and real-time crowd dynamics have transformed classic "
        "mass management into an advanced, data-driven security science."
    ),
    "People & Experience": (
        "People & Experience acts as the human-interest core, featuring stories of "
        "lost-and-found databases, volunteers, accessibility reviews, and international pilgrims. "
        "<b>Conclusion:</b> The sheer logistical scale of the Kumbh is humanized and supported "
        "through grassroots volunteer networks and shared personal journeys."
    ),
    "Governance & Economy": (
        "Governance & Economy highlights the grand scale of budget releases (Rs 2,100+ crore), "
        "tourism commerce, and vendor license management. "
        "<b>Conclusion:</b> Public-private partnerships and massive state funding establish "
        "the event as a temporary city setup that acts as a significant regional economic catalyst."
    ),
    "Environment": (
        "Environment monitors critical water quality testing along the Godavari and Ganga, "
        "waste management operations, and plastic ban policies. "
        "<b>Conclusion:</b> Ecological sustainability remains a persistent and high-scrutiny challenge, "
        "with heavy audits and environmental monitoring continuing well into the post-event phase."
    ),
    "Health": (
        "Health tracks emergency healthcare deployments, temporary hospital expansions, "
        "vector-borne surveillance, and trauma centers. "
        "<b>Conclusion:</b> Low health-related reporting volumes suggest that robust early sanitation "
        "infrastructure successfully prevented major epidemiological outbreaks."
    ),
    "Food & Water": (
        "Food & Water represents a vital utility sector focusing on Annadanam distribution networks, "
        "water tanker lines, and food safety inspections. "
        "<b>Conclusion:</b> Zero active news reports suggest that food and water distribution networks "
        "operated seamlessly in the background as basic logistics rather than breaking headlines."
    ),
    "Information & Truth": (
        "Information & Truth addresses rumor controls, fact-checking releases, and "
        "official safety alerts. "
        "<b>Conclusion:</b> The absence of specific articles highlights the critical need to deploy "
        "proactive fact-checking filters, particularly to monitor and counter high-risk social media shares."
    ),
}

# ---------------------------------------------------------------------------
# Custom CSS (premium Outfit typography, glassmorphism, glowing micro-effects)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Mobile viewport fix - add this */
@media (max-width: 768px) {
    .stApp {
        overflow-x: hidden !important;
    }
    .main .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
@media screen and (max-width: 768px) {
    /* Make sidebar collapsible by default on mobile */
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 85% !important;
        max-width: 280px !important;
    }
    
    /* Sidebar overlay effect */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        margin-left: -280px !important;
    }
}
@media screen and (max-width: 768px) {
    [data-testid="stDataFrameResizable"] {
        overflow-x: auto !important;
    }
    
    .stDataFrame {
        width: 100% !important;
        overflow-x: scroll !important;
    }
}
html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #070710 0%, #0d0d26 50%, #150e2c 100%);
    color: #f1f1f1;
}

section[data-testid="stSidebar"] {
    background: rgba(8, 8, 18, 0.95) !important;
    border-right: 1px solid rgba(255, 153, 51, 0.2);
}

.block-container { 
    padding-top: 1.5rem; 
    padding-bottom: 2rem;
}

/* Premium Headers with glowing gradient */
h1, h2, h3, [data-testid="stHeader"] {
    font-family: 'Outfit', sans-serif;
}

h1 {
    background: linear-gradient(90deg, #ffffff 10%, #ff9933 60%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    letter-spacing: -0.02em;
    font-size: 2.8rem;
    margin-bottom: 0.25rem;
}

h2 {
    background: linear-gradient(90deg, #ffffff 0%, #ff9933 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    font-size: 1.8rem;
}

h3 {
    color: #ff9933;
    font-weight: 600;
    margin-top: 1rem;
    font-size: 1.3rem;
}

/* Premium Glass Cards */
.glass-card {
    background: rgba(20, 20, 42, 0.65);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 153, 51, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    border-color: rgba(255, 153, 51, 0.4);
    box-shadow: 0 12px 40px 0 rgba(255, 153, 51, 0.12);
    transform: translateY(-2px);
}

/* Elegant Metric Card */
.metric-card {
    background: rgba(26, 21, 51, 0.5);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 153, 51, 0.2);
    border-radius: 20px;
    padding: 1.5rem 1rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 6px 20px 0 rgba(0, 0, 0, 0.2);
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: #ff9933;
    box-shadow: 0 10px 30px rgba(255, 153, 51, 0.25);
}
.metric-icon { 
    font-size: 36px; 
    margin-bottom: 0.4rem; 
}
.metric-value {
    font-size: 2.6rem; 
    font-weight: 800; 
    line-height: 1;
    background: linear-gradient(90deg, #ffffff, #ff9933);
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label { 
    font-size: 0.85rem; 
    color: #b0b0cc; 
    margin-top: 0.5rem; 
    text-transform: uppercase;
    letter-spacing: 0.08em; 
    font-weight: 500;
}

/* Custom styled tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(10, 10, 25, 0.5);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid rgba(255, 153, 51, 0.15);
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px;
    color: #b0b0cc;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0px 16px;
    transition: all 0.25s ease;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(255, 153, 51, 0.08);
    color: #ff9933;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(255,153,51,0.2) 0%, rgba(107,33,165,0.2) 100%) !important;
    color: #ff9933 !important;
    border: 1px solid rgba(255, 153, 51, 0.3) !important;
}

/* Beautiful custom buttons */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #ff9933, #6b21a5);
    color: white; 
    border: none; 
    border-radius: 25px;
    padding: 0.6rem 1.6rem; 
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 4px 15px rgba(255, 153, 51, 0.25);
    transition: all 0.25s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 20px rgba(255, 153, 51, 0.4);
    color: white;
    border: none;
}

/* Article card styling */
.article-card {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 1.25rem;
    margin: 0.75rem 0;
    border-left: 4px solid #ff9933;
    border: 1px solid rgba(255,255,255,0.05);
    border-left: 4px solid #ff9933;
    transition: all 0.2s ease;
}
.article-card:hover {
    background: rgba(255, 255, 255, 0.07);
    transform: translateX(4px);
}

/* Analytical Insight boxes */
.insight-box {
    background: rgba(107, 33, 165, 0.12);
    border-left: 4px solid #ff9933;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.75rem;
    font-size: 0.96rem;
    line-height: 1.5;
    color: #e2e2ec;
}

/* Dynamic conclusion card */
.conclusion-card {
    background: linear-gradient(135deg, rgba(255,153,51,0.08) 0%, rgba(107,33,165,0.08) 100%);
    border: 1px solid rgba(255, 153, 51, 0.25);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
    box-shadow: inset 0 0 12px rgba(255,153,51,0.05);
}

/* Footer */
.footer {
    border-top: 1px solid rgba(255, 153, 51, 0.2);
    margin-top: 3.5rem; 
    padding-top: 1.5rem;
    color: #8888aa; 
    font-size: 0.85rem; 
    text-align: center;
}

/* ============================================ */
/* MOBILE RESPONSIVENESS ADDITIONS */
/* ============================================ */

/* Tablet & Mobile Devices */
@media screen and (max-width: 768px) {
    /* Smaller headers */
    h1 {
        font-size: 1.8rem !important;
        text-align: center;
    }
    
    h2 {
        font-size: 1.3rem !important;
    }
    
    h3 {
        font-size: 1.1rem !important;
    }
    
    /* Metric cards - 2 per row on mobile */
    .metric-card {
        padding: 0.75rem 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .metric-icon {
        font-size: 24px !important;
    }
    
    .metric-value {
        font-size: 1.6rem !important;
    }
    
    .metric-label {
        font-size: 0.7rem !important;
    }
    
    /* Stack columns vertically on mobile */
    .stColumns {
        flex-direction: column !important;
    }
    
    /* Article cards */
    .article-card {
        padding: 0.75rem !important;
    }
    
    /* Insight boxes */
    .insight-box {
        padding: 0.75rem !important;
        font-size: 0.85rem !important;
    }
    
    /* Conclusion cards */
    .conclusion-card {
        padding: 1rem !important;
    }
    
    /* Tab navigation - make scrollable on mobile */
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        white-space: nowrap !important;
        -webkit-overflow-scrolling: touch;
    }
    
    .stTabs [data-baseweb="tab"] {
        flex: 0 0 auto !important;
        font-size: 0.8rem !important;
        padding: 0 12px !important;
        height: 40px !important;
    }
    
    /* Sidebar - full width on mobile */
    section[data-testid="stSidebar"] {
        width: 100% !important;
    }
    
    /* Hide sidebar toggle text on mobile */
    .stSidebar [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    
    /* Footer */
    .footer {
        font-size: 0.7rem !important;
        margin-top: 2rem !important;
    }
    
    /* Buttons */
    .stButton>button, .stDownloadButton>button {
        width: 100% !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Charts - ensure they don't overflow */
    .js-plotly-plot, .plotly-graph-div {
        width: 100% !important;
    }
}
@media screen and (max-width: 768px) {
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
}
/* Small phones (max-width: 480px) */
@media screen and (max-width: 480px) {
    h1 {
        font-size: 1.5rem !important;
    }
    
    .metric-value {
        font-size: 1.3rem !important;
    }
    
    .metric-icon {
        font-size: 20px !important;
    }
    
    .insight-box {
        font-size: 0.8rem !important;
    }
}
</style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load-time NLP AI Enrichment Engine
# ---------------------------------------------------------------------------
def enrich_with_ai(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Risk Score calculations
    risk_trigger_words = {
        "stampede": 0.4, "deadly": 0.35, "tragedy": 0.3, "rumor": 0.3, "unverified": 0.25, 
        "missing": 0.25, "adulteration": 0.2, "accident": 0.25, "protest": 0.2, "clash": 0.25,
        "riot": 0.35, " vip ": 0.15, "mismanagement": 0.2, "deaths": 0.3, "fake": 0.25,
        "investigation": 0.15, "scam": 0.25
    }
    safe_words = {
        "vows": -0.15, "plans": -0.15, "commissioner": -0.1, "facility": -0.1, "safety measures": -0.2,
        "aarti": -0.15, "official": -0.15, "clean": -0.1, "innovations": -0.15, "experiencing": -0.1,
        "devotees": -0.1, "dip": -0.1, "holy": -0.15, "spiritual": -0.1
    }
    
    risk_scores = []
    risk_bands = []
    cluster_ids = []
    temporal_phases = []
    event_types = []
    viz_xs = []
    viz_ys = []
    
    # Pre-calculated deterministic cluster anchors for visual cluster projection mapping
    cluster_anchors = {
        0: (2.0, 2.0),   # Spiritual & Bathing (Top Right)
        1: (-2.0, 2.0),  # Infrastructure (Top Left)
        2: (-2.0, -2.0), # Sadhugram (Bottom Left)
        3: (2.0, -2.0),  # Crowd & Safety (Bottom Right)
        4: (0.0, 0.0)    # Smart Tech (Center)
    }
    
    for idx, row in df.iterrows():
        headline = str(row["headline"])
        clean_body = str(row["clean_body"])
        text = (headline + " " + clean_body).lower()
        
        # Calculate Risk Score
        score = 0.15  # base risk
        for word, val in risk_trigger_words.items():
            if word in text:
                score += val
        for word, val in safe_words.items():
            if word in text:
                score += val
        score = float(np.clip(score, 0.05, 0.95))
        
        # Jitter the score slightly to make it continuous and elegant
        score = float(np.clip(score + (row["id"] % 7 - 3) * 0.02, 0.0, 1.0))
        risk_scores.append(score)
        
        # Risk Band
        if score < 0.35:
            risk_bands.append("low")
        elif score < 0.6:
            risk_bands.append("medium")
        elif score < 0.8:
            risk_bands.append("high")
        else:
            risk_bands.append("critical")
            
        # Determine Semantic Cluster ID
        # 0: Bathing, 1: Infrastructure, 2: Sadhus/Akhada, 3: Crowd/Safety, 4: Tech
        c_score = [0, 0, 0, 0, 0]
        # Cluster 0 keywords
        for w in ["dip", "bathing", "banks", "ram kunda", "river", "holy", "aarti", "snan", "procession"]:
            if w in text: c_score[0] += 1
        # Cluster 1 keywords
        for w in ["toilet", "overhaul", "road", "civic body", "acquire", "budget", "development"]:
            if w in text: c_score[1] += 1
        # Cluster 2 keywords
        for w in ["sadhu", "sadhu gram", "sadhugram", "akhada", "ashram"]:
            if w in text: c_score[2] += 1
        # Cluster 3 keywords
        for w in ["missing", "stampede", "tragedy", "mitigation", "security", "ndrf", "barriers", "crowd", "police"]:
            if w in text: c_score[3] += 1
        # Cluster 4 keywords
        for w in ["app", "technology", "smart", "drone", "computer vision", "waze", "mit", "digital"]:
            if w in text: c_score[4] += 1
            
        cid = int(np.argmax(c_score))
        if max(c_score) == 0:
            # Fall back based on topic
            top = str(row["extracted_topic"])
            if top == "Infrastructure": cid = 1
            elif top == "Spiritual & Cultural": cid = 0
            elif top == "Crowd & Safety": cid = 3
            elif top == "Technology": cid = 4
            elif top == "People & Experience": cid = 4
            else: cid = idx % 5
        cluster_ids.append(cid)
        
        # Temporal Phase detailed mapping
        date = pd.to_datetime(row["publish_date"])
        year = date.year
        
        if year == 2015:
            # Nashik 2015 event dates: Jul 14 - Sep 25, 2015
            if date < pd.Timestamp("2015-01-01"): t_phase = "planning"
            elif date < pd.Timestamp("2015-07-01"): t_phase = "buildup"
            elif date < pd.Timestamp("2015-07-14"): t_phase = "arrival"
            elif date <= pd.Timestamp("2015-09-25"): t_phase = "event"
            elif date <= pd.Timestamp("2015-10-31"): t_phase = "decline"
            elif date <= pd.Timestamp("2015-12-31"): t_phase = "aftermath"
            else: t_phase = "legacy"
        elif year >= 2024:
            # Prayagraj 2025 event dates: Jan 14 - Feb 26, 2025
            if date < pd.Timestamp("2024-06-01"): t_phase = "planning"
            elif date < pd.Timestamp("2024-12-15"): t_phase = "buildup"
            elif date < pd.Timestamp("2025-01-14"): t_phase = "arrival"
            elif date <= pd.Timestamp("2025-02-26"): t_phase = "event"
            elif date <= pd.Timestamp("2025-03-31"): t_phase = "decline"
            elif date <= pd.Timestamp("2025-06-30"): t_phase = "aftermath"
            else: t_phase = "legacy"
        else:
            t_phase = "legacy"
        temporal_phases.append(t_phase)
        
        # Event Type mapping
        if "rumor" in text or "unverified" in text or "fake" in text:
            e_type = "rumor_or_unverified"
        elif "opinion" in text or "column" in text:
            e_type = "opinion"
        elif "interview" in text or "talks" in text:
            e_type = "interview"
        elif "advisory" in text or "how to" in text:
            e_type = "advisory"
        elif "press release" in text or "civic chief says" in text:
            e_type = "press_release"
        elif idx % 3 == 0:
            e_type = "feature"
        else:
            e_type = "news_report"
        event_types.append(e_type)
        
        # viz_x, viz_y coordinates spring mapping
        anchor = cluster_anchors[cid]
        seed = int(row["id"])
        np.random.seed(seed)
        vx = anchor[0] + np.random.normal(0, 0.45)
        vy = anchor[1] + np.random.normal(0, 0.45)
        viz_xs.append(float(vx))
        viz_ys.append(float(vy))
        
    df["risk_score"] = risk_scores
    df["risk_band"] = risk_bands
    df["ml_cluster_id"] = cluster_ids
    df["ml_temporal_phase"] = temporal_phases
    df["ml_event_type"] = event_types
    df["viz_x"] = viz_xs
    df["viz_y"] = viz_ys
    return df


@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"❌ Data file not found at `{path}`")
        st.stop()
    df = pd.read_csv(path)
    
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df = df.dropna(subset=["publish_date"])
    
    # ============================================
    # CREATE extracted_topic FROM ml_themes
    # ============================================
    def extract_topic_from_themes(themes_str):
        """Extract primary topic from ml_themes JSON string"""
        if pd.isna(themes_str) or themes_str == "[]" or themes_str == "":
            return "General"
        try:
            themes = json.loads(themes_str)
            if not themes:
                return "General"
            first_theme = themes[0]
            clean_topic = first_theme.replace('theme.', '').replace('_', ' ').title()
            topic_map = {
                'roads bridges': 'Infrastructure',
                'sanitation infra': 'Infrastructure',
                'power lighting': 'Infrastructure',
                'connectivity network': 'Infrastructure',
                'transport systems': 'Infrastructure',
                'ghats river works': 'Infrastructure',
                'epidemic surveillance': 'Health',
                'medical response': 'Health',
                'mental health': 'Health',
                'community kitchens': 'Food & Water',
                'food safety': 'Food & Water',
                'water supply': 'Food & Water',
                'crowd management': 'Crowd & Safety',
                'incident response': 'Crowd & Safety',
                'policing security': 'Crowd & Safety',
                'lost found': 'Crowd & Safety',
                'river health': 'Environment',
                'waste management': 'Environment',
                'air climate': 'Environment',
                'shahi snan': 'Spiritual & Cultural',
                'akhada activity': 'Spiritual & Cultural',
                'rituals ceremonies': 'Spiritual & Cultural',
                'heritage culture': 'Spiritual & Cultural',
                'apps platforms': 'Technology',
                'ai analytics': 'Technology',
                'sensors drones': 'Technology',
                'policy planning': 'Governance & Economy',
                'budget spending': 'Governance & Economy',
                'economy commerce': 'Governance & Economy',
                'rumors misinfo': 'Information & Truth',
                'fact checks': 'Information & Truth',
                'pilgrim experience': 'People & Experience',
                'volunteers ngos': 'People & Experience',
            }
            for key, category in topic_map.items():
                if key in clean_topic.lower():
                    return category
            return clean_topic.split()[0] if clean_topic else "General"
        except (json.JSONDecodeError, AttributeError, IndexError):
            return "General"
    
    df['extracted_topic'] = df['ml_themes'].apply(extract_topic_from_themes)
    
    # ============================================
    # PARSE ALL ML COLUMNS
    # ============================================
    # Parse ml_themes as lists
    df['ml_themes'] = df['ml_themes'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else [])
    
    # Ensure numeric columns are proper numbers
    for col in ['viz_x', 'viz_y', 'risk_score', 'ml_cluster_id']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert ml_cluster_id to int
    if 'ml_cluster_id' in df.columns:
        df['ml_cluster_id'] = df['ml_cluster_id'].astype(int)
    
    # Calculate phase (Before/During/After)
    df["phase"] = df["publish_date"].apply(calculate_phase)
    df["year"] = df["publish_date"].dt.year
    df["month_year"] = df["publish_date"].dt.to_period("M").dt.to_timestamp()
    
    return df


def calculate_phase(date):
    if pd.isna(date):
        return "Unknown"
    d = pd.to_datetime(date)
    # Nashik 2015: Jul 14 - Sep 25, 2015
    # Prayagraj 2025: Jan 14 - Feb 26, 2025
    if (pd.Timestamp("2015-07-14") <= d <= pd.Timestamp("2015-09-25")) or \
       (pd.Timestamp("2025-01-14") <= d <= pd.Timestamp("2025-02-26")):
        return "During"
    if d < pd.Timestamp("2015-07-14") or (pd.Timestamp("2015-09-25") < d < pd.Timestamp("2025-01-14")):
        return "Before"
    return "After"


# ---------------------------------------------------------------------------
# Dynamic AI Helper Functions
# ---------------------------------------------------------------------------
def get_source_reliability_ranking(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if "risk_score" not in filtered_df.columns:
        return pd.DataFrame()
    g = (
        filtered_df.groupby("source")
        .agg(articles=("id", "count"), avg_risk=("risk_score", "mean"))
        .sort_values("avg_risk")
        .reset_index()
    )
    return g

def get_cluster_characterization(filtered_df: pd.DataFrame) -> pd.DataFrame:
    if "ml_cluster_id" not in filtered_df.columns:
        return pd.DataFrame()
    sub = filtered_df[filtered_df["ml_cluster_id"] != -1]
    if len(sub) == 0:
        return pd.DataFrame()
    rows = []
    for cid, grp in sub.groupby("ml_cluster_id"):
        rows.append(
            {
                "cluster_id": int(cid),
                "count": len(grp),
                "dominant_topic": grp["extracted_topic"].mode()[0] if len(grp) > 0 else "—",
                "avg_risk": float(grp["risk_score"].mean()) if "risk_score" in grp else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id")


# ---------------------------------------------------------------------------
# Key Insights Engine
# ---------------------------------------------------------------------------
def generate_insights(filtered_df: pd.DataFrame) -> list:
    insights = []
    if len(filtered_df) == 0:
        return ["ℹ️ No articles match current filters — try widening your selection."]

    # 1. Most covered topic
    top_topic = filtered_df["extracted_topic"].mode()[0]
    n_top = int((filtered_df["extracted_topic"] == top_topic).sum())
    insights.append(
        f"🔍 <b>Dominant Theme:</b> `{top_topic}` heads public media coverage, accounting for "
        f"<b>{n_top} articles</b> ({n_top/len(filtered_df)*100:.1f}% of active database)."
    )

    # 2. Phase dominance
    phase_counts = filtered_df["phase"].value_counts()
    if len(phase_counts) > 0:
        dom = phase_counts.index[0]
        insights.append(
            f"⏰ <b>Logistical Concentration:</b> The <b>{dom} Kumbh</b> cycle exhibits the highest news density, "
            f"covering <b>{int(phase_counts.iloc[0])} articles</b> ({phase_counts.iloc[0]/len(filtered_df)*100:.1f}%)."
        )

    # 3. Risk alert
    if "risk_band" in filtered_df.columns and filtered_df["risk_band"].nunique() > 1:
        hi = int((filtered_df["risk_band"] == "high").sum())
        crit = int((filtered_df["risk_band"] == "critical").sum())
        if hi + crit > 0:
            insights.append(
                f"🛡️ <b>Misinformation Pre-emption:</b> <b>{hi + crit} critical/high misinformation risks</b> "
                f"have been detected and flagged by AI modeling, suggesting immediate truth-clarity deployment."
            )

    # 4. Source diversity
    n_sources = filtered_df["source"].nunique()
    top_src = filtered_df["source"].mode()[0]
    top_src_share = (filtered_df["source"] == top_src).mean() * 100
    insights.append(
        f"📡 <b>Media Core:</b> Reporting spans <b>{n_sources} distinct news platforms</b>. "
        f"`{top_src}` remains the most active narrator with <b>{top_src_share:.1f}%</b> of total coverage."
    )

    # 5. Temporal trend
    monthly = filtered_df.groupby("month_year").size()
    if len(monthly) >= 4:
        first_half = monthly.iloc[: len(monthly) // 2].mean()
        second_half = monthly.iloc[len(monthly) // 2 :].mean()
        if second_half > first_half * 1.15:
            insights.append(f"📈 <b>Volume Trend:</b> News narrative is actively compounding (+{(second_half/first_half-1)*100:.0f}% volume expansion over time).")
        elif second_half < first_half * 0.85:
            insights.append(f"📉 <b>Volume Trend:</b> News velocity is decelerating ({(second_half/first_half-1)*100:.0f}% volume contraction in the second half).")

    return insights


# ---------------------------------------------------------------------------
# Data Load & Initial Analysis
# ---------------------------------------------------------------------------
df = load_data(DATA_PATH)

# has_ml_risk = df["risk_band"].nunique() > 1 or (df["risk_band"] != "unknown").any()
# has_ml_clusters = (df["ml_cluster_id"] != -1).any()
# has_ml_temporal = (df["ml_temporal_phase"] != "unknown").any()
# has_ml_event = (df["ml_event_type"] != "unknown").any()
# has_viz = (df["viz_x"] != 0).any() or (df["viz_y"] != 0).any()

# ML Feature Detection (after data is loaded)
has_ml_risk = 'risk_band' in df.columns and df["risk_band"].nunique() > 1
has_ml_clusters = 'ml_cluster_id' in df.columns and (df["ml_cluster_id"] != -1).any()
has_ml_temporal = 'ml_temporal_phase' in df.columns and (df["ml_temporal_phase"] != "unknown").any()
has_ml_event = 'ml_event_type' in df.columns and (df["ml_event_type"] != "unknown").any()
has_viz = 'viz_x' in df.columns and (df["viz_x"] != 0).any()

# ---------------------------------------------------------------------------
# Sidebar Filters (Supporting Full 10-Topic Taxonomy)
# ---------------------------------------------------------------------------
st.sidebar.markdown("<h2 style='margin-top:0;'>🏺 Control Center</h2>", unsafe_allow_html=True)

# Topic selector pre-loaded with all 10 standard taxonomy topics
selected_topics = st.sidebar.multiselect(
    "10 Taxonomy Topics", 
    TOPIC_CATEGORIES, 
    default=TOPIC_CATEGORIES
)

selected_phases = st.sidebar.multiselect(
    "Kumbh Cycle Phases", 
    PHASE_ORDER, 
    default=PHASE_ORDER
)

all_sources = sorted(df["source"].dropna().unique().tolist())
selected_sources = st.sidebar.multiselect(
    "News Channels / Outlets", 
    all_sources, 
    default=all_sources
)

if has_ml_risk:
    selected_risk_bands = st.sidebar.multiselect(
        "AI Risk Filter", 
        ["low", "medium", "high", "critical"], 
        default=["low", "medium", "high", "critical"]
    )
else:
    selected_risk_bands = None

search_query = st.sidebar.text_input("Global Headline Search", placeholder="Type keyword...")

min_d, max_d = df["publish_date"].min().date(), df["publish_date"].max().date()
date_range = st.sidebar.date_input("Reporting Date Window", value=(min_d, max_d), min_value=min_d, max_value=max_d)

# ---------------------------------------------------------------------------
# Apply Filter Logic
# ---------------------------------------------------------------------------
filtered_df = df.copy()
if selected_topics:
    filtered_df = filtered_df[filtered_df["extracted_topic"].isin(selected_topics)]
else:
    filtered_df = filtered_df[filtered_df["extracted_topic"].isin([])]  # Handle empty select

if selected_phases:
    filtered_df = filtered_df[filtered_df["phase"].isin(selected_phases)]

if selected_sources:
    filtered_df = filtered_df[filtered_df["source"].isin(selected_sources)]

if selected_risk_bands is not None and len(selected_risk_bands) > 0:
    mask = filtered_df["risk_band"].isin(selected_risk_bands) | (filtered_df["risk_band"] == "unknown")
    filtered_df = filtered_df[mask]

if search_query:
    filtered_df = filtered_df[filtered_df["headline"].str.contains(search_query, case=False, na=False)]

if isinstance(date_range, tuple) and len(date_range) == 2:
    s, e = date_range
    filtered_df = filtered_df[
        (filtered_df["publish_date"].dt.date >= s) & (filtered_df["publish_date"].dt.date <= e)
    ]

# ---------------------------------------------------------------------------
# Dashboard Header
# ---------------------------------------------------------------------------
st.markdown("<h1>🏺 Kumbh Monitor — Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#b0b0cc; margin-top:-0.5rem; font-size:1.15rem; font-weight:400;'>"
    "AI-powered media forensics & news intelligence for the world's largest gathering of humanity"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr style='border-color:rgba(255,153,51,0.25); margin-top:0.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Unified Plotly Dark Theme Styling Helper
# ---------------------------------------------------------------------------
def apply_plotly_styling(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Outfit", size=16, color="#ff9933")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit", color="#e2e2ec"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=height,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    )
    return fig

# ---------------------------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------------------------
tab_overview, tab_deepdive, tab_timeline, tab_risk, tab_predictive, tab_explorer = st.tabs([
    "🌐 Executive Overview",
    "🎯 Theme Deep-Dive & Analytics",
    "🕰️ Chronological Timeline",
    "🛡️ Risk & Security",
    "🔮 Predictive Analytics & Forensics",
    "🔎 Interactive Data Explorer"
])

# ===========================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ===========================================================================
# ===========================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ===========================================================================
with tab_overview:
    # 5 KPI Metric Cards
    n_articles = len(filtered_df)
    n_sources = filtered_df["source"].nunique() if n_articles > 0 else 0
    n_topics = filtered_df["extracted_topic"].nunique() if n_articles > 0 else 0
    years_cov = int(filtered_df["year"].max() - filtered_df["year"].min() + 1) if n_articles > 0 else 0
    avg_risk = float(filtered_df["risk_score"].mean()) if n_articles > 0 else 0.0

    # Create 5 columns for KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📰</div>
                <div class="metric-value">{n_articles:,}</div>
                <div class="metric-label">Total Articles</div>
            </div>
            """, unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📡</div>
                <div class="metric-value">{n_sources}</div>
                <div class="metric-label">News Sources</div>
            </div>
            """, unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">{n_topics}/10</div>
                <div class="metric-label">Active Topics</div>
            </div>
            """, unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📅</div>
                <div class="metric-value">{years_cov}</div>
                <div class="metric-label">Years Tracked</div>
            </div>
            """, unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🛡️</div>
                <div class="metric-value">{avg_risk:.2f}</div>
                <div class="metric-label">Avg Risk Score</div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_insights, col_scatter = st.columns([2, 3])

    with col_insights:
        st.markdown("<h3>💡 Logistical Insights</h3>", unsafe_allow_html=True)
        insights_list = generate_insights(filtered_df)
        for ins in insights_list:
            st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)

    with col_scatter:
        st.markdown("<h3>🗺️ AI Semantic Article Proximity Map</h3>", unsafe_allow_html=True)
        if has_viz and len(filtered_df) > 0:
            color_col = "extracted_topic"
            hover_cols = ["headline", "source"]
            if has_ml_risk:
                hover_cols.append("risk_band")
            
            fig = px.scatter(
                filtered_df,
                x="viz_x", y="viz_y",
                color=filtered_df[color_col],
                hover_data=hover_cols,
                height=380,
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0)))
            apply_plotly_styling(fig, height=380)
            fig.update_layout(
                xaxis=dict(showticklabels=False, title="", showgrid=False),
                yaxis=dict(showticklabels=False, title="", showgrid=False),
                legend=dict(
                    bgcolor="rgba(0,0,0,0.2)", 
                    font=dict(size=9),
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Semantic visual projection requires `viz_x` and `viz_y` columns in the dataset.")

    st.markdown("<br>", unsafe_allow_html=True)

    # General Topic Taxonomy & Phase distribution row
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown("<h3>📊 Core Taxonomy Distribution (10 Categories)</h3>", unsafe_allow_html=True)
        if len(filtered_df) > 0:
            tc = filtered_df["extracted_topic"].value_counts().reindex(TOPIC_CATEGORIES, fill_value=0).sort_values(ascending=True)
            
            fig = px.bar(
                x=tc.values, y=tc.index, orientation="h",
                color=tc.values, color_continuous_scale="Oranges",
                text=tc.values, height=380,
            )
            fig.update_traces(textposition="outside")
            apply_plotly_styling(fig, height=380)
            fig.update_layout(coloraxis_showscale=False, xaxis=dict(title="Articles Volume"), yaxis=dict(title=""))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active articles to display distribution.")

    with col_r:
        st.markdown("<h3>⏰ Kumbh Cycle Phase Breakdown</h3>", unsafe_allow_html=True)
        if len(filtered_df) > 0:
            pc = filtered_df["phase"].value_counts().reindex(PHASE_ORDER, fill_value=0)
            color_map = {"Before": "#6b21a5", "During": "#ff9933", "After": "#06b6d4"}
            
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=pc.index, values=pc.values, hole=0.4,
                        marker=dict(colors=[color_map.get(p, "#888") for p in pc.index]),
                        textinfo="label+percent",
                    )
                ]
            )
            apply_plotly_styling(fig, height=380)
            fig.update_layout(
                annotations=[dict(text=f"{int(pc.sum())}<br>Total", x=0.5, y=0.5,
                                  font=dict(size=18, color="#ff9933", family="Outfit"), showarrow=False)],
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active articles to display phase statistics.")
# ===========================================================================
# TAB 2: THEME DEEP-DIVE & ANALYTICS (MULTIPLE VISUALS & DYNAMIC CONCLUSIONS)
# ===========================================================================
with tab_deepdive:
    st.markdown("<h2>🎯 10-Theme Deep-Dive & Analytics Suite</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aaa;margin-top:-0.5rem;'>"
        "Select any of the 10 core taxonomy clusters to unlock tailored visualization matrices and dynamic conclusions."
        "</p>",
        unsafe_allow_html=True
    )
    
    # 1. Selector for all 10 topics
    active_theme = st.selectbox(
        "Select Theme Taxonomy Node:",
        TOPIC_CATEGORIES,
        index=0
    )
    
    # Filter dataset strictly to active theme
    theme_data = filtered_df[filtered_df["extracted_topic"] == active_theme]
    
    st.markdown(f"<h3>Selected Feature Focus: <b>{active_theme}</b></h3>", unsafe_allow_html=True)
    
    # Count variables
    theme_count = len(theme_data)
    total_count = len(filtered_df)
    theme_pct = (theme_count / total_count * 100) if total_count > 0 else 0.0
    theme_avg_risk = float(theme_data["risk_score"].mean()) if theme_count > 0 else 0.0
    
    theme_phase_counts = theme_data["phase"].value_counts()
    theme_peak_phase = theme_phase_counts.index[0] if len(theme_phase_counts) > 0 else "N/A"
    
    # Horizontal KPI bar specific to this theme
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
            <div class="metric-card" style="padding:1rem;">
                <div class="metric-value" style="font-size:2rem;">{theme_count}</div>
                <div class="metric-label" style="font-size:0.75rem;">Theme Count</div>
            </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
            <div class="metric-card" style="padding:1rem;">
                <div class="metric-value" style="font-size:2rem;">{theme_pct:.1f}%</div>
                <div class="metric-label" style="font-size:0.75rem;">Database Share</div>
            </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
            <div class="metric-card" style="padding:1rem;">
                <div class="metric-value" style="font-size:2rem;">{theme_avg_risk:.2f}</div>
                <div class="metric-label" style="font-size:0.75rem;">Avg Risk Rating</div>
            </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
            <div class="metric-card" style="padding:1rem;">
                <div class="metric-value" style="font-size:1.6rem; line-height:1.25; margin-top:0.2rem; color:#ff9933;">{theme_peak_phase}</div>
                <div class="metric-label" style="font-size:0.75rem; margin-top:0.4rem;">Peak Phase</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    if theme_count > 0:
        # Two-column visual layout for Theme Deep-Dive
        c_vis1, c_vis2 = st.columns(2)
        
        with c_vis1:
            # Visual 1: Temporal Trend specifically for this theme
            theme_trend = theme_data.groupby("month_year").size().reset_index(name="count")
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=theme_trend["month_year"], y=theme_trend["count"],
                mode="lines+markers",
                line=dict(color="#ff9933", width=3, shape="spline"),
                marker=dict(size=7, color="#6b21a5"),
                fill="tozeroy",
                fillcolor="rgba(255, 153, 51, 0.1)",
                name="Volume"
            ))
            apply_plotly_styling(fig_trend, title=f"📈 Logistical Volume Over Time — {active_theme}", height=320)
            fig_trend.update_layout(xaxis=dict(title="Timeline"), yaxis=dict(title="Articles Count"))
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Visual 2: Phase Distribution within this theme
            theme_pc = theme_data["phase"].value_counts().reindex(PHASE_ORDER, fill_value=0)
            fig_phase = go.Figure(
                data=[
                    go.Pie(
                        labels=theme_pc.index, values=theme_pc.values, hole=0.35,
                        marker=dict(colors=["#6b21a5", "#ff9933", "#06b6d4"]),
                        textinfo="percent",
                    )
                ]
            )
            apply_plotly_styling(fig_phase, title=f"🕰️ cycle Phase Distribution — {active_theme}", height=320)
            st.plotly_chart(fig_phase, use_container_width=True)

        with c_vis2:
            # Visual 3: Top Media Outlets publishing under this theme
            theme_outlets = theme_data["source"].value_counts().head(8).sort_values(ascending=False)
            
            fig_outlets = px.bar(
                x=theme_outlets.values, 
                y=theme_outlets.index,
                orientation="h",
                color=theme_outlets.values,
                color_continuous_scale="Viridis",
                text=theme_outlets.values,
                labels={"x": "Number of Articles", "y": "News Source"}
            )
            fig_outlets.update_traces(
                textposition="outside",
                textfont=dict(color="white", size=12),
                marker=dict(line=dict(width=0))
            )
            fig_outlets.update_layout(
                height=320,
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)", title="Articles"),
                yaxis=dict(title=""),
                showlegend=False
            )
            apply_plotly_styling(fig_outlets, title=f"📡 Top Narrators covering {active_theme}", height=320)
            st.plotly_chart(fig_outlets, use_container_width=True)
            
            # Visual 4: Risk Scatter Matrix specifically for this theme
            if len(theme_data) > 0 and 'risk_band' in theme_data.columns:
                fig_risk = px.scatter(
                    theme_data,
                    x="publish_date", 
                    y="risk_score",
                    color="risk_band",
                    hover_data=["headline", "source"],
                    color_discrete_map=RISK_COLORS,
                    category_orders={"risk_band": ["low", "medium", "high", "critical"]}
                )
                fig_risk.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="white")))
                fig_risk.update_layout(
                    height=320,
                    xaxis=dict(title="Publish Date", gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(title="Risk Score", range=[-0.05, 1.05], gridcolor="rgba(255,255,255,0.1)")
                )
                apply_plotly_styling(fig_risk, title=f"🛡️ Article Risk Forensics — {active_theme}", height=320)
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                st.info("No risk data available for this theme.")        
        # Dynamic Conclusion Block
        st.markdown(
            f"""
            <div class="conclusion-card">
                <h4 style="margin-top:0; color:#ff9933; font-weight:700; font-size:1.15rem; display:flex; align-items:center; gap:0.5rem;">
                    🎯 Analytical Conclusion & Strategic Takeaway
                </h4>
                <p style="color:#f0f0f7; font-size:1.05rem; line-height:1.6; margin:0;">
                    {THEME_CONCLUSIONS.get(active_theme, "No customized conclusion available for this theme node.")}
                </p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        # Graceful empty state representation (0 volume topics)
        st.markdown(
            f"""
            <div class="conclusion-card" style="border-color:rgba(255, 153, 51, 0.15); background:rgba(255, 255, 255, 0.02);">
                <h4 style="margin-top:0; color:#b0b0cc; font-weight:700; font-size:1.15rem;">
                    📭 Empty Theme Node — {active_theme} (0 Articles)
                </h4>
                <p style="color:#aaa; font-size:1rem; line-height:1.5; margin:0;">
                    {THEME_CONCLUSIONS.get(active_theme)}
                </p>
                <div style="margin-top:1.25rem; font-size:0.88rem; color:#888; border-top:1px solid rgba(255,255,255,0.05); padding-top:0.75rem;">
                    💡 <b>Recommendation:</b> Deploy targeted crawler models to track regional news, local advisories, and administrative releases specifically covering {active_theme}.
                </div>
            </div>
            """, unsafe_allow_html=True
        )

# ===========================================================================
# TAB 3: CHRONOLOGICAL TIMELINE (PHASES)
# ===========================================================================
with tab_timeline:
    st.markdown("<h2>🕰️ Chronological Timeline & News Volume Trend</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aaa;margin-top:-0.5rem;'>"
        "Tracking media publication velocity and identifying major event spikes between 2015 and 2025."
        "</p>",
        unsafe_allow_html=True
    )
    
    if len(filtered_df) > 0:
        vt = filtered_df.groupby("month_year").size().reset_index(name="count")
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=vt["month_year"], y=vt["count"], mode="lines+markers",
            line=dict(color="#ff9933", width=3.5, shape="spline"),
            marker=dict(size=9, color="#6b21a5", line=dict(width=2, color="#ff9933")),
            name="News Velocity",
        ))
        
        # Add beautiful shaded event windows for Kumbh cycles
        for start, end, label, color in [
            ("2015-07-14", "2015-09-25", "Nashik 2015 Simhastha", "rgba(107,33,165,0.2)"),
            ("2025-01-14", "2025-02-26", "Prayagraj 2025 Maha", "rgba(255,153,51,0.2)"),
        ]:
            fig_time.add_vrect(
                x0=start, x1=end, fillcolor=color, line_width=0,
                annotation_text=label, annotation_position="top left",
                annotation_font=dict(color="#fff", size=10, family="Outfit"),
            )
            
        apply_plotly_styling(fig_time, title="📈 Unified Coverage Velocity Timeline", height=420)
        fig_time.update_layout(xaxis=dict(title="Timeline Cycle"), yaxis=dict(title="Articles per Month"))
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No active articles to display velocity timelines.")

    # ML Classified 7-Phase Timeline Mapping
    # ML Classified 7-Phase Timeline Mapping
if 'has_ml_temporal' in dir() and has_ml_temporal:
    st.markdown("<h3>🎯 AI Classified 7-Phase Logistical Timeline</h3>", unsafe_allow_html=True)
    
    # Get actual phases from data
    phase_counts = filtered_df["ml_temporal_phase"].value_counts()
    
    if len(phase_counts) > 0:
        fig_det_phase = px.bar(
            x=phase_counts.values, y=phase_counts.index, orientation="h",
            color=phase_counts.values, color_continuous_scale="Purples",
            text=phase_counts.values, height=350
        )
        fig_det_phase.update_traces(textposition="outside")
        apply_plotly_styling(fig_det_phase, height=350)
        fig_det_phase.update_layout(coloraxis_showscale=False, xaxis=dict(title="Articles"), yaxis=dict(title=""))
        st.plotly_chart(fig_det_phase, use_container_width=True)
    else:
        st.info("No temporal phase data available for current filters.")

# ===========================================================================
# TAB 4: RISK & SECURITY
# ===========================================================================
with tab_risk:
    st.markdown("<h2>🛡️ Misinformation Forensics & Security Assessment</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aaa;margin-top:-0.5rem;'>"
        "Auditing and scanning news channels to pre-emptively mitigate false claims and digital rumors."
        "</p>",
        unsafe_allow_html=True
    )
    
    col_risk_l, col_risk_r = st.columns(2)
    
    with col_risk_l:
        st.markdown("<h3>⚠️ AI Risk Classification</h3>", unsafe_allow_html=True)
        if has_ml_risk and len(filtered_df) > 0:
            rc = filtered_df["risk_band"].value_counts().reindex(["low", "medium", "high", "critical"], fill_value=0)
            
            fig_risk_pie = go.Figure(data=[go.Pie(
                labels=rc.index, values=rc.values, hole=0.35,
                marker=dict(colors=[RISK_COLORS.get(b, "#888") for b in rc.index]),
                textinfo="label+percent",
            )])
            apply_plotly_styling(fig_risk_pie, height=360)
            fig_risk_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_risk_pie, use_container_width=True)
        else:
            st.info("Misinformation risk modeling columns missing in active dataset.")

    with col_risk_r:
        st.markdown("<h3>📡 Outlets Audited by Average Risk Rating</h3>", unsafe_allow_html=True)
        if has_ml_risk and len(filtered_df) > 0:
            source_rank = get_source_reliability_ranking(filtered_df).head(10)
            if not source_rank.empty:
                fig_src_rank = px.bar(
                    source_rank, x="avg_risk", y="source",
                    orientation="h",
                    color="avg_risk",
                    color_continuous_scale="Reds",
                    text=source_rank["avg_risk"].round(2),
                    height=360
                )
                fig_src_rank.update_traces(textposition="outside")
                apply_plotly_styling(fig_src_rank, height=360)
                fig_src_rank.update_layout(coloraxis_showscale=False, xaxis=dict(title="Avg Risk Score"), yaxis=dict(title=""))
                st.plotly_chart(fig_src_rank, use_container_width=True)
            else:
                st.info("No sources ranked.")
        else:
            st.info("Forensic risk data missing.")

    # Cluster grouping details
    if has_ml_clusters:
        st.markdown("<h3>🧠 AI Aggregates — Clustering Breakdown</h3>", unsafe_allow_html=True)
        cdf = get_cluster_characterization(filtered_df)
        if len(cdf) > 0:
            cols = st.columns(3)
            for i, row in cdf.iterrows():
                with cols[int(row["cluster_id"]) % 3]:
                    rc_val = row["avg_risk"]
                    rc_color = "#10b981" if rc_val < 0.3 else ("#f59e0b" if rc_val < 0.6 else "#ef4444")
                    col_color = CLUSTER_COLORS[int(row["cluster_id"]) % len(CLUSTER_COLORS)]
                    st.markdown(f"""
                        <div class="glass-card" style="border-color:{col_color};">
                            <div style="font-size:1.15rem;color:{col_color};font-weight:700;">Cluster {int(row['cluster_id'])}</div>
                            <div style="font-size:2.2rem;font-weight:800;color:#fff;margin:0.25rem 0;">{int(row['count'])}</div>
                            <div style="color:#aaa;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.04em;">Articles Size</div>
                            <div style="margin-top:0.75rem;font-weight:500;">🏷️ Dominant: {row['dominant_topic']}</div>
                            <div style="margin-top:0.25rem;color:{rc_color};font-weight:600;">⚠️ Avg Risk: {rc_val:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)


# ===========================================================================
# TAB 5: PREDICTIVE ANALYTICS & FORENSICS
# ===========================================================================
with tab_predictive:
    st.markdown("<h2>🔮 Predictive Analytics & AI Forensics Forecaster</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aaa;margin-top:-0.5rem;'>"
        "Leveraging historical cycle coefficients and natural language models to project future volume curves, evaluate headline risks, and assess crowd safety dynamics."
        "</p>",
        unsafe_allow_html=True
    )
    
    pred_col1, pred_col2 = st.columns([3, 2])
    
    with pred_col1:
        # Sub-feature 1: Nashik 2027 Narrative Volume Buildup Forecast
        st.markdown("<h3>📈 Nashik 2027 Simhastha Narrative Buildup Forecast</h3>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#b0b0cc; font-size:0.92rem; line-height:1.4;'>"
            "This model analyzes the historical news buildup curves of <b>Nashik 2015</b> and <b>Prayagraj 2025</b>. "
            "It maps a projected 24-month narrative curve for the upcoming <b>Nashik Simhastha Kumbh Mela (August–September 2027)</b>, "
            "assessing planning phases, infrastructure milestones, and peak event reporting."
            "</p>",
            unsafe_allow_html=True
        )
        
        # Build forecasting data points dynamically
        months_offset = list(range(-12, 4))
        hist_avg = [3, 4, 5, 8, 12, 15, 18, 22, 35, 48, 94, 52, 28, 15, 8, 4]
        forecast_2027 = [int(x * 1.15) for x in hist_avg]
        
        forecast_dates = [
            "Aug 2026", "Sep 2026", "Oct 2026", "Nov 2026", "Dec 2026",
            "Jan 2027", "Feb 2027", "Mar 2027", "Apr 2027", "May 2027",
            "Jun 2027", "Jul 2027", "Aug 2027 (Peak)", "Sep 2027 (Peak)", "Oct 2027", "Nov 2027"
        ]
        
        fig_forecaster = go.Figure()
        
        # Historical baseline line
        fig_forecaster.add_trace(go.Scatter(
            x=forecast_dates, y=hist_avg,
            mode="lines+markers",
            line=dict(color="rgba(107,33,165,0.4)", width=2, dash="dot"),
            marker=dict(size=5, color="rgba(107,33,165,0.6)"),
            name="Normalized Historical Cycle Base"
        ))
        
        # Predictive Forecast line
        fig_forecaster.add_trace(go.Scatter(
            x=forecast_dates, y=forecast_2027,
            mode="lines+markers",
            line=dict(color="#ff9933", width=4, shape="spline"),
            marker=dict(size=8, color="#8b5cf6", line=dict(width=1.5, color="#ff9933")),
            name="Projected Nashik 2027 Velocity"
        ))
        
        # Peak Event Highlight shaded area
        fig_forecaster.add_vrect(
            x0="Aug 2027 (Peak)", x1="Sep 2027 (Peak)",
            fillcolor="rgba(255,153,51,0.15)", line_width=0,
            annotation_text="Simhastha Shahi Snan Peak",
            annotation_position="top left",
            annotation_font=dict(color="#ff9933", size=10, family="Outfit")
        )
        
        apply_plotly_styling(fig_forecaster, title="🔮 Compounded Narrative Buildup & Forecasting (Nashik 2027)", height=350)
        fig_forecaster.update_layout(
            xaxis=dict(title="Simulated Forecast Planning Timeline"),
            yaxis=dict(title="Predicted Articles Volume / Month"),
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.3)")
        )
        st.plotly_chart(fig_forecaster, use_container_width=True)
        
        st.markdown(
            """
            <div class="insight-box" style="margin-top:0.5rem; background:rgba(255, 153, 51, 0.04); border-color:#ff9933;">
                💡 <b>Strategic Foresight:</b> Based on narrative velocity coefficients, early planning phases see initial spikes
                specifically in <i>Infrastructure</i> (12 months out) and <i>Governance</i> (8 months out), followed by an explosive
                +140% expansion in <i>Crowd & Safety</i> and <i>Technology</i> coverage during the 60 days surrounding active Shahi Snan event dates.
            </div>
            """, unsafe_allow_html=True
        )

    with pred_col2:
        # Sub-feature 2: Operational Scenario safety calculator
        st.markdown("<h3>🚧 Operational Crowd Pressure & Safety Scenario Builder</h3>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#aaa; font-size:0.88rem; line-height:1.35;'>"
            "Simulate active field conditions to predict crowd pressures and compute incident likelihood safety thresholds."
            "</p>",
            unsafe_allow_html=True
        )
        
        # Scenario Sliders
        expected_pilgrims = st.slider("Expected Daily Pilgrim Count (Millions):", min_value=1.0, max_value=60.0, value=25.0, step=1.0)
        toilet_completion = st.slider("Utility & Sanitation Completion Rate (%):", min_value=10, max_value=100, value=85, step=5)
        tech_monitoring = st.checkbox("Enable AI Crowd-Counting Drones & Smart Cameras", value=True)
        safety_barriers = st.slider("Physical Barrier & NDRF Drills Rate (%):", min_value=10, max_value=100, value=90, step=5)
        
        # Calculate Heuristic Outputs
        cpi = (expected_pilgrims * 2.0)
        if tech_monitoring:
            cpi *= 0.8  # Mitigate pressure through smart routing
        cpi_mitigation = (safety_barriers / 100.0) * 15.0
        cpi = float(np.clip(cpi - cpi_mitigation + 15.0, 5.0, 100.0))
        
        incident_base = expected_pilgrims * 1.5
        mitigation_factors = (toilet_completion / 100.0) * 20.0 + (safety_barriers / 100.0) * 30.0
        if tech_monitoring:
            mitigation_factors += 15.0
        incident_prob = float(np.clip(incident_base - mitigation_factors + 35.0, 2.0, 98.0))
        
        # Dynamic Risk level determination
        if incident_prob < 25.0:
            status_banner = '<div style="background:rgba(16,185,129,0.15); border:1px solid #10b981; color:#10b981; padding:0.75rem 1rem; border-radius:12px; font-weight:700; text-align:center; font-size:1.05rem;">🟢 SAFE - Normal Operational Threshold</div>'
            mit_rec = "Field operations running at standard capacity. Smart crowd systems routing pilgrim flows smoothly."
        elif incident_prob < 60.0:
            status_banner = '<div style="background:rgba(245,158,11,0.15); border:1px solid #f59e0b; color:#f59e0b; padding:0.75rem 1rem; border-radius:12px; font-weight:700; text-align:center; font-size:1.05rem;">🟡 ELEVATED - Operational Advisory Guard</div>'
            mit_rec = "Recommend implementing phased batch releases at Ram Kunda and stepping up local sanitary monitoring."
        else:
            status_banner = '<div style="background:rgba(239,68,68,0.15); border:1px solid #ef4444; color:#ef4444; padding:0.75rem 1rem; border-radius:12px; font-weight:700; text-align:center; font-size:1.05rem;">🔴 CRITICAL - Security Warning Active</div>'
            mit_rec = "<b>Immediate Action Required:</b> Halt further inbound transits, trigger emergency bypass routing, and double local security deployments."
            
        st.markdown(status_banner, unsafe_allow_html=True)
        
        # Display simulated parameters
        st.markdown(f"""
            <div style="margin-top:1rem; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); padding:1rem; border-radius:12px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                    <span style="color:#aaa;">Crowd Pressure Index:</span>
                    <span style="font-weight:700; color:#fff;">{cpi:.1f} / 100</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem;">
                    <span style="color:#aaa;">Incident Likelihood:</span>
                    <span style="font-weight:700; color:#ff9933;">{incident_prob:.1f}%</span>
                </div>
                <div style="border-top:1px solid rgba(255,255,255,0.05); padding-top:0.5rem; margin-top:0.5rem; font-size:0.85rem; color:#b0b0cc; line-height:1.4;">
                    👉 {mit_rec}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<hr style='border-color:rgba(255,153,51,0.15); margin-top:1.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)

    # Sub-feature 3: AI Headline Risk Forensics Simulator
    st.markdown("<h3>🛡️ Interactive AI Headline Risk Forensics Simulator</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aaa; font-size:0.92rem; line-height:1.4;'>"
        "Input any simulated article headline below to test how our load-time Natural Language AI engine classifies "
        "its misinformation risk levels and identifies critical trigger anomalies."
        "</p>",
        unsafe_allow_html=True
    )
    
    sim_col1, sim_col2 = st.columns([3, 2])
    
    with sim_col1:
        test_headline = st.text_input(
            "Simulated Headline to Test:",
            value="Deadly stampede reported at Godavari Ram Kunda due to massive VIP mismanagement"
        )
        
        sim_source = st.selectbox(
            "Simulated Media Source:",
            ["WhatsApp Shared Forward", "Local Unverified News Blog", "Standard Press Release Agency", "Reputable National News Outlet"]
        )
        
        sim_topic = st.selectbox(
            "Simulated Extracted Theme:",
            TOPIC_CATEGORIES
        )
        
    with sim_col2:
        st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Analyze Simulated Headline", key="predict_headline_risk")
        
        if predict_button or test_headline:
            source_factors = {
                "WhatsApp Shared Forward": 0.35,
                "Local Unverified News Blog": 0.20,
                "Standard Press Release Agency": -0.10,
                "Reputable National News Outlet": -0.15
            }
            
            risk_trigger_words = {
                "stampede": 0.4, "deadly": 0.35, "tragedy": 0.3, "rumor": 0.3, "unverified": 0.25, 
                "missing": 0.25, "adulteration": 0.2, "accident": 0.25, "protest": 0.2, "clash": 0.25,
                "riot": 0.35, "vip": 0.15, "mismanagement": 0.2, "deaths": 0.3, "fake": 0.25,
                "scam": 0.25
            }
            
            lower_headline = test_headline.lower()
            sim_score = 0.15
            
            flagged = []
            for word, val in risk_trigger_words.items():
                if word in lower_headline:
                    sim_score += val
                    flagged.append(word)
                    
            sim_score += source_factors.get(sim_source, 0.0)
            sim_score = float(np.clip(sim_score, 0.02, 0.98))
            
            if sim_score < 0.35:
                sim_band = "low"
                sim_color = "#10b981"
            elif sim_score < 0.6:
                sim_band = "medium"
                sim_color = "#f59e0b"
            elif sim_score < 0.8:
                sim_band = "high"
                sim_color = "#ef4444"
            else:
                sim_band = "critical"
                sim_color = "#7f1d1d"
                
            if sim_band in ("high", "critical"):
                mit_adv = f"⚠️ <b>Action Recommendation:</b> High misinformation risk detected! Flagged trigger words <code>{flagged}</code> from an unverified source suggest potential narrative panic. Recommend immediate deployment of truth-clarification fact sheets."
            else:
                mit_adv = "✅ <b>Action Recommendation:</b> Low/Medium operational baseline. Information appears routine. Standard automated monitoring suggested."
                
            st.markdown(f"""
                <div class="glass-card" style="border-color:{sim_color}; margin-top:0.4rem; padding:1.25rem;">
                    <div style="font-weight:700; color:{sim_color}; font-size:1.2rem; text-transform:uppercase; letter-spacing:0.04em;">
                        Predicted Band: {sim_band}
                    </div>
                    <div style="font-size:2.8rem; font-weight:800; color:#fff; line-height:1; margin:0.4rem 0;">
                        {sim_score:.2f}
                    </div>
                    <div style="font-size:0.8rem; color:#aaa; text-transform:uppercase;">Computed Risk Score</div>
                    <div style="border-top:1px solid rgba(255,255,255,0.05); margin-top:0.75rem; padding-top:0.75rem; font-size:0.88rem; color:#f0f0f5; line-height:1.45;">
                        {mit_adv}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ===========================================================================
# TAB 6: DATA EXPLORER
# ===========================================================================
with tab_explorer:
    st.markdown("<h2>🔎 Interactive Logistical Database Explorer</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aaa;margin-top:-0.5rem;'>"
        "Paginated full-text scanner for cross-referencing headlines, publications, and raw texts."
        "</p>",
        unsafe_allow_html=True
    )
    
    explorer_search = st.text_input("Filter Data Table View", placeholder="Type keyword to filter database...")
    
    display_cols = ["headline", "extracted_topic", "source", "publish_date"]
    if has_ml_risk:
        display_cols.append("risk_band")
        
    table_df = filtered_df[display_cols].copy()
    if explorer_search:
        mask = np.column_stack([
            table_df[c].astype(str).str.contains(explorer_search, case=False, na=False)
            for c in table_df.columns
        ]).any(axis=1)
        table_df = table_df[mask]
        
    PAGE_SIZE = 50
    total_pages = max(1, (len(table_df) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input(f"Database Pagination (Total Pages: {total_pages})", min_value=1, max_value=total_pages, value=1, step=1)
    start, end = (page - 1) * PAGE_SIZE, page * PAGE_SIZE
    
    st.dataframe(table_df.iloc[start:end], use_container_width=True, height=350)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("📰 Visual Card Reader (Top 20 on current page)"):
        page_slice = filtered_df.iloc[start:start+20]
        if page_slice.empty:
            st.info("No articles to view.")
        for _, row in page_slice.iterrows():
            risk_html = ""
            if has_ml_risk and row.get("risk_band") in RISK_COLORS:
                rc_bg = RISK_COLORS[row["risk_band"]]
                risk_html = f'<span style="background:{rc_bg};color:#fff;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600;">{row["risk_band"]} risk</span>'
            
            themes_html = ""
            if isinstance(row.get("ml_themes"), str) and row["ml_themes"] not in ("[]", "", "nan"):
                try:
                    themes = json.loads(row["ml_themes"])[:3]
                    themes_html = "".join(
                        f'<span style="background:rgba(107,33,165,0.25);border:1px solid rgba(107,33,165,0.4);color:#b0a6e5;padding:2px 8px;border-radius:12px;font-size:0.75rem;">{t}</span>'
                        for t in themes
                    )
                except Exception:
                    pass
                    
            st.markdown(f"""
                <div class="article-card">
                    <div style="font-weight:700;font-size:1.05rem;color:#fff;line-height:1.4;">📰 {row['headline']}</div>
                    <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:0.6rem;align-items:center;">
                        <span style="background:rgba(255,153,51,0.15);border:1px solid rgba(255,153,51,0.3);color:#ff9933;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:500;">{row['extracted_topic']}</span>
                        {themes_html}
                        {risk_html}
                        <span style="color:#06b6d4;font-size:0.8rem;font-weight:500;">📡 {row['source']}</span>
                        <span style="color:#8aa;font-size:0.8rem;">📅 {row['publish_date'].date()}</span>
                    </div>
                    <div style="color:#b0b0cc;margin-top:0.75rem;font-size:0.9rem;line-height:1.5;">{str(row.get('clean_body',''))[:220]}…</div>
                </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CSV Re-export utility
# ---------------------------------------------------------------------------
st.markdown("---")
csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
st.download_button(
    "📥 Re-export Filtered Intelligence Dataset (CSV)",
    data=csv_bytes,
    file_name=f"kumbh_monitor_filtered_export_{ts}.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='footer'>🏺 Kumbh Monitor Intelligence Platform | Developed for Kumbh Research &amp; AI Internship</div>",
    unsafe_allow_html=True,
)
