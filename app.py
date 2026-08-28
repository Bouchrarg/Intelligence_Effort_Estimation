import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import os
import sys
import re

# ── Chemin vers le scraper ──────────────────────────────────────────────────
_SCRAPER_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__) or '.'), 'ScrappingEnsam')
if _SCRAPER_DIR not in sys.path:
    sys.path.insert(0, _SCRAPER_DIR)

try:
    import scraper as _scraper_module
    from scraper import GitHubClient, extract_features
except ImportError:
    _scraper_module = None
    GitHubClient = None
    extract_features = None

# ── Box-Cox (λ=0.287) ───────────────────────────────────────────────────────
LAMBDA = 0.287

def inverse_transform_target(y_t):
    return float(np.power(max(float(y_t), 0), 1.0 / LAMBDA) - 1)

# ── Chargement modèle ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(
        os.path.abspath(os.path.dirname(__file__) or '.'),
        'models', 'best_v5_model_pipeline.joblib'
    )
    return joblib.load(path)

# ── Parse URL ────────────────────────────────────────────────────────────────
def parse_github_url(url):
    url = url.strip().rstrip('/')
    m = re.search(r"github\.com/([^/]+)/([^/?#]+)", url)
    if m:
        return m.group(1), m.group(2)
    m2 = re.match(r"^([^/]+)/([^/?#]+)$", url)
    if m2:
        return m2.group(1), m2.group(2)
    return None, None

# ── Build features + prédiction ──────────────────────────────────────────────
def build_and_predict(feat, size_kb=0):
    total_commits = feat.get("total_commits", 50)

    if size_kb > 0 and feat.get("net_loc", 0) == 0:
        net_loc = (size_kb * 1024) / 35
    else:
        net_loc = feat.get("net_loc") or total_commits * 15
    churn_loc = total_commits * 30

    feat["net_loc"]   = round(net_loc, 1)
    feat["churn_loc"] = round(churn_loc, 1)
    feat["active_days"] = feat.get("active_days") or max(total_commits // 2, 1)
    feat["code_churn_normalized"] = round(churn_loc / max(feat["active_days"], 1), 2)

    kloc = max(net_loc / 1000.0, 0.1)
    cocomo_pm = 2.4 * (kloc ** 1.05)
    if feat.get("bus_factor_ratio", 0) > 0.7:
        cocomo_pm *= 1.2
    cocomo_hours = round(cocomo_pm * 160, 1)

    churn_hours      = round(churn_loc / 15.0, 1)
    cycle_time_hours = round(
        feat.get("pr_merge_time_median_h", 24.0)
        * max(feat.get("active_contributors", 1), 1), 1
    )
    feat["cycle_time_hours"] = cycle_time_hours
    effort_target = round(0.5 * churn_hours + 0.3 * cycle_time_hours + 0.2 * cocomo_hours, 1)
    feat["effort_target"] = effort_target

    ac = max(feat.get("active_contributors", 1), 1)
    feat["effort_contributors_seen"]   = ac
    feat["effort_was_capped"]          = False
    feat["contributors_x_release"]     = ac * feat.get("release_regularity", 0)
    feat["contributors_x_busfactor"]   = ac * feat.get("bus_factor_ratio", 0)
    feat["pr_per_contributor"]         = feat.get("pr_count_merged", 0) / (ac + 1)
    feat["process_maturity"]           = (
        feat.get("ci_success_rate", 0) * 0.4
        + float(feat.get("has_tests", 0)) * 0.3
        + feat.get("release_regularity", 0) * 0.3
    )
    feat["inactivity_burden"]          = feat.get("days_inactive", 0) * np.log1p(ac)
    feat["experience_per_contributor"] = feat.get("weighted_experience", 0) / (ac + 1)

    COLS = [
        'full_name', 'stars', 'created_at', 'days_inactive', 'code_churn_normalized',
        'pr_merge_time_median_h', 'issues_resolution_time_h', 'active_contributors',
        'bus_factor_ratio', 'pr_count_merged', 'review_cycle_count', 'has_ci',
        'ci_success_rate', 'has_tests', 'weighted_experience', 'commit_velocity_trend',
        'release_regularity', 'weekend_commit_ratio', 'dependency_count', 'language_diversity',
        'total_commits', 'net_loc', 'active_days', 'cycle_time_hours', 'effort_target',
        'reliability_score', 'effort_contributors_seen', 'effort_was_capped',
        'contributors_x_release', 'contributors_x_busfactor', 'pr_per_contributor',
        'process_maturity', 'inactivity_burden', 'experience_per_contributor'
    ]
    for k in COLS:
        if k not in feat:
            feat[k] = 0.0
    feat['full_name']  = 0.0
    feat['created_at'] = 0.0

    df = pd.DataFrame([feat])[COLS]
    model = load_model()
    bc = model.predict(df)[0]
    pred = max(inverse_transform_target(bc), 0.0)

    decomp = {
        "churn_hours":  churn_hours,
        "cycle_hours":  cycle_time_hours,
        "cocomo_hours": cocomo_hours,
        "net_loc":      net_loc,
    }
    return pred, decomp


# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GitHub Effort Estimator · ENSAM v5",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, .stApp {
    background: #0a0f1a !important;
    color: #dce6f5 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Inputs */
.stTextInput > label, .stNumberInput > label {
    color: #7a90b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: .02em !important;
}
.stTextInput input, .stNumberInput input {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #dce6f5 !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}

/* Button */
.stButton > button {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 13px 0 !important;
    width: 100% !important;
    letter-spacing: .02em !important;
    transition: background .15s !important;
}
.stButton > button:hover {
    background: #2563eb !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}
[data-testid="metric-container"] label {
    color: #4b6a9b !important;
    font-size: 10px !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #dce6f5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Divider */
hr { border-color: #1e2d45 !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #7a90b8 !important;
    font-size: 13px !important;
}

/* Hide chrome */
#MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1.5rem;">
    <div style="
        display:inline-block; font-size:11px; letter-spacing:.18em;
        text-transform:uppercase; color:#3b82f6;
        border:1px solid rgba(59,130,246,.3); padding:4px 16px;
        border-radius:4px; background:rgba(59,130,246,.06); margin-bottom:18px;
    ">ENSAM · ML Pipeline v5 · Stacking Ensemble</div>
    <h1 style="
        font-family:'Inter',sans-serif; font-size:36px; font-weight:600;
        color:#dce6f5; margin:0 0 10px; letter-spacing:-.02em;
    ">GitHub Effort Estimator</h1>
    <p style="color:#6b84aa; font-size:14px; line-height:1.7; margin:0;">
        Estimez l'effort de développement d'un dépôt GitHub en <strong style="color:#dce6f5">heures·personne</strong><br>
        Voting Ensemble (RandomForest + LightGBM + GradientBoosting) · +1 300 repos d'entraînement
    </p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# FORMULAIRE URL
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="
    background:#111827; border:1px solid #1e2d45; border-radius:14px;
    padding:24px 28px; margin-bottom:8px;
">
    <div style="font-size:11px; letter-spacing:.12em; text-transform:uppercase;
                color:#4b6a9b; margin-bottom:16px;">
        // Dépôt GitHub
    </div>
""", unsafe_allow_html=True)

repo_url = st.text_input(
    "URL du dépôt",
    placeholder="https://github.com/owner/repository",
    label_visibility="collapsed"
)
gh_token = st.text_input(
    "GitHub Token (optionnel)",
    type="password",
    placeholder="GitHub Personal Access Token (optionnel — recommandé pour éviter le rate-limit)"
)

st.markdown("""
    <p style="font-size:12px; color:#3d5478; margin-top:-6px; margin-bottom:4px;">
        ⏱ Le scraping prend ~15–30 s selon la taille du repo.
    </p>
</div>
""", unsafe_allow_html=True)

btn = st.button("⚡  Analyser ce dépôt")


# ════════════════════════════════════════════════════════════════════════════
# ANALYSE
# ════════════════════════════════════════════════════════════════════════════
if btn:
    if not repo_url.strip():
        st.error("Veuillez entrer une URL GitHub.")
        st.stop()

    owner, repo_name = parse_github_url(repo_url)
    if not owner:
        st.error("URL invalide. Exemple : https://github.com/tensorflow/tensorflow")
        st.stop()

    if _scraper_module is None:
        st.error(f"Module 'scraper' introuvable dans : {_SCRAPER_DIR}")
        st.stop()

    with st.spinner(f"Analyse de **{owner}/{repo_name}** en cours…"):
        try:
            orig_sleep = time.sleep
            _scraper_module.time.sleep = lambda x: None

            class FastClient(GitHubClient):
                def _get(self, path, params=None, retries=1):
                    u = path if path.startswith("http") else f"{self.BASE}{path}"
                    for _ in range(retries):
                        try:
                            r = self.session.get(u, params=params, timeout=8)
                            if r.status_code == 204: return {}
                            if r.status_code == 404: return None
                            if r.status_code == 202: return None
                            if r.status_code in (403, 429):
                                raise Exception("Rate limit GitHub — ajoutez un Token.")
                            r.raise_for_status()
                            return r.json()
                        except Exception as e:
                            if "Rate limit" in str(e): raise e
                    return None

                def get_code_frequency(self, full_name):
                    return []

            client = FastClient(gh_token or None)
            repo_info = client._get(f"/repos/{owner}/{repo_name}")
            if not repo_info:
                st.error(f"Dépôt introuvable : {owner}/{repo_name}")
                st.stop()

            feat = extract_features(client, repo_info)
            _scraper_module.time.sleep = orig_sleep

            if not feat:
                st.error("Dépôt filtré (trop petit, inactif ou trop peu de stars).")
                st.stop()

            size_kb = repo_info.get("size", 0)
            pred, decomp = build_and_predict(feat, size_kb=size_kb)
            st.session_state["result"] = {
                "pred": pred, "feat": feat, "decomp": decomp,
                "source": f"{owner}/{repo_name}"
            }

        except Exception as e:
            _scraper_module.time.sleep = orig_sleep
            st.error(f"Erreur : {e}")
            with st.expander("Détail de l'erreur"):
                import traceback
                st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════
# RÉSULTATS
# ════════════════════════════════════════════════════════════════════════════
if "result" in st.session_state:
    r      = st.session_state["result"]
    pred   = r["pred"]
    feat   = r["feat"]
    decomp = r["decomp"]

    pred_low  = max(pred * 0.50, 1)
    pred_high = pred * 1.50
    equiv_pm  = pred / 160.0
    ac        = max(int(feat.get("active_contributors", 1)), 1)
    per_contrib = pred / ac
    weeks     = pred / 40.0

    st.divider()

    # ── Chiffre principal ──────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        background:#111827; border:1px solid #1e2d45; border-radius:14px;
        padding:32px; text-align:center; margin-bottom:16px;
    ">
        <div style="font-size:10px; letter-spacing:.18em; text-transform:uppercase;
                    color:#4b6a9b; margin-bottom:10px;">
            // {r["source"]} · Effort estimé
        </div>
        <div style="
            font-family:'JetBrains Mono',monospace; font-size:64px; font-weight:600;
            color:#60a5fa; line-height:1; margin-bottom:12px;
        ">{int(pred):,} <span style="font-size:28px; color:#7a90b8; font-weight:400">h</span></div>
        <div style="
            display:inline-block; background:rgba(59,130,246,.1);
            border:1px solid rgba(59,130,246,.25); border-radius:20px;
            padding:6px 22px; font-size:14px; color:#93c5fd;
        ">≈ {equiv_pm:.1f} personne·mois</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Intervalle de confiance ────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Borne basse (−50%)", f"{int(pred_low):,} h")
    c2.metric("Estimation ML",      f"{int(pred):,} h")
    c3.metric("Borne haute (+50%)", f"{int(pred_high):,} h")

    st.caption("MAPE modèle ≈ 50–60% · PRED(50) ≈ 61%")

