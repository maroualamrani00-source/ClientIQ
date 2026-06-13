"""
ClientIQ — Version Finale
Sprint 5 — Master Ingenierie de Decision — PFE 2026
Le dashboard montre ce qui s'est passe ; ClientIQ dit quoi faire et le fait executer.
"""
import os, sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

st.set_page_config(page_title="ClientIQ", page_icon="💎", layout="wide", initial_sidebar_state="expanded")
from utils.data_processor import DataProcessor
from utils.decision_engine import DecisionEngine

# ═══════════════════════════════════════════
#  PALETTE — Blanc / Beige / Teal Profonde
# ═══════════════════════════════════════════
C = {
    'bg':       '#FAFBFC',
    'bg2':      '#F5F0EB',
    'bg3':      '#F0ECE6',
    'bg4':      '#E8E4DE',
    'card':     '#FFFFFF',
    'card2':    '#FDFCFA',
    'sf':       '#F7F5F2',
    'sf2':      '#F0ECE6',
    'bd':       '#E8E4DE',
    'bd2':      '#D6D0C8',
    'tx':       '#1A1A2E',
    'tx2':      '#5A5A72',
    'tx3':      '#8A8A9A',
    'teal':     '#0D7377',
    'teal_l':   '#14919B',
    'teal_d':   '#0A5C5F',
    'teal_bg':  '#E6F5F5',
    'amb':      '#D4A017',
    'amb_l':    '#E8B830',
    'amb_bg':   '#FFF8E1',
    'coral':    '#E07A5F',
    'coral_l':  '#EC9A85',
    'coral_bg': '#FFF0EB',
    'mint':     '#2D9B83',
    'mint_l':   '#4BC5A8',
    'mint_bg':  '#E8F8F4',
    'sky':      '#3B82C4',
    'sky_l':    '#5DA0D8',
    'sky_bg':   '#EBF3FB',
    'rose':     '#C74060',
    'rose_l':   '#E06585',
    'rose_bg':  '#FDE8EE',
    'purple':   '#7C5CBF',
    'purple_l': '#9A7CD4',
    'purple_bg':'#F0EBF8',
    'orange':   '#D68C45',
    'orange_l': '#E8A860',
    'orange_bg':'#FFF3E0',
}

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ===== BASE ===== */
.stApp {
    background: linear-gradient(170deg, #FAFBFC 0%, #F5F0EB 40%, #FDFCFA 70%, #F0ECE6 100%);
    color: #1A1A2E;
    font-family: 'Inter', system-ui, sans-serif;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F7F5F2 40%, #F0ECE6 100%) !important;
    border-right: 1px solid #E8E4DE !important;
}
section[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 11px 16px !important; border-radius: 12px !important;
    border: 1px solid transparent !important;
    transition: all .3s cubic-bezier(.4,0,.2,1) !important;
    color: #5A5A72 !important; font-size: .86rem !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(13,115,119,.06) !important;
    border-color: rgba(13,115,119,.15) !important; color: #1A1A2E !important;
}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(13,115,119,.10), rgba(45,155,131,.06)) !important;
    border-color: #0D7377 !important; color: #0D7377 !important; font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(13,115,119,.08) !important;
}

/* ===== HEADINGS ===== */
h1, h2, h3, h4, h5, h6 {
    color: #1A1A2E !important; font-family: 'Inter', system-ui, sans-serif !important;
}

/* ===== GLASS CARD ===== */
.glass {
    background: #FFFFFF; border: 1px solid #E8E4DE; border-radius: 16px;
    padding: 24px; margin: 8px 0;
    box-shadow: 0 2px 12px rgba(0,0,0,.04), 0 1px 3px rgba(0,0,0,.02);
    transition: all .3s ease;
}
.glass:hover {
    border-color: #D6D0C8; box-shadow: 0 4px 20px rgba(0,0,0,.06);
}

/* ===== KPI CARDS ===== */
.kpi {
    background: #FFFFFF; border-radius: 14px;
    border-left: 4px solid #0D7377; padding: 16px 20px; margin: 4px 0;
    transition: all .3s ease; box-shadow: 0 2px 8px rgba(0,0,0,.03);
}
.kpi:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,.06); }
.kpi.teal { border-left-color: #0D7377; }
.kpi.amb  { border-left-color: #D4A017; }
.kpi.rose { border-left-color: #C74060; }
.kpi.mint { border-left-color: #2D9B83; }
.kpi.sky  { border-left-color: #3B82C4; }
.kpi.pink { border-left-color: #E07A5F; }
.kpi.orange { border-left-color: #D68C45; }
.kpi h3 { font-size: .68rem; color: #8A8A9A; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 5px; font-weight: 600; }
.kpi .v { font-size: 1.45rem; font-weight: 700; color: #0D7377; font-family: 'Inter', sans-serif; }
.kpi .s { font-size: .68rem; color: #8A8A9A; margin-top: 2px; }

/* ===== HERO BANNER ===== */
.hero {
    background: linear-gradient(135deg, #0D7377 0%, #14919B 30%, #2D9B83 55%, #3B82C4 80%, #7C5CBF 100%);
    padding: 34px 44px; border-radius: 20px; margin-bottom: 20px;
    text-align: center; position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(13,115,119,.18);
}
.hero::before {
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(212,160,23,.06) 0%, transparent 50%);
    animation: hGlow 6s ease-in-out infinite;
}
@keyframes hGlow { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(3%,3%) scale(1.1)} }
.hero h1 {
    font-size: 2.1rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #FFFFFF, #FFD54F, #FFFFFF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-size: 300% auto; animation: gMove 5s ease-in-out infinite;
}
@keyframes gMove { 0%{background-position:0% center} 50%{background-position:100% center} 100%{background-position:0% center} }
.hero p { color: rgba(255,255,255,.80); font-size: .92rem; margin: 8px 0 0; }

/* ===== PROFILE HEADER ===== */
.phdr {
    background: #FFFFFF; border-radius: 18px;
    border: 1px solid #E8E4DE; padding: 26px 30px; margin-bottom: 20px;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 16px rgba(0,0,0,.04);
}
.phdr::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #0D7377, #2D9B83, #D4A017, #E07A5F, #3B82C4, #7C5CBF, #0D7377);
    background-size: 300% auto; animation: gMove 3s ease-in-out infinite;
}

/* ===== BADGES ===== */
.badge { display:inline-block; padding:4px 14px; border-radius:20px; font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.5px; }
.b-mint   { background:#E8F8F4; color:#0A7B68; border:1px solid #B8E8DA; }
.b-rose   { background:#FDE8EE; color:#A03050; border:1px solid #F0C0D0; }
.b-teal   { background:#E6F5F5; color:#0A5C5F; border:1px solid #B0DDE0; }
.b-amb    { background:#FFF8E1; color:#9A7A10; border:1px solid #E8D88A; }
.b-mag    { background:#F0EBF8; color:#5A3FA0; border:1px solid #CFC0E8; }
.b-orange { background:#FFF3E0; color:#A06820; border:1px solid #E8C888; }
.b-sky    { background:#EBF3FB; color:#2A60A0; border:1px solid #B0CCE8; }
.b-pink   { background:#FFF0EB; color:#B05838; border:1px solid #F0B8A8; }

/* ===== ACTION ITEMS ===== */
.act {
    background: #FDFCFA; border-radius: 12px; border: 1px solid #E8E4DE;
    padding: 14px 18px; margin: 5px 0; transition: all .2s ease;
}
.act:hover { background: #F7F5F2; border-color: #D6D0C8; transform: translateX(4px); }

/* ===== DIVIDERS ===== */
.dv { height: 1px; background: linear-gradient(90deg, transparent, #D6D0C8, #0D7377, #D6D0C8, transparent); margin: 20px 0; }

/* ===== DATAFRAMES ===== */
.dataframe { background-color: #FFFFFF !important; color: #1A1A2E !important; border-radius: 12px !important; overflow: hidden !important; }
.dataframe th { background: linear-gradient(135deg, #0D7377, #14919B) !important; color: white !important; border: none !important; padding: 12px !important; font-weight: 600 !important; font-size: .82rem !important; }
.dataframe td { background-color: #FDFCFA !important; color: #1A1A2E !important; border-bottom: 1px solid #E8E4DE !important; padding: 10px !important; font-size: .82rem !important; }

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #0D7377, #14919B) !important; color: white !important;
    border: 1px solid #0D7377 !important; border-radius: 12px !important;
    padding: 12px 28px !important; font-weight: 600 !important; transition: all .3s !important;
    box-shadow: 0 4px 16px rgba(13,115,119,.18) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #D4A017, #E8B830) !important;
    border-color: #D4A017 !important; transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(212,160,23,.25) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #2D9B83, #4BC5A8) !important; color: white !important;
    border: 1px solid #2D9B83 !important; border-radius: 12px !important; font-weight: 600 !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #F5F0EB; border-radius: 12px; padding: 4px; border: 1px solid #E8E4DE; }
.stTabs [data-baseweb="tab"] { border-radius: 10px !important; color: #5A5A72 !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0D7377, #14919B) !important; color: #FFFFFF !important; }

/* ===== INPUTS ===== */
.stSelectbox > div > div, .stNumberInput > div > div, .stTextInput > div > div {
    background: #FFFFFF !important; border-color: #E8E4DE !important; color: #1A1A2E !important; border-radius: 10px !important;
}
.stCheckbox label { color: #1A1A2E !important; }
.stMetric { background: transparent !important; }
.stMetric label { color: #8A8A9A !important; }
.stMetric [data-testid="stMetricValue"] { color: #0D7377 !important; }

/* ===== EXPANDER ===== */
.streamlit-expanderHeader { background: #FDFCFA !important; border: 1px solid #E8E4DE !important; border-radius: 12px !important; color: #1A1A2E !important; }

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 7px; }
::-webkit-scrollbar-track { background: #F5F0EB; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #0D7377, #14919B); border-radius: 4px; }

/* ===== PLOTLY ===== */
.js-plotly-plot .plotly .modebar { background: transparent !important; }
.js-plotly-plot .plotly .modebar-btn path { fill: #8A8A9A !important; }
.js-plotly-plot .plotly .modebar-btn:hover path { fill: #0D7377 !important; }

/* ===== DECISION PRIORITIES ===== */
.d-crit { border-left: 4px solid #C74060; }
.d-haut { border-left: 4px solid #D4A017; }
.d-moy  { border-left: 4px solid #3B82C4; }

/* ===== ALERTS ===== */
.al-ok   { background: linear-gradient(135deg, #E8F8F4, #F0FBF8); border: 1px solid #B0E0D0; border-radius: 14px; padding: 18px 24px; color: #0A5C5F; }
.al-warn { background: linear-gradient(135deg, #FFF8E1, #FFFBF0); border: 1px solid #E8D88A; border-radius: 14px; padding: 18px 24px; color: #7A6A10; }
.al-info { background: linear-gradient(135deg, #EBF3FB, #F0F7FD); border: 1px solid #B0CCE8; border-radius: 14px; padding: 18px 24px; color: #2A60A0; }

/* ===== SLIDER & RADIO ===== */
.stSlider > div > div > div > div { background: #E8E4DE !important; }
.stRadio div { color: #1A1A2E !important; }
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════
def kpi_card(label, value, cls='', sub=''):
    s = f'<div class="s">{sub}</div>' if sub else ''
    return f'<div class="kpi {cls}"><h3>{label}</h3><div class="v">{value}</div>{s}</div>'


def chart_style(fig, h=380):
    fig.update_layout(
        height=h, margin={'l': 40, 'r': 20, 't': 45, 'b': 40},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1A1A2E', 'family': 'Inter,system-ui,sans-serif', 'size': 12},
        xaxis=dict(gridcolor='#E8E4DE', color='#5A5A72', zerolinecolor='#E8E4DE'),
        yaxis=dict(gridcolor='#E8E4DE', color='#5A5A72', zerolinecolor='#E8E4DE'),
        legend=dict(font=dict(color='#5A5A72'), bgcolor='rgba(0,0,0,0)', bordercolor='#E8E4DE', borderwidth=1),
    )
    return fig


def gauge(val, title, color, mx=None):
    if mx is None:
        mx = val * 1.5 if val > 0 else 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 12, 'color': '#5A5A72'}},
        number={'font': {'size': 24, 'color': color, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, mx], 'tickcolor': '#8A8A9A', 'tickfont': {'size': 8, 'color': '#8A8A9A'}, 'dtick': mx / 5},
            'bar': {'color': color}, 'bgcolor': '#F5F0EB', 'borderwidth': 2, 'bordercolor': '#E8E4DE',
            'steps': [{'range': [0, mx * .3], 'color': '#FDFCFA'}, {'range': [mx * .3, mx * .7], 'color': '#F7F5F2'}],
            'threshold': {'line': {'color': color, 'width': 3}, 'thickness': .8, 'value': val},
        },
    ))
    fig.update_layout(
        height=230, margin={'l': 15, 'r': 15, 't': 50, 'b': 15},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#1A1A2E'},
    )
    return fig


# ═══════════════════════════════════════════
#  LOAD ML PIPELINE
# ═══════════════════════════════════════════
@st.cache_resource
def load_all():
    proc = DataProcessor()
    proc.run_full_pipeline(path=os.path.join(BASE_DIR, 'data', 'ifood_cleaned.csv'))
    eng = DecisionEngine(proc.df_sprint3, proc.models, proc)
    return proc, eng


with st.spinner("Initialisation du moteur ML..."):
    proc, eng = load_all()
df = proc.df_sprint3
models = proc.models


# ═══════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:28px 12px 22px;'>
        <div style='font-size:2.6rem;'>💎</div>
        <div style='font-size:1.25rem;font-weight:800;color:#0D7377;letter-spacing:1px;margin-top:10px;'>ClientIQ</div>
        <div style='font-size:.66rem;color:#8A8A9A;margin-top:6px;letter-spacing:.8px;text-transform:uppercase;line-height:1.4;'>Version Finale<br><span style='color:#0D7377;'>Scorez · Decidez · Executez</span></div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    st.markdown("""<p style='font-size:.66rem;color:#8A8A9A;text-transform:uppercase;letter-spacing:2px;font-weight:600;margin-bottom:10px;padding-left:16px;'>Navigation</p>""", unsafe_allow_html=True)
    PAGES = [
        "🏠  Tableau de Bord",
        "🧠  Scoreur Client",
        "🎯  Campagnes",
        "⚡  Centre de Decisions",
        "👤  Profil Client",
        "🔬  Simulateur What-If",
    ]
    page = st.radio("", PAGES, label_visibility="collapsed")
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    st.markdown("""<div style='padding:12px 16px;background:#FDFCFA;border-radius:12px;border:1px solid #E8E4DE;'>
        <div style='font-size:.64rem;color:#8A8A9A;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px;'>Pipeline ML</div>
        <div style='font-size:.78rem;color:#1A1A2E;'>K-Means · XGBoost · RandomForest<br><span style='color:#0D7377;'>2 205 clients</span> · <span style='color:#2D9B83;'>6 profils</span> · <span style='color:#D4A017;'>Score Unifie</span></div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    st.markdown("""<div style='text-align:center;padding:10px;font-size:.64rem;color:#8A8A9A;'>
        <b style='color:#1A1A2E;'>Maroua Lamrani</b><br><span style='color:#0D7377;'>Master Ingenierie de Decision</span><br><span style='font-size:.58rem;'>Encadrant : Pr. SAHID Abdelkebir<br>Sprint 5 — PFE 2026</span></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  PAGE 1 — TABLEAU DE BORD
# ═══════════════════════════════════════════
if page == "🏠  Tableau de Bord":
    st.markdown("""<div class="hero"><h1>ClientIQ — De l'Analyse a la Decision</h1><p>Le dashboard montre ce qui s'est passe. ClientIQ dit quoi faire maintenant et le fait executer.</p></div>""", unsafe_allow_html=True)
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(kpi_card('Clients', '2 205', '', 'Base totale'), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card('Profils', '6', 'teal', '6 segments'), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card('CLV Moyen', f'{df["CLV_pred_XGBoost"].mean():.1f} €', 'amb', 'XGBoost'), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card('Top 20%', f'{int(df["Top20_Score"].sum()):,}', 'pink', 'Score'), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card('Reponse', f'{df["Response"].mean()*100:.1f}%', 'mint', 'Campagnes'), unsafe_allow_html=True)

    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(gauge(models['kmeans_metrics']['silhouette'], 'Silhouette', C['mint'], 1.0), width='stretch')
    with c2: st.plotly_chart(gauge(models['xgb_clv_metrics']['R2'], 'R² CLV', C['sky'], 1.0), width='stretch')
    with c3: st.plotly_chart(gauge(models['rf_metrics']['ROC_AUC'], 'ROC-AUC', C['amb'], 1.0), width='stretch')

    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    st.markdown("### Distribution des Profils Marketing")

    # Classify all clients
    pc = {}
    for i in range(len(df)):
        pk = eng.classify(df.iloc[i].to_dict())
        pc[pk] = pc.get(pk, 0) + 1

    c1, c2 = st.columns(2)
    with c1:
        sp = sorted(pc.items(), key=lambda x: x[1], reverse=True)
        nm = [eng.PROFILES[p]['titre'] for p, _ in sp]
        cn = [c for _, c in sp]
        cl = [eng.PROFILES[p]['couleur'] for p, _ in sp]
        fig = go.Figure(go.Bar(
            x=nm, y=cn, marker_color=cl,
            text=[f'{c:,}' for c in cn], textposition='outside',
            textfont=dict(color='#1A1A2E', size=11),
        ))
        chart_style(fig.update_layout(
            yaxis=dict(title='Nombre'), xaxis=dict(tickfont=dict(size=9)),
            showlegend=False, bargap=.35,
        ), 420)
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = go.Figure(go.Pie(
            labels=nm, values=cn, hole=.65,
            marker=dict(colors=cl, line=dict(color='#FFFFFF', width=3)),
            textinfo='percent', textfont=dict(size=11, color='#1A1A2E'),
            pull=[.03] * len(nm),
        ))
        fig.update_layout(
            height=420, margin={'l': 10, 'r': 10, 't': 10, 'b': 10},
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', y=-.08, font=dict(color='#5A5A72', size=9)),
            annotations=[dict(text='<b>2 205</b><br>clients', x=.5, y=.5, font_size=16, font_color='#0D7377', showarrow=False)],
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    st.markdown("### Top 5 Clients — Score Unifie")
    top5 = df.nlargest(5, 'Score_Unifie')
    for rank, (idx, row) in enumerate(top5.iterrows()):
        pk = eng.classify(row.to_dict())
        pi = eng.PROFILES[pk]
        fn, fs, fc = eng.fidelity(row.to_dict())
        st.markdown(f"""<div class="glass" style="display:flex;align-items:center;gap:20px;padding:16px 24px;">
            <div style="font-size:1.8rem;font-weight:800;color:#D6D0C8;min-width:40px;text-align:center;">#{rank+1}</div>
            <div style="flex:1;"><div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                <span style="font-weight:700;color:#1A1A2E;font-size:1rem;">Client #{idx}</span>
                <span class="badge b-mag">{pi['icone']} {pi['titre']}</span>
                <span class="badge" style="background:{fc}18;color:{fc};border:1px solid {fc}33;">{fs}/100 — {fn}</span></div>
                <div style="color:#5A5A72;font-size:.76rem;margin-top:5px;">CLV: <b style="color:#2D9B83;">{row['CLV_pred_XGBoost']:.1f}€</b> · P(R): <b style="color:#3B82C4;">{row['P_Response']:.3f}</b> · Score: <b style="color:#D4A017;">{row['Score_Unifie']:.1f}</b> · RFM: <b style="color:#7C5CBF;">{row['RFM_Score']:.0f}/15</b></div></div></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  PAGE 2 — SCOREUR CLIENT
# ═══════════════════════════════════════════
elif page == "🧠  Scoreur Client":
    st.markdown("""<div class="hero"><h1>Scoreur de Nouveaux Clients</h1><p>Entrez les donnees d'un client ou uploadez un CSV — le systeme predit son profil et recommande des actions</p></div>""", unsafe_allow_html=True)
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["✍️ Saisie Manuelle", "📁 Upload CSV"])

    with t1:
        st.markdown("### Saisir un Nouveau Client")
        st.caption("Remplissez les champs. Le systeme calcule automatiquement les metriques derivees.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Informations")
            n_inc = st.number_input("Revenu Annuel (€)", 0, 200000, 40000, key='n1')
            n_age = st.number_input("Age", 18, 100, 40, key='n2')
            n_kid = st.number_input("Enfants (-12 ans)", 0, 5, 0, key='n3')
            n_teen = st.number_input("Adolescents", 0, 5, 0, key='n4')
        with c2:
            st.markdown("#### Activite")
            n_rec = st.number_input("Jours depuis dernier achat", 0, 365, 30, key='n5')
            n_wp = st.number_input("Achats Web", 0, 30, 3, key='n6')
            n_cp = st.number_input("Achats Catalogue", 0, 30, 1, key='n7')
            n_sp = st.number_input("Achats Magasin", 0, 30, 4, key='n8')
            n_wv = st.number_input("Visites Web/mois", 0, 20, 5, key='n9')
            n_dl = st.number_input("Achats avec promo", 0, 15, 2, key='n10')
        with c3:
            st.markdown("#### Depenses (€)")
            n_w = st.number_input("Vins", 0, 1500, 100, key='n11')
            n_fr = st.number_input("Fruits", 0, 200, 20, key='n12')
            n_mt = st.number_input("Viandes", 0, 1000, 80, key='n13')
            n_fi = st.number_input("Poissons", 0, 500, 30, key='n14')
            n_sw = st.number_input("Sucreries", 0, 300, 15, key='n15')
            n_go = st.number_input("Premium", 0, 400, 25, key='n16')
            n_cm = st.number_input("Campagnes acceptees", 0, 6, 1, key='n17')

        if st.button("🧠 Scorez ce Client", type="primary", use_container_width=True):
            data = {
                'Income': n_inc, 'Age': n_age, 'Kidhome': n_kid, 'Teenhome': n_teen,
                'Recency': n_rec, 'NumWebPurchases': n_wp, 'NumCatalogPurchases': n_cp,
                'NumStorePurchases': n_sp, 'NumWebVisitsMonth': n_wv, 'NumDealsPurchases': n_dl,
                'MntWines': n_w, 'MntFruits': n_fr, 'MntMeatProducts': n_mt,
                'MntFishProducts': n_fi, 'MntSweetProducts': n_sw, 'MntGoldProds': n_go,
                'AcceptedCmpTotal': n_cm, 'DaysAsCustomer': 180,
            }
            r = eng.score_new_client(data)
            c = r['client']; pi = r['profil']; fid = r['fidelity']
            st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="phdr"><div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
                <div style="font-size:2.2rem;">{pi['icone']}</div><div>
                <h2 style="margin:0;color:#1A1A2E;font-size:1.5rem;">{pi['titre']}</h2>
                <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;">
                    <span class="badge b-mag">{pi['titre']}</span>
                    <span class="badge" style="background:{fid['color']}18;color:{fid['color']};border:1px solid {fid['color']}33;">Fidelite: {fid['name']} ({fid['score']}/100)</span>
                    <span class="badge b-teal">Segment: {c.get('Segment','N/A')}</span>
                    {'<span class="badge b-amb">⭐ TOP 20%</span>' if r['is_top20'] else ''}</div></div></div>
                <p style="color:#5A5A72;margin:14px 0 0;font-size:.88rem;line-height:1.7;">{pi['description']}</p></div>""", unsafe_allow_html=True)
            kc1, kc2, kc3, kc4, kc5 = st.columns(5)
            with kc1: st.markdown(kpi_card('CLV', f"{c['CLV_pred_XGBoost']:.1f} €", 'mint'), unsafe_allow_html=True)
            with kc2: st.markdown(kpi_card('P(Reponse)', f"{c['P_Response']:.1%}", 'sky'), unsafe_allow_html=True)
            with kc3: st.markdown(kpi_card('Score', f"{c['Score_Unifie']:.1f}", 'amb'), unsafe_allow_html=True)
            with kc4: st.markdown(kpi_card('RFM', f"{c['RFM_Score']:.0f}/15", 'pink'), unsafe_allow_html=True)
            with kc5: st.markdown(kpi_card('Budget', pi['budget_priorite'], 'orange'), unsafe_allow_html=True)
            st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
            ac1, ac2 = st.columns(2)
            with ac1:
                st.markdown("#### Actions Recommandees")
                for i, a in enumerate(r['actions'], 1):
                    st.markdown(f"""<div class="act"><div style="display:flex;align-items:center;gap:10px;">
                        <div style="min-width:30px;height:30px;border-radius:50%;background:{pi['couleur']}15;display:flex;align-items:center;justify-content:center;border:1px solid {pi['couleur']}30;flex-shrink:0;">
                        <span style="color:{pi['couleur']};font-weight:700;font-size:.8rem;">{i}</span></div>
                        <span style="font-size:.88rem;color:#1A1A2E;">{a}</span></div></div>""", unsafe_allow_html=True)
            with ac2:
                st.markdown("#### Details du Scoring")
                dets = [
                    ('Revenu', f"{c.get('Income',0):,.0f} €"),
                    ('Frequence', f"{c.get('Frequency',0)} achats"),
                    ('Panier moyen', f"{c.get('AvgBasketSize',0):.1f} €"),
                    ('Total depenses', f"{c.get('MntTotal',0):,.0f} €"),
                    ('Engagement digital', f"{c.get('DigitalEngagement',0)}"),
                    ('Sensibilite promo', f"{c.get('DealSensitivity',0):.2f}"),
                    ('Canal recommande', pi['canal']),
                    ('ROI attendu', pi['roi_attendu']),
                ]
                for l, v in dets:
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:8px 12px;border-bottom:1px solid #E8E4DE;">
                        <span style="color:#5A5A72;font-size:.84rem;">{l}</span><span style="color:#1A1A2E;font-weight:600;font-size:.84rem;">{v}</span></div>""", unsafe_allow_html=True)

    with t2:
        st.markdown("### Upload CSV de Nouveaux Clients")
        st.caption("Colonnes requises : Income, Age, Recency, NumWebPurchases, MntWines, MntFruits...")
        upl = st.file_uploader("Choisir un CSV", type=['csv'], key='csvu')
        if upl:
            try:
                ndf = pd.read_csv(upl)
                st.markdown(f"""<div class="al-info"><b>✅ Fichier charge : {len(ndf)} nouveaux clients</b></div>""", unsafe_allow_html=True)
                st.dataframe(ndf.head(10), width='stretch', hide_index=True)
                if st.button("🧠 Scorez tous les clients", type="primary", use_container_width=True):
                    res = eng.score_new_clients_batch(ndf)
                    rows = []
                    for rr in res:
                        if 'error' in rr:
                            rows.append({'Statut': 'ERREUR', 'Erreur': rr['error']})
                        else:
                            s = rr['scored']
                            rows.append({
                                'Statut': 'SCORE', 'Profil': s['profil']['titre'],
                                'Fidelite': s['fidelity']['name'],
                                'CLV': round(s['client'].get('CLV_pred_XGBoost', 0), 1),
                                'P(Reponse)': round(s['client'].get('P_Response', 0), 3),
                                'Score': round(s['client'].get('Score_Unifie', 0), 1),
                                'Canal': s['canal'], 'ROI': s['roi_attendu'],
                                'Priorite': s['budget_priorite'],
                            })
                    sdf = pd.DataFrame(rows)
                    st.markdown(f"""<div class="al-ok"><b>✅ {sum(1 for rr in rows if rr['Statut']=='SCORE')} clients scores</b></div>""", unsafe_allow_html=True)
                    st.dataframe(sdf, width='stretch', hide_index=True,
                                 column_config={
                                     'CLV': st.column_config.NumberColumn(format='%.1f €'),
                                     'P(Reponse)': st.column_config.ProgressColumn(format='%.0f%%', min_value=0, max_value=1),
                                 })
                    buf = io.StringIO()
                    sdf.to_csv(buf, index=False)
                    st.download_button("📥 Telecharger CSV", buf.getvalue(), "clients_scores.csv", "text/csv")
            except Exception as e:
                st.markdown(f"""<div class="al-warn"><b>❌ Erreur : {str(e)}</b></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  PAGE 3 — CAMPAGNES
# ═══════════════════════════════════════════
elif page == "🎯  Campagnes":
    st.markdown("""<div class="hero"><h1>Generateur de Campagnes Marketing</h1><p>Ciblez un segment, definissez un budget, lancez une campagne avec liste et KPIs</p></div>""", unsafe_allow_html=True)
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: tgt = st.selectbox("Cible", ["Top 20% Score Unifie", "Clients à Haute Valeur", "Clients Sensibles aux Promotions"])
    with c2: bgt = st.number_input("Budget (€)", 100, 50000, 2000, step=100)
    with c3: go_btn = st.button("🎯 Generer", type="primary", use_container_width=True)

    if go_btn:
        sm = {
            "Clients à Haute Valeur": "Clients à Haute Valeur",
            "Clients Sensibles aux Promotions": "Clients Sensibles aux Promotions",
        }
        camp = eng.generate_campaign(segment_name=sm.get(tgt), budget_total=bgt)
        if camp:
            pc = C['rose'] if camp['priorite'] == 'HAUTE' else C['amb'] if camp['priorite'] == 'MOYENNE' else C['sky']
            uc = C['rose'] if camp['urgence'] == 'IMMÉDIATE' else C['amb']
            st.markdown(f"""<div class="phdr"><h2 style="margin:0 0 8px;color:#1A1A2E;">{camp['nom']}</h2>
                <div style="display:flex;gap:8px;flex-wrap:wrap;">
                    <span class="badge b-mag">{camp['nb_cibles']} clients</span>
                    <span class="badge b-teal">Budget: {camp['budget']}€</span>
                    <span class="badge" style="background:{pc}18;color:{pc};border:1px solid {pc}33;">Priorite: {camp['priorite']}</span>
                    <span class="badge" style="background:{uc}18;color:{uc};border:1px solid {uc}33;">{camp['urgence']}</span>
                    <span class="badge b-mint">ROI: {camp['est_roi']}%</span></div></div>""", unsafe_allow_html=True)
            kc1, kc2, kc3, kc4 = st.columns(4)
            with kc1: st.markdown(kpi_card('Conversion', f"{camp['est_conversion']}%", 'mint'), unsafe_allow_html=True)
            with kc2: st.markdown(kpi_card('Revenu', f"{camp['est_revenu']:,.0f} €", 'amb'), unsafe_allow_html=True)
            with kc3: st.markdown(kpi_card('Profit', f"{camp['est_profit']:,.0f} €", camp['est_profit'] > 0 and 'mint' or 'rose'), unsafe_allow_html=True)
            with kc4: st.markdown(kpi_card('ROI', f"{camp['est_roi']}%", camp['est_roi'] > 100 and 'amb' or 'sky'), unsafe_allow_html=True)
            st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
            ac1, ac2 = st.columns(2)
            with ac1:
                st.markdown("#### Plan d'Action")
                for i, a in enumerate(camp['actions'], 1):
                    st.markdown(f"""<div class="act"><div style="display:flex;align-items:center;gap:10px;">
                        <div style="min-width:30px;height:30px;border-radius:50%;background:rgba(13,115,119,.12);display:flex;align-items:center;justify-content:center;border:1px solid rgba(13,115,119,.25);">
                        <span style="color:#0D7377;font-weight:700;font-size:.8rem;">{i}</span></div>
                        <span style="font-size:.88rem;color:#1A1A2E;">{a}</span></div></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div style="margin-top:14px;padding:14px 18px;background:#E6F5F5;border:1px solid #B0DDE0;border-radius:12px;">
                    <div style="font-size:.7rem;color:#8A8A9A;text-transform:uppercase;letter-spacing:1px;">Canal Recommande</div>
                    <div style="color:#0D7377;font-weight:600;margin-top:4px;">{camp['canal']}</div>
                    <div style="color:#5A5A72;font-size:.78rem;margin-top:2px;">Frequence: {camp['frequence']}</div></div>""", unsafe_allow_html=True)
            with ac2:
                st.markdown("#### Liste des Clients Cibles")
                cldf = pd.DataFrame(camp['clients'])
                st.dataframe(cldf, width='stretch', hide_index=True,
                             column_config={
                                 'CLV_Predit': st.column_config.NumberColumn(format='%.1f €'),
                                 'P_Reponse': st.column_config.ProgressColumn(format='%.0f%%', min_value=0, max_value=1),
                             })
                buf = io.StringIO()
                cldf.to_csv(buf, index=False)
                st.download_button("📥 Exporter Campagne CSV", buf.getvalue(), f"campagne_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

            # ── Moteur de Recommandation Marketing — Message marketing généré par LLM ──
            st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
            st.markdown("### Moteur de Recommandation Marketing — Contenu Marketing")
            mkt = eng.generate_marketing_message(camp)
            if mkt['llm_used']:
                st.markdown(f'<span class="badge b-amb">🤖 IA ({mkt["llm_model"]})</span> &nbsp; <span class="badge b-teal">Généré par LLM</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge b-orange">📋 Template (règles métier)</span> &nbsp; <span class="badge b-sky">Définissez ANTHROPIC_API_KEY pour le mode LLM</span>', unsafe_allow_html=True)
            st.markdown(f"""<div class="glass" style="font-family:Georgia,serif;line-height:1.8;color:#1A1A2E;white-space:pre-wrap;">{mkt['message']}</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  PAGE 4 — CENTRE DE DECISIONS
# ═══════════════════════════════════════════
elif page == "⚡  Centre de Decisions":
    st.markdown("""<div class="hero"><h1>Centre de Decisions</h1><p>Actions prioritaires a prendre maintenant — Du data analyst au decideur</p></div>""", unsafe_allow_html=True)
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    nd = st.slider("Nombre de decisions", 10, 200, 50, step=10)
    decs = eng.get_decisions(top_n=nd)
    if decs:
        nc = sum(1 for d in decs if d['priorite'] == 'CRITIQUE')
        nh = sum(1 for d in decs if d['priorite'] == 'HAUTE')
        clv_r = sum(d['clv_perdu'] for d in decs)
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1: st.markdown(kpi_card('Decisions', str(len(decs)), 'rose', 'Total'), unsafe_allow_html=True)
        with kc2: st.markdown(kpi_card('Critiques', str(nc), 'rose', 'Urgence'), unsafe_allow_html=True)
        with kc3: st.markdown(kpi_card('Hautes', str(nh), 'amb', 'Fidelisation'), unsafe_allow_html=True)
        with kc4: st.markdown(kpi_card('CLV a Risque', f'{clv_r:,.0f} €', 'orange', 'Si aucune action'), unsafe_allow_html=True)
        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
        filt = st.radio("Filtrer", ["Toutes", "CRITIQUE", "HAUTE", "MOYENNE"], horizontal=True)
        fd = [d for d in decs if d['priorite'] == filt] if filt != "Toutes" else decs
        st.markdown(f"### {len(fd)} decision(s) — {filt}")
        for d in fd[:30]:
            pcls = 'd-crit' if d['priorite'] == 'CRITIQUE' else 'd-haut' if d['priorite'] == 'HAUTE' else 'd-moy'
            pclr = C['rose'] if d['priorite'] == 'CRITIQUE' else C['amb'] if d['priorite'] == 'HAUTE' else C['sky']
            fc = d.get('fidelity_color', '#8A8A9A')
            st.markdown(f"""<div class="glass {pcls}" style="padding:16px 22px;">
                <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px;">
                    <span class="badge" style="background:{pclr}18;color:{pclr};border:1px solid {pclr}33;">{d['priorite']}</span>
                    <span class="badge b-mag">{d['type']}</span>
                    <span class="badge b-teal">Client #{d['client_idx']}</span>
                    <span class="badge b-amb">{d['profil']}</span>
                    <span style="color:#8A8A9A;font-size:.72rem;">Delai: {d['delai']}</span></div>
                <p style="color:#1A1A2E;font-size:.88rem;margin:0 0 6px;line-height:1.6;">{d['action']}</p>
                <div style="display:flex;gap:16px;flex-wrap:wrap;">
                    <span style="color:#5A5A72;font-size:.76rem;">Canal: <b style="color:#0D7377;">{d['canal']}</b></span>
                    <span style="color:#5A5A72;font-size:.76rem;">CLV risque: <b style="color:#C74060;">{d['clv_perdu']:,.0f}€</b></span>
                    <span style="color:#5A5A72;font-size:.76rem;">Fidelite: <b style="color:{fc};">{d['fidelite']}</b></span></div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
        ddf = eng.export_decisions_csv(decs)
        buf = io.StringIO()
        ddf.to_csv(buf, index=False)
        st.download_button("📥 Exporter Decisions CSV", buf.getvalue(), f"decisions_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")


# ═══════════════════════════════════════════
#  PAGE 5 — PROFIL CLIENT  (FIXED BUTTONS)
# ═══════════════════════════════════════════
elif page == "👤  Profil Client":
    st.markdown("""<div class="hero"><h1>Profil Client Detaille</h1><p>Explorez chaque dimension d'un client — Profil, fidelite, depenses, percentiles, actions</p></div>""", unsafe_allow_html=True)
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

    # ——— FIXED: separate session state to avoid widget key conflict ———
    if 'profil_idx' not in st.session_state:
        st.session_state.profil_idx = 0

    c1, c2 = st.columns([1, 3])
    with c1:
        new_idx = st.number_input("Index Client", 0, len(df) - 1, int(st.session_state.profil_idx))
        st.session_state.profil_idx = new_idx
    with c2:
        ca, cb, cc = st.columns(3)
        with ca:
            if st.button("⭐ Top Score", use_container_width=True):
                top_idx = int(df.nlargest(1, 'Score_Unifie').index[0])
                st.session_state.profil_idx = top_idx
                st.rerun()
        with cb:
            if st.button("🎲 Aleatoire", use_container_width=True):
                rand_idx = int(df.sample(1).index[0])
                st.session_state.profil_idx = rand_idx
                st.rerun()
        with cc:
            if st.button("⏰ A Reactiver", use_container_width=True):
                cds = df[df['Recency'] > 50]
                if len(cds) > 0:
                    react_idx = int(cds.nlargest(1, 'Score_Unifie').index[0])
                    st.session_state.profil_idx = react_idx
                    st.rerun()

    # Use the session state index
    idx = int(st.session_state.profil_idx)
    if idx < 0 or idx >= len(df):
        idx = 0
        st.session_state.profil_idx = 0

    pr = eng.full_profile(idx)
    c = pr['client']; pi = pr['pi']; fid = pr['fidelity']; pcts = pr['pct']; sp = pr['spend']

    st.markdown(f"""<div class="phdr"><div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <div style="font-size:2.2rem;">{pi['icone']}</div><div>
        <h2 style="margin:0;color:#1A1A2E;font-size:1.5rem;">Client #{idx}</h2>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;">
            <span class="badge b-mag">{pi['titre']}</span>
            <span class="badge" style="background:{fid['color']}18;color:{fid['color']};border:1px solid {fid['color']}33;">Fidelite: {fid['name']} ({fid['score']}/100)</span>
            <span class="badge b-teal">{c.get('Segment','N/A')}</span></div></div></div>
        <p style="color:#5A5A72;margin:14px 0 0;font-size:.88rem;line-height:1.7;">{pi['description']}</p></div>""", unsafe_allow_html=True)

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    with kc1: st.markdown(kpi_card('CLV', f"{c['CLV_pred_XGBoost']:.1f} €", 'mint'), unsafe_allow_html=True)
    with kc2: st.markdown(kpi_card('P(Rep)', f"{c['P_Response']:.1%}", 'sky'), unsafe_allow_html=True)
    with kc3: st.markdown(kpi_card('Score', f"{c['Score_Unifie']:.1f}", 'amb'), unsafe_allow_html=True)
    with kc4: st.markdown(kpi_card('RFM', f"{c['RFM_Score']:.0f}/15", 'pink'), unsafe_allow_html=True)
    with kc5: st.markdown(kpi_card('Revenu', f"{c['Income']:,.0f} €", 'orange'), unsafe_allow_html=True)

    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["📊 Percentiles", "🛒 Depenses", "🎯 Actions"])

    with t1:
        st.markdown("### Positionnement vs Population")
        for l, d in pcts.items():
            pv = d['pct']; df_ = d['diff']
            cl = C['mint'] if df_ > 10 else C['amb'] if df_ > 0 else C['rose'] if df_ < -10 else C['tx2']
            ar = '▲' if df_ > 0 else '▼' if df_ < 0 else '●'
            st.markdown(f"""<div style="display:flex;align-items:center;gap:14px;padding:9px 0;border-bottom:1px solid #E8E4DE;">
                <div style="min-width:100px;font-weight:600;color:#1A1A2E;font-size:.84rem;">{l}</div>
                <div style="flex:1;height:7px;background:#F0ECE6;border-radius:4px;overflow:hidden;">
                    <div style="height:100%;width:{pv}%;background:linear-gradient(90deg,#0D7377,{cl});border-radius:4px;"></div></div>
                <div style="min-width:50px;text-align:right;font-size:.78rem;color:#8A8A9A;">{pv:.0f}e</div>
                <div style="min-width:100px;text-align:right;"><span style="font-size:.78rem;color:{cl};font-weight:600;">{ar} {df_:+.1f}%</span></div>
                <div style="min-width:70px;text-align:right;font-weight:700;color:#1A1A2E;font-size:.86rem;">{d['val']}</div></div>""", unsafe_allow_html=True)

    with t2:
        st.markdown("### Depenses par Categorie")
        cd = {k: v for k, v in sp.items() if k not in ['total', 'top']}
        cc1, cc2 = st.columns(2)
        with cc1:
            nms = list(cd.keys()); vls = [cd[k]['montant'] for k in nms]; avs = [cd[k]['avg'] for k in nms]
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Client', x=nms, y=vls, marker_color=C['teal'], opacity=.85))
            fig.add_trace(go.Bar(name='Moy. Pop.', x=nms, y=avs, marker_color=C['tx3'], opacity=.35))
            chart_style(fig.update_layout(
                barmode='group', yaxis=dict(title='€'),
                legend=dict(orientation='h', y=-.15),
                xaxis=dict(tickfont=dict(size=8)),
            ), 400)
            st.plotly_chart(fig, width='stretch')
        with cc2:
            ps = [cd[k]['pct'] for k in nms]
            cls = [C['rose'], C['mint'], C['amb'], C['sky'], C['purple_l'], C['orange_l']]
            fig = go.Figure(go.Pie(
                labels=nms, values=ps, hole=.6,
                marker=dict(colors=cls, line=dict(color='#FFFFFF', width=2)),
                textinfo='percent', textfont=dict(size=10, color='#1A1A2E'),
            ))
            fig.update_layout(
                height=400, margin={'l': 10, 'r': 10, 't': 10, 'b': 10},
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', y=-.08, font=dict(color='#5A5A72', size=8)),
                annotations=[dict(text=f'<b>{sp["total"]:.0f}€</b><br>Total', x=.5, y=.5, font_size=14, font_color='#2D9B83', showarrow=False)],
            )
            st.plotly_chart(fig, width='stretch')

    with t3:
        st.markdown(f"### Actions Marketing — {pi['titre']}")
        st.markdown(f"""<div style="margin-bottom:16px;display:flex;gap:8px;flex-wrap:wrap;">
            <span class="badge b-mag">Canal: {pi['canal']}</span>
            <span class="badge b-teal">Frequence: {pi['frequence']}</span>
            <span class="badge b-amb">ROI: {pi['roi_attendu']}</span></div>""", unsafe_allow_html=True)
        for i, a in enumerate(pi['actions'], 1):
            st.markdown(f"""<div class="act"><div style="display:flex;align-items:center;gap:10px;">
                <div style="min-width:30px;height:30px;border-radius:50%;background:{pi['couleur']}15;display:flex;align-items:center;justify-content:center;border:1px solid {pi['couleur']}30;flex-shrink:0;">
                <span style="color:{pi['couleur']};font-weight:700;font-size:.8rem;">{i}</span></div>
                <span style="font-size:.88rem;color:#1A1A2E;">{a}</span></div></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  PAGE 6 — SIMULATEUR WHAT-IF
# ═══════════════════════════════════════════
elif page == "🔬  Simulateur What-If":
    st.markdown("""<div class="hero"><h1>Simulateur What-If</h1><p>Modifiez les caracteristiques d'un client et observez l'impact en temps reel</p></div>""", unsafe_allow_html=True)
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

    # Use separate session state for What-If index too
    if 'sim_idx' not in st.session_state:
        st.session_state.sim_idx = 0

    si = st.number_input("Index Client", 0, len(df) - 1, int(st.session_state.sim_idx), key='si_input')
    st.session_state.sim_idx = si

    dv = eng.df.iloc[si].to_dict()

    st.markdown("### Modifiez les Parametres")
    co1, co2, co3 = st.columns(3)
    with co1:
        st.markdown("#### Revenu & Recence")
        m_i = st.number_input("Revenu (€)", 0, 200000, int(dv.get('Income', 40000)), key='m_i')
        m_r = st.number_input("Recence (j)", 0, 365, int(dv.get('Recency', 30)), key='m_r')
    with co2:
        st.markdown("#### Canaux")
        m_w = st.number_input("Achats Web", 0, 30, int(dv.get('NumWebPurchases', 3)), key='m_w')
        m_cp = st.number_input("Achats Catalogue", 0, 30, int(dv.get('NumCatalogPurchases', 1)), key='m_cp')
        m_sp = st.number_input("Achats Magasin", 0, 30, int(dv.get('NumStorePurchases', 4)), key='m_sp')
        m_wv = st.number_input("Visites Web/mois", 0, 20, int(dv.get('NumWebVisitsMonth', 5)), key='m_wv')
    with co3:
        st.markdown("#### Depenses (€)")
        m_wi = st.number_input("Vins", 0, 1500, int(dv.get('MntWines', 100)), key='m_wi')
        m_mt = st.number_input("Viandes", 0, 1000, int(dv.get('MntMeatProducts', 80)), key='m_mt')
        m_fi = st.number_input("Poissons", 0, 500, int(dv.get('MntFishProducts', 30)), key='m_fi')
        m_go = st.number_input("Premium", 0, 400, int(dv.get('MntGoldProds', 25)), key='m_go')

    if st.button("🔬 Simuler", type="primary", use_container_width=True):
        mods = {
            'Income': m_i, 'Recency': m_r,
            'NumWebPurchases': m_w, 'NumCatalogPurchases': m_cp,
            'NumStorePurchases': m_sp, 'NumWebVisitsMonth': m_wv,
            'MntWines': m_wi, 'MntMeatProducts': m_mt,
            'MntFishProducts': m_fi, 'MntGoldProds': m_go,
        }
        sim = eng.simulate(si, mods)
        dc = sim['delta']; cli = sim['client']
        opk = sim['old_pk']; npk = sim['new_pk']
        opi = sim['old_pi']; npi = sim['new_pi']
        ofn = sim['old_f']; nfn, nfs = sim['new_f']
        changed = sim['changed']

        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
        if changed:
            st.markdown("""<div class="al-ok"><b>🔄 Changement detecte !</b> Le profil ou la fidelite du client a change suite aux modifications.</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="al-info"><b>📊 Aucun changement</b> Les modifications n'ont pas impacte le profil ni la fidelite.</div>""", unsafe_allow_html=True)

        kc1, kc2, kc3 = st.columns(3)
        with kc1:
            st.markdown(f"""<div class="phdr" style="text-align:center;">
                <div style="font-size:.72rem;color:#8A8A9A;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Avant</div>
                <div style="font-size:1.1rem;">{opi['icone']} <b>{opi['titre']}</b></div>
                <div style="font-size:.78rem;color:#5A5A72;">Fidelite: {ofn}</div></div>""", unsafe_allow_html=True)
        with kc2:
            st.markdown(f"""<div style="text-align:center;padding:20px;font-size:2rem;color:#D4A017;">➡️</div>""", unsafe_allow_html=True)
        with kc3:
            st.markdown(f"""<div class="phdr" style="text-align:center;">
                <div style="font-size:.72rem;color:#8A8A9A;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Apres</div>
                <div style="font-size:1.1rem;">{npi['icone']} <b>{npi['titre']}</b></div>
                <div style="font-size:.78rem;color:#5A5A72;">Fidelite: {nfn} ({nfs}/100)</div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
        st.markdown("### Impact Detaille")
        for key, d in dc.items():
            label_map = {
                'CLV_pred_XGBoost': 'CLV', 'P_Response': 'P(Reponse)',
                'Score_Unifie': 'Score Unifie', 'RFM_Score': 'RFM',
                'Frequency': 'Frequence', 'AvgBasketSize': 'Panier Moyen',
            }
            label = label_map.get(key, key)
            av = d['avant']; ap = d['apres']; dl = d['delta']; pt = d['pct']
            cl = C['mint'] if dl > 0 else C['rose'] if dl < 0 else C['tx2']
            ic = '▲' if dl > 0 else '▼' if dl < 0 else '●'
            st.markdown(f"""<div style="display:flex;align-items:center;gap:14px;padding:9px 0;border-bottom:1px solid #E8E4DE;">
                <div style="min-width:110px;font-weight:600;color:#1A1A2E;font-size:.84rem;">{label}</div>
                <div style="min-width:80px;text-align:right;color:#8A8A9A;font-size:.84rem;">{av:.2f}</div>
                <div style="min-width:40px;text-align:center;color:{cl};font-weight:700;font-size:.84rem;">{ic}</div>
                <div style="min-width:80px;text-align:right;font-weight:700;color:#1A1A2E;font-size:.84rem;">{ap:.2f}</div>
                <div style="min-width:80px;text-align:right;"><span style="font-size:.78rem;color:{cl};font-weight:600;">{dl:+.2f} ({pt:+.1f}%)</span></div></div>""", unsafe_allow_html=True)
