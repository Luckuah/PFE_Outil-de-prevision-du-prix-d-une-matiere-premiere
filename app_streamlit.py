# ==================== STREAMLIT DASHBOARD – VERSION REFACTORISÉE (FIXES UX + BUGS) ====================
# Corrections :
# 1. Bug NameError corrigé (fonction définie AVANT usage)
# 2. Page AVANT /analyze inchangée (aucun graphique affiché)
# 3. Graphique prix pétrole déplacé dans l'onglet 2 (post-analyse uniquement)
# 4. Jauge refaite : aiguille + label LOW / MEDIUM / HIGH (pas de %)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from config_param import DataConfig, ModelConfig, CrisesConfig, UIConfig
from data_utils import prepare_full_dataset
from model_training import load_and_predict, prepare_data, forecast_future

# ==================== CONFIG PAGE ====================
st.set_page_config(
    page_title=UIConfig.APP_TITLE,
    page_icon="🛢️",
    layout="wide"
)

# ==================== SESSION STATE ====================
if "crises" not in st.session_state:
    st.session_state.crises = CrisesConfig.CRISES.copy()

if "results" not in st.session_state:
    st.session_state.results = None

# ==================== DATA ====================
@st.cache_data(ttl=UIConfig.CACHE_TTL)
def get_data(crises_dict):
    return prepare_full_dataset(crises_dict=crises_dict)

# ==================== GRAPHIQUE FUTUR ====================
def afficher_graphique_futur(df_historique, resultats):
    df_recent = df_historique.tail(15).copy()
    preds_futures = resultats['future_predictions']
    basse, haute = resultats['intervals']['future']

    derniere_date = df_recent.index[-1]
    dates_futures = pd.date_range(
        start=derniere_date + pd.Timedelta(days=1),
        periods=len(preds_futures)
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_recent.index,
        y=df_recent['Close'],
        name="Historique récent",
        line=dict(color=UIConfig.HISTORICAL_COLOR, width=3)
    ))

    fig.add_trace(go.Scatter(
        x=dates_futures,
        y=preds_futures,
        name="Prédiction LSTM",
        line=dict(color=UIConfig.PREDICTION_COLOR, dash="dash", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_futures, dates_futures[::-1]]),
        y=np.concatenate([haute, basse[::-1]]),
        fill='toself',
        fillcolor='rgba(214,39,40,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name="Intervalle de confiance"
    ))

    fig.update_layout(template=UIConfig.PLOT_TEMPLATE, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ==================== SIDEBAR ====================
st.sidebar.title("🛢️ Oil Market AI")
st.sidebar.button("✅ Health")
st.sidebar.markdown("""
### 🎓 Projet PFE
Analyse et prédiction du prix du pétrole
via LSTM + indicateurs macro-financiers
avec explications RAG.
""")

# ==================== PAGE ANALYZE ====================
st.title("📊 Analyse & Prédiction du Marché du Pétrole")

# --------- AVANT ANALYSE (inchangé) ---------
st.info("Sélectionnez une période et lancez l'analyse pour obtenir les prédictions.")

df = get_data(st.session_state.crises)

col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("Période d'analyse")
    min_date = df.index.min().to_pydatetime()
    max_date = df.index.max().to_pydatetime()

    selected_dates = st.date_input(
        "",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# AUCUN graphique ici AVANT le clic

# --------- LANCER ANALYSE ---------
if st.button("🔍 Lancer l'analyse"):
    with st.spinner("Analyse en cours..."):
        bundle = load_and_predict(ModelConfig.MODEL_PATH, st.session_state.crises)
        df_recent = bundle['df']
        model = bundle['model']

        X_scaled, y_scaled, scaler_X, scaler_y = prepare_data(df_recent, DataConfig.FEATURE_COLS)
        last_sequence = X_scaled[-ModelConfig.LOOKBACK:]

        preds, low, high = forecast_future(
            model,
            last_sequence,
            ModelConfig.FUTURE_STEPS,
            len(DataConfig.FEATURE_COLS),
            scaler_y,
            std_val=1.5,
            lookback=ModelConfig.LOOKBACK
        )

        st.session_state.results = {
            "future_predictions": preds,
            "intervals": {"future": (low, high)},
            "df": df_recent,
            "rag_text": bundle.get("rag_text"),
            "confidence_level": bundle.get("confidence_level", "medium"),
            "rag_sources": bundle.get("rag_sources", [])
        }

# ==================== APRÈS ANALYSE ====================
if st.session_state.results:
    res = st.session_state.results

    st.markdown("---")

    left, right = st.columns([3, 1])

    # --------- TEXTE RAG ---------
    with left:
        st.subheader("🧠 Explications & Facteurs")
        st.markdown(res.get("rag_text", ""))

    # --------- RÉSUMÉ + JAUGE ---------
    with right:
        st.subheader("📈 Résumé de la prédiction")

        current_price = df['Close'].iloc[-1]
        predicted_price = res['future_predictions'][0]
        pct = (predicted_price - current_price) / current_price * 100

        st.metric("Prix actuel", f"${current_price:.2f}")
        st.metric("Prix prédit", f"${predicted_price:.2f}", f"{pct:.2f}%")

        # ----- Jauge aiguille (LOW / MEDIUM / HIGH) -----
        conf = res['confidence_level']
        conf_map = {"low": 10, "medium": 50, "high": 90}
        conf_color = {"low": "#e74c3c", "medium": "#f1c40f", "high": "#2ecc71"}

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge",
            value=conf_map[conf],
            gauge={
                'axis': {'range': [0, 100], 'tickvals': []},
                'bar': {'color': 'rgba(0,0,0,0)'},
                'steps': [
                    {'range': [0, 33], 'color': '#fbeaea'},
                    {'range': [33, 66], 'color': '#fdf3d6'},
                    {'range': [66, 100], 'color': '#eaf7f0'},
                ],
                'threshold': {
                    'line': {'color': conf_color[conf], 'width': 6},
                    'thickness': 0.75,
                    'value': conf_map[conf]
                }
            }
        ))

        fig_gauge.add_annotation(
            x=0.5,
            y=0.15,
            text=f"<b><span style='color:{conf_color[conf]}'>{conf.upper()}</span></b>",
            showarrow=False,
            font=dict(size=18)
        )

        fig_gauge.update_layout(height=260, margin=dict(t=20, b=0, l=0, r=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # ==================== ONGLETS ====================
    tab1, tab2, tab3 = st.tabs([
        "🔮 Prévision à 10 jours",
        "📊 Données d'entraînement",
        "📚 Sources RAG"
    ])

    with tab1:
        afficher_graphique_futur(res['df'], res)

    with tab2:
        mask = (df.index >= pd.Timestamp(selected_dates[0])) & (df.index <= pd.Timestamp(selected_dates[1]))
        df_filtered = df.loc[mask]
        fig_train = px.line(
            df_filtered,
            x=df_filtered.index,
            y='Close',
            title="Cours du pétrole sur la période d'entraînement",
            template=UIConfig.PLOT_TEMPLATE
        )
        st.plotly_chart(fig_train, use_container_width=True)

    with tab3:
        for src in res.get("rag_sources", []):
            st.markdown(f"- {src}")
