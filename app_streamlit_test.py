import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from urllib.parse import urlparse
from config_param import DataConfig, ModelConfig, CrisesConfig, UIConfig
from data_utils import prepare_full_dataset
from model_training import load_and_predict, prepare_data, forecast_future

# =========================
# CONFIG GÉNÉRALE STREAMLIT
# =========================

st.set_page_config(
    page_title="Prévision du prix du pétrole",
    page_icon="🛢️",
    layout="wide"
)

# =========================
# CONFIG API
# =========================

DEFAULT_API_URL = "http://127.0.0.1:8000"

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL

# ==================== SESSION STATE ====================
if "crises" not in st.session_state:
    st.session_state.crises = CrisesConfig.CRISES.copy()

if "results" not in st.session_state:
    st.session_state.results = None

    # ==================== DATA ====================
@st.cache_data(ttl=UIConfig.CACHE_TTL)
def get_data(crises_dict):
    return prepare_full_dataset(crises_dict=crises_dict)

# ============== FONCTIONS UTILITAIRES API ==============

def call_api(method: str, endpoint: str, json=None, timeout: int = 60):
    """
    Petit wrapper pour appeler l'API FastAPI proprement.
    NE MODIFIE PAS LE BACKEND, juste la façon d'appeler.
    """
    url = st.session_state.api_url.rstrip("/") + endpoint
    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        elif method == "POST":
            resp = requests.post(url, json=json, timeout=timeout)
        else:
            raise ValueError("Méthode HTTP non supportée")

        return resp
    except Exception as e:
        st.error(f"❌ Erreur de connexion à l'API ({url}) : {e}")
        return None


def extract_predicted_price(predicted_price_10d):
    """
    Gère les deux cas possibles :
    - liste de prix (J+1..J+10)
    - simple float
    Retourne (array_pred, mean_pred).
    """
    if isinstance(predicted_price_10d, list) and len(predicted_price_10d) > 0:
        arr = np.array(predicted_price_10d, dtype=float)
        return arr, float(arr.mean())
    elif isinstance(predicted_price_10d, (int, float)):
        return np.array([float(predicted_price_10d)]), float(predicted_price_10d)
    else:
        return None, None


def get_last_spot_price_from_market_summary(market_summary: dict):
    """
    Essaie de récupérer un dernier prix 'spot' depuis market_data_summary/daily.
    """
    if not isinstance(market_summary, dict):
        return None

    # Certaines versions utilisent "daily", d'autres "1D"
    daily = (
        market_summary.get("daily")
        or market_summary.get("1D")
        or market_summary.get("Daily")
    )

    if isinstance(daily, dict):
        return daily.get("latest_close")
    return None


# ============== FONCTIONS UI RÉUTILISABLES ==============

def get_confidence_label(conf_level: str) -> str:
    """
    Transforme LOW/MEDIUM/HIGH en un label lisible avec emoji.
    """
    if not isinstance(conf_level, str):
        return "N/A"
    level = conf_level.upper()
    if level == "HIGH":
        return "🟢 Élevée (HIGH)"
    if level == "MEDIUM":
        return "🟠 Moyenne (MEDIUM)"
    if level == "LOW":
        return "🔴 Faible (LOW)"
    return level


def render_summary_header(preds_mean, last_spot, conf_level, timestamp: str = ""):
    """
    Affiche les métriques clés en haut de la page de prévision :
    - Prix moyen prévu à J+10
    - Dernier spot
    - Confiance
    Inclut la variation % entre spot et prévision si possible.
    """
    st.markdown("### 📌 Synthèse rapide")

    m1, m2, m3 = st.columns(3)

    delta_pct_str = None
    if isinstance(preds_mean, (int, float)) and isinstance(last_spot, (int, float)) and last_spot != 0:
        delta_pct = (preds_mean - last_spot) / last_spot * 100
        delta_pct_str = f"{delta_pct:+.2f} %"

    with m1:
        if isinstance(preds_mean, (int, float)):
            st.metric(
                "Prix moyen prévu à J+10",
                f"{preds_mean:.2f} $",
                delta=delta_pct_str
            )
        else:
            st.metric("Prix moyen prévu à J+10", "N/A")

    with m2:
        if isinstance(last_spot, (int, float)):
            st.metric("Dernier prix spot (daily)", f"{last_spot:.2f} $")
        else:
            st.metric("Dernier prix spot (daily)", "N/A")

    with m3:
        st.metric("Confiance de l'agent", get_confidence_label(conf_level))

    if timestamp:
        st.caption(f"⏱️ Analysé le : `{timestamp}`")

    st.markdown("---")


def render_prediction_tab(preds_array, last_spot):
    """
    Affiche le graphique de prévision + tableau.
    Utilisé dans le premier onglet "Prévision 10J".
    """
    if preds_array is None or len(preds_array) == 0:
        st.info("Aucune prévision exploitable trouvée dans `predicted_price_10d`.")
        return

    st.subheader("📈 Prévision sur 10 jours")

    jours = [f"J+{i+1}" for i in range(len(preds_array))]
    df_preds = pd.DataFrame({
        "Jour": jours,
        "Prix prédit ($)": preds_array
    }).set_index("Jour")

    st.line_chart(df_preds, height=300)

    if isinstance(last_spot, (int, float)):
        st.caption(
            f"Dernier spot connu : **{last_spot:.2f} $**. "
            "Les points J+1..J+10 représentent la trajectoire prévue du modèle."
        )

    with st.expander("📋 Détail des valeurs prédites"):
        st.table(df_preds.style.format("{:.2f}"))


def render_timeframe_card(tf_name: str, tf_data: dict):
    """
    Affiche un bloc (card logique) pour un timeframe :
    - 3 métriques principales (close, tendance, volatilité)
    - 2 métriques secondaires (plus haut, plus bas)
    - sparkline des recent_closes
    """
    if not isinstance(tf_data, dict):
        st.write(f"Format inattendu pour `{tf_name}`.")
        return

    st.markdown(f"#### Timeframe : `{tf_name}`")

    latest_close = tf_data.get("latest_close")
    period_high = tf_data.get("period_high")
    period_low = tf_data.get("period_low")
    volatility = tf_data.get("volatility")
    trend = tf_data.get("trend")

    if trend == "UP":
        trend_label = "↗️ Haussière"
    elif trend == "DOWN":
        trend_label = "↘️ Baissière"
    else:
        trend_label = "➖ Neutre"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Dernier close",
            f"{latest_close:.2f} $" if isinstance(latest_close, (int, float)) else "N/A"
        )
    with c2:
        st.metric("Tendance", trend_label)
    with c3:
        st.metric(
            "Volatilité (σ)",
            f"{volatility:.4f}" if isinstance(volatility, (int, float)) else "N/A"
        )

    c4, c5, _ = st.columns(3)
    with c4:
        st.metric(
            "Plus haut période",
            f"{period_high:.2f} $" if isinstance(period_high, (int, float)) else "N/A"
        )
    with c5:
        st.metric(
            "Plus bas période",
            f"{period_low:.2f} $" if isinstance(period_low, (int, float)) else "N/A"
        )

    recent = tf_data.get("recent_closes")
    if isinstance(recent, list) and recent:
        df_recent = pd.DataFrame(
            {"Index": list(range(len(recent))), "Close": recent}
        ).set_index("Index")
        st.line_chart(df_recent, height=150)

    st.markdown("---")


def render_market_timeframes_block(timeframes: dict):
    """
    Affiche l'ensemble des timeframes (15min, 4h, daily, etc.).
    Réutilisé à la fois dans :
    - l'onglet "Contexte de marché" de /analyze
    - la page "Données de marché" (/market-data)
    """
    if not isinstance(timeframes, dict) or not timeframes:
        st.info("Aucun timeframe trouvé dans les données de marché.")
        return

    for tf_name, tf_data in timeframes.items():
        render_timeframe_card(tf_name, tf_data)


def render_explanation_and_factors(explanation: str, key_factors):
    """
    Texte explicatif + liste de facteurs clés, dans l'onglet dédié.
    """
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📝 Explication de l'agent")
        if explanation:
            st.write(explanation)
        else:
            st.info("Aucune explication fournie dans la réponse.")

    with col_right:
        st.subheader("📌 Facteurs clés")
        if isinstance(key_factors, list) and key_factors:
            for f in key_factors:
                st.markdown(f"- 🏷️ {f}")
        else:
            st.write("_Aucun facteur clé explicite._")


def parse_source_item(item):
    """
    Tente de parser un élément de la liste `sources`.
    Les données peuvent arriver sous forme :
    - de dict: {"article_title": "...", "source_url": "..."}
    - ou de string: "article_title: '...' source_url: '...'"
    On renvoie (title, url) au mieux, éventuellement (None, None).
    """
    title, url = None, None

    if isinstance(item, dict):
        title = item.get("article_title") or item.get("title")
        url = item.get("source_url") or item.get("url")
        return title, url

    if isinstance(item, str):
        # extraction très simple, suffisante pour le format d'exemple
        text = item.strip()

        # article_title: '...'
        if "article_title" in text:
            try:
                part_title = text.split("article_title:")[1]
                # on coupe au début de source_url si présent
                if "source_url" in part_title:
                    part_title = part_title.split("source_url")[0]
                title = part_title.strip(" ':-")
            except Exception:
                pass

        if "source_url" in text:
            try:
                part_url = text.split("source_url:")[1]
                # on enlève les quotes simples/doubles et espaces
                url = part_url.strip(" '\"")
            except Exception:
                pass

    return title, url


def domain_from_url(url: str) -> str:
    """
    Extrait le domaine d'une URL pour l'afficher à côté du lien.
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""


def render_sources_tab(sources):
    """
    Affiche la liste des sources dans l'onglet "Sources".
    """
    st.subheader("📰 Articles & sources utilisées")

    if not isinstance(sources, list) or not sources:
        st.info("Aucune source fournie dans la réponse.")
        return

    for item in sources:
        title, url = parse_source_item(item)
        if url:
            domain = domain_from_url(url)
            if not title:
                title = url
            if domain:
                st.markdown(f"- [{title}]({url})  \n  _({domain})_")
            else:
                st.markdown(f"- [{title}]({url})")
        else:
            # fallback : affichage brut
            st.markdown(f"- {item}")

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

# =========================
# SIDEBAR / NAVIGATION
# =========================

with st.sidebar:
    st.title("🛢️ Pétrole – Dashboard")
    st.caption("Interface Streamlit connectée à FastAPI")

    st.markdown("---")

    if st.button("Tester /health"):
        resp = call_api("GET", "/health", timeout=20)
        if resp is not None:
            try:
                resp.raise_for_status()
                st.success(f"✅ /health OK : {resp.json()}")
            except Exception as e:
                st.error(f"❌ /health renvoie une erreur : {e}")
                st.write(resp.text)

    st.markdown("---")
    st.caption(
        "Projet de PFE : prévision du prix du pétrole via **LSTM**, "
        "données de marché multi-timeframes et **RAG + LLM**."
    )


# =========================
# PAGE 1 – ASSISTANT DE PRÉVISION
# =========================

st.title("🔮 Assistant de prévision du prix du pétrole")

st.markdown(
    "Cette page combine : un **modèle LSTM**, des **données de marché** "
    "et un **agent d'analyse (LLM)** pour produire une prévision à 10 jours "
    "et une explication lisible."
)

st.markdown("### 💬 Pose ta question")
default_question = (
    "Quelle est ta prédiction pour le prix du pétrole dans les 10 prochains jours "
    "et quels sont les principaux facteurs à surveiller ?"
)

user_query = st.text_area(
    "Question à envoyer à l'API `/analyze` :",
    value=default_question,
    height=120
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    launch = st.button("Lancer l'analyse", type="primary")

if launch:

    if not user_query.strip():
        st.warning("Merci d'entrer une question avant de lancer l'analyse.")
    else:
        payload = {"user_query": user_query}

        with st.spinner("Analyse en cours..."):
            resp = call_api("POST", "/analyze", json=payload, timeout=600)
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
            df = get_data(st.session_state.crises)

            min_date = df.index.min().to_pydatetime()
            max_date = df.index.max().to_pydatetime()

        if resp is None:
            st.stop()

        # Si l'API renvoie une erreur, on reste concis, détails dans expander.
        if resp.status_code != 200:
            st.error("❌ L'API a renvoyé une erreur.")
            with st.expander("Détails techniques de l'erreur", expanded=False):
                try:
                    st.json(resp.json())
                except Exception:
                    st.write(resp.text)
                st.code(f"Status code: {resp.status_code}")
            st.stop()

        data = resp.json()
        st.success("✅ Analyse terminée")

        # ====== EXTRACTION DES CHAMPS ======

        predicted_price_10d = data.get("predicted_price_10d")
        conf_level = data.get("confidence", "N/A")
        explanation = data.get("explanation", "")
        key_factors = data.get("key_factors", [])
        timestamp = data.get("timestamp", "")
        market_summary = data.get("market_data_summary") or data.get("market_data") or {}
        lstm_input = data.get("lstm_input")
        sources = data.get("sources", [])

        preds_array, preds_mean = extract_predicted_price(predicted_price_10d)
        last_spot = get_last_spot_price_from_market_summary(market_summary)

        if st.session_state.results:
            res = st.session_state.results

            st.markdown("---")

            # ====== BLOC METRIQUES HAUT DE PAGE ======

            left, right = st.columns([3, 1])

            # --------- TEXTE RAG ---------
            with left:
                st.subheader("🧠 Explications & Facteurs")
                render_explanation_and_factors(explanation, key_factors)

            with right:
                st.subheader("📈 Résumé de la prédiction")

                current_price = df['Close'].iloc[-1]
                predicted_price = res['future_predictions'][0]
                pct = (predicted_price - current_price) / current_price * 100

                left_in, right_in = st.columns(2)

                with left_in:
                    st.metric("Prix actuel", f"${current_price:.2f}")
                
                with right_in:
                    st.metric("Prix prédit", f"${predicted_price:.2f}", f"{pct:.2f}%")
                

                # ----- Jauge aiguille (LOW / MEDIUM / HIGH) -----
                conf = res['confidence_level']
                conf_map = {"low": 10, "medium": 50, "high": 90}
                conf_color = {"low": "#e74c3c", "medium": "#f1c40f", "high": "#2ecc71"}
                default_color = {"low": "#fbeaea", "medium": "#fdf3d6", "high": "#eaf7f0"}
                final_color = default_color
                final_color[conf] = conf_color[conf]

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge",
                    value=conf_map[conf],
                    gauge={
                        'axis': {'range': [0, 100], 'tickvals': []},
                        'bar': {'color': 'rgba(0,0,0,0)'},
                        'steps': [
                            {'range': [0, 33], 'color': final_color['low']},
                            {'range': [33, 66], 'color': final_color['medium']},
                            {'range': [66, 100], 'color': final_color['high']},
                        ]
                    }
                ))

                fig_gauge.add_annotation(
                    x=0.5,
                    y=0.15,
                    text=f"<b><span style='color:{conf_color[conf]}'>{conf.upper()}</span></b>",
                    showarrow=False,
                    font=dict(size=26)
                )

                fig_gauge.update_layout(height=260, margin=dict(t=20, b=0, l=0, r=0))
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("---")

            # ====== TABS PRINCIPAUX ======

            tab1, tab2, tab3 = st.tabs([
            "🔮 Prévision à 10 jours",
            "📊 Cours actuel",
            "📚 Sources RAG"
            ])

            with tab1:
                afficher_graphique_futur(res['df'], res)

            with tab2:
                min_date = df.index.min().to_pydatetime()
                max_date = df.index.max().to_pydatetime()

                df_filtered = df.loc[min_date:max_date]

                fig_train = px.line(
                    df_filtered,
                    x=df_filtered.index,
                    y='Close',
                    title="Cours du pétrole sur la période d'entraînement",
                    template=UIConfig.PLOT_TEMPLATE
                )

                st.plotly_chart(fig_train, use_container_width=True)


            with tab3:
                render_sources_tab(sources)
            # ====== BLOC DEBUG TECHNIQUE (OPT.) ======
            with st.expander("🔧 Détails techniques (pour développeurs)", expanded=False):
                st.subheader("Requête envoyée à /analyze")
                st.json(payload)
                st.subheader("Réponse brute de /analyze")
                st.json(data)
                st.code(f"Status code: {resp.status_code}")
                if isinstance(lstm_input, list) and lstm_input:
                    st.subheader("Input LSTM (liste complète)")
                    st.write(lstm_input)
                elif isinstance(lstm_input, (int, float)):
                    st.subheader("Input LSTM (scalaire)")
                    st.write(lstm_input)