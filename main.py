import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import de vos modules personnalisés
from config_param import DataConfig, ModelConfig, CrisesConfig, UIConfig
from data_utils import prepare_full_dataset
from model_training import train_full_pipeline, load_and_predict, prepare_data, forecast_future

# Configuration de la page Streamlit
st.set_page_config(
    page_title=UIConfig.APP_TITLE,
    page_icon="🛢️",
    layout="wide"
)

# ==================== GESTION DE L'ÉTAT (SESSION STATE) ====================
if "crises" not in st.session_state:
    st.session_state.crises = CrisesConfig.CRISES.copy()

if "results" not in st.session_state:
    st.session_state.results = None

# ==================== CHARGEMENT DES DONNÉES ====================
@st.cache_data(ttl=UIConfig.CACHE_TTL)
def get_data(crises_dict):
    # Utilise le pipeline de data_utils pour avoir tous les indicateurs
    return prepare_full_dataset(crises_dict=crises_dict)

# ==================== PAGES DE L'APPLICATION ====================

def page_dashboard():
    st.title("📊 Dashboard du Marché du Pétrole")
    
    df = get_data(st.session_state.crises)
    
    # --- FILTRES TEMPORELS ---
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Options d'affichage")
        min_date = df.index.min().to_pydatetime()
        max_date = df.index.max().to_pydatetime()
        
        selected_dates = st.date_input(
            "Période d'analyse",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

    if len(selected_dates) == 2:
        mask = (df.index >= pd.Timestamp(selected_dates[0])) & (df.index <= pd.Timestamp(selected_dates[1]))
        df_filtered = df.loc[mask]
        
        with col2:
            # Graphique principal
            fig = px.line(df_filtered, x=df_filtered.index, y='Close', 
                          title=f"Prix du Brent ({DataConfig.OIL_TICKER})",
                          labels={'Close': 'Prix ($)', 'index': 'Date'},
                          template=UIConfig.PLOT_TEMPLATE)
            
            # Ajout des zones de crises
            for nom, (debut, fin) in st.session_state.crises.items():
                d_start = pd.to_datetime(debut)
                d_end = pd.to_datetime(fin)
                # On n'affiche que si la crise est dans la zone visible
                if d_start <= df_filtered.index.max() and d_end >= df_filtered.index.min():
                    fig.add_vrect(x0=debut, x1=fin, fillcolor="red", opacity=0.1, 
                                 layer="below", line_width=0, annotation_text=nom)
            
            st.plotly_chart(fig, use_container_width=True)

        # Métriques techniques
        st.subheader("Indicateurs Techniques (Dernières valeurs)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prix Actuel", f"${df['Close'].iloc[-1]:.2f}")
        m2.metric("RSI (14)", f"{df['RSI_14'].iloc[-1]:.2f}")
        m3.metric("VIX", f"{df['VIX_Close'].iloc[-1]:.2f}")
        m4.metric("Volatilité (ATR)", f"{df['ATR_14'].iloc[-1]:.2f}")

def page_predictions():
    st.title("🔮 Prédictions LSTM")
    
    mode = st.radio("Stratégie de prédiction :", 
                    ["Utiliser le modèle pré-entraîné", "Réentraîner le modèle complet"])
    
    if st.button("Lancer les calculs"):
        with st.spinner("Analyse des séries temporelles en cours..."):
            try:
                if mode == "Réentraîner le modèle complet":
                    # Utilise le dictionnaire de crises de la session
                    results = train_full_pipeline(crises_dict=st.session_state.crises)
                    st.session_state.results = results
                    st.success("Modèle réentraîné avec succès !")
                else:
                    # Chargement et prédiction directe
                    bundle = load_and_predict(ModelConfig.MODEL_PATH, st.session_state.crises)
                    df_recent = bundle['df']
                    model = bundle['model']
                    
                    # Préparation manuelle pour l'inférence
                    X_scaled, y_scaled, scaler_X, scaler_y = prepare_data(df_recent, DataConfig.FEATURE_COLS)
                    last_sequence = X_scaled[-ModelConfig.LOOKBACK:]
                    
                    # Simulation des résultats pour l'affichage
                    preds, low, high = forecast_future(
                        model, last_sequence, ModelConfig.FUTURE_STEPS, 
                        len(DataConfig.FEATURE_COLS), scaler_y, std_val=1.5, lookback=ModelConfig.LOOKBACK
                    )
                    
                    st.session_state.results = {
                        'future_predictions': preds,
                        'intervals': {'future': (low, high)},
                        'df': df_recent
                    }

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {str(e)}")

    # Affichage des résultats si disponibles
    if st.session_state.results:
        res = st.session_state.results
        df_hist = res.get('df', get_data(st.session_state.crises))
        
        # Graphique des prévisions
        st.subheader("Prévisions à 10 jours")
        afficher_graphique_futur(df_hist, res)
        
        # Tableau des valeurs
        with st.expander("Voir les valeurs prédites"):
            preds = res['future_predictions']
            low, high = res['intervals']['future']
            df_future = pd.DataFrame({
                'Jour': [f"J+{i+1}" for i in range(len(preds))],
                'Prix Prédit': preds.round(2),
                'Min (CI)': low.round(2),
                'Max (CI)': high.round(2)
            })
            st.table(df_future)

def page_parametres():
    st.title("⚙️ Paramètres & Crises")
    
    st.subheader("Gestion des événements historiques")
    st.info("Ajoutez ou supprimez des crises pour modifier le 'Market_Regime' du modèle.")
    
    # Formulaire pour ajouter une crise
    with st.expander("➕ Ajouter une nouvelle période de crise"):
        with st.form("new_crisis"):
            name = st.text_input("Nom de la crise")
            start_c = st.date_input("Date de début", value=datetime(2023, 1, 1))
            end_c = st.date_input("Date de fin", value=datetime(2023, 12, 31))
            if st.form_submit_button("Ajouter"):
                st.session_state.crises[name] = (start_c.strftime('%Y-%m-%d'), end_c.strftime('%Y-%m-%d'))
                st.rerun()

    # Liste des crises actuelles
    st.write("### Crises enregistrées")
    for nom in list(st.session_state.crises.keys()):
        col1, col2 = st.columns([4, 1])
        dates = st.session_state.crises[nom]
        col1.write(f"**{nom}** : du {dates[0]} au {dates[1]}")
        if col2.button("Supprimer", key=nom):
            del st.session_state.crises[nom]
            st.rerun()

# ==================== FONCTIONS GRAPHIQUES ====================

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
    # Historique
    fig.add_trace(go.Scatter(x=df_recent.index, y=df_recent['Close'], name="Historique récent",
                             line=dict(color=UIConfig.HISTORICAL_COLOR, width=3)))
    # Prédiction
    fig.add_trace(go.Scatter(x=dates_futures, y=preds_futures, name="Prédiction LSTM",
                             line=dict(color=UIConfig.PREDICTION_COLOR, dash="dash", width=3)))
    # Intervalle
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_futures, dates_futures[::-1]]),
        y=np.concatenate([haute, basse[::-1]]),
        fill='toself', fillcolor='rgba(214, 39, 40, 0.2)',
        line=dict(color='rgba(255,255,255,0)'), name="Confiance 95%"
    ))
    
    fig.update_layout(template=UIConfig.PLOT_TEMPLATE, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ==================== MAIN LOOP ====================
def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Aller vers :", UIConfig.PAGES)
    
    if selection == "Dashboard":
        page_dashboard()
    elif selection == "Prédictions":
        page_predictions()
    elif selection == "Paramètres":
        page_parametres()

if __name__ == "__main__":
    main()