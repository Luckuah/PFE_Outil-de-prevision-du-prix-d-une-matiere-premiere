import streamlit as st
import pandas as pd
from recup_data import creer_dataframe_brent_avec_plages,create_crisis
import plotly.express as px
import plotly.graph_objects as go
from model_training_prod_ready import train_full_pipeline, load_and_predict, forecast_future, prepare_data, CONFIG
import numpy as np

CRISES = {
    "Crise financière mondiale": ("2007-08-01", "2009-06-30"),
    "Crise financière 2008": ("2008-09-01", "2009-06-30"),
    "Crise de la dette européenne": ("2010-01-01", "2012-12-31"),
    "Printemps arabe": ("2010-12-01", "2012-12-31"),
    "Sanctions Iran": ("2012-01-01", "2015-07-14"),
    "Effondrement du pétrole (OPEP+)": ("2014-06-01", "2016-02-29"),
    "Guerre civile en Libye": ("2011-02-15", "2011-10-23"),
    "Accord nucléaire iranien (JCPOA)": ("2015-07-14", "2018-05-08"),
    "Sanctions Iran (reprise)": ("2018-05-08", "2019-12-31"),
    "Covid-19 (choc de la demande)": ("2020-02-01", "2020-05-31"),
    "Guerre des prix du pétrole (Russie/OPEP)": ("2020-03-01", "2020-04-30"),
    "Crise énergétique post-Covid": ("2021-01-01", "2022-12-31"),
    "Guerre Ukraine": ("2022-02-24", "2023-12-31"),
}

@st.cache_data
def load_data():
    return creer_dataframe_brent_avec_plages()

def page_dashboard(df):
    st.title("Dashboard des données Brent")

    # Sécurité
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # ----- CALENDRIER -----
    st.subheader("📅 Sélection de la période")

    dates = st.date_input(
    "Choisissez la période du marché",
    value=[df.index.min(), df.index.max()],
    min_value=df.index.min(),
    max_value=df.index.max()
    )

    if not isinstance(dates, (list, tuple)) or len(dates) != 2:
        st.warning("Veuillez sélectionner une période valide (début et fin).")
        st.stop()

    date_debut, date_fin = dates

    date_debut = pd.to_datetime(date_debut)
    date_fin = pd.to_datetime(date_fin)

    date_debut = max(date_debut, df.index.min())
    date_fin = min(date_fin, df.index.max())

    # Filtrage
    mask = (df.index >= pd.to_datetime(date_debut)) & \
           (df.index <= pd.to_datetime(date_fin))

    df_filtre = df.loc[mask]

    # ----- TABLE -----
    st.subheader("Aperçu des données")
    st.dataframe(df_filtre)

    # ----- GRAPHIQUE -----
    plot_df = df_filtre.reset_index()
    plot_df.columns = ["Date", "Brent_Price","Is_Crisis"]

    fig = px.line(
        plot_df,
        x="Date",
        y="Brent_Price",
        title="Évolution du prix du Brent (BZ=F)"
    )

    crises_a_afficher = st.session_state.get("crises", {})

    # On ajoute chaque crise sélectionnée sur le graphique
    for nom, (debut, fin) in crises_a_afficher.items():
        fig.add_vrect(
            x0=debut, 
            x1=fin,
            fillcolor="red", 
            opacity=0.2, 
            layer="below", 
            line_width=0,
            annotation_text=nom, 
            annotation_position="top left"
        )

    st.plotly_chart(fig, use_container_width=True)


def afficher_graphique_futur(df_historique, resultats):
    """
    Affiche les 7 derniers jours réels + les prédictions futures
    avec intervalles de confiance.
    """
    # 1. Extraire les 7 derniers jours réels
    df_recent = df_historique.tail(7).copy()
    
    # 2. Récupérer les prédictions et intervalles depuis le dictionnaire 'resultats'
    preds_futures = resultats['future_predictions']
    basse, haute = resultats['intervals']['future']
    
    # 3. Créer les dates futures
    derniere_date = df_recent.index[-1]
    dates_futures = pd.date_range(
        start=derniere_date + pd.Timedelta(days=1), 
        periods=len(preds_futures)
    )

    # 4. Créer le graphique
    fig = go.Figure()

    # Trace : Historique récent
    fig.add_trace(go.Scatter(
        x=df_recent.index, 
        y=df_recent.iloc[:, 0], # Prend la première colonne de prix
        name="Historique (7j)",
        mode="lines+markers",
        line=dict(color="#1f77b4", width=3)
    ))

    # Trace : Prédiction
    fig.add_trace(go.Scatter(
        x=dates_futures, 
        y=preds_futures,
        name="Prédiction",
        mode="lines+markers",
        line=dict(color="#d62728", dash="dash")
    ))

    # Trace : Intervalle de confiance (Ombrage)
    fig.add_trace(go.Scatter(
        x=pd.concat([pd.Series(dates_futures), pd.Series(dates_futures)[::-1]]),
        y=pd.concat([pd.Series(haute), pd.Series(basse)[::-1]]),
        fill='toself',
        fillcolor='rgba(214, 39, 40, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name="Intervalle de confiance (95%)"
    ))

    fig.update_layout(
        title="Zoom : 7 derniers jours et Prévisions",
        xaxis_title="Date",
        yaxis_title="Prix du Brent ($)",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------- PAGE 2 : PRÉDICTIONS ----------
def page_predictions(df):
    st.title("Prédictions du modèle LSTM")

    # Mode d'utilisation
    mode = st.radio("Mode d'exécution :", ["Utiliser le modèle sauvegardé (.h5)", "Réentraîner le modèle"])
    # --- NOUVELLE OPTION POUR LES CRISES ---
    utiliser_crises = st.checkbox("Prendre en compte les crises définies dans 'Paramètres' pour le réentrainement", value=True)
    
    crises_a_injecter = None
    if utiliser_crises:
        crises_a_injecter = st.session_state.get("crises", {})
        if crises_a_injecter:
            st.info(f"ℹ️ {len(crises_a_injecter)} crises seront marquées comme régime '-1' pendant l'entraînement.")
            with st.expander("🔍 Voir les crises qui seront appliquées"):
                # Création d'un petit tableau pour la clarté
                df_crises = pd.DataFrame([
                    {"Crise": nom, "Début": start, "Fin": end} 
                    for nom, (start, end) in crises_a_injecter.items()
                ])
                st.table(df_crises)
        else:
            st.warning("⚠️ Aucune crise n'est sélectionnée dans l'onglet 'Paramètres'.")
    if st.button("Lancer les prévisions"):
        with st.spinner("Traitement en cours..."):
            try:
                if mode == "Utiliser le modèle sauvegardé (.h5)":
                    # --- LOGIQUE DE CHARGEMENT ---
                    # Charge le modèle et télécharge les données récentes
                    bundle = load_and_predict('lstm_oil_model.h5')
                    model = bundle['model']
                    df_recent = bundle['df']
                    
                    # Préparation des données pour l'inférence
                    feature_cols = ['Open', 'High', 'Low', 'Volume', 'VIX_Close', 
                                    'RSI_14', 'MACD', 'ATR_14', 'ADX_14', 'VROC',
                                    'BB_Upper', 'BB_Mid', 'BB_Lower', 'Market_Regime']
                    
                    X_scaled, y_scaled, scaler_X, scaler_y = prepare_data(df_recent, feature_cols)
                    
                    # Extraction de la dernière séquence pour prédire le futur
                    last_sequence = X_scaled[-CONFIG['LOOKBACK']:]
                    
                    # Calcul des prédictions futures
                    # Note: std_val est fixé ici à une valeur type car non sauvegardé séparément
                    preds, low, high = forecast_future(
                        model, last_sequence, CONFIG['FUTURE_STEPS'], 
                        len(feature_cols), scaler_y, std_val=1.5, lookback=CONFIG['LOOKBACK']
                    )
                    
                    # On formate les résultats pour l'affichage
                    resultats = {
                        'future_predictions': preds,
                        'intervals': {'future': (low, high)}
                    }
                
                else:
                    # --- LOGIQUE D'ENTRAÎNEMENT COMPLET ---
                    resultats = train_full_pipeline(CONFIG,crises_a_injecter)
                
                # --- AFFICHAGE COMMUN ---
                st.success("Analyses terminées !")
                
                # Tableau des prix
                preds_df = pd.DataFrame({
                    "Date": [f"J+{i+1}" for i in range(len(resultats['future_predictions']))],
                    "Prix Prédit": resultats['future_predictions'],
                    "Borne Basse": resultats['intervals']['future'][0],
                    "Borne Haute": resultats['intervals']['future'][1]
                })
                cols_numeriques = ["Prix Prédit", "Borne Basse", "Borne Haute"]
                # On applique le formatage uniquement sur ces colonnes
                st.table(preds_df.style.format({col: "{:.2f} $" for col in cols_numeriques}))
                # Graphique dynamique
                st.subheader("📈 Projection des prix")
                afficher_graphique_futur(df, resultats)

            except Exception as e:
                st.error(f"Erreur : {e}")



def page_parametres():
    st.title("⚙️ Paramètres")

    # --- SECTION 1 : AJOUTER UNE CRISE PERSONNALISÉE ---
    st.subheader("🆕 Ajouter une crise personnalisée")
    
    # Initialisation de la liste des crises perso si elle n'existe pas
    if "crises_perso" not in st.session_state:
        st.session_state["crises_perso"] = {}

    with st.form("form_crise_perso"):
        nom_crise = st.text_input("Nom de la crise")
        col1, col2 = st.columns(2)
        with col1:
            debut_perso = st.date_input("Date de début")
        with col2:
            fin_perso = st.date_input("Date de fin")
        
        submit = st.form_submit_button("Ajouter à la liste")
        
        if submit:
            if nom_crise:
                st.session_state["crises_perso"][nom_crise] = (debut_perso, fin_perso)
                st.success(f"Crise '{nom_crise}' ajoutée !")
            else:
                st.error("Veuillez donner un nom à la crise.")

    st.divider()

    # --- SECTION 2 : SÉLECTION ET ÉDITION ---
    st.subheader("Sélection des crises pour le Dashboard")

    # On fusionne le dictionnaire fixe (CRISES) et le dictionnaire dynamique (crises_perso)
    toutes_les_options = {**CRISES, **st.session_state["crises_perso"]}
    
    default_selection = list(st.session_state.get("crises", {}).keys())

    crises_selectionnees = st.multiselect(
        "Sélectionnez les crises à afficher sur le graphique",
        options=list(toutes_les_options.keys()),
        default=[s for s in default_selection if s in toutes_les_options]
    )

    crises_actives = {}
    for crise in crises_selectionnees:
        st.write(f"**Édition : {crise}**")
        col1, col2 = st.columns(2)
        
        # Récupération des valeurs actuelles
        val_debut, val_fin = toutes_les_options[crise]
        
        with col1:
            start = st.date_input(f"Début - {crise}", value=pd.to_datetime(val_debut), key=f"start_{crise}")
        with col2:
            end = st.date_input(f"Fin - {crise}", value=pd.to_datetime(val_fin), key=f"end_{crise}")

        crises_actives[crise] = (start, end)

    # Sauvegarde finale pour le dashboard
    st.session_state["crises"] = crises_actives
    
    # Option pour vider les crises personnalisées
    if st.button("Réinitialiser les crises personnalisées"):
        st.session_state["crises_perso"] = {}
        st.rerun()



# ---------- APPLICATION PRINCIPALE ----------
def dashboard():
    st.sidebar.title("Navigation")

    page = st.sidebar.selectbox(
        "Choisir une page",
        ["Dashboard", "Prédictions", "Paramètres"]
    )

    df = load_data()

    if page == "Dashboard":
        page_dashboard(df)

    elif page == "Prédictions":
        page_predictions(df)

    elif page == "Paramètres":
        page_parametres()




if __name__ == "__main__":
    dashboard()