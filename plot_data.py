import pandas as pd
import matplotlib.pyplot as plt

def plot_data(data: pd.DataFrame, columns=["Close", "MA30"]):
    """Visualise les colonnes sélectionnées."""
    plt.figure(figsize=(12,6))
    for col in columns:
        if col in data.columns:
            plt.plot(data[col], label=col)
    plt.title("Évolution des prix normalisés")
    plt.xlabel("Date")
    plt.ylabel("Valeur normalisée")
    plt.legend()
    plt.show()