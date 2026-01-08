from Pipeline_Data.models.trieur_agent import TrieurAgent


# Initialisation
agent = TrieurAgent()

# 1. Test avec une nouvelle CRUCIALE (devrait avoir un score de 3 ou 4)
article_important = {
    "title": "L'OPEP+ réduit sa production de 2 millions de barils",
    "content": "Dans une décision surprise, les pays membres ont décidé de réduire massivement l'offre pour soutenir les cours du pétrole..."
}

# 2. Test avec une nouvelle INUTILE (devrait avoir un score de 0 ou 1)
article_inutile = {
    "title": "Nouveau café ouvert dans une station-service à Lyon",
    "content": "Une petite boutique propose désormais des croissants frais pour les automobilistes locaux."
}

print("--- Test Article Majeur ---")
score1, just1 = agent.score_article(article_important["title"], article_important["content"])
print(f"Score: {score1}/4\nJustification: {just1}")

print("\n--- Test Article Mineur ---")
score2, just2 = agent.score_article(article_inutile["title"], article_inutile["content"])
print(f"Score: {score2}/4\nJustification: {just2}")