


-- Création de la table principale
CREATE TABLE gdelt_articles_scored (
    -- Identifiants
    id INT AUTO_INCREMENT PRIMARY KEY,
    global_event_id BIGINT UNIQUE NOT NULL,
    day DATE NOT NULL,
    month_year INT,
    year INT,
    date_added DATETIME NOT NULL,
    
    -- Acteurs
    actor1_name VARCHAR(500),
    actor1_country_code VARCHAR(3),
    actor2_name VARCHAR(500),
    actor2_country_code VARCHAR(3),
    
    -- Événement
    event_code VARCHAR(10),
    event_root_code VARCHAR(2),
    quad_class TINYINT,
    goldstein_scale FLOAT,
    
    -- Métriques GDELT
    num_mentions INT DEFAULT 0,
    num_sources INT DEFAULT 0,
    num_articles INT DEFAULT 0,
    avg_tone FLOAT,
    
    -- Géographie
    actor1_geo_country_code VARCHAR(3),
    actor2_geo_country_code VARCHAR(3),
    action_geo_country_code VARCHAR(3),
    action_geo_fullname VARCHAR(500),
    action_geo_lat FLOAT,
    action_geo_long FLOAT,
    
    -- Article complet
    source_url TEXT,
    article_title TEXT,
    article_content LONGTEXT,
    article_language VARCHAR(5),
    article_author VARCHAR(255),
    article_publish_date DATETIME,
    
    -- Scoring
    llm_score TINYINT COMMENT 'Score LLM de 0 à 4',
    llm_justification TEXT,
    final_score FLOAT COMMENT 'Score final combiné de 0 à 100',
    
    -- Features calculées
    is_oil_country BOOLEAN DEFAULT FALSE,
    keyword_matches TEXT COMMENT 'Mots-clés matchés (JSON)',
    
    -- Metadata
    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Index pour optimiser les requêtes
    INDEX idx_day (day),
    INDEX idx_date_added (date_added),
    INDEX idx_final_score (final_score),
    INDEX idx_action_country (action_geo_country_code),
    INDEX idx_event_code (event_code),
    INDEX idx_actor1_country (actor1_country_code),
    INDEX idx_actor2_country (actor2_country_code),
    INDEX idx_language (article_language),
    INDEX idx_combined_score_date (final_score, day)
    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Articles GDELT scorés pour prédiction prix pétrole';

-- Table pour les embeddings (pour RAG)
CREATE TABLE article_embeddings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    global_event_id BIGINT UNIQUE NOT NULL,
    embedding BLOB COMMENT 'Vecteur embedding (768 dimensions)',
    embedding_model VARCHAR(100) DEFAULT 'paraphrase-multilingual-mpnet-base-v2',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (global_event_id) REFERENCES gdelt_articles_scored(global_event_id) ON DELETE CASCADE,
    INDEX idx_event_id (global_event_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Embeddings vectoriels pour RAG';

-- Table pour les requêtes utilisateurs (historique)
CREATE TABLE user_queries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding BLOB,
    num_results INT DEFAULT 5,
    execution_time_ms INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Historique des requêtes utilisateurs';

-- Table de liaison pour les résultats de requêtes
CREATE TABLE query_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_id INT NOT NULL,
    global_event_id BIGINT NOT NULL,
    rank_position INT NOT NULL,
    similarity_score FLOAT,
    rerank_score FLOAT,
    
    FOREIGN KEY (query_id) REFERENCES user_queries(id) ON DELETE CASCADE,
    FOREIGN KEY (global_event_id) REFERENCES gdelt_articles_scored(global_event_id) ON DELETE CASCADE,
    INDEX idx_query_id (query_id),
    INDEX idx_event_id (global_event_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Résultats des requêtes RAG';

-- Table pour les statistiques quotidiennes
CREATE TABLE daily_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    total_events_downloaded INT DEFAULT 0,
    events_after_filtering INT DEFAULT 0,
    articles_fetched INT DEFAULT 0,
    articles_scored INT DEFAULT 0,
    articles_kept INT DEFAULT 0,
    avg_score FLOAT,
    processing_time_seconds INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_date (date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Statistiques quotidiennes du pipeline';

-- Vue pour les articles les plus pertinents
CREATE VIEW top_articles AS
SELECT 
    id,
    global_event_id,
    day,
    article_title,
    actor1_name,
    actor2_name,
    action_geo_fullname,
    final_score,
    llm_score,
    source_url
FROM gdelt_articles_scored
WHERE final_score >= 70
ORDER BY final_score DESC, day DESC;

-- Vue pour les statistiques par pays
CREATE VIEW country_stats AS
SELECT 
    action_geo_country_code AS country_code,
    COUNT(*) AS num_articles,
    AVG(final_score) AS avg_score,
    MAX(final_score) AS max_score,
    MIN(day) AS first_article_date,
    MAX(day) AS last_article_date
FROM gdelt_articles_scored
GROUP BY action_geo_country_code
HAVING num_articles >= 5
ORDER BY num_articles DESC;

-- Vue pour les tendances hebdomadaires
CREATE VIEW weekly_trends AS
SELECT 
    YEAR(day) AS year,
    WEEK(day) AS week,
    COUNT(*) AS num_articles,
    AVG(final_score) AS avg_score,
    AVG(goldstein_scale) AS avg_goldstein,
    AVG(avg_tone) AS avg_tone,
    SUM(CASE WHEN quad_class IN (3, 4) THEN 1 ELSE 0 END) AS conflict_articles,
    SUM(CASE WHEN quad_class IN (1, 2) THEN 1 ELSE 0 END) AS cooperation_articles
FROM gdelt_articles_scored
GROUP BY year, week
ORDER BY year DESC, week DESC;

-- Procédure stockée pour nettoyer les vieux articles
DELIMITER //
CREATE PROCEDURE clean_old_articles(IN days_to_keep INT)
BEGIN
    DELETE FROM gdelt_articles_scored 
    WHERE day < DATE_SUB(CURDATE(), INTERVAL days_to_keep DAY)
    AND final_score < 60;
    
    SELECT ROW_COUNT() AS deleted_rows;
END //
DELIMITER ;

-- Procédure pour obtenir les statistiques d'un pays
DELIMITER //
CREATE PROCEDURE get_country_articles(IN country_code VARCHAR(3), IN limit_count INT)
BEGIN
    SELECT 
        global_event_id,
        day,
        article_title,
        final_score,
        llm_score,
        source_url
    FROM gdelt_articles_scored
    WHERE action_geo_country_code = country_code
    ORDER BY final_score DESC, day DESC
    LIMIT limit_count;
END //
DELIMITER ;

-- Insertion d'un enregistrement de test
INSERT INTO daily_stats (date, total_events_downloaded, events_after_filtering, articles_fetched, articles_scored, articles_kept, avg_score, processing_time_seconds)
VALUES (CURDATE(), 0, 0, 0, 0, 0, 0.0, 0)
ON DUPLICATE KEY UPDATE date=date;

-- Affichage des tables créées
SHOW TABLES;

-- Affichage de la structure de la table principale
DESCRIBE gdelt_articles_scored;