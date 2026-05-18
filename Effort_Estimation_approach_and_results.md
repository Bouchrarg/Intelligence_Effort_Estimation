# Synthèse du projet Effort Estimation

## 1. Contexte général
Ce projet vise à estimer l'effort logiciel d'un projet GitHub en heures à partir de métriques extraites du dépôt.
Le pipeline actuel est composé de trois étapes principales :

1. Scraping GitHub (`ScrappingEnsam/`)
2. Préparation des données et nettoyage
3. Entraînement du modèle de prédiction (`Effort_Target_formula/train_model.py`)


---

## 2. Architecture actuelle du pipeline

### 2.1 Scraping et checkpoint
- Le scraping se fait dans `ScrappingEnsam/scraper.py`.
- Les résultats bruts sont écrits dans `ScrappingEnsam/Scrapped_Data/features_raw_LOT{N}.csv`.
- Les checkpoints sont sauvegardés dans `ScrappingEnsam/Checkpoints/checkpoint_LOT{N}.json`.
- Si les CSV manquent, `ScrappingEnsam/Checkpoints/extract_from_checkpoints.py` permet de recréer les CSV depuis les checkpoints.
- Le merge global se fait avec `ScrappingEnsam/merge_all.py`.

### 2.2 Nettoyage et feature engineering
- Le jeu de données utilisé par `Effort_Target_formula/train_model.py` est `../ScrappingEnsam/features_merged_fixed.csv`.
- Cette version corrigée doit contenir les nettoyages et les ajustements nécessaires avant entraînement.

### 2.3 Modélisation
- Le script `Effort_Target_formula/train_model.py` réalise :
  - nettoyage des données
  - retrait des outliers
  - encodage des variables catégorielles
  - suppression des fuites de données (`leakage`)
  - log-transform de la target
  - benchmark de plusieurs modèles
  - entraînement final de XGBoost
  - sauvegarde du modèle et du jeu de test
  - explication avec SHAP

---

## 3. Préparation des données

### 3.1 Filtrage qualité
- `reliability_score >= 65` est utilisé comme filtre de qualité.
- Cela permet de retirer les projets qui ont des métriques GitHub incomplètes ou peu fiables.

### 3.2 Suppression des outliers
- Le script retire les 1 % les plus bas et les 1 % les plus hauts de `effort_target`.
- Objectif : limiter l'influence des projets anormalement petits et des très gros projets.

### 3.3 Remplissage et encodage
- `language` manquant est remplacé par `Unknown`.
- `avg_file_size_loc` et `comment_per_pr_avg` manquants sont remplacés par `0.0`.
- `language` est converti en variables dummy via `pd.get_dummies(..., drop_first=True)`.

### 3.4 Retrait du leakage
Le script retire explicitement ces colonnes de `X` :

- `full_name`, `url`, `created_at`, `lot`
- `net_loc`, `churn_loc`, `active_days`, `total_commits`
- `pr_merge_time_median_h`, `active_contributors`
- `code_churn_normalized`
- `churn_hours`, `cycle_time_hours`, `cocomo_pm`, `cocomo_hours`
- `reliability_score`

Ces colonnes sont considérées comme des fuites parce qu'elles reconstruisent directement la target théorique.

### 3.5 Target engineering
- La target est transformée avec `log1p` : `y_log = np.log1p(y)`.
- Le split train/test est stratifié selon des quantiles de `y_log` pour conserver la distribution.

---

## 4. Benchmark et sélection de modèle

Les modèles testés sont :
- Ridge
- Random Forest
- Gradient Boosting
- XGBoost

Les résultats observés sont :
- Ridge : `MAE ≈ 815,713 h`, `R² ≈ -0.010`
- Random Forest : `MAE ≈ 771,872 h`, `R² ≈ 0.007`
- Gradient Boosting : `MAE ≈ 769,351 h`, `R² ≈ 0.039`
- XGBoost : `MAE ≈ 772,850 h`, `R² ≈ 0.034`

Le modèle final retenu est **XGBoost**.

---

## 5. Résultats finaux

Sur le jeu de test, le modèle final XGBoost donne :

- **MAE** : `766,272 heures`
- **R²** : `0.043`
- **Erreur médiane** : `74 %`

Validation croisée 5-fold sur le train :

- **R² moyen** : `0.306 ± 0.085`
- **MAE log** : `1.1669 ± 0.0692`

Fichiers générés :
- `effort_model.pkl`
- `test_set_unseen.csv`
- `correlation_matrix.png`
- `target_distribution.png`
- `models_comparison.png`
- `actual_vs_predicted.png`
- `feature_importance_xgb.png`
- `shap_summary.png`
- `shap_bar.png`
- `shap_waterfall_0.png`

---

## 6. Problèmes identifiés

### 6.1 Performance faible
- Le R² de test est très faible (`0.043`).
- Le MAE est extrêmement élevé en heures, ce qui signifie que le modèle n’est pas précis en absolu.
- L’erreur médiane de `74 %` indique une large dispersion des prédictions.

### 6.2 Données très bruitées
- La target couvre des valeurs très larges (`5k` à `15M` heures).
- La distribution est fortement biaisée et difficile à modéliser.

### 6.3 Risque de leakage
- Le pipeline actuel retire des variables à forte fuite, mais la target est elle-même dérivée d’une formule théorique.
- Si `effort_target` est calculé à partir d’une formule interne, il est possible que les features restantes ne soient pas suffisantes pour reconstruire fidèlement cette formule.

### 6.4 Explicabilité
- Le SHAP met en avant des features comme `bus_factor_ratio`, `days_inactive`, `language_diversity`, `dependency_count` et `weekend_commit_ratio`.
- Cela suggère que le modèle utilise surtout des signaux structurels et non pas des mesures directes d’effort.

---

## 7. Recommandations et prochaines étapes

1. **Vérifier que `features_merged_fixed.csv` contient bien les corrections attendues** avant entraînement.
2. **Tester un autre format de target**, par exemple un score normalisé ou un ratio plutôt que des heures brutes.
3. **Comparer le modèle aux poids théoriques** de l’effort si la target est construite par formule.
4. **Analyser `test_set_unseen.csv`** pour comprendre les cas où l’erreur est la plus forte.
5. **Essayer d’autres approches** telles que des modèles sur target log-transformée directement ou des modèles plus robustes aux outliers.

---

## 8. Plan de branches Git

- `main` : conserver la version actuelle du pipeline `ScrappingEnsam/` et la structure de scraping.
- `clean-and-train` ou `first_try_cleaning_training` : brancher `CleanAndTrain/` et le workflow de nettoyage/training isolé.
