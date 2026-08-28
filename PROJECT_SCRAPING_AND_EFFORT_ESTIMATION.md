# Projet d'estimation d'effort GitHub

## 1. Vue d'ensemble du projet

Le projet collecte des données GitHub sur des repositories publics, construit un dataset de features, puis tente d'estimer l'effort de développement.

Il y a deux grandes approches présentes dans la branche `feature/ml-notebooks` :

- la première approche : `effort_target` synthétique calculé dans `ScrappingEnsam/scraper.py`,
- la nouvelle approche : `effort_hours_real` estimé à partir de l'historique de commits (`effort_estimator.py` / `effort_estimator2.py`).

Le pipeline global est :

1. Scraping distribué des repos GitHub avec `ScrappingEnsam/scraper.py`.
2. Checkpoint et sauvegarde partielle des résultats pour tolérer les interruptions.
3. Fusion des CSV de chaque lot avec `ScrappingEnsam/merge_all.py`.
4. Enrichissement du dataset avec une vraie target d'effort via `ScrappingEnsam/effort_estimator.py` ou `ScrappingEnsam/effort_estimator2.py`.
5. Expérimentation ML dans les notebooks `notebooks/*.ipynb`.

---

## 2. Scraping GitHub : `ScrappingEnsam/scraper.py`

### 2.1 Objectif

Collecter un ensemble de features qualitatives et quantitatives pour chaque repository GitHub sélectionné, puis calculer une target d'effort synthétique.

### 2.2 Architecture

Le script contient :

- une classe `GitHubClient` pour les appels à l'API GitHub,
- des fonctions de collecte de données par dépôt,
- un système de checkpoint JSON (`Checkpoints/checkpoint_LOT{N}.json`),
- une sauvegarde CSV par lot dans `Scrapped_Data/features_raw_LOT{N}.csv`.

### 2.3 Requêtes par lot

Le scraping est organisé en lots, chaque membre du groupe exécute un lot différent :

- `LOT 1` : Python / JavaScript,
- `LOT 2` : TypeScript / Java,
- `LOT 3` : Go / Rust / C++,
- `LOT 4` : machine-learning / data-science / devtools / web-framework.

Chaque lot utilise plusieurs requêtes de recherche GitHub avec des filtres `stars`, `pushed`, `language`, `forks` ou `topic`.

### 2.4 Corrections intégrées

Le script a ajouté plusieurs correctifs importants par rapport à une version initiale :

- `get_code_frequency` : attente longue plus retry 90s pour éviter les 202 GitHub,
- `get_commit_stats` : utilise `/repos/{full_name}/contributors` synchrone au lieu de `/stats/contributors`,
- `get_closed_issues_count` : compte les issues fermées réelles via `/issues?state=closed`,
- `get_review_cycle_count` : ajoute une feature sur les cycles de revue par PR,
- `weighted_experience` : calcule l'expérience pondérée des contributeurs.

### 2.5 Extraction des features

Pour chaque dépôt, `extract_features(...)` construit un dictionnaire de valeurs, notamment :

- `active_contributors`, `bus_factor_ratio`, `total_commits`,
- `net_loc`, `churn_loc`, `code_churn_normalized`,
- `pr_merge_time_median_h`, `comment_per_pr_avg`,
- `closed_issues`, `review_cycle_count`,
- `has_ci`, `ci_success_rate`, `has_tests`,
- `dependency_count`, `language_diversity`,
- `commit_velocity_trend`, `release_regularity`, `weekend_commit_ratio`,
- `weighted_experience`,
- `cocomo_hours`, `cocomo_pm`,
- `effort_target`.

### 2.6 Filtrage qualité

Le scraper applique des règles de filtrage avant d'ajouter un dépôt au dataset :

- minimum `50` commits,
- minimum `2` contributeurs,
- maximum `180` jours d'inactivité,
- minimum `30` étoiles,
- minimum `1` issue fermée,
- si `net_loc < 100` ou `churn_loc < 500`, le dépôt est filtré.

### 2.7 Checkpoint et sortie

Le script sauvegarde :

- `Checkpoints/checkpoint_LOT{N}.json` : liste des repos traités et résultats intermédiaires,
- `Scrapped_Data/features_raw_LOT{N}.csv` : dataset brut de chaque lot.

Il peut être relancé et continue après le dernier dépôt traité.

---

## 3. Fusion des lots : `ScrappingEnsam/merge_all.py`

### Fonction

`merge_all.py` prend tous les fichiers `Scrapped_Data/features_raw_LOT*.csv` et produit un dataset unique :

- `features_merged.csv`

### Étapes

1. Lire tous les CSV matching `Scrapped_Data/features_raw_LOT*.csv`.
2. Concaténer les DataFrames.
3. Supprimer les doublons sur `full_name` en gardant la première occurrence.
4. Afficher des statistiques rapides : distribution par lot, par langage, description de `effort_target`.

Ce fichier est la base pour les notebooks et l’enrichissement ultérieur.

---

## 4. Première approche : `effort_target`

### Définition

`effort_target` est une target synthétique calculée dans `ScrappingEnsam/scraper.py` à partir de trois composantes :

- `churn_hours` : effort estimé à partir du churn LOC,
- `cycle_time_hours` : temps de cycle des PRs mergées,
- `cocomo_hours` : estimation COCOMO II basée sur la taille du code.

### Formule

`effort_target = 0.5 × churn_hours + 0.3 × cycle_time_hours + 0.2 × cocomo_hours`

#### Détails des composantes

- `churn_hours = churn_loc / PRODUCTIVITY_LOC_PER_HOUR`
  - `PRODUCTIVITY_LOC_PER_HOUR = 15`,
- `cycle_time_hours = median_h_per_pr × active_contributors`,
- `cocomo_pm = 2.4 × (kloc ** 1.05)`,
  - `kloc = max(net_loc / 1000, 0.1)`,
  - si `bus_factor_ratio > 0.7`, un multiplicateur `1.2` est appliqué,
  - `cocomo_hours = cocomo_pm × 160`.

### Objectif

Cette target cherche à combiner :

- la quantité de code modifié,
- le temps de revue de PR,
- une estimation de taille logicielle.

C’est une première approximation d’effort, facile à calculer depuis les features extraites par le scraper.

### Limites de cette approche

- elle reste heuristique et dépend de coefficients fixes,
- elle ne reflète pas directement l’activité réelle des contributeurs,
- elle peut être influencée par des variations de style de commit ou de processus de revue,
- elle ne tient pas compte d’un historique complet de commits.

---

## 5. Nouvelle approche : `effort_hours_real`

### `ScrappingEnsam/effort_estimator.py` (v1)

La première version du calcul réel d’effort :

- les commits sont récupérés pour chaque dépôt,
- les commits sont regroupés par auteur,
- on reconstitue des sessions de travail pour chaque auteur,
- on additionne les durées de session pour obtenir l’effort total.

### Méthode de session

- gap entre deux commits ≤ `2 heures` → même session,
- session d’un seul commit → minimum `30 minutes`,
- session plafonnée à `8 heures`.

### Avantages

- cible beaucoup plus ancrée dans l’activité réelle,
- reprend une méthode utilisée dans des travaux de mining OSS,
- ne dépend pas directement de métriques de churn ou de COCOMO.

### Inconvénients

- dépend de la qualité des données de commit,
- ne couvre qu’un échantillon limité de commits (`MAX_PAGES = 10` pages),
- peut sous-estimer pour les gros projets si seuls les commits récents sont analysés.

---

### `ScrappingEnsam/effort_estimator2.py` (v2)

La version améliorée propose une estimation à deux phases :

1. **Sample** : récupérer un échantillon de commits récents et d’anciens commits,
2. **Scale** : extrapoler cet effort échantillonné sur l’historique complet du repo.

#### Pourquoi c’est nécessaire

La v1 ne voit que les derniers commits. Pour un dépôt avec 50 000 commits, cela peut donner une estimation incomparablement plus basse que l’effort réel.

#### Ce qui change

- `SAMPLE_PAGES = 5` pages de commits récents,
- `SAMPLE_PAGES_OLD = 3` pages de commits anciens,
- utilisation de `total_commits` pour l’extrapolation,
- `DECAY_FACTOR = 0.85` pour corriger les commits anciens.

#### Formule de scaling

- `rate_recent = effort_recent_h / commits_recent`
- `rate_old = (effort_old_h × DECAY_FACTOR) / commits_old`
- `rate_middle = (rate_recent + rate_old) / 2`
- `commits_middle = total_commits - commits_sampled`
- `effort_middle = rate_middle × commits_middle`
- `effort_total = effort_recent_h + effort_old_h × DECAY_FACTOR + effort_middle`

Cette formule considère que l’effort des commits intermédiaires est une interpolation entre le rythme ancien et le rythme récent.

#### Diagnostics importants

La fonction renvoie aussi :

- `effort_hours_sampled`,
- `effort_coverage_ratio`,
- `effort_scaling_factor`,
- `effort_total_commits`,
- `effort_commits_sampled`,
- `effort_contributors_seen`,
- `effort_was_capped`.

#### Bornes de plausibilité

- `MIN_EFFORT_H = 10`
- `MAX_EFFORT_H = 5_000_000`

Cela permet d’accepter des projets très volumineux comme des dépôts majeurs, tout en filtrant les valeurs aberrantes extrêmes.

---

## 6. Fichier de sortie enrichi

Le dataset final de cette branche est :

- `ScrappingEnsam/features_with_real_effort_v2.csv`

Il contient à la fois :

- les features extraites par `scraper.py`,
- `effort_target` (ancienne target synthétique),
- `effort_hours_real` (nouveau target réel),
- les diagnostics de scaling et de couverture.

Ce fichier est la base pour les notebooks ML de la branche.

---

## 7. Notebooks et modèles

### Notebooks présents

- `notebooks/ml_pipeline.ipynb`
- `notebooks/ml_pipeline_v4_fixed.ipynb`
- `notebooks/ml_pipeline_v5_improved.ipynb`

Ils représentent l’évolution du pipeline ML :

- exploration initiale,
- corrections et stabilisation,
- version améliorée finale.

### Modèles présents

- `models/best_v5_model_pipeline.joblib` : probablement le meilleur pipeline global,
- `models/go_rust_rf_model.joblib` : probablement un modèle spécialisé sur Go/Rust ou un essai de pipeline alternatif.

---

## 8. Conclusions et recommandations

### Ce qui est déjà solide

- le scraper est robuste et intègre des correctifs importants,
- la fusion des lots est simple et fiable,
- la première target synthétique est disponible pour benchmark,
- la nouvelle target basée sur les commits est bien plus réaliste.

### Ce qu’il reste à valider

- la qualité des estimations `effort_hours_real` sur un grand nombre de repos,
- la fiabilité du scaling de v2 pour des dépôts à très longue histoire,
- l’impact de `effort_hours_real` sur les performances des modèles ML par rapport à `effort_target`.

### Recommandation d’usage

- utiliser `effort_target` comme baseline simple,
- utiliser `effort_hours_real` pour les études plus avancées et la modélisation finale,
- analyser `effort_coverage_ratio` et `effort_scaling_factor` pour détecter les estimations peu fiables.

---

## 9. Commandes importantes

### Lancer le scraping d’un lot

```powershell
python .\ScrappingEnsam\scraper.py
```

### Fusionner les résultats

```powershell
python .\ScrappingEnsam\merge_all.py
```

### Enrichir avec la nouvelle target

Utiliser `ScrappingEnsam/effort_estimator2.py` et sa fonction `add_real_effort_target(...)` dans un script ou notebook.

---

## 10. Fichiers clés à connaître

- `ScrappingEnsam/scraper.py`
- `ScrappingEnsam/merge_all.py`
- `ScrappingEnsam/effort_estimator.py`
- `ScrappingEnsam/effort_estimator2.py`
- `ScrappingEnsam/features_with_real_effort_v2.csv`
- `notebooks/ml_pipeline_v5_improved.ipynb`
- `models/best_v5_model_pipeline.joblib`
