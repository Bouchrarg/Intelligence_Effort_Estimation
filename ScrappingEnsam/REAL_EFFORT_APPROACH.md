# Nouvelle approche d'estimation d'effort de la branche `feature/ml-notebooks`

## 1. Contexte général

Cette branche vise à remplacer la target synthétique existante (`effort_target`) par une target d'effort réellement estimée à partir de l'historique de commits GitHub.

Le pipeline proposé se compose de trois étapes principales :

1. 
Scraping distribué des repos GitHub avec `ScrappingEnsam/scraper.py`.
2. Fusion des lots en un dataset unique avec `ScrappingEnsam/merge_all.py`.
3. Enrichissement du dataset avec un effort réel calculé par `ScrappingEnsam/effort_estimator.py` ou `ScrappingEnsam/effort_estimator2.py`.

Le fichier de sortie enrichi présent dans la branche est :

- `ScrappingEnsam/features_with_real_effort_v2.csv`

Il contient notamment le nouveau target : `effort_hours_real`.

---

## 2. Nouveau target : `effort_hours_real`

### Ce que c'est

`effort_hours_real` est une estimation en heures de l'effort réellement dépensé sur un repository, reconstruite à partir de l'activité de commit.

### Colonnes associées

Dans `features_with_real_effort_v2.csv`, les colonnes liées au nouveau target sont :

- `effort_hours_real`
- `effort_hours_sampled`
- `effort_coverage_ratio`
- `effort_commits_sampled`
- `effort_total_commits`
- `effort_contributors_seen`
- `effort_scaling_factor`
- `effort_was_capped`

Ces colonnes donnent à la fois la target finale et des diagnostics de qualité.

---

## 3. Comment le nouveau target est calculé

La branche contient deux versions d'estimation :

### 3.1 `ScrappingEnsam/effort_estimator.py` — version 1

Cette version utilise une reconstruction de sessions de travail basée sur les commits.

#### Principe

Pour chaque auteur du dépôt :

- les commits sont triés par date,
- si deux commits sont espacés de moins de `SESSION_GAP_H = 2.0` heures, ils appartiennent à la même session,
- la durée d'une session est `dernier_commit - premier_commit`,
- une session isolée d'un seul commit est au moins `DEFAULT_SOLO_MIN = 30` minutes,
- chaque session est plafonnée à `MAX_SESSION_H = 8.0` heures.

La somme de toutes les sessions donne l'effort total estimé.

#### Formule de base

Pour un auteur :

- si la session dure moins de 30 min, on remplace par 30 min,
- si la session dure plus de 8 h, on la limite à 8 h,
- la durée d'une session est donc :
  `max(30 min, min(8 h, last - first))`.

Pour un repo :

- `effort_hours_real = somme des durées de session de tous les contributeurs`

#### Diagnostic

La version 1 ajoute aussi :

- `effort_contributors_seen`
- `effort_commits_sampled`
- `effort_pages_fetched`

Ces informations servent surtout à vérifier la qualité de l'estimation et la taille de l'échantillon.

---

### 3.2 `ScrappingEnsam/effort_estimator2.py` — version 2 (approche améliorée)

La version 2 corrige le principal biais de v1 : l'échantillonnage limité aux commits récents ne reflète pas l'effort historique complet des gros projets.

#### Deux phases

1. **Phase d'échantillonnage**
   - on récupère `SAMPLE_PAGES = 5` pages de commits récents,
   - on récupère `SAMPLE_PAGES_OLD = 3` pages de commits les plus anciens,
   - chaque page = `COMMITS_PER_PAGE = 100` commits.

2. **Phase de scaling**
   - on utilise `total_commits` pour extrapoler l'effort estimé à toute l'histoire du projet.

#### Pourquoi ce choix ?

- les commits récents reflètent l'activité actuelle de l'équipe,
- les commits anciens reflètent l'effort de lancement et la structure historique,
- l'interpolation entre les deux réduit le biais d'une estimation purement récente.

#### Calcul détaillé

1. On calcule `effort_recent_h` à partir des commits récents.
2. On calcule `effort_old_h` à partir des commits anciens.
3. On récupère `total_commits` via l'API GitHub si disponible.
4. On calcule le ratio d'échantillonnage :
   `commits_sampled = commits_recent + commits_old`.
5. On extrapole l'effort historique total en tenant compte d'une correction pour les commits anciens.

La formule de scaling est implémentée dans `scale_effort_to_full_history(...)`.

#### Formule de scaling utilisée

- `rate_recent = effort_recent_h / commits_recent`
- `rate_old = (effort_old_h × DECAY_FACTOR) / commits_old`
- `rate_middle = (rate_recent + rate_old) / 2`
- `commits_middle = total_commits - commits_sampled`
- `effort_middle = rate_middle × commits_middle`
- `effort_total = effort_recent_h + effort_old_h × DECAY_FACTOR + effort_middle`

avec

- `DECAY_FACTOR = 0.85`

Ce facteur signifie que l'on considère les commits anciens comme un peu moins « denses » en effort que les commits récents, ce qui est cohérent avec l'hypothèse que les premiers commits étaient plus gros mais moins fréquents.

#### Diagnostic et filtrage

La version 2 produit :

- `effort_hours_sampled` : effort brut des commits échantillonnés,
- `effort_coverage_ratio` : `commits_sampled / total_commits`,
- `effort_scaling_factor` : `1 / coverage`,
- `effort_was_capped` : vrai si le résultat dépasse la borne maximale,
- `effort_total_commits` : le total des commits du dépôt.

Elle applique également des bornes de plausibilité plus larges :

- `MIN_EFFORT_H = 10` h,
- `MAX_EFFORT_H = 5_000_000` h.

Cela permet d’inclure des projets très volumineux, tout en filtrant les valeurs aberrantes extrêmes.

---

## 4. Avantages de la nouvelle approche

### 4.1 Réalisme supérieur

- La target n'est plus artificielle ni dérivée d'une formule statique.
- Elle est construite à partir de l'activité de contribution réelle du repo.
- Elle capture le comportement des contributeurs au niveau des sessions de travail.

### 4.2 Diagnostics plus riches

- La version 2 fournit des métriques de confiance (`coverage`, `scaling_factor`).
- On sait combien de commits et de contributeurs ont réellement participé à l'estimation.
- Cela permet de repérer les estimations peu fiables sans casser tout le dataset.

### 4.3 Meilleure robustesse pour les gros projets

- En échantillonnant les commits anciens et récents, la méthode corrige le biais d'un simple historique récent.
- La correction `DECAY_FACTOR` réduit le surestimation due à des anciennes contributions plus lourdes.

### 4.4 Intégration facile au dataset existant

- La fonction `add_real_effort_target(...)` enrichit simplement le DataFrame existant produit par `merge_all.py`.
- Le dataset final reste compatible avec les notebooks ML présents dans la branche.

---

## 5. Inconvénients et limites

### 5.1 Hypothèses fortes

- L'approche assume que l'effort est proportionnel au nombre de commits.
- Elle suppose que les sessions de commits de 2 h sont représentatives du vrai travail.
- Les commits « fantômes » (merge, reformat, nettoyage) peuvent fausser l'effort.

### 5.2 Dépendance à l'API GitHub

- La méthode nécessite des appels API pour chaque repo : commits, total commits, éventuellement stats contributrices.
- Si `total_commits` n'est pas disponible, le scaling devient moins précis.
- Des limites de rate limit peuvent ralentir ou arrêter le traitement.

### 5.3 Biais des commits anciens

- La correction appliquée avec `DECAY_FACTOR = 0.85` est heuristique.
- Si le projet a un rythme de commit très variable, l'interpolation peut rester imparfaite.

### 5.4 Peut être sensible aux outliers

- Un repo avec un très petit nombre de commits peut recevoir une estimation instable.
- Les commits isolés donnent au minimum 30 min, ce qui peut exagérer l'effort des projets presque inactifs.

### 5.5 Complexité et coût

- v2 est plus coûteuse à calculer que v1.
- Le pipeline ajoute des étapes de collecte supplémentaires et des calculs de diagnostic.

---

## 6. Pourquoi il y a deux modules de modèles entraînés

La branche contient deux artefacts de modèle :

- `models/best_v5_model_pipeline.joblib`
- `models/go_rust_rf_model.joblib`

### Interprétation probable

- `best_v5_model_pipeline.joblib` est très probablement le meilleur pipeline général issu de l'itération `ml_pipeline_v5_improved.ipynb`.
- `go_rust_rf_model.joblib` suggère un modèle Random Forest entraîné sur un sous-ensemble spécifique (Go/Rust), ou bien une expérience dédiée pour ces langages.

### Pourquoi deux modèles ?

- l'équipe a expérimenté plusieurs pipelines et architectures,
- un modèle peut être général (`best_v5_model_pipeline`) tandis qu'un autre est spécialisé sur des langages ou des caractéristiques différentes,
- cela correspond à une démarche d'exploration : comparer un pipeline « général » et un modèle plus ciblé pour améliorer les performances.

Ce choix est cohérent avec la présence de plusieurs notebooks : `ml_pipeline.ipynb`, `ml_pipeline_v4_fixed.ipynb`, `ml_pipeline_v5_improved.ipynb`.

---

## 7. Nouvelles ajouts et modifications majeures de la branche

### 7.1 Fichiers ajoutés / nouveaux

- `ScrappingEnsam/effort_estimator.py`
- `ScrappingEnsam/effort_estimator2.py`
- `ScrappingEnsam/features_with_real_effort_v2.csv`
- `notebooks/ml_pipeline_v4_fixed.ipynb`
- `notebooks/ml_pipeline_v5_improved.ipynb`
- `models/best_v5_model_pipeline.joblib`
- `models/go_rust_rf_model.joblib`

### 7.2 Modifications importantes

#### `ScrappingEnsam/scraper.py`

- chemins d'export modifiés vers `Scrapped_Data/` et `Checkpoints/`,
- fix pour `get_code_frequency` avec attente 10s + retry 90s,
- fix pour `get_commit_stats` en utilisant `/contributors` synchrone plutôt que `/stats/contributors`,
- ajout de `review_cycle_count`,
- amélioration des filtres qualité et de la génération des features.

#### `ScrappingEnsam/merge_all.py`

- fusion des CSV LOT en un seul dataset,
- déduplication par `full_name`,
- impression de statistiques de target `effort_target` avant export.

#### `ScrappingEnsam/effort_estimator.py`

- nouvelle target basée sur la reconstruction de sessions,
- mise en place de seuils de plausibilité,
- enrichissement batch via `add_real_effort_target(...)`.

#### `ScrappingEnsam/effort_estimator2.py`

- nouvelle version de l'estimation avec échantillonnage récent + ancien,
- scaling vers l'historique complet (`total_commits`),
- diagnostics détaillés pour chaque repo,
- bornes plausibles étendues pour les grands dépôts.

---

## 8. Recommandations d'utilisation

### Utilisation idéale

1. exécuter `ScrappingEnsam/scraper.py` pour générer les CSV LOT,
2. exécuter `ScrappingEnsam/merge_all.py` pour créer `features_merged.csv`,
3. charger `features_merged.csv` dans Python,
4. enrichir avec `ScrappingEnsam/effort_estimator2.py` via `add_real_effort_target(...)`,
5. utiliser `effort_hours_real` comme nouvelle target dans les notebooks ML.

### Pourquoi choisir `effort_estimator2.py`

- c'est la version la plus aboutie,
- elle est conçue pour réduire le biais des gros projets,
- elle fournit des métriques de confiance qui ne sont pas présentes dans la version 1.

---

## 9. Conclusion

La branche `feature/ml-notebooks` introduit une vraie cible d'effort basée sur l'historique de commit plutôt que sur une formule statique.

- `effort_estimator.py` est la version de base basée sur la reconstruction de sessions,
- `effort_estimator2.py` est la version améliorée avec échantillonnage des extrémités et scaling full-history,
- les deux modèles enregistrés montrent une exploration de pipelines ML distincts,
- les nouveaux diagnostics rendent l'estimation beaucoup plus traçable.

> En résumé : la nouvelle target est `effort_hours_real`; elle est construite par session de commits puis éventuellement extrapolée à l'ensemble de l'historique du repo. Cette approche est plus réaliste que la formule ancienne, mais elle reste dépendante des hypothèses sur les commits et de la qualité des données GitHub.
