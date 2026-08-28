<!-- Slide number: 1 -->

![](Image1.jpg)

<!-- Slide number: 2 -->
Plan de la Présentation

01
Contexte & Architecture

02
Données & Feature Engineering

03
Modèles & Résultats

04
Difficultés & Perspectives

### Notes:

<!-- Slide number: 3 -->

![](Image3.jpg)

<!-- Slide number: 4 -->

Contexte & Problématique

Le Défi

Notre Approche
Estimer l'effort logiciel de projets GitHub uniquement à partir des traces observables du processus de développement — sans LOC, sans feuilles de temps.
50 ans de recherche en Software Effort Estimation (SEE)

Modèles classiques (COCOMO) basés sur les LOC

①
Méthodologie structurée : 1347 dépôts GitHub
R² typique littérature : 0,50–0,75 ; MdMRE 30–50%

②
Cible d'effort : reconstruction de sessions de commits
LOC varient ×10–20 selon le langage pour une même fonctionnalité

③
Box-Cox λ=0,287 + 6 features d'interaction équipe×processus
Projets open source : équipes distribuées, bénévoles, commits hétérogènes

④
Voting Ensemble RF + LightGBM + XGBoost optimisé Optuna

### Notes:

<!-- Slide number: 5 -->

Architecture du Pipeline — 7 Étapes

1

2

3

4

5

6

7
scraper.py
effort_estimator_v2
Filtres qualité
Feature engineering
Preprocessing
Modélisation+Optuna
Évaluation
▶
▶
▶
▶
▶
▶

features_merged.csv
1347 × 44 colonnes
features_with_real_effort_v2.csv
989 dépôts retenus
(couverture ≥10%, top 2%)
34 features finales
(28 orig. + 6 interact.)
Box-Cox λ=0,287
Split 80/20 · StandardScaler
5 modèles tunés +
2 ensembles (Stacking, Voting)
R², PRED(25/50), MAPE
Permutation · LOGO-CV

Chaque étape produit un artefact persisté (CSV / modèle sérialisé) permettant de reprendre l'exécution à n'importe quel point — reproductibilité garantie par random_state = 42.
Stratification par lots thématiques

Lot 1
Python / JS
~277
scikit-learn, React, Django

Lot 2
TypeScript / Java
~261
Angular, Spring Boot, VS Code

Lot 3
Go / Rust
~536
Kubernetes, Tokio, Terraform

Lot 4
ML / Web
~326
TensorFlow, FastAPI, Next.js

### Notes:

<!-- Slide number: 6 -->

![](Image1.jpg)

<!-- Slide number: 7 -->

Collecte de Données & Estimation d'Effort Commit-Based

Collecte via API GitHub REST

Méthode Robles et al. (2014)
1347 dépôts publics (stars>30, commits>50, contrib.≥2)
Reconstruction de sessions de travail :

Si deux commits d'un même développeur sont séparés par < 2h → même session de travail.
Gestion rate-limiting : X-RateLimit-Remaining < 100

δ(S) = max(t) − min(t)  |  min 30 min, max 8 h
21 features : processus + qualité + dynamique d'équipe

Extrapolation bidirectionnelle (500 commits récents + 300 anciens) × facteur décroissance 0,85
Endpoints : /commits, /stats/contributors, /git/trees

Filtrage Qualité
Filtre couverture : suppression des dépôts avec couverture < 10% (sauf projets < 1500 commits) → −222 dépôts
Filtre outliers : suppression du 98ème percentile → −13 dépôts  |  Dataset final : 989 dépôts, effort 12–2640 h

### Notes:

<!-- Slide number: 8 -->

Feature Engineering Avancé

Transformation Box-Cox de la cible

Gestion du Target Leakage
Distribution brute : skewness = +0,80
3 vagues de détection :
log1p standard → skewness = −0,63  (sur-correction ❌)
⚠ 1ère : R²=0,99 — total_commits via cocomo_hours
Box-Cox λ = 0,287 → skewness = −0,10  ✓ quasi-normal
⚠ 2ème : R²=0,70 — reliability_score indirect
y(λ) = (yλ − 1) / λ
⚠ 3ème : R²=0,994 — colonnes diagnostics estimateur
→ Règle : tracer le graphe de dépendance de toutes les colonnes calculées
6 Features d'Interaction Équipe×Processus — Contribution : 28,3% de l'importance totale

experience_per_contributor ★
weighted_exp / (contrib+1)
2ème feature : qualité > quantité (11%)

contributors_x_release
contrib × release_regularity
Overhead de coordination (Brooks 1995)

contributors_x_busfactor
contrib × bus_factor_ratio
Risque de silos de connaissance

pr_per_contributor
pr_count / (contrib+1)
Bottleneck de revue individuelle

process_maturity
0,4×CI + 0,3×tests + 0,3×release
Indice composite de maturité

inactivity_burden
days_inactive × log(contrib+1)
Coût de remise en route équipe large

### Notes:

<!-- Slide number: 9 -->

![](Image1.jpg)

<!-- Slide number: 10 -->

Modèles Machine Learning & Optimisation Optuna

Ridge / Lasso

Random Forest

LightGBM
Régularisation L2/L1
α=10 par CV
Sélection de features implicite (Lasso)
200 arbres · Bootstrap
n_estimators, max_depth tunés
200 trials Optuna
Croissance leaf-wise
Histogramme-based (O(b))
300 trials Optuna · 9 params

R²(BC)
R²(BC)
R²(BC)
0,49
0,910
0,914
⚙ 200 trials
⚙ 300 trials

XGBoost

Stacking Ensemble

Voting Ensemble ★
Croissance level-wise
Régularisation conservative
200 trials Optuna
Méta-learner RidgeCV
Prédictions out-of-fold
Pondération optimale modèles
Moyenne des 3 modèles
Réduction de variance
Meilleur modèle final

R²(BC)
R²(BC)
R²(BC)
0,911
0,918
0,921
⚙ 200 trials
🏆 Meilleur modèle

### Notes:

<!-- Slide number: 11 -->

Résultats Comparatifs & Performances Finales

0,893
26,0%
69,7%
88,9%
R² espace original
MAPE
PRED(25)
PRED(50)
Test set 198 dépôts
Erreur relative moyenne
Seuil acceptabilité > 50%
Seuil acceptabilité > 65%
Progression des modèles (espace original)
| Modèle | R²(orig) | MAPE | PRED(25) | PRED(50) |
| --- | --- | --- | --- | --- |
| Baseline mono-feature | 0,398 | — | 22,0% | 46,3% |
| RandomForest v4 (sans tuning) | 0,483 | 59,6% | 28,5% | 64,2% |
| LightGBM (300 trials) | 0,881 | 27,7% | 68,7% | 89,4% |
| RandomForest (200 trials) | 0,886 | 27,8% | 71,2% | 88,9% |
| XGBoost (200 trials) | 0,873 | 27,7% | 65,7% | 89,9% |
| Stacking Ensemble | 0,888 | 26,8% | 70,7% | 87,9% |
| Voting Ensemble ★ | 0,893 | 26,0% | 69,7% | 88,9% |

### Notes:

<!-- Slide number: 12 -->

Importance des Features & Validation LOGO-CV

### Chart: Importance des features (Voting Ensemble)

| Category | Importance |
|---|---|
| active_contrib. | 33.5 |
| experience/contrib ★ | 11.0 |
| bus_factor_ratio | 4.3 |
| contrib×busfactor ★ | 3.6 |
| language_div. | 4.0 |
| process_maturity ★ | 2.5 |

Validation LOGO-CV

Python / JS
R² = 0,567
✓ Bonne généralisation

TypeScript / Java
R² = 0,566
✓ Pratiques similaires

Go / Rust
R² = 0,316
⚠ Limitation — commits
moins fréquents

Insight clé :

ML / Web
R² = 0,545
experience_per_contributor (★) atteint 11% d'importance — 2ème rang. La qualité de l'équipe est un prédicteur d'effort plus puissant que sa taille seule. Ce résultat remet en question les hypothèses fondamentales des modèles COCOMO.
✓ Projets hétérogènes
La limitation sur Go/Rust reflète des pratiques de commits moins fréquentes — biais structurel de la reconstruction.

### Notes:

<!-- Slide number: 13 -->

![](Image1.jpg)

<!-- Slide number: 14 -->

Difficultés Rencontrées

Target Leakage — Détection progressive
1
3 vagues : R²=0,99 (total_commits) → R²=0,70 (reliability_score) → R²=0,994 (colonnes diagnostics estimateur). Règle : tracer le graphe de dépendance de toutes les colonnes calculées.

Troncature de la cible commit-based
2
Limitation API : 1000 commits max → cible tronquée entre 20–550h dans v1. Solution : extrapolation bidirectionnelle (500 récents + 300 anciens) × facteur décroissance 0,85 + filtre couverture 10%.

Choix de la transformation de la cible
3
log1p standard → skewness −0,63 (sur-correction). Box-Cox λ=0,287 → skewness −0,10 (quasi-normal). Investigation approfondie nécessaire — impact significatif sur les modèles linéaires.

Stabilité des ensembles
4
Corrélation élevée RF/LGB/XGB (r>0,95) → gain limité des ensembles. Voting bat Stacking : méta-learner en surapprentissage sur 491 exemples. La moyenne simple se révèle plus robuste.

### Notes:

<!-- Slide number: 15 -->

Perspectives d'Amélioration & Déploiement

🎯  Amélioration de la cible

⚙️  Feature Engineering avancé
Données GitHub Issues avec estimations de temps (labels estimate: Xh)
Features réseau de collaboration (co-modifications de fichiers — Bird 2011)

Intégration Wakatime / Copilot Telemetry pour temps de codage effectif
Features temporelles de dérive (concept drift — dynamique équipe)

Meilleure détection des bots et commits automatisés
Métriques de complexité architecturale (modules, couplage)

📊  Amélioration des modèles

🚀  Déploiement & Généralisation
Régression par quantiles → intervalles de prédiction [P10, P90]
API REST publique : URL repo → estimation + intervalle de confiance

Active Learning : identifier les dépôts les plus informatifs pour Go/Rust
Généralisation GitLab / Bitbucket / Azure DevOps (features génériques Git)

Modèles multi-tâches séparés par communauté de langage
Interface web pour chefs de projet (évaluation de projets open source à reprendre)

### Notes:

<!-- Slide number: 16 -->

Conclusion Générale

Ce projet démontre la faisabilité d'une estimation d'effort logiciel fondée exclusivement sur des métriques de processus GitHub, surpassant les seuils d'acceptabilité industrielle définis dans la littérature académique — sans recours aux LOC ni aux modèles COCOMO.

R² = 0,893
MAPE = 26%
PRED(25) = 69,7%
+85% R²
Test set 198 dépôts
vs 59,6% baseline
Seuil > 50% ✓
vs RandomForest v4
Contributions scientifiques principales
✓  Reconstruction de sessions commit → cible d'effort viable pour projets open source
✓  experience_per_contributor : 2ème feature la plus importante — qualité > quantité
✓  Modèle basé sur métriques processus uniquement comparable aux outils industriels LOC-based

ENSAM — Département Informatique & IA  |  Projet Machine Learning 2025–2026

### Notes: