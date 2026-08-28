# Guide Détaillé : Présentation de la Collecte de Données et Feature Engineering

Ce document est conçu pour t'aider à comprendre et présenter parfaitement la partie "Collecte de données et Feature Engineering". C'est effectivement le cœur du projet, car **de bonnes données font de bons modèles**. 

Voici une explication étape par étape, avec le "Quoi", le "Comment" et surtout le "Pourquoi" (ce qui fera la différence devant un jury ou une audience).

---

## 1. La Collecte des Données (Scraping)

**Ce qu'il y a actuellement dans tes slides :** Mention de l'API GitHub REST et des 1347 dépôts.

**Ce que tu dois dire et comprendre (Le "Pourquoi") :**
*   **L'Objectif :** Les modèles classiques d'estimation d'effort (comme COCOMO) se basent sur les Lignes de Code (LOC). Mais les LOC varient énormément selon le langage (10 lignes de Java = 1 ligne de Python). Notre but était d'estimer l'effort basé sur le **processus réel de développement collaboratif** (commits, pull requests, issues).
*   **La Méthode (`scraper.py`) :** L'API GitHub étant très restrictive (limite de requêtes/Rate Limit), nous avons dû diviser le travail en **4 lots thématiques** répartis dans l'équipe.
*   **La Tolérance aux pannes :** Nous avons implémenté un système de "Checkpoint" pour sauvegarder la progression et reprendre le téléchargement en cas d'interruption ou de blocage par GitHub.

**💡 Suggestion visuelle pour les slides :** Un schéma de flux simple : `API GitHub` ➔ `Scraper (Rate Limit Handling)` ➔ `Checkpoints JSON` ➔ `Fichiers CSV bruts`.

---

## 2. Le Filtrage de Qualité

**Ce que tu dois dire :** On ne peut pas faire du Machine Learning robuste sur des dépôts étudiants abandonnés après 2 jours.
*   **Les Règles de sélection :** Nous avons imposé des filtres stricts : au moins 50 commits, au moins 2 contributeurs, au moins 30 étoiles, et des projets actifs (max 180 jours d'inactivité).
*   **Le Résultat :** Cela garantit que notre modèle s'entraîne uniquement sur des **projets open source matures et collaboratifs**, réduisant le "bruit" dans nos données. On passe de milliers de dépôts potentiels à 989 dépôts finaux de très haute qualité.

---

## 3. Le Point Crucial : La Cible d'Effort (Effort Estimator)

*C'est la partie la plus technique et la plus innovante de votre approche. C'est là-dessus qu'il faut insister.*

**Le Problème :** Comment savoir combien d'heures un développeur a travaillé sur un projet open source, alors qu'il n'y a pas de pointeuse ni de feuille de temps ?
**La Solution (Méthode de Robles) :** On reconstruit des "sessions de travail" à partir de l'heure des commits.
*   **Le Concept :** Si un développeur fait un commit à 14h00 et un autre à 15h30, on considère qu'il a travaillé en continu (car écart < 2h). On regroupe cela en une "session de travail".
*   **La Limite de l'API (Le "Scaling") :** L'API GitHub ne nous donne que les derniers commits (max 1000). Pour un gros projet comme *Kubernetes*, on sous-estimerait massivement l'effort.
*   **Notre Innovation (Le script v2) :** Nous avons mis en place une méthode d'**extrapolation bidirectionnelle**. On échantillonne des commits récents et anciens, on calcule un rythme moyen de travail, et on multiplie ce rythme par le nombre total de commits du projet (avec un facteur de décroissance pour les vieux commits).

**💡 Suggestion visuelle pour les slides :** Un petit graphique temporel (timeline) montrant un point (commit), un trait (temps écoulé), un autre point (commit). Avec la règle écrite en gros : `Si T2 - T1 < 2h ➔ 1 seule session de travail`.

---

## 4. Feature Engineering (Création de variables intelligentes)

C'est ici que tu montres l'intelligence "métier" du projet.

**Le Quoi :** Nous n'avons pas juste pris les données brutes (nombre de commits, nombre d'auteurs). Nous avons créé 6 variables d'interaction complexes.
**Le Pourquoi :** Dans la vraie vie, 10 développeurs ne produisent pas 10 fois plus vite qu'un seul développeur à cause de l'overhead (le coût) de communication (Loi de Brooks). Nos variables doivent capturer la "dynamique" de l'équipe.

**Les variables "Star" à présenter et expliquer :**
1.  **`experience_per_contributor`** : L'expérience globale divisée par le nombre de contributeurs. 
    *   *Argument à l'oral :* "Cela prouve que la **qualité** de l'équipe compte plus que la **quantité**. C'est d'ailleurs devenu la 2ème feature la plus importante de notre modèle final !"
2.  **`contributors_x_busfactor`** : Le Bus Factor représente le nombre de personnes clés qui détiennent tout le savoir du projet.
    *   *Argument à l'oral :* "Combiné au nombre de contributeurs, cela mesure le risque de silos de connaissance dans des équipes larges."
3.  **`process_maturity`** : Un indice composite combinant la présence d'Intégration Continue (CI), de tests et la régularité des releases.

---

## 5. La Rigueur Statistique : Transformation & Target Leakage

*C'est la partie "Science des données" de ta présentation, qui montre votre rigueur.*

*   **Box-Cox :** La répartition des heures de travail (notre "target") était très asymétrique : beaucoup de petits projets, et quelques projets gigantesques. Les algorithmes de Machine Learning n'aiment pas ça. La transformation Box-Cox (avec un lambda de 0.287) a permis de rendre la distribution "normale" (en forme de cloche), ce qui améliore considérablement la précision des modèles.
*   **Le Piège du "Target Leakage" (Fuite de données) :** 
    *   *L'Anecdote à raconter :* "Au début de nos essais, notre modèle avait un score suspectement parfait (R²=0.99). Nous avons enquêté et découvert un *Target Leakage* : certaines variables d'entrée (comme `total_commits`) étaient mathématiquement utilisées pour calculer la variable à prédire. Le modèle ne prédisait rien, il trichait en résolvant l'équation à l'envers. Nous avons donc mis en place 3 vagues de nettoyage pour garantir un modèle 100% honnête."

---

## Ce que tu dois rajouter/modifier sur tes slides actuels :

1.  **Graphe visuel d'une "Session de Commit" :** (À mettre près du Slide 7). Montre visuellement la règle des 2 heures. C'est très conceptuel, un dessin aide beaucoup le public.
2.  **Graphe de distribution Box-Cox :** (À rajouter pour le Slide 8). Un histogramme "Avant" (tout écrasé à gauche à cause du Skewness) et "Après Box-Cox" (une belle courbe en cloche). Tu peux générer ça facilement en Python dans vos notebooks avec `seaborn.histplot`.
3.  **Focus sur les features d'interaction :** Sur ton slide 8, il y a beaucoup de texte. Regroupe-les sous un grand titre "Capturer l'Intelligence Métier" et insiste surtout sur le *pourquoi* de `experience_per_contributor` (Qualité > Quantité).

---
### 🎤 Ton argumentaire de conclusion pour clore ta partie :
*"En conclusion de cette phase : nous ne nous sommes pas contentés de télécharger des données brutes. Nous avons réussi à **reconstituer l'effort humain** à partir de traces numériques complexes, nous avons **nettoyé rigoureusement les biais** (leakage, asymétries de distribution), et nous avons injecté notre **compréhension de la gestion de projet logiciel** à travers des variables d'interaction inédites. C'est ce socle de données fiable et intelligent qui a permis à nos modèles de Machine Learning d'atteindre par la suite d'excellentes performances."*
