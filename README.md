# Ordonnancement Équitable de Tournois Round-Robin

Ce projet propose différentes méthodes pour générer des calendriers de tournois round-robin équitables, en minimisant une combinaison pondérée de métriques d'équité normalisées analytiquement. Le projet gère désormais les nombres de joueurs pairs et impairs (en ajoutant un joueur fictif pour les cas impairs).

## Exécution des Solveurs

Assurez-vous d'être dans le répertoire racine du projet. Utilisez `python3 -m` pour exécuter les modules.

### Solveur Exact (MILP)

Utilise un programme linéaire en nombres entiers mixtes (MILP) pour trouver des solutions optimales pour des instances de petite taille.

Pour exécuter le solveur exact :

```bash
python3 -m src.exact_model <n_players> [<poids_ps> <poids_md> [<time_limit>]]
```

Paramètres :
- `n_players` (int, requis) : Nombre de joueurs.
- `poids_ps` (float, optionnel) : Poids pour la métrique Penalty Sequence. Par défaut, utilise la valeur `ALPHA` dans `src/config.py` (actuellement 0.8).
- `poids_md` (float, optionnel) : Poids pour la métrique Max Deviation. Par défaut, utilise la valeur `BETA` dans `src/config.py` (actuellement 1.2).
- `time_limit` (float, optionnel) : Limite de temps pour le solveur en secondes. Par défaut : Aucune.

Exemple (avec poids par défaut et sans limite de temps) :
```bash
python3 -m src.exact_model 6
```
Exemple (avec poids spécifiés et limite de temps de 10 secondes) :
```bash
python3 -m src.exact_model 6 0.8 1.2 10
```
Note : Le solveur exact minimise une combinaison pondérée des métriques normalisées. Le poids pour Home Strength est implicitement 1.0 dans l'objectif.

### Recuit Simulé (Non Optimisé)

Une implémentation basique de l'heuristique de Recuit Simulé.

Pour exécuter le solveur SA non optimisé :

```bash
python3 -m src.sa_solver_non_opti <n_players> [<iterations> [<poids_ps> <poids_md> [<time_budget>]]]
```

Paramètres :
- `n_players` (int, requis) : Nombre de joueurs.
- `iterations` (int, optionnel) : Nombre d'itérations SA. Par défaut : 10000. Ignoré si `time_budget` est fourni.
- `poids_ps` (float, optionnel) : Poids pour la métrique Penalty Sequence. Par défaut, utilise la valeur `ALPHA` dans `src/config.py` (actuellement 0.8).
- `poids_md` (float, optionnel) : Poids pour la métrique Max Deviation. Par défaut, utilise la valeur `BETA` dans `src/config.py` (actuellement 1.2).
- `time_budget` (float, optionnel) : Budget temps en secondes. Si défini, remplace `iterations`.

Exemple (avec itérations par défaut) :
```bash
python3 -m src.sa_solver_non_opti 8
```
Exemple (avec 50000 itérations et poids spécifiés) :
```bash
python3 -m src.sa_solver_non_opti 8 50000 0.8 1.2
```
Exemple (avec budget temps de 10 secondes) :
```bash
python3 -m src.sa_solver_non_opti 10 10
```

### Recuit Simulé (Optimisé Numba)

Une version optimisée avec Numba de l'heuristique de Recuit Simulé pour de meilleures performances. Ce solveur peut également sauvegarder et charger le meilleur calendrier trouvé pour un `n` donné.

Pour exécuter le solveur SA optimisé :

```bash
python3 -m src.sa_solver <n_players> [-i <iterations>] [-t <time_budget>] [<alpha> <beta>] [<runs>]
```

Paramètres :
- `n_players` (int, requis) : Nombre de joueurs.
- `-t, --time_budget` (float, optionnel) : Budget temps en secondes. Si défini, remplace `--iterations`.
- `alpha` (float, optionnel) : Poids pour la métrique Penalty Sequence. Par défaut, utilise la valeur `ALPHA` dans `src/config.py` (actuellement 0.8).
- `beta` (float, optionnel) : Poids pour la métrique Max Deviation. Par défaut, utilise la valeur `BETA` dans `src/config.py` (actuellement 1.2).
- `runs` (int, optionnel) : Nombre d'exécutions parallèles. Par défaut : 1.

Exemple (avec budget temps de 10 secondes) :
```bash
python3 -m src.sa_solver 10 -t 10
```

## Génération et Visualisation des Résultats de Calibration

Le projet inclut des scripts pour exécuter des calibrations du solveur SA optimisé sur une grille de paramètres (alpha, beta) et visualiser les résultats, notamment un graphique interactif en 3D de la frontière de Pareto.

### Exécuter la Calibration

Le script `src/run_calibration.py` exécute le solveur SA optimisé pour différentes combinaisons de poids `alpha` et `beta` et sauvegarde les résultats dans un fichier CSV.

Pour exécuter la calibration (par défaut pour n=300) :

```bash
python3 -m src.run_calibration
```

Le fichier de sortie par défaut est `calibration_results_n300_analytical_norm.csv`.
### Générer les Graphiques de Calibration

Le script `src/plot_calibration_results.py` lit un fichier de résultats de calibration CSV et génère plusieurs graphiques (2D et 3D, statiques et interactifs) dans un répertoire de sortie.

Pour générer les graphiques (par défaut à partir de `calibration_results_n300_analytical_norm.csv`) :

```bash
python3 -m src.plot_calibration_results
```

Les graphiques seront sauvegardés dans le répertoire `calibration_plots_n300_analytical_norm/`.

### Ouvrir le Graphique Interactif 3D

Le graphique interactif 3D est sauvegardé sous forme de fichier HTML. Vous pouvez l'ouvrir directement dans votre navigateur web.

Pour ouvrir le graphique 3D interactif pour n=500 (celui utilisé pour le rapport) :

```bash
open calibration_plots_n500_analytical_norm/pareto_3d_interactive_n500_anal_norm.html
```

Si vous avez exécuté la calibration et la génération de graphiques pour un autre nombre de joueurs (par exemple, n=300), le chemin du fichier HTML sera différent (par exemple, `calibration_plots_n300_analytical_norm/pareto_3d_interactive_n300_anal_norm.html`).
