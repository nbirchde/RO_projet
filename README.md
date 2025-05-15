# Ordonnancement Équitable de Tournois Round-Robin

Ce projet propose différentes méthodes pour générer des calendriers de tournois round-robin équitables, en minimisant une combinaison pondérée de métriques d'équité normalisées analytiquement. Le projet gère désormais les nombres de joueurs pairs et impairs (en ajoutant un joueur fictif pour les cas impairs).

## Exécution des Solveurs

Assurez-vous d'être dans le répertoire racine du projet (`/Users/nicholasbirchdelacalle/Documents/BA3/RO/ro-projet`). Utilisez `python3 -m` pour exécuter les modules.

### Solveur Exact (MILP)

Utilise un programme linéaire en nombres entiers mixtes (MILP) pour trouver des solutions optimales pour des instances de petite taille.

Pour exécuter le solveur exact :

```bash
python3 -m src.exact_model [n_players] [poids_hs] [poids_ps] [poids_md]
```

Paramètres :
- `n_players` (int, requis) : Nombre de joueurs.
- `poids_hs` (float, optionnel) : Poids pour la métrique Home Strength. Par défaut, utilise la valeur dans `src/config.py`.
- `poids_ps` (float, optionnel) : Poids pour la métrique Penalty Sequence. Par défaut, utilise la valeur dans `src/config.py`.
- `poids_md` (float, optionnel) : Poids pour la métrique Max Deviation. Par défaut, utilise la valeur dans `src/config.py`.

Exemple :
```bash
python3 -m src.exact_model 6 1.0 0.8 1.2
```
Note : Le solveur exact ne modélise pas directement la Penalty Sequence dans son objectif, mais la métrique est calculée et normalisée après la résolution.

### Recuit Simulé (Non Optimisé)

Une implémentation basique de l'heuristique de Recuit Simulé.

Pour exécuter le solveur SA non optimisé :

```bash
python3 -m src.sa_solver_non_opti [n_players] [iterations] [poids_hs] [poids_ps] [poids_md]
```

Paramètres :
- `n_players` (int, requis) : Nombre de joueurs.
- `iterations` (int, optionnel) : Nombre d'itérations SA. Par défaut : 10000.
- `poids_hs` (float, optionnel) : Poids pour la métrique Home Strength. Par défaut, utilise la valeur dans `src/config.py`.
- `poids_ps` (float, optionnel) : Poids pour la métrique Penalty Sequence. Par défaut, utilise la valeur dans `src/config.py`.
- `poids_md` (float, optionnel) : Poids pour la métrique Max Deviation. Par défaut, utilise la valeur dans `src/config.py`.

Exemple :
```bash
python3 -m src.sa_solver_non_opti 8 50000 1.0 0.8 1.2
```

### Recuit Simulé (Optimisé Numba)

Une version optimisée avec Numba de l'heuristique de Recuit Simulé pour de meilleures performances. Ce solveur peut également sauvegarder et charger le meilleur calendrier trouvé pour un `n` donné.

Pour exécuter le solveur SA optimisé :

```bash
python3 -m src.sa_solver [n_players] [iterations] [poids_hs] [poids_ps] [poids_md] [runs]
```

Paramètres :
- `n_players` (int, requis) : Nombre de joueurs.
- `iterations` (int, optionnel) : Nombre d'itérations SA par exécution. Par défaut : 100000.
- `poids_hs` (float, optionnel) : Poids pour la métrique Home Strength. Par défaut, utilise la valeur dans `src/config.py`.
- `poids_ps` (float, optionnel) : Poids pour la métrique Penalty Sequence. Par défaut, utilise la valeur dans `src/config.py`.
- `poids_md` (float, optionnel) : Poids pour la métrique Max Deviation. Par défaut, utilise la valeur dans `src/config.py`.
- `runs` (int, optionnel) : Nombre d'exécutions parallèles. Par défaut : 1.

Exemple :
```bash
python3 -m src.sa_solver 10 100000 1.0 0.8 1.2 4
