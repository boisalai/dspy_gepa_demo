# Scripts

Voici l'ordre d'apprentissage progressif recommandé pour maîtriser DSPy et GEPA :

## 📚 Ordre d'apprentissage recommandé

### Niveau 1 : Fondamentaux (Les bases)

1. `config.py` - Configuration initiale

```bash
uv run src/config.py
```

- Configure DSPy avec Ollama
- Comprend les différents fournisseurs (Ollama, OpenAI, Anthropic)
- Configure le modèle de réflexion pour GEPA

2. `data.py` - Données du tutoriel

```bash
uv run src/data.py
```
- Explore le dataset de tickets IT
- Comprend les catégories et priorités
- Voit des exemples d'entraînement et validation

3. `signatures.py` - Définition des contrats d'entrée/sortie

```bash
uv run src/signatures.py
```
- Découvre différents types de signatures DSPy
- Comprend les contraintes et descriptions de champs

### Niveau 2 : Construction de modules (Créer des classifieurs)

4. `modules.py` - Modules DSPy de base

```bash
uv run src/modules.py
```
- Crée un classifieur simple
- Découvre différents types de signatures DSPy
- Comprend les contraintes et descriptions de champs

### Niveau 2 : Construction de modules (Créer des classifieurs)

4. `modules.py` - Modules DSPy de base

```bash
uv run src/modules.py
```
- Crée un classifieur simple
- Découvre les modules composés (Sequential, Validated, Ensemble)
- Test les prédictions en direct

5. `metrics.py` - Métriques d'évaluation

```bash
uv run src/metrics.py
```
- Comprend exact_match vs partial_match
- Teste différentes métriques

6. `evaluation.py` - Évaluation des modules
```bash
uv run src/evaluation.py
```
- Évalue un module sur le dataset
- Compare plusieurs modules

### Niveau 3 : Optimisation (Améliorer les performances)

7. `optimizers.py` - Optimiseurs classiques

```bash
uv run src/optimizers.py
```
- Utilise BootstrapFewShot (démonstrations automatiques)
- Découvre MIPRO (optimisation d'instructions)
- Compare baseline vs optimisé

8. `patterns.py` - Patterns de production
```bash
uv run src/patterns.py  
```

- Validation automatique des sorties
- Retry avec backoff exponentiel
- Fallback entre modèles
- Ensemble voting

### Niveau 4 : Avancé (GEPA et architectures complexes)

9. `gepa_utils.py` - Optimisation GEPA

```bash
uv run src/gepa_utils.py
```
- Optimisation génétique + Pareto + réflexion LLM
- Modes light/medium/heavy
- Inspection des prompts générés
- ⚠️ Le plus long : 5-40 minutes selon le mode

10. `multi_model.py` - Architectures multi-modèles

```bash
uv run src/multi_model.py
```
- Benchmark de différents modèles
- Architectures hybrides (différents modèles pour différentes tâches)

### Niveau 5 : Synthèse (Tout mettre ensemble)

```bash
uv run src/synthesis.py
```
- Combine les modules et les optimisations
- Crée un pipeline de bout en bout

11. `examples.py` - Exemples complets

# Exemple basique
```bash
uv run src/examples.py 1
```

# Évaluation
```bash
uv run src/examples.py 2
```

# Optimisation BootstrapFewShot
```bash
uv run src/examples.py 3
```

# Optimisation GEPA
```bash
uv run src/examples.py 4
```

# Comparaison de modules
```bash
uv run src/examples.py 5
```

# Tous les exemples
```bash
uv run src/examples.py all
```

🎯 Parcours suggérés

Pour un apprentissage rapide (1-2h) :
```bash
uv run src/config.py
uv run src/signatures.py
uv run src/modules.py
uv run src/examples.py 1
uv run src/examples.py 2
```

Pour maîtriser l'optimisation (3-4h) :
Parcours rapide + :
```bash
uv run src/optimizers.py
uv run src/examples.py 3
uv run src/gepa_utils.py  # Mode 'light'
```

Pour devenir expert (journée complète) :
Suivez tous les scripts dans l'ordre 1-11.

```bash
uv run src/config.py
uv run src/data.py
uv run src/signatures.py
uv run src/modules.py
uv run src/metrics.py   # OK
uv run src/evaluation.py  # OK
uv run src/optimizers.py  # OK
uv run src/patterns.py  #OK
uv run src/gepa_utils.py  # Mode 'heavy'
uv run src/multi_model.py
uv run src/synthesis.py
uv run src/examples.py all
```