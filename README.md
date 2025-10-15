# Tutoriel DSPy et GEPA : Classification automatique de billets d'assistance informatique

Ce tutoriel vous guide à travers **DSPy** (Declarative Self-improving Language Programs) et **GEPA** (Genetic-Pareto Algorithm), deux outils puissants pour développer des applications utilisant des modèles de langage.

## Vue d'ensemble

DSPy est un cadre de développement qui permet de créer des programmes utilisant des modèles de langage de manière déclarative et maintenable. Plutôt que d'écrire des instructions manuellement, vous définissez ce que vous voulez accomplir, et DSPy optimise automatiquement les instructions et les exemples.

GEPA est l'optimiseur le plus sophistiqué de DSPy. Il utilise des algorithmes génétiques et la réflexion par modèle de langage pour trouver automatiquement les meilleures instructions possibles.

## Objectifs d'apprentissage

À la fin de ce tutoriel, vous serez en mesure de :

1. Définir des **signatures** DSPy pour déclarer vos tâches
2. Créer et composer des **modules** DSPy
3. Évaluer la performance avec des **métriques personnalisées**
4. Optimiser automatiquement avec les **optimiseurs** (BootstrapFewShot, MIPRO, GEPA)
5. Utiliser différents **modèles de langage** de manière interchangeable
6. Implémenter des **motifs avancés** pour la production

## Prérequis

### Logiciels requis

- **Python 3.9+**
- **Ollama** (pour exécuter des modèles localement)
- **uv** (gestionnaire de paquets Python recommandé)

### Installation

1. **Installer Ollama**

   Visitez [ollama.ai](https://ollama.ai) et suivez les instructions pour votre système d'exploitation.

2. **Télécharger un modèle**

   ```bash
   ollama pull llama3.1:8b
   ```

3. **Cloner ce dépôt**

   ```bash
   git clone https://github.com/votre-utilisateur/dspy_gepa_demo.git
   cd dspy_gepa_demo
   ```

4. **Installer les dépendances**

   Avec uv (recommandé) :
   ```bash
   uv pip install dspy-ai
   ```

   Ou avec pip :
   ```bash
   pip install dspy-ai
   ```

## Structure du projet

```
dspy_gepa_demo/
├── README.md                 # Ce fichier
├── src/
│   ├── config.py            # Configuration de DSPy et des modèles
│   ├── data.py              # Données d'entraînement et de validation
│   ├── signatures.py        # Définitions des signatures DSPy
│   ├── modules.py           # Modules DSPy (Predict, ChainOfThought, etc.)
│   ├── metrics.py           # Métriques d'évaluation
│   ├── evaluation.py        # Fonctions d'évaluation
│   ├── optimizers.py        # Optimiseurs (BootstrapFewShot, MIPRO)
│   ├── gepa_utils.py        # Utilitaires GEPA
│   ├── patterns.py          # Motifs avancés (validation, retry, etc.)
│   ├── multi_model.py       # Configuration multi-modèles
│   └── examples.py          # Exemples d'utilisation
└── images/                   # Images pour la documentation
```

## Concepts fondamentaux

### 1. Signatures : Déclarer ce que vous voulez faire

Une **signature** définit le contrat d'entrée-sortie de votre tâche. C'est comme une interface qui décrit ce que votre programme doit accomplir, sans spécifier comment.

**Exemple simple :**

```python
import dspy

class TicketClassifier(dspy.Signature):
    """Classer un billet d'assistance informatique selon sa catégorie et sa priorité."""

    ticket = dspy.InputField(desc="Description du billet d'assistance informatique")
    category = dspy.OutputField(desc="Catégorie parmi: Hardware, Software, Network, etc.")
    priority = dspy.OutputField(desc="Priorité parmi: Low, Medium, High, Urgent, Critical")
```

**Composants d'une signature :**

- **Docstring** : Description de la tâche
- **InputField** : Champs d'entrée avec descriptions
- **OutputField** : Champs de sortie attendus

### 2. Modules : Exécuter vos tâches

Un **module** utilise une signature pour générer des prédictions. DSPy offre plusieurs types de modules :

#### Predict (simple)

Génération directe sans raisonnement explicite.

```python
import dspy

class SimpleClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(TicketClassifier)

    def forward(self, ticket):
        return self.predictor(ticket=ticket)
```

#### ChainOfThought (avec raisonnement)

Le modèle raisonne avant de répondre, ce qui améliore la précision.

```python
class ThinkingClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(TicketClassifier)

    def forward(self, ticket):
        return self.predictor(ticket=ticket)
```

**Avantage :** Amélioration typique de 5 à 15 % de la précision.

#### Modules composés

Vous pouvez composer plusieurs modules pour créer des architectures plus complexes.

```python
class SequentialClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.category_predictor = dspy.ChainOfThought(CategoryClassifier)
        self.priority_predictor = dspy.ChainOfThought(PriorityClassifier)

    def forward(self, ticket):
        # Étape 1 : Prédire la catégorie
        category_result = self.category_predictor(ticket=ticket)

        # Étape 2 : Prédire la priorité en utilisant la catégorie
        priority_result = self.priority_predictor(
            ticket=ticket,
            category=category_result.category
        )

        return dspy.Prediction(
            category=category_result.category,
            priority=priority_result.priority
        )
```

### 3. Métriques : Mesurer la performance

Une **métrique** est une fonction qui évalue la qualité d'une prédiction. Elle retourne un score entre 0 et 1.

**Exemple : Correspondance exacte**

```python
def exact_match_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Retourne 1 si la catégorie ET la priorité sont correctes, 0 sinon.
    """
    category_match = prediction.category.strip().lower() == example['category'].strip().lower()
    priority_match = prediction.priority.strip().lower() == example['priority'].strip().lower()

    return 1.0 if (category_match and priority_match) else 0.0
```

### 4. Optimiseurs : Améliorer automatiquement

Les **optimiseurs** améliorent vos modules en optimisant les instructions et en générant des exemples de démonstration.

#### BootstrapFewShot

L'optimiseur le plus simple. Il génère des exemples de démonstration en exécutant votre module sur les données d'entraînement et en gardant les prédictions correctes.

**Amélioration typique :** 5 à 15 %

#### MIPRO

Optimise à la fois les instructions et les exemples en générant plusieurs variantes et en testant différentes combinaisons.

**Amélioration typique :** 10 à 25 %

#### GEPA (Genetic-Pareto Algorithm)

L'optimiseur le plus sophistiqué. Il combine :
- **Algorithmes génétiques** : Évolution de populations d'instructions
- **Réflexion par modèle de langage** : Analyse des erreurs et propositions d'améliorations
- **Optimisation Pareto** : Équilibre de multiples objectifs

**Amélioration typique :** 15 à 30 %

**Niveaux d'optimisation :**

| Niveau   | Durée      | Appels LLM | Amélioration | Utilisation             |
|----------|------------|------------|--------------|-------------------------|
| `light`  | 5-10 min   | ~200-400   | 10-20%       | Tests, prototypage      |
| `medium` | 10-20 min  | ~400-800   | 15-25%       | Production légère       |
| `heavy`  | 20-40 min  | ~800-1600  | 20-30%       | Performance maximale    |

## Comment GEPA fonctionne

GEPA utilise une approche inspirée de l'évolution biologique :

1. **Population initiale** : Génère plusieurs variantes d'instructions
2. **Évaluation** : Teste chaque variante sur les données d'entraînement
3. **Sélection** : Garde les meilleures (front de Pareto)
4. **Réflexion** : Un modèle de langage analyse les erreurs et propose des améliorations
5. **Mutation** : Génère de nouvelles variantes basées sur la réflexion
6. **Répétition** : Continue jusqu'à convergence

**Avantages de GEPA :**
- Trouve des instructions que les humains n'auraient pas imaginées
- Apprend de ses erreurs de manière itérative
- Équilibre plusieurs objectifs (précision, concision, etc.)

**Quand utiliser GEPA :**
- Vous visez la meilleure performance possible
- Vous avez suffisamment de données (20+ exemples)
- Vous avez du temps pour l'optimisation (10-30 minutes)
- La tâche est critique pour votre application

## Guide d'utilisation rapide

### Exemple 1 : Utilisation de base

```python
# 1. Configurer DSPy avec Ollama
from src.config import configure_ollama

lm = configure_ollama()

# 2. Créer un module
from src.modules import SimpleTicketClassifier

classifier = SimpleTicketClassifier()

# 3. Faire des prédictions
ticket = "Mon ordinateur ne démarre plus, j'ai une présentation dans 1 heure"
result = classifier(ticket=ticket)

print(f"Catégorie : {result.category}")
print(f"Priorité : {result.priority}")
```

### Exemple 2 : Évaluation

```python
from src.evaluation import evaluate_module
from src.metrics import exact_match_metric
from src.data import get_val_examples

# Évaluer sur l'ensemble de validation
val_examples = get_val_examples()
score = evaluate_module(classifier, val_examples, exact_match_metric)

print(f"Score : {score:.2%}")
```

### Exemple 3 : Optimisation avec BootstrapFewShot

```python
from src.optimizers import optimize_with_bootstrap
from src.data import get_train_examples

# Données d'entraînement
train_examples = get_train_examples()

# Optimiser
optimized = optimize_with_bootstrap(
    classifier,
    train_examples,
    exact_match_metric,
    max_bootstrapped_demos=4
)

# Évaluer l'amélioration
score_optimized = evaluate_module(optimized, val_examples, exact_match_metric)
print(f"Score optimisé : {score_optimized:.2%}")
```

### Exemple 4 : Optimisation avec GEPA

```python
from src.gepa_utils import optimize_with_gepa
from src.config import configure_reflection_lm

# Configurer le modèle de réflexion
reflection_lm = configure_reflection_lm()

# Optimiser avec GEPA (mode léger pour débuter)
gepa_optimized = optimize_with_gepa(
    classifier,
    train_examples,
    val_examples,
    exact_match_metric,
    reflection_lm,
    auto='light'
)

# Évaluer
score_gepa = evaluate_module(gepa_optimized, val_examples, exact_match_metric)
print(f"Score GEPA : {score_gepa:.2%}")
```

### Exemple 5 : Exécuter tous les exemples

```bash
# Exemple de base
python src/examples.py 1

# Évaluation
python src/examples.py 2

# Optimisation
python src/examples.py 3

# GEPA
python src/examples.py 4

# Comparaison de modules
python src/examples.py 5

# Tous les exemples
python src/examples.py all
```

## Meilleures pratiques

### Données

- **Minimum 20-30 exemples** pour l'entraînement
- **Exemples diversifiés** couvrant tous les cas d'usage
- **Ensemble de validation séparé** (15-20 % des données)
- **Étiquettes cohérentes** et vérifiées

### Métriques

- **Commencer simple** : exact_match pour débuter
- **Ajouter des nuances** : partial_match pour plus de signal
- **Tester sur des cas limites** : exemples ambigus, courts, longs
- **Documenter clairement** : expliquer ce que signifie chaque score

### Optimisation

- **Phase 1** : Module de base sans optimisation
- **Phase 2** : BootstrapFewShot pour amélioration rapide
- **Phase 3** : MIPRO si les résultats sont prometteurs
- **Phase 4** : GEPA pour performance maximale

### Production

- **Toujours valider** les sorties
- **Implémenter un réessai** pour les APIs
- **Avoir un plan B** (modèle de secours)
- **Surveiller la performance** en continu
- **Collecter les erreurs** pour amélioration continue

## Dépannage

### Ollama ne démarre pas

```bash
# Vérifier qu'Ollama est installé
ollama --version

# Démarrer Ollama
ollama serve

# Vérifier les modèles disponibles
ollama list
```

### Le modèle n'est pas trouvé

```bash
# Télécharger le modèle
ollama pull llama3.1:8b
```

### Erreur de mémoire insuffisante

- Utiliser un modèle plus petit : `mistral:7b`
- Réduire `max_tokens` du reflection_lm
- Utiliser `auto='light'` au lieu de `medium` ou `heavy`
- Fermer les autres applications

### GEPA ne s'améliore pas

- Vérifier la qualité des données (minimum 20 exemples)
- Vérifier que la métrique est bien définie
- Essayer un niveau plus élevé (`medium` ou `heavy`)
- Augmenter le nombre d'exemples d'entraînement

## Ressources

### Documentation officielle

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Article DSPy](https://arxiv.org/abs/2310.03714)

### GEPA

- [Article GEPA](https://arxiv.org/abs/2507.19457)
- [GEPA GitHub](https://github.com/gepa-ai/gepa)

### Ollama

- [Documentation Ollama](https://ollama.ai/docs)
- [Bibliothèque de modèles Ollama](https://ollama.ai/library)

### Communauté

- [Discord DSPy](https://discord.gg/dspy)
- [Discord Ollama](https://discord.gg/ollama)

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bogues
- Proposer des améliorations
- Ajouter des exemples
- Améliorer la documentation

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Remerciements

- L'équipe DSPy de Stanford pour ce cadre de développement remarquable
- Les créateurs de GEPA pour cet optimiseur sophistiqué
- La communauté Ollama pour les modèles locaux accessibles

---

**Bon développement avec DSPy et GEPA !**
