# Tutoriel DSPy et GEPA : Classification automatique de billets d'assistance informatique

Ce tutoriel explique **DSPy** (Declarative Self-improving Language Programs) et **GEPA** (Genetic-Pareto Algorithm), deux outils pour développer des applications utilisant des modèles de langage.

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

### 1. Les signatures DSPy

#### Qu'est-ce qu'une signature ?

Une signature dans DSPy est une déclaration de ce que votre programme doit faire, pas comment le faire.

#### Analogie

Imaginez que vous engagez un assistant :

- ❌ Sans DSPy : Vous lui donnez des instructions détaillées ("d'abord lis le ticket, ensuite regarde si c'est hardware...")
- ✅ Avec DSPy : Vous lui donnez simplement le contrat ("je te donne un ticket, tu me donnes la catégorie et la priorité")

#### Composants d'une signature

- **Docstring** : Description de la tâche
- **Champs d'entrée** (`InputField`) : Ce que vous fournissez
- **Champs de sortie** (`OutputField`) : Ce que vous attendez
- **Descriptions** (optionnelles) : Précisions sur chaque champ

#### Exemple 1 : Signature simple

La forme la plus basique d'une signature.

```python
import dspy

class BasicSignature(dspy.Signature):
    """Classifier un ticket IT."""
    
    ticket = dspy.InputField()
    category = dspy.OutputField()
```

#### Exemple 2 : Signature avec descriptions

Ajouter des descriptions aide le modèle à mieux comprendre la tâche.

```python
class DescriptiveSignature(dspy.Signature):
    """Classifier un ticket de support IT selon sa catégorie."""
    
    ticket = dspy.InputField(desc="Description du problème rapporté par l'utilisateur")
    category = dspy.OutputField(desc="Catégorie technique du problème")
```

#### Exemple 3 : Signature avec contraintes

Spécifier les valeurs possibles dans la description.

```python
import dspy

class TicketClassifier(dspy.Signature):
    """Classer un billet d'assistance informatique selon sa catégorie et sa priorité."""

    ticket = dspy.InputField(desc="Description du billet d'assistance informatique")
    category = dspy.OutputField(desc="Catégorie parmi: Hardware, Software, Network, etc.")
    priority = dspy.OutputField(desc="Priorité parmi: Low, Medium, High, Urgent, Critical")
```

#### ✅ Bonnes Pratiques pour les signatures

- **Docstring claire** : Décrivez la tâche en une phrase
- **Noms explicites** : ticket plutôt que input, category plutôt que output
- **Descriptions précises** : Ajoutez desc pour guider le modèle
- **Contraintes claires** : Listez les valeurs possibles quand applicable
- **Commencer simple** : Ajoutez des champs progressivement


### 2. Modules : Exécuter vos tâches

## Qu'est-ce qu'un module ?

Un **module** dans DSPy est un composant qui **utilise une signature** pour générer des prédictions.

**Analogie :**
- **Signature** = Le contrat ("je te donne X, tu me donnes Y")
- **Module** = L'employé qui exécute le contrat (avec sa propre méthode de travail)

DSPy offre plusieurs types de modules, chacun avec une stratégie différente :

#### Module 1 : Predict (le plus simple)

**Predict** est le module de base : il génère directement une réponse.

**Fonctionnement :**
1. Reçoit les entrées
2. Génère immédiatement les sorties
3. Retourne le résultat

**Quand l'utiliser :**
- Tâches simples
- Besoin de rapidité
- Première version d'un système

```python
import dspy

class SimpleClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(TicketClassifier)

    def forward(self, ticket):
        return self.predictor(ticket=ticket)
```

#### Module 2 : ChainOfThought (avec raisonnement)

**ChainOfThought** demande au modèle de raisonner avant de répondre.

**Fonctionnement :**
1. Reçoit les entrées
2. **Génère d'abord un raisonnement**
3. Génère ensuite les sorties basées sur ce raisonnement
4. Retourne le résultat (avec le raisonnement)

**Quand l'utiliser :**
- Tâches complexes nécessitant de la réflexion
- Besoin d'expliquer les décisions
- Améliorer la précision (+5-15% typiquement)

**Avantages :**
- ✅ Meilleure précision
- ✅ Raisonnement explicite et auditable
- ✅ Meilleure gestion des cas limites

**Inconvénients :**
- ❌ Plus lent (génère plus de tokens)
- ❌ Plus coûteux en appels LLM

```python
class ThinkingClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(TicketClassifier)

    def forward(self, ticket):
        return self.predictor(ticket=ticket)
```

**Exemple de sortie avec raisonnement :**
```
Raisonnement: "L'utilisateur mentionne que son ordinateur ne démarre plus.
Il s'agit clairement d'un problème matériel. De plus, il a une présentation
dans 1 heure, ce qui rend le problème urgent."

Catégorie: Hardware
Priorité: Urgent
```

#### Module 3 : ReAct (raisonnement + actions)

**ReAct** alterne entre raisonnement et actions.

**Fonctionnement :**
1. Raisonne sur le problème
2. Décide d'une action à faire (ex: chercher dans une base de données)
3. Observe le résultat de l'action
4. Raisonne à nouveau avec cette nouvelle information
5. Répète jusqu'à avoir la réponse

**Quand l'utiliser :**
- Besoin d'interactions avec des outils externes
- Recherche d'informations nécessaire
- Tâches multi-étapes

**Note :** ReAct nécessite de définir des outils (fonctions) que le modèle peut appeler.

#### Module 4 : ProgramOfThought (génération de code)

**ProgramOfThought** génère du code Python pour raisonner.

**Fonctionnement :**
1. Analyse le problème
2. Génère du code Python pour le résoudre
3. Exécute le code
4. Utilise le résultat pour générer la réponse

**Quand l'utiliser :**
- Problèmes mathématiques
- Calculs complexes
- Manipulation de données structurées

**Exemple typique :** "Combien font 347 * 892 + 123 / 7 ?"
- Le modèle génère : `result = 347 * 892 + 123 / 7`
- Exécute le code : `309541.57`
- Retourne la réponse avec le calcul exact

#### Module 5 : Modules personnalisés (composition)

Vous pouvez créer vos propres modules en **composant** plusieurs modules existants.

**Pourquoi composer des modules ?**
- Décomposer une tâche complexe en sous-tâches
- Réutiliser des modules existants
- Créer des pipelines sophistiqués

**Exemple 1 : Pipeline séquentiel**

Classifier d'abord la catégorie, puis la priorité en fonction de la catégorie.

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

**Exemple 2 : Module avec validation**

Ajouter une étape de validation pour vérifier que les prédictions sont valides.

```python
class ValidatedClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(TicketClassifier)
        self.valid_categories = {"Hardware", "Software", "Network", "Access"}
        self.valid_priorities = {"Low", "Medium", "High", "Urgent", "Critical"}

    def forward(self, ticket):
        result = self.predictor(ticket=ticket)

        # Validation et normalisation
        if result.category not in self.valid_categories:
            result.category = "Software"  # Catégorie par défaut
        if result.priority not in self.valid_priorities:
            result.priority = "Medium"  # Priorité par défaut

        return result
```

**Exemple 3 : Module avec consensus (ensemble)**

Utiliser plusieurs modules et combiner leurs prédictions.

```python
class EnsembleClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifiers = [
            dspy.Predict(TicketClassifier),
            dspy.ChainOfThought(TicketClassifier),
            dspy.ChainOfThought(TicketClassifier)  # 2x ChainOfThought
        ]

    def forward(self, ticket):
        predictions = [clf(ticket=ticket) for clf in self.classifiers]

        # Vote majoritaire pour la catégorie
        categories = [p.category for p in predictions]
        category = max(set(categories), key=categories.count)

        # Vote majoritaire pour la priorité
        priorities = [p.priority for p in predictions]
        priority = max(set(priorities), key=priorities.count)

        return dspy.Prediction(category=category, priority=priority)
```

#### 💡 Bonnes pratiques pour les modules

**✅ À faire :**

1. **Commencer simple** : Utilisez d'abord `Predict`, puis `ChainOfThought` si besoin
2. **Nommer clairement** : `TicketClassifier` plutôt que `Classifier1`
3. **Un module = une tâche** : Gardez les modules focalisés
4. **Composer progressivement** : Testez chaque module individuellement
5. **Documenter** : Ajoutez des docstrings à vos modules personnalisés

**❌ À éviter :**

1. **Utiliser ChainOfThought partout** : Plus lent et plus coûteux
2. **Trop de composition** : Gardez les pipelines compréhensibles
3. **Oublier la validation** : Vérifiez toujours les sorties
4. **Ne pas mesurer** : Utilisez des métriques pour comparer les modules

### 3. Métriques : Mesurer la performance

## Pourquoi évaluer ?

Jusqu'à présent, nous avons créé des modules et observé leurs sorties qualitativement. Mais pour :
- **Comparer** différents modules
- **Mesurer** les améliorations
- **Optimiser** automatiquement (avec GEPA)

...nous avons besoin de **mesures quantitatives** : les **métriques**.

## Qu'est-ce qu'une métrique ?

Une **métrique** est une fonction qui prend :
- Un **exemple** avec la vraie réponse (ground truth)
- Une **prédiction** du modèle
- Des paramètres optionnels (trace, pred_name, pred_trace) pour les optimiseurs

Et retourne un **score entre 0.0 et 1.0** :
- **0.0** = Complètement incorrect
- **1.0** = Parfaitement correct
- **0.5** = Partiellement correct

#### Métrique 1 : Exact Match (correspondance exacte)

La métrique la plus stricte : tout doit être parfait.

**Avantages :**
- ✅ Simple à comprendre
- ✅ Pas d'ambiguïté
- ✅ Facile à interpréter (0% ou 100%)

**Inconvénients :**
- ❌ Très stricte
- ❌ Ne donne pas de crédit partiel
- ❌ Peut décourager si le score est trop bas

```python
def exact_match_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Retourne 1.0 si la catégorie ET la priorité sont correctes, 0.0 sinon.

    Args:
        example: Dict avec les vraies valeurs {'ticket': ..., 'category': ..., 'priority': ...}
        prediction: Objet Prediction avec .category et .priority
        trace: Trace d'exécution (optionnel, pour GEPA)
        pred_name: Nom du predictor (optionnel, pour GEPA)
        pred_trace: Trace du predictor (optionnel, pour GEPA)

    Returns:
        float: 1.0 si exact match, 0.0 sinon
    """
    category_match = prediction.category.strip().lower() == example['category'].strip().lower()
    priority_match = prediction.priority.strip().lower() == example['priority'].strip().lower()

    return 1.0 if (category_match and priority_match) else 0.0
```

**Exemple d'utilisation :**
```python
example = {'ticket': 'Mon PC ne démarre pas', 'category': 'Hardware', 'priority': 'Urgent'}
prediction = dspy.Prediction(category='Hardware', priority='Urgent')

score = exact_match_metric(example, prediction)
print(f"Score: {score}")  # Output: Score: 1.0
```

#### Métrique 2 : Partial Match (correspondance partielle)

Plus nuancée : donne des points partiels si au moins un champ est correct.

**Avantages :**
- ✅ Plus de nuance
- ✅ Donne du crédit partiel
- ✅ Meilleur signal d'apprentissage

**Inconvénients :**
- ❌ Moins binaire (interprétation plus complexe)

```python
def partial_match_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Retourne un score partiel basé sur les champs corrects.

    Score:
    - 1.0 si catégorie ET priorité correctes
    - 0.5 si seulement l'une des deux est correcte
    - 0.0 si les deux sont incorrectes

    Returns:
        float: Score entre 0.0 et 1.0
    """
    category_match = prediction.category.strip().lower() == example['category'].strip().lower()
    priority_match = prediction.priority.strip().lower() == example['priority'].strip().lower()

    if category_match and priority_match:
        return 1.0
    elif category_match or priority_match:
        return 0.5
    else:
        return 0.0
```

**Exemple de comparaison :**
```python
example = {'ticket': 'Mon PC ne démarre pas', 'category': 'Hardware', 'priority': 'Urgent'}

# Cas 1: Tout correct
pred1 = dspy.Prediction(category='Hardware', priority='Urgent')
print(f"Exact: {exact_match_metric(example, pred1)}")    # 1.0
print(f"Partial: {partial_match_metric(example, pred1)}") # 1.0

# Cas 2: Seulement la catégorie correcte
pred2 = dspy.Prediction(category='Hardware', priority='Low')
print(f"Exact: {exact_match_metric(example, pred2)}")    # 0.0
print(f"Partial: {partial_match_metric(example, pred2)}") # 0.5

# Cas 3: Tout incorrect
pred3 = dspy.Prediction(category='Software', priority='Low')
print(f"Exact: {exact_match_metric(example, pred3)}")    # 0.0
print(f"Partial: {partial_match_metric(example, pred3)}") # 0.0
```

#### Fonction d'évaluation réutilisable

Créons une fonction pour évaluer n'importe quel module sur un dataset complet.

```python
def evaluate_module(module, dataset, metric):
    """
    Évalue un module DSPy sur un dataset avec une métrique.

    Args:
        module: Module DSPy à évaluer
        dataset: Liste de dictionnaires avec 'ticket', 'category', 'priority'
        metric: Fonction de métrique (exact_match_metric ou partial_match_metric)

    Returns:
        float: Score moyen sur le dataset (entre 0.0 et 1.0)
    """
    scores = []

    for example in dataset:
        # Faire la prédiction
        prediction = module(ticket=example['ticket'])

        # Calculer le score
        score = metric(example, prediction)
        scores.append(score)

    # Retourner le score moyen
    return sum(scores) / len(scores) if scores else 0.0
```

**Exemple d'utilisation :**
```python
from src.data import get_val_examples
from src.modules import SimpleTicketClassifier

# Charger les données
val_examples = get_val_examples()

# Créer le classifier
classifier = SimpleTicketClassifier()

# Évaluer
score_exact = evaluate_module(classifier, val_examples, exact_match_metric)
score_partial = evaluate_module(classifier, val_examples, partial_match_metric)

print(f"Exact match: {score_exact:.1%}")      # ex: 65.0%
print(f"Partial match: {score_partial:.1%}")  # ex: 82.5%
```

#### 💡 Bonnes pratiques pour l'évaluation

**✅ À faire :**

1. **Toujours avoir un dataset de validation séparé** : Ne jamais évaluer sur les données d'entraînement
2. **Utiliser plusieurs métriques** : Exact match + partial match donnent une vue complète
3. **Tester sur des cas limites** : Tickets ambigus, très courts, très longs
4. **Documenter vos métriques** : Expliquez ce que signifie chaque score
5. **Comparer de manière équitable** : Même dataset, même métrique

**❌ À éviter :**

1. **Une seule métrique** : Peut ne pas capturer toute la complexité
2. **Dataset trop petit** : Minimum 10-15 exemples de validation
3. **Évaluer sur le trainset** : Donnera des scores artificiellement élevés
4. **Ignorer les cas limites** : Les erreurs se cachent souvent dans les edge cases

### 4. Optimiseurs : Améliorer automatiquement

## Introduction : Modules vs Optimiseurs

Jusqu'à présent, nous avons vu des **modules** (Predict, ChainOfThought, ReAct, etc.). Ces modules **exécutent** des tâches en interrogeant le LLM.

Les **optimiseurs**, quant à eux, **améliorent** les modules en :
- Ajoutant des exemples de démonstration (few-shot learning)
- Optimisant les instructions (prompts)
- Ajustant les paramètres
- Sélectionnant les meilleures configurations

**Analogie :** Si un module est comme un employé qui exécute des tâches, un optimiseur est comme un coach qui entraîne l'employé à s'améliorer.

## BootstrapFewShot : Générer des exemples de démonstration

**BootstrapFewShot** est l'optimiseur le plus simple de DSPy. Il fonctionne en :

1. Exécutant votre module sur les données d'entraînement
2. Gardant les prédictions correctes (validées par votre métrique)
3. Utilisant ces prédictions comme exemples de démonstration (few-shot)
4. Injectant ces exemples dans le prompt du module optimisé

**Avantages :**
- ✅ Simple à comprendre et à utiliser
- ✅ Rapide à exécuter
- ✅ Amélioration typique de 5-15%
- ✅ Pas besoin de configuration complexe

**Inconvénients :**
- ❌ N'optimise pas les instructions
- ❌ Amélioration limitée comparé à MIPRO ou GEPA

**Quand l'utiliser :**
- Première optimisation
- Tests rapides
- Vous avez peu de temps
- Vous voulez comprendre comment fonctionne l'optimisation

**Exemple d'utilisation :**
```python
from dspy.teleprompt import BootstrapFewShot
from src.data import get_train_examples
from src.metrics import exact_match_metric

# Créer votre module
classifier = SimpleTicketClassifier()

# Configurer l'optimiseur
optimizer = BootstrapFewShot(
    metric=exact_match_metric,
    max_bootstrapped_demos=4,  # Nombre max d'exemples à générer
    max_labeled_demos=0,        # Pas d'exemples manuels
)

# Optimiser
train_examples = get_train_examples()
optimized_classifier = optimizer.compile(
    student=classifier,
    trainset=train_examples
)

# Le module optimisé inclut maintenant 4 exemples de démonstration
```

## BootstrapFewShotWithRandomSearch

Une variante améliorée de BootstrapFewShot qui teste plusieurs combinaisons d'exemples.

**Fonctionnement :**
1. Génère plusieurs ensembles d'exemples
2. Teste différentes combinaisons
3. Garde la meilleure configuration selon la métrique

**Amélioration typique :** 8-18%

**Quand l'utiliser :** Quand vous avez un peu plus de temps que BootstrapFewShot standard.

## SignatureOptimizer

**SignatureOptimizer** se concentre uniquement sur l'optimisation des instructions de votre signature, sans ajouter d'exemples de démonstration.

**Fonctionnement :**
1. Génère plusieurs variantes d'instructions pour votre signature
2. Teste chaque variante
3. Garde la meilleure

**Configuration :**
```python
from dspy.teleprompt import SignatureOptimizer

optimizer = SignatureOptimizer(
    metric=exact_match_metric,
    breadth=10,  # Nombre de variantes à générer
    depth=3      # Nombre d'itérations de raffinement
)

optimized = optimizer.compile(
    student=classifier,
    trainset=train_examples,
    valset=val_examples
)
```

**Amélioration typique :** 5-12%

**Quand l'utiliser :**
- Vous voulez améliorer vos prompts sans ajouter d'exemples
- Vous avez des contraintes de latence (les exemples ajoutent des tokens)

## MIPRO : Optimisation des instructions et exemples

**MIPRO** (Multi-prompt Instruction Proposal Optimizer) est un optimiseur plus avancé qui :

1. **Génère plusieurs variantes d'instructions** pour votre signature
2. **Sélectionne les meilleurs exemples** de démonstration
3. **Teste différentes combinaisons** (instructions × exemples)
4. **Garde la meilleure configuration** selon votre métrique

**Avantages :**
- ✅ Optimise à la fois les instructions et les exemples
- ✅ Amélioration typique de 10-25%
- ✅ Recherche systématique de la meilleure configuration

**Inconvénients :**
- ❌ Plus lent que BootstrapFewShot
- ❌ Nécessite plus d'appels LLM
- ❌ Configuration plus complexe

**Quand l'utiliser :**
- Après avoir testé BootstrapFewShot
- Vous voulez une meilleure performance
- Vous avez 10-20 minutes pour l'optimisation

**Configuration :**
```python
from dspy.teleprompt import MIPRO

optimizer = MIPRO(
    metric=exact_match_metric,
    num_candidates=10,      # Nombre de variantes d'instructions à tester
    init_temperature=1.0    # Température pour la génération
)

optimized = optimizer.compile(
    student=classifier,
    trainset=train_examples,
    valset=val_examples,
    num_trials=10,          # Nombre d'essais
    max_bootstrapped_demos=4
)
```

## GEPA (Genetic-Pareto Algorithm)

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

## Comparaison des optimiseurs

| Optimiseur | Ce qu'il optimise | Vitesse | Amélioration typique | Complexité | Quand l'utiliser |
|------------|------------------|---------|---------------------|------------|------------------|
| **BootstrapFewShot** | Exemples uniquement | ⚡⚡⚡ Rapide | 5-15% | Simple | Première optimisation, tests rapides |
| **BootstrapFewShotWithRandomSearch** | Exemples (avec recherche) | ⚡⚡ Moyen | 8-18% | Simple | Quand vous avez un peu plus de temps |
| **SignatureOptimizer** | Instructions uniquement | ⚡⚡ Moyen | 5-12% | Moyen | Améliorer les prompts sans exemples |
| **MIPRO** | Instructions + exemples | ⚡ Lent | 10-25% | Moyen | Production, bonne performance |
| **GEPA** | Instructions + exemples + réflexion | 🐌 Très lent | 15-30% | Élevée | Performance maximale, tâches critiques |

## Stratégie d'optimisation recommandée

1. **Phase 1 : Baseline** - Commencez sans optimisation pour avoir un point de référence
2. **Phase 2 : BootstrapFewShot** - Première amélioration rapide (5-10 minutes)
3. **Phase 3 : MIPRO** - Si les résultats sont prometteurs (10-20 minutes)
4. **Phase 4 : GEPA** - Pour la performance maximale sur les tâches critiques (20-40 minutes)

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

## Multi-modèles et flexibilité

### Introduction : Pourquoi utiliser plusieurs modèles ?

DSPy offre une **abstraction puissante** : votre code reste le même quel que soit le modèle utilisé. Vous pouvez :

1. **Changer de fournisseur** facilement (Ollama → OpenAI → Anthropic)
2. **Comparer les performances** de différents modèles
3. **Créer des architectures hybrides** (modèle rapide pour catégorie, modèle précis pour priorité)
4. **Optimiser coût vs performance**

**Avantages du multi-modèles :**
- 💰 **Optimisation des coûts** : Utilisez des modèles gratuits (Ollama) pour le développement
- 🎯 **Meilleure performance** : Testez plusieurs modèles pour trouver le meilleur
- 🔒 **Confidentialité** : Modèles locaux pour les données sensibles
- ⚡ **Flexibilité** : Changez de modèle sans réécrire votre code

### Configuration de différents fournisseurs

#### Ollama (local, gratuit)

Ollama permet d'exécuter des modèles **localement** sans API key ni coûts.

**Modèles recommandés :**
- `llama3.1:8b` - Équilibré, bon pour la plupart des tâches (4.7 GB)
- `mistral:7b` - Rapide, bon pour les tâches simples (4.1 GB)
- `qwen2.5:7b` - Haute qualité, excellent pour les tâches complexes (4.7 GB)
- `gemma2:9b` - Alternative de Google, très performant (5.4 GB)

**Configuration :**
```python
import dspy

# Configurer Ollama
lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',
    api_base='http://localhost:11434',
    temperature=0.0  # Déterministe pour la classification
)

dspy.configure(lm=lm)
```

#### OpenAI (API, payant)

OpenAI propose des modèles très performants via API.

**Modèles recommandés :**
- `gpt-4o-mini` - Rapide et économique, bon rapport qualité/prix
- `gpt-4o` - Haute performance, multimodal
- `gpt-4-turbo` - Équilibré performance/coût

**Configuration :**
```python
import dspy
import os

# Votre clé API OpenAI
os.environ['OPENAI_API_KEY'] = 'sk-...'

# Configurer OpenAI
lm = dspy.LM(
    model='openai/gpt-4o-mini',
    temperature=0.0
)

dspy.configure(lm=lm)
```

#### Anthropic (API, payant)

Anthropic propose les modèles Claude, connus pour leur qualité et leur sécurité.

**Modèles recommandés :**
- `claude-3-5-haiku-20241022` - Rapide et économique
- `claude-3-5-sonnet-20241022` - Équilibré, excellent pour la plupart des tâches
- `claude-3-opus-20240229` - Maximum de performance

**Configuration :**
```python
import dspy
import os

# Votre clé API Anthropic
os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'

# Configurer Anthropic
lm = dspy.LM(
    model='anthropic/claude-3-5-sonnet-20241022',
    temperature=0.0
)

dspy.configure(lm=lm)
```

### Comparer les performances de différents modèles

Créons une fonction de benchmarking pour comparer les modèles :

```python
import time
from src.data import get_val_examples
from src.metrics import exact_match_metric

def benchmark_model(model_config, val_examples, metric):
    """
    Benchmark un modèle sur un dataset de validation.

    Args:
        model_config: Configuration du modèle (nom, api_base, etc.)
        val_examples: Dataset de validation
        metric: Métrique d'évaluation

    Returns:
        dict: Résultats avec score, temps, et modèle
    """
    # Configurer le modèle
    lm = dspy.LM(**model_config)
    dspy.configure(lm=lm)

    # Créer le classifier
    classifier = SimpleTicketClassifier()

    # Mesurer le temps
    start_time = time.time()

    # Évaluer
    scores = []
    for example in val_examples:
        pred = classifier(ticket=example['ticket'])
        score = metric(example, pred)
        scores.append(score)

    elapsed_time = time.time() - start_time
    avg_score = sum(scores) / len(scores)

    return {
        'model': model_config['model'],
        'score': avg_score,
        'time': elapsed_time,
        'time_per_example': elapsed_time / len(val_examples)
    }

# Comparer plusieurs modèles Ollama
models = [
    {'model': 'ollama_chat/llama3.1:8b', 'api_base': 'http://localhost:11434'},
    {'model': 'ollama_chat/mistral:7b', 'api_base': 'http://localhost:11434'},
    {'model': 'ollama_chat/qwen2.5:7b', 'api_base': 'http://localhost:11434'},
]

val_examples = get_val_examples()
results = [benchmark_model(m, val_examples, exact_match_metric) for m in models]

# Afficher les résultats
for r in results:
    print(f"{r['model']:30} | Score: {r['score']:.1%} | Temps: {r['time']:.1f}s ({r['time_per_example']:.2f}s/ex)")
```

### Architectures hybrides : utiliser différents modèles pour différentes tâches

Une **architecture hybride** utilise différents modèles pour différentes parties de votre pipeline. Par exemple :

- Modèle **rapide et économique** pour la catégorisation
- Modèle **précis mais coûteux** pour la priorisation

**Avantages :**
- 💰 **Optimisation des coûts** : Utiliser des modèles coûteux uniquement quand nécessaire
- ⚡ **Optimisation de la vitesse** : Modèles rapides pour les tâches simples
- 🎯 **Optimisation de la qualité** : Modèles précis pour les tâches critiques

**Exemple :**
```python
class HybridTicketClassifier(dspy.Module):
    """
    Classifier hybride utilisant différents modèles pour différentes tâches.
    """

    def __init__(self, category_lm, priority_lm):
        super().__init__()
        self.category_lm = category_lm
        self.priority_lm = priority_lm

    def forward(self, ticket):
        # Étape 1 : Catégorie avec modèle rapide (Ollama)
        with dspy.context(lm=self.category_lm):
            category_pred = dspy.Predict(CategoryClassifier)
            category_result = category_pred(ticket=ticket)

        # Étape 2 : Priorité avec modèle précis (OpenAI)
        with dspy.context(lm=self.priority_lm):
            priority_pred = dspy.ChainOfThought(PriorityClassifier)
            priority_result = priority_pred(
                ticket=ticket,
                category=category_result.category
            )

        return dspy.Prediction(
            category=category_result.category,
            priority=priority_result.priority
        )

# Configuration
fast_lm = dspy.LM(model='ollama_chat/llama3.1:8b', api_base='http://localhost:11434')
precise_lm = dspy.LM(model='openai/gpt-4o-mini')

# Créer le classifier hybride
hybrid_classifier = HybridTicketClassifier(
    category_lm=fast_lm,
    priority_lm=precise_lm
)
```

### Guide de sélection de modèles

| Critère | Ollama (local) | OpenAI | Anthropic |
|---------|---------------|--------|-----------|
| **Coût** | Gratuit (matériel local) | Payant à l'usage | Payant à l'usage |
| **Vitesse** | Dépend du matériel | Rapide (API cloud) | Rapide (API cloud) |
| **Confidentialité** | ✅ 100% local | ⚠️ Données envoyées à OpenAI | ⚠️ Données envoyées à Anthropic |
| **Qualité** | Bonne (7-8B params) | Excellente | Excellente |
| **Disponibilité** | Dépend de votre machine | Haute (99.9% uptime) | Haute (99.9% uptime) |
| **Setup** | Installation locale requise | API key uniquement | API key uniquement |

**Recommandations :**

- **Développement/tests** : Ollama (gratuit, rapide à itérer)
- **Production avec données sensibles** : Ollama (confidentialité)
- **Production haute performance** : OpenAI ou Anthropic (qualité maximale)
- **Production économique** : Hybride (Ollama pour les tâches simples, API pour les tâches complexes)

## Patterns avancés (Production)

Cette section couvre des **patterns de production** pour rendre vos modules DSPy plus robustes, fiables et performants.

### Pourquoi utiliser ces patterns ?

En production, les LLMs peuvent :
- ❌ Générer des sorties invalides (mauvais format, valeurs hors limites)
- ❌ Échouer temporairement (timeout, rate limiting)
- ❌ Produire des résultats incohérents
- ❌ Être indisponibles (downtime API)

Les **patterns avancés** permettent de :
- ✅ Valider et corriger les sorties
- ✅ Réessayer automatiquement en cas d'erreur
- ✅ Basculer vers un modèle de secours
- ✅ Combiner plusieurs prédictions pour plus de robustesse

### Pattern 1 : Validation

Le **pattern de validation** vérifie que les sorties du LLM respectent les contraintes de votre application.

**Problème :**
Les LLMs peuvent générer :
- Des catégories qui n'existent pas ("Matériel" au lieu de "Hardware")
- Des priorités invalides ("Très urgent" au lieu de "Urgent")
- Des formats incorrects (minuscules au lieu de majuscules)

**Solution :**
```python
class ValidatedTicketClassifier(dspy.Module):
    """
    Classifier avec validation et correction des sorties.
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(TicketClassifier)

        # Valeurs valides
        self.valid_categories = {"Hardware", "Software", "Network", "Access", "Other"}
        self.valid_priorities = {"Low", "Medium", "High", "Urgent", "Critical"}

        # Mapping pour la normalisation
        self.category_mapping = {
            "matériel": "Hardware",
            "logiciel": "Software",
            "réseau": "Network",
            "accès": "Access",
        }
        self.priority_mapping = {
            "faible": "Low",
            "moyen": "Medium",
            "haute": "High",
            "urgent": "Urgent",
            "critique": "Critical",
        }

    def _normalize(self, value, valid_set, mapping, default):
        """Normalise et valide une valeur."""
        # Nettoyer
        value = value.strip()

        # Déjà valide ?
        if value in valid_set:
            return value

        # Essayer en minuscules
        lower_value = value.lower()
        if lower_value in mapping:
            return mapping[lower_value]

        # Fallback : valeur par défaut
        return default

    def forward(self, ticket):
        # Prédiction
        result = self.predictor(ticket=ticket)

        # Validation et normalisation
        result.category = self._normalize(
            result.category,
            self.valid_categories,
            self.category_mapping,
            "Other"
        )
        result.priority = self._normalize(
            result.priority,
            self.valid_priorities,
            self.priority_mapping,
            "Medium"
        )

        return result
```

### Pattern 2 : Retry (réessayer en cas d'erreur)

Le **pattern de retry** réessaye automatiquement une opération en cas d'échec temporaire.

**Solution :**
```python
import time
from typing import Optional

class RetryTicketClassifier(dspy.Module):
    """
    Classifier avec retry automatique en cas d'erreur.
    """

    def __init__(self, max_retries=3, backoff_factor=2):
        super().__init__()
        self.predictor = dspy.ChainOfThought(TicketClassifier)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def forward(self, ticket):
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Essayer la prédiction
                result = self.predictor(ticket=ticket)
                return result

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Attendre avec backoff exponentiel
                    wait_time = self.backoff_factor ** attempt
                    time.sleep(wait_time)

        # Si tous les essais ont échoué
        raise Exception(f"Failed after {self.max_retries} attempts: {last_error}")
```

### Pattern 3 : Fallback (modèle de secours)

Le **pattern de fallback** utilise un modèle de secours si le modèle principal échoue.

**Solution :**
```python
class FallbackTicketClassifier(dspy.Module):
    """
    Classifier avec fallback vers un modèle de secours.
    """

    def __init__(self, primary_lm, fallback_lm):
        super().__init__()
        self.primary_lm = primary_lm
        self.fallback_lm = fallback_lm

    def forward(self, ticket):
        # Essayer avec le modèle principal
        try:
            with dspy.context(lm=self.primary_lm):
                predictor = dspy.ChainOfThought(TicketClassifier)
                result = predictor(ticket=ticket)
                result.model_used = "primary"
                return result

        except Exception as e:
            # Fallback vers le modèle de secours
            with dspy.context(lm=self.fallback_lm):
                predictor = dspy.Predict(TicketClassifier)
                result = predictor(ticket=ticket)
                result.model_used = "fallback"
                return result
```

### Pattern 4 : Ensemble (combiner plusieurs prédictions)

Le **pattern d'ensemble** combine les prédictions de plusieurs modèles pour améliorer la robustesse.

**Solution :**
```python
class EnsembleTicketClassifier(dspy.Module):
    """
    Classifier d'ensemble combinant plusieurs modèles.
    """

    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, ticket):
        predictions = []

        # Obtenir les prédictions de chaque modèle
        for lm in self.models:
            with dspy.context(lm=lm):
                predictor = dspy.ChainOfThought(TicketClassifier)
                pred = predictor(ticket=ticket)
                predictions.append(pred)

        # Vote majoritaire pour la catégorie
        categories = [p.category for p in predictions]
        category = max(set(categories), key=categories.count)

        # Vote majoritaire pour la priorité
        priorities = [p.priority for p in predictions]
        priority = max(set(priorities), key=priorities.count)

        return dspy.Prediction(category=category, priority=priority)
```

### Combiner les patterns

En production, vous pouvez **combiner plusieurs patterns** :

```python
class ProductionTicketClassifier(dspy.Module):
    """
    Classifier de production avec tous les patterns.
    """

    def __init__(self, primary_lm, fallback_lm, max_retries=3):
        super().__init__()
        self.primary_lm = primary_lm
        self.fallback_lm = fallback_lm
        self.max_retries = max_retries

        # Valeurs valides
        self.valid_categories = {"Hardware", "Software", "Network", "Access", "Other"}
        self.valid_priorities = {"Low", "Medium", "High", "Urgent", "Critical"}

    def _validate(self, result):
        """Valide et normalise les sorties."""
        if result.category not in self.valid_categories:
            result.category = "Other"
        if result.priority not in self.valid_priorities:
            result.priority = "Medium"
        return result

    def forward(self, ticket):
        last_error = None

        # Pattern Retry
        for attempt in range(self.max_retries):
            try:
                # Pattern Fallback
                try:
                    with dspy.context(lm=self.primary_lm):
                        predictor = dspy.ChainOfThought(TicketClassifier)
                        result = predictor(ticket=ticket)
                except Exception:
                    with dspy.context(lm=self.fallback_lm):
                        predictor = dspy.Predict(TicketClassifier)
                        result = predictor(ticket=ticket)

                # Pattern Validation
                result = self._validate(result)
                return result

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise Exception(f"Failed after {self.max_retries} attempts: {last_error}")
```

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

### Erreurs GEPA courantes

#### Erreur: `TypeError: GEPA.__init__() got an unexpected keyword argument`

**Symptôme:**
```
TypeError: GEPA.__init__() got an unexpected keyword argument 'breadth'
```

**Cause:** Utilisation de paramètres de l'ancienne API GEPA (pre-3.0)

**Solution:**
```python
# ❌ Ancienne API (ne fonctionne plus)
optimizer = GEPA(
    metric=my_metric,
    breadth=10,
    depth=3
)

# ✅ Nouvelle API (DSPy 3.0+)
optimizer = GEPA(
    metric=my_metric,
    auto='light'  # ou 'medium' ou 'heavy'
)
```

#### Erreur: `reflection_lm` manquant

**Symptôme:**
```
ValueError: reflection_lm is required for GEPA
```

**Solution:** Configurer un modèle de réflexion séparé:
```python
# Configurer le modèle de réflexion
reflection_lm = dspy.LM(
    model='ollama_chat/qwen2.5:7b',
    api_base='http://localhost:11434',
    temperature=1.0,       # Important : température élevée
    max_tokens=8000        # Important : suffisamment de tokens
)

# Utiliser avec GEPA
optimizer = GEPA(
    metric=my_metric,
    auto='light',
    reflection_lm=reflection_lm
)
```

#### GEPA est trop lent

**Solutions:**
1. Utiliser `auto='light'` au lieu de `medium` ou `heavy`
2. Réduire le nombre d'exemples d'entraînement (min 20, idéal 30-50)
3. Utiliser un modèle de réflexion plus rapide (ex: llama3.1:8b au lieu de qwen2.5:7b)
4. Vérifier que votre machine a suffisamment de RAM

## Conseils GEPA pour une optimisation réussie

### Qualité des données

✅ **Bonnes pratiques:**
- Minimum 15-20 exemples d'entraînement (idéalement 30-50)
- Exemples **diversifiés** couvrant tous les cas d'usage
- Labels **corrects** et **cohérents**
- Données de validation **séparées** du trainset

❌ **À éviter:**
- Trop peu d'exemples (<10)
- Exemples répétitifs ou très similaires
- Labels incohérents ou ambigus
- Utiliser le même dataset pour l'entraînement et la validation

### Configuration du reflection_lm

**Paramètres essentiels:**
- **temperature = 1.0** : Température élevée pour encourager la créativité
- **max_tokens = 8000** : Suffisamment de tokens pour l'analyse et les suggestions

**Choix du modèle:**
- `qwen2.5:7b` - Excellent pour la réflexion, recommandé
- `llama3.1:8b` - Bon équilibre qualité/vitesse
- `mistral:7b` - Plus rapide mais moins performant

### Choisir le bon niveau d'optimisation

| Niveau | Quand l'utiliser |
|--------|------------------|
| **light** | - Première utilisation de GEPA<br>- Tests et prototypage<br>- Budget temps limité (5-10 min)<br>- Validation du concept |
| **medium** | - Production légère<br>- Bons résultats avec light<br>- Budget modéré (10-20 min)<br>- Tâche importante |
| **heavy** | - Performance maximale requise<br>- Tâche critique<br>- Budget temps généreux (20-40 min)<br>- Après succès avec medium |

### Interpréter les résultats

**Score après GEPA:**
- **<60%** : Problème probable avec les données ou la tâche
- **60-75%** : Résultats OK, peut être amélioré
- **75-85%** : Bons résultats
- **>85%** : Excellents résultats

**Si les résultats ne sont pas satisfaisants:**
1. Vérifier la qualité des données (labels corrects ?)
2. Vérifier que la métrique mesure bien ce que vous voulez
3. Essayer un niveau plus élevé (`medium` → `heavy`)
4. Ajouter plus d'exemples d'entraînement
5. Essayer un meilleur modèle de réflexion

## Checklist de mise en production

Avant de déployer votre application DSPy en production, voici une checklist complète :

### Données et métriques

- [ ] **Données d'entraînement de qualité**
  - Au moins 30-50 exemples diversifiés
  - Labels vérifiés et cohérents
  - Couverture de tous les cas d'usage importants

- [ ] **Données de validation séparées**
  - 15-20% des données totales
  - Jamais utilisées pour l'entraînement
  - Représentatives de la production

- [ ] **Métriques bien définies**
  - Alignées avec les objectifs business
  - Testées sur des cas limites
  - Documentées clairement

### Module et optimisation

- [ ] **Module de base testé**
  - Fonctionne sur tous les cas d'usage
  - Performance de base mesurée
  - Code propre et documenté

- [ ] **Optimisation effectuée**
  - Au moins BootstrapFewShot testé
  - MIPRO ou GEPA si performance critique
  - Gain de performance mesuré et documenté

- [ ] **Module optimisé sauvegardé**
  - Utiliser `module.save()` pour sauvegarder
  - Version control (git) des prompts optimisés
  - Documentation des paramètres d'optimisation

### Robustesse et fiabilité

- [ ] **Validation des sorties**
  - Pattern de validation implémenté
  - Valeurs par défaut définies
  - Gestion des cas limites

- [ ] **Gestion des erreurs**
  - Pattern de retry implémenté
  - Pattern de fallback si nécessaire
  - Logging des erreurs

- [ ] **Tests complets**
  - Tests unitaires pour chaque module
  - Tests d'intégration
  - Tests sur des cas limites

### Performance et monitoring

- [ ] **Performance acceptable**
  - Latence mesurée (<2s idéalement)
  - Throughput suffisant pour la charge attendue
  - Coûts estimés et acceptables

- [ ] **Monitoring mis en place**
  - Tracking des métriques de performance
  - Alertes sur les erreurs
  - Logs structurés et accessibles

- [ ] **Plan de rollback**
  - Possibilité de revenir à la version précédente
  - Tests de rollback effectués

### Sécurité et confidentialité

- [ ] **Données sensibles protégées**
  - Pas de données personnelles dans les logs
  - Chiffrement si nécessaire
  - Conformité RGPD/autres réglementations

- [ ] **API keys sécurisées**
  - Stockage sécurisé (variables d'environnement)
  - Rotation régulière si possible
  - Pas de keys dans le code source

### Documentation

- [ ] **Documentation technique**
  - Architecture documentée
  - Instructions de déploiement
  - Guide de troubleshooting

- [ ] **Documentation utilisateur**
  - Cas d'usage supportés
  - Limites connues
  - Exemples d'utilisation

## Adapter ce tutoriel à votre cas d'usage

Ce tutoriel utilise la classification de tickets IT comme exemple, mais DSPy peut être appliqué à de nombreux cas d'usage. Voici comment adapter ce code à votre problème.

### Définir votre tâche

**Questions à se poser:**
1. Quelle est mon entrée ? (texte, image, tableau, etc.)
2. Quelle est ma sortie attendue ? (classification, extraction, génération, etc.)
3. Ai-je des exemples d'entrée/sortie ?
4. Comment mesurer si la sortie est correcte ?

### Étapes d'adaptation

**1. Créer votre signature**
```python
class MaSignature(dspy.Signature):
    """Description claire de ma tâche."""

    mon_entree = dspy.InputField(desc="Description de l'entrée")
    ma_sortie = dspy.OutputField(desc="Description de la sortie attendue")
```

**2. Préparer vos données**
```python
# Format : liste de dictionnaires
mes_donnees = [
    {'mon_entree': "exemple 1", 'ma_sortie': "résultat 1"},
    {'mon_entree': "exemple 2", 'ma_sortie': "résultat 2"},
    # ...
]

# Séparer train/val (80/20)
split_idx = int(len(mes_donnees) * 0.8)
train = mes_donnees[:split_idx]
val = mes_donnees[split_idx:]
```

**3. Créer votre module**
```python
class MonModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(MaSignature)

    def forward(self, mon_entree):
        return self.predictor(mon_entree=mon_entree)
```

**4. Définir votre métrique**
```python
def ma_metrique(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """Comparer la prédiction avec la ground truth."""
    # Votre logique de comparaison
    correct = prediction.ma_sortie == example['ma_sortie']
    return 1.0 if correct else 0.0
```

**5. Évaluer et optimiser**
```python
# Évaluer
module = MonModule()
score = evaluate_module(module, val, ma_metrique)
print(f"Score baseline: {score:.1%}")

# Optimiser avec BootstrapFewShot
from dspy.teleprompt import BootstrapFewShot
optimizer = BootstrapFewShot(metric=ma_metrique)
optimized = optimizer.compile(student=module, trainset=train)

# Réévaluer
score_opt = evaluate_module(optimized, val, ma_metrique)
print(f"Score optimisé: {score_opt:.1%}")
```

### Exemples de cas d'usage

| Cas d'usage | Signature | Type de module recommandé |
|-------------|-----------|---------------------------|
| **Classification de texte** | text → category | ChainOfThought |
| **Extraction d'informations** | document → entities | ChainOfThought |
| **Génération de résumé** | long_text → summary | Predict ou ChainOfThought |
| **Question-Réponse** | question, context → answer | ChainOfThought |
| **Traduction** | text, target_lang → translation | Predict |
| **Analyse de sentiment** | review → sentiment, score | ChainOfThought |
| **Génération de code** | description → code | ProgramOfThought |
| **RAG (Retrieval-Augmented Generation)** | query → answer | ReAct avec Retrieve |

## Le pouvoir de DSPy

DSPy représente un **changement de paradigme** dans le développement avec les LLMs :

**Avant DSPy:**
- ❌ Prompts écrits manuellement et fragiles
- ❌ Difficile de maintenir la cohérence
- ❌ Optimisation par essai-erreur
- ❌ Code spécifique à chaque modèle

**Avec DSPy:**
- ✅ Prompts optimisés automatiquement
- ✅ Abstraction propre et maintenable
- ✅ Optimisation algorithmique (GEPA, MIPRO)
- ✅ Indépendance du fournisseur LLM

### Principes clés à retenir

1. **Commencez simple** : Signature basique → Module Predict → Métrique simple
2. **Itérez rapidement** : Testez, mesurez, optimisez, répétez
3. **Utilisez les optimiseurs** : BootstrapFewShot → MIPRO → GEPA
4. **Pensez production** : Validation, retry, fallback, monitoring
5. **Documentez tout** : Code, décisions, résultats, problèmes rencontrés

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
