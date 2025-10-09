# 🔄 Notes sur les versions de GEPA

## ⚠️ Changements d'API entre versions

L'API de GEPA a évolué entre les versions. Si vous rencontrez des erreurs comme :
```
TypeError: GEPA.__init__() got an unexpected keyword argument 'breadth'
AssertionError: GEPA requires a reflection language model to be provided
```

Cela signifie que vous utilisez DSPy 3.0+ avec l'ancienne API.

## ✅ Paramètres actuels (DSPy 3.0+)

### Configuration de base
```python
from dspy.teleprompt import GEPA
import dspy

# REQUIS : Un modèle de réflexion pour analyser les erreurs
reflection_lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',  # ou autre modèle
    api_base='http://localhost:11434',
    temperature=1.0,
    max_tokens=8000
)

optimizer = GEPA(
    metric=your_metric_function,
    auto='light',  # 'light', 'medium', ou 'heavy'
    reflection_lm=reflection_lm,  # REQUIS
)
```

### Configuration avancée
```python
optimizer = GEPA(
    metric=your_metric_function,
    auto='heavy',  # Optimisation approfondie
    reflection_lm=reflection_lm,  # REQUIS
    max_full_evals=30,  # Alternative à 'auto'
)
```

## ❌ Anciens paramètres (obsolètes dans DSPy 3.0+)

Ces paramètres ne fonctionnent **plus** dans DSPy 3.0+ :
- `breadth` → Remplacé par `auto` ou `max_full_evals`
- `depth` → Remplacé par `auto` ou `max_full_evals`
- `max_bootstrapped_demos` → Remplacé par le système interne
- `max_labeled_demos` → Remplacé par le système interne
- `num_candidates` → Remplacé par `auto`

## 📊 Guide de migration

Si vous avez du code ancien :

### Avant (DSPy < 3.0)
```python
optimizer = GEPA(
    metric=my_metric,
    breadth=5,
    depth=2
)
```

### Après (DSPy 3.0+)
```python
import dspy

# NOUVEAU : reflection_lm est maintenant REQUIS
reflection_lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',
    api_base='http://localhost:11434',
    temperature=1.0,
    max_tokens=8000
)

optimizer = GEPA(
    metric=my_metric,
    auto='light',  # Remplace breadth/depth
    reflection_lm=reflection_lm,  # REQUIS
)
```

## 🎯 Explication des paramètres actuels (DSPy 3.0+)

### `auto` (budget automatique)
- **Défaut** : None (un de `auto`, `max_full_evals`, ou `max_metric_calls` est requis)
- **Description** : Preset de budget d'optimisation
- **Options** :
  - `'light'` : Optimisation rapide (~5-10 min avec Ollama)
  - `'medium'` : Optimisation équilibrée (~10-20 min)
  - `'heavy'` : Optimisation approfondie (~20-40 min)
- **Recommandation** :
  - Test rapide : `'light'`
  - Production : `'medium'` ou `'heavy'`

### `reflection_lm` (modèle de réflexion)
- **Requis** : **OUI** (obligatoire dans DSPy 3.0+)
- **Type** : `dspy.LM`
- **Description** : Modèle utilisé pour analyser les erreurs et proposer des améliorations
- **Recommandation** : Utiliser un modèle puissant avec `temperature=1.0`
- **Exemple** :
```python
reflection_lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',
    api_base='http://localhost:11434',
    temperature=1.0,
    max_tokens=8000
)
```

### `metric` (fonction d'évaluation)
- **Requis** : Oui
- **Type** : Fonction avec signature `(example, prediction, trace=None, pred_name=None, pred_trace=None)`
- **Retour** : Float entre 0.0 et 1.0 OU dict `{'score': float, 'feedback': str}`
- **Exemple** :
```python
def my_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    if prediction.output == example.output:
        return 1.0
    else:
        return 0.0
```

### `max_full_evals` (alternative à `auto`)
- **Défaut** : None
- **Description** : Nombre maximum d'évaluations complètes sur train+val
- **Utilisation** : À la place de `auto` pour un contrôle précis du budget

## ⏱️ Temps d'exécution estimés

Avec Ollama local (llama3.1:8b) et DSPy 3.0+ :

| Configuration | Temps estimé | Usage |
|--------------|--------------|-------|
| auto='light' | ~5-10 min | Test rapide |
| auto='medium' | ~10-20 min | Standard (recommandé) |
| auto='heavy' | ~20-40 min | Optimisation maximale |
| max_full_evals=10 | ~10-15 min | Contrôle précis |

## 🔍 Vérifier votre version de DSPy

```python
import dspy
print(f"Version DSPy: {dspy.__version__}")
```

Si version < 2.5 :
```bash
pip install --upgrade dspy-ai
```

## 💡 Conseils pratiques

### Pour économiser du temps
1. Commencez avec `breadth=3, depth=1` pour valider
2. Puis augmentez progressivement
3. Utilisez un petit subset de données pour tester

### Pour maximiser les résultats
1. Utilisez `breadth=10, depth=3`
2. Assurez-vous d'avoir au moins 15 exemples d'entraînement
3. Testez votre métrique avant d'optimiser

### Pour déboguer
```python
# Activer les logs détaillés
import logging
logging.basicConfig(level=logging.INFO)

# Puis lancer GEPA
optimizer = GEPA(metric=my_metric, breadth=3, depth=1)
```

## 🐛 Problèmes courants

### Erreur : "AssertionError: GEPA requires a reflection language model"
**Cause** : Le paramètre `reflection_lm` n'est pas fourni

**Solution** :
```python
reflection_lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',
    api_base='http://localhost:11434',
    temperature=1.0,
    max_tokens=8000
)
optimizer = GEPA(metric=my_metric, auto='light', reflection_lm=reflection_lm)
```

### Erreur : "unexpected keyword argument 'breadth'"
**Cause** : Utilisation de l'ancienne API avec DSPy 3.0+

**Solution** : Remplacer `breadth` et `depth` par `auto='light'` (voir ci-dessus)

### Optimisation trop lente
**Cause** : `breadth` ou `depth` trop élevés

**Solution** :
```python
# Réduire les paramètres
optimizer = GEPA(metric=my_metric, breadth=3, depth=1)
```

### Aucune amélioration après GEPA
**Causes possibles** :
1. Pas assez de données d'entraînement (< 10 exemples)
2. Métrique mal définie
3. Tâche trop simple (déjà à 90%+)

**Solutions** :
1. Ajouter plus d'exemples dans `trainset`
2. Vérifier que la métrique retourne bien 0-1
3. Augmenter `breadth` et `depth`

## 📚 Ressources

- **Documentation officielle** : https://dspy-docs.vercel.app/
- **Paper GEPA** : https://arxiv.org/abs/2507.19457
- **GitHub DSPy** : https://github.com/stanfordnlp/dspy
- **GitHub GEPA** : https://github.com/gepa-ai/gepa

## ✅ Validation de votre configuration

Testez ce code pour vérifier que GEPA fonctionne avec DSPy 3.0+ :

```python
import dspy
from dspy.teleprompt import GEPA

# Configuration simple
lm = dspy.LM('ollama_chat/llama3.1:8b', api_base='http://localhost:11434')
dspy.configure(lm=lm)

# IMPORTANT : Créer le modèle de réflexion (REQUIS)
reflection_lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',
    api_base='http://localhost:11434',
    temperature=1.0,
    max_tokens=8000
)

# Données minimales
train_examples = [
    dspy.Example(input="Hello", output="Bonjour").with_inputs('input'),
    dspy.Example(input="Goodbye", output="Au revoir").with_inputs('input'),
]

# Métrique simple (NOUVELLE SIGNATURE avec pred_name et pred_trace)
def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return 1.0 if prediction.output == example.output else 0.0

# Test GEPA
class SimpleTranslator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.translate = dspy.Predict("input -> output")

    def forward(self, input):
        return self.translate(input=input)

translator = SimpleTranslator()

try:
    optimizer = GEPA(
        metric=metric,
        auto='light',
        reflection_lm=reflection_lm  # REQUIS
    )
    optimized = optimizer.compile(translator, trainset=train_examples)
    print("✅ GEPA fonctionne correctement!")
except Exception as e:
    print(f"❌ Erreur : {e}")
```

Si ce test fonctionne, votre configuration est correcte ! 🎉
