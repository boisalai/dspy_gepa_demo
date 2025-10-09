# ✅ GEPA mis à jour pour DSPy 3.0+

## 🎉 Problème résolu

Le code a été mis à jour pour fonctionner avec **DSPy 3.0+**. Vous pouvez maintenant utiliser GEPA directement :

```bash
uv run gepa_guide.py
```

## 📊 Changements d'API (DSPy 3.0+)

L'API GEPA a évolué dans DSPy 3.0+. Voici les changements principaux :

### Avant (DSPy < 3.0)
```python
optimizer = GEPA(
    metric=my_metric,
    breadth=5,
    depth=2
)
```

### Maintenant (DSPy 3.0+)
```python
import dspy
from dspy.teleprompt import GEPA

# NOUVEAU : Un modèle de réflexion est maintenant REQUIS
reflection_lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',
    api_base='http://localhost:11434',
    temperature=1.0,
    max_tokens=8000
)

optimizer = GEPA(
    metric=my_metric,
    auto='light',  # Options: 'light', 'medium', 'heavy'
    reflection_lm=reflection_lm  # REQUIS
)
```

### Métriques mises à jour

Les métriques doivent maintenant accepter des paramètres supplémentaires :

```python
# Ancienne signature
def my_metric(example, prediction, trace=None):
    return 1.0 if correct else 0.0

# Nouvelle signature (DSPy 3.0+)
def my_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return 1.0 if correct else 0.0
```

## 🚀 Utilisation recommandée

```bash
# Lancer GEPA avec DSPy 3.0+
uv run gepa_guide.py

# Choisir une option :
# 1. Optimisation basique (rapide, ~5-10 min)
# 2. Optimisation avancée (approfondie, ~10-20 min)
```

## 🔍 Évolution de l'API GEPA

| Version DSPy | API GEPA | Status |
|--------------|----------|--------|
| < 2.5  | `max_bootstrapped_demos`, `max_labeled_demos` | ❌ Obsolète |
| 2.5-2.9   | `breadth`, `depth` | ❌ Obsolète |
| **3.0+**  | `auto='light'/'medium'/'heavy'`, `reflection_lm` | ✅ **Actuel** |

## 📚 Fichiers du projet

- ✅ **gepa_guide.py** - Guide GEPA pour DSPy 3.0+ (recommandé)
- ✅ **diagnose_gepa.py** - Diagnostic de votre environnement
- ✅ **GEPA_API_CHANGES.md** - Documentation détaillée des changements d'API
- ✅ **main_simple.py** - Alternative sans GEPA (toujours fonctionnel)

## 🎯 Workflow recommandé

```bash
# 1. Vérifier que le reste fonctionne
uv run main_simple.py

# 2. Explorer les exemples avancés
uv run advanced_examples.py

# 3. (Optionnel) Diagnostiquer votre environnement
uv run diagnose_gepa.py

# 4. Optimiser avec GEPA
uv run gepa_guide.py
```

## ✨ Résultat attendu

```
🎓 Guide GEPA : Optimisation automatique de prompts

======================================================================
Choisissez un exemple :
1. Optimisation basique (recommandé pour débuter)
2. Optimisation avancée (avec paramètres personnalisés)
3. Juste afficher les conseils
======================================================================

Votre choix (1-3) : 1

======================================================================
🔧 Optimisation GEPA - Exemple basique
======================================================================

✅ DSPy configuré pour GEPA
📊 Évaluation AVANT optimisation...
   Score: 42.86%

🧬 Lancement de l'optimisation GEPA...
   (Cela peut prendre 5-10 minutes avec Ollama)

[... progression ...]

✅ Optimisation terminée!

📊 Évaluation APRÈS optimisation...
   Score: 57.14%

📈 Amélioration: +33.3%
```

## ❓ Questions fréquentes

**Q : J'ai une ancienne version de DSPy, que faire ?**
R : Mettez à jour avec `pip install --upgrade dspy-ai` pour obtenir DSPy 3.0+

**Q : GEPA ne fonctionne toujours pas, que faire ?**
R : Vérifiez votre version avec `pip show dspy-ai`. Si le problème persiste, utilisez `main_simple.py` qui fonctionne sans GEPA.

**Q : Dois-je absolument utiliser GEPA ?**
R : Non ! GEPA est optionnel. Les scripts principaux (main_simple.py, advanced_examples.py) fonctionnent parfaitement sans GEPA.

**Q : Puis-je utiliser GEPA avec Ollama local ?**
R : Oui ! Tous les exemples utilisent Ollama par défaut. Prévoir 5-10 minutes pour l'optimisation.

## 📞 Support

Si vous rencontrez des problèmes :

1. Vérifiez votre version de DSPy : `pip show dspy-ai`
2. Lancez le diagnostic : `uv run diagnose_gepa.py`
3. Consultez [GEPA_API_CHANGES.md](GEPA_API_CHANGES.md) pour plus de détails

## 📖 Documentation complémentaire

- **[GEPA_API_CHANGES.md](GEPA_API_CHANGES.md)** - Guide de migration détaillé
- **[README.md](README.md)** - Documentation complète du projet
- **[Paper GEPA](https://arxiv.org/abs/2507.19457)** - Paper de recherche original

---

**TL;DR : Le code fonctionne maintenant avec DSPy 3.0+. Utilisez `uv run gepa_guide.py`** ✨
