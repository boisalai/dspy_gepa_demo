# 🔧 Guide de dépannage

## ❌ Erreur : 'Prediction' object has no attribute 'rationale'

### Cause
Cette erreur se produit avec certaines versions de DSPy où l'attribut pour le raisonnement a un nom différent.

### ✅ Solution rapide

Utilisez la version simplifiée du code :

```bash
python main_simple.py
```

Cette version évite complètement le problème et fonctionne avec toutes les versions de DSPy.

### ✅ Solution alternative

Si vous voulez continuer à utiliser `main.py`, le fichier a été corrigé automatiquement. Si vous avez encore des problèmes, voici le correctif manuel :

Dans `main.py`, remplacez :

```python
def forward(self, ticket):
    result = self.classifier(ticket=ticket)
    return dspy.Prediction(
        category=result.category,
        priority=result.priority,
        reasoning=result.rationale  # ← PROBLÈME ICI
    )
```

Par :

```python
def forward(self, ticket):
    result = self.classifier(ticket=ticket)
    # Accès robuste au raisonnement
    reasoning = getattr(result, 'rationale', None) or getattr(result, 'reasoning', '') or ''
    return dspy.Prediction(
        category=result.category,
        priority=result.priority,
        reasoning=reasoning
    )
```

## ❌ Erreur : GEPA got an unexpected keyword argument 'max_bootstrapped_demos'

### Cause
L'API de GEPA a changé dans les versions récentes de DSPy. Les anciens paramètres ne sont plus supportés.

### ✅ Solution

Les fichiers ont été corrigés pour utiliser les nouveaux paramètres. Si vous rencontrez encore cette erreur :

**Anciens paramètres (obsolètes) :**
```python
optimizer = GEPA(
    metric=my_metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=5,
    num_candidates=5
)
```

**Nouveaux paramètres (actuels) :**
```python
optimizer = GEPA(
    metric=my_metric,
    breadth=5,  # Nombre de candidats par itération
    depth=2     # Nombre d'itérations
)
```

**Consultez le guide complet :** `GEPA_API_CHANGES.md`

## ❌ Erreur : Connection refused to localhost:11434

### Cause
Ollama n'est pas en cours d'exécution.

### ✅ Solution

Dans un terminal séparé, lancez :
```bash
ollama serve
```

Puis relancez votre script Python.

## ❌ Erreur : Model 'llama3.1:8b' not found

### Cause
Le modèle n'est pas téléchargé.

### ✅ Solution

```bash
ollama pull llama3.1:8b
```

Attendez que le téléchargement soit terminé (environ 5 minutes selon votre connexion).

## ❌ Erreur : ModuleNotFoundError: No module named 'dspy'

### Cause
DSPy n'est pas installé.

### ✅ Solution

```bash
pip install dspy-ai
# OU
uv pip install dspy-ai
```

## ❌ Performance très lente

### Causes possibles
1. Modèle trop gros pour votre machine
2. Pas de GPU disponible
3. Peu de RAM

### ✅ Solutions

**Option 1 : Utiliser un modèle plus petit**
```bash
ollama pull phi3:3.8b  # Seulement 2.3 GB
```

Puis dans le code :
```python
lm = dspy.LM(
    model='ollama_chat/phi3:3.8b',  # Plus rapide
    api_base='http://localhost:11434'
)
```

**Option 2 : Réduire la température**
```python
lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',
    api_base='http://localhost:11434',
    temperature=0.1  # Plus déterministe = plus rapide
)
```

## ❌ Résultats incohérents ou mauvais

### Causes
1. Pas assez de données d'entraînement
2. Modèle pas assez puissant
3. Besoin d'optimisation

### ✅ Solutions

**Option 1 : Ajouter plus de données**
Éditez `data.py` et ajoutez au moins 20-30 exemples.

**Option 2 : Utiliser un meilleur modèle**
```bash
ollama pull llama3.1:70b  # Nécessite ~40GB RAM
```

**Option 3 : Optimiser avec GEPA**
Exécutez :
```bash
python gepa_guide.py
```

## ❌ Erreur lors de l'optimisation GEPA

### Cause
GEPA nécessite plusieurs appels au LLM et peut prendre du temps.

### ✅ Solutions

**Option 1 : Être patient**
L'optimisation GEPA avec Ollama local prend 5-10 minutes. C'est normal.

**Option 2 : Réduire le budget**
Dans le code GEPA :
```python
optimizer = GEPA(
    metric=your_metric,
    num_candidates=3,  # Au lieu de 10
    max_rounds=1       # Au lieu de 3
)
```

**Option 3 : Utiliser un service cloud**
Si vous avez une clé API OpenAI :
```python
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

L'optimisation sera beaucoup plus rapide.

## ❌ ImportError: cannot import name 'GEPA'

### Cause
Version de DSPy trop ancienne.

### ✅ Solution

```bash
pip install --upgrade dspy-ai
```

## 🆘 Autres problèmes

### Vérifier votre installation

Exécutez ce script de diagnostic :

```python
import sys
print(f"Python version: {sys.version}")

try:
    import dspy
    print(f"DSPy version: {dspy.__version__}")
except:
    print("❌ DSPy non installé")

try:
    import requests
    r = requests.get('http://localhost:11434/api/tags')
    if r.status_code == 200:
        print("✅ Ollama fonctionne")
        models = r.json()
        print(f"Modèles disponibles: {[m['name'] for m in models.get('models', [])]}")
    else:
        print("❌ Ollama ne répond pas correctement")
except:
    print("❌ Impossible de se connecter à Ollama")
```

### Réinitialisation complète

Si rien ne fonctionne :

```bash
# 1. Désinstaller DSPy
pip uninstall dspy-ai

# 2. Réinstaller DSPy
pip install dspy-ai

# 3. Redémarrer Ollama
pkill -f ollama
ollama serve &

# 4. Retélécharger un modèle
ollama pull llama3.1:8b

# 5. Tester avec la version simple
python main_simple.py
```

## 📚 Ressources d'aide

- **DSPy GitHub Issues** : https://github.com/stanfordnlp/dspy/issues
- **DSPy Discord** : https://discord.gg/VzS6RHHK6F
- **Ollama GitHub** : https://github.com/ollama/ollama
- **Documentation DSPy** : https://dspy-docs.vercel.app/

## 💡 Astuces

### Pour déboguer

Activez les logs détaillés :
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Pour tester la connexion Ollama

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Why is the sky blue?"
}'
```

Si ça fonctionne, Ollama est OK. Le problème est dans le code Python.

---

**Besoin d'aide supplémentaire ?**
Consultez le fichier `fixes.py` pour des solutions alternatives testées.
