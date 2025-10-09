"""
Exemple d'utilisation de DSPy + GEPA + Ollama
Cas d'usage : Classification automatique de tickets IT

Prérequis:
1. Installer Ollama: brew install ollama
2. Télécharger un modèle: ollama pull llama3.1:8b
3. Installer les dépendances: pip install dspy-ai gepa

Usage:
python main.py
"""

import dspy
from data import trainset, valset, CATEGORIES, PRIORITIES

# =============================================================================
# ÉTAPE 1: Configuration du modèle (Ollama local - 100% gratuit)
# =============================================================================

print("🚀 Configuration de DSPy avec Ollama...")

# Configurer DSPy pour utiliser Ollama
lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',  # ou mistral:7b, qwen2.5:7b, etc.
    api_base='http://localhost:11434',
    temperature=0.3  # Basse température pour plus de cohérence
)

dspy.configure(lm=lm)

print("✅ Ollama configuré avec succès!\n")

# =============================================================================
# ÉTAPE 2: Définir la signature DSPy (entrées/sorties)
# =============================================================================

class TicketClassifier(dspy.Signature):
    """Classifier un ticket de support IT selon sa catégorie et sa priorité."""
    
    ticket = dspy.InputField(desc="Description du ticket de support IT")
    category = dspy.OutputField(desc=f"Catégorie du ticket parmi: {', '.join(CATEGORIES)}")
    priority = dspy.OutputField(desc=f"Priorité du ticket parmi: {', '.join(PRIORITIES)}")

# =============================================================================
# ÉTAPE 3: Créer un module DSPy
# =============================================================================

class TicketClassificationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought ajoute du raisonnement avant la classification
        self.classifier = dspy.ChainOfThought(TicketClassifier)
    
    def forward(self, ticket):
        result = self.classifier(ticket=ticket)
        # Récupérer le raisonnement s'il existe (nom peut varier selon version DSPy)
        reasoning = getattr(result, 'rationale', None) or getattr(result, 'reasoning', '') or ''
        return dspy.Prediction(
            category=result.category,
            priority=result.priority,
            reasoning=reasoning
        )

# =============================================================================
# ÉTAPE 4: Tester le modèle AVANT optimisation
# =============================================================================

print("📊 Test du modèle AVANT optimisation GEPA...\n")

# Créer une instance du module
classifier = TicketClassificationModule()

# Tester sur quelques exemples
test_examples = valset[:3]

for i, example in enumerate(test_examples, 1):
    print(f"--- Exemple {i} ---")
    print(f"Ticket: {example['ticket']}")
    
    # Prédiction
    prediction = classifier(ticket=example['ticket'])
    
    print(f"✨ Prédiction: {prediction.category} | {prediction.priority}")
    print(f"✅ Attendu: {example['category']} | {example['priority']}")
    if hasattr(prediction, 'reasoning') and prediction.reasoning:
        print(f"💭 Raisonnement: {prediction.reasoning}")
    print()

# =============================================================================
# ÉTAPE 5: Définir une métrique d'évaluation
# =============================================================================

def evaluate_classifier(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Métrique simple: 1 point si catégorie ET priorité sont correctes
    0.5 point si une seule des deux est correcte
    Compatible avec GEPA DSPy 3.0+
    """
    category_correct = prediction.category.strip().lower() == example['category'].strip().lower()
    priority_correct = prediction.priority.strip().lower() == example['priority'].strip().lower()

    if category_correct and priority_correct:
        return 1.0
    elif category_correct or priority_correct:
        return 0.5
    else:
        return 0.0

# =============================================================================
# ÉTAPE 6: Optimiser avec GEPA (optionnel - décommenter pour l'utiliser)
# =============================================================================

"""
ATTENTION: GEPA nécessite plusieurs appels au LLM pour l'optimisation.
Avec Ollama local, cela peut prendre plusieurs minutes.

Pour activer l'optimisation GEPA, décommentez le code ci-dessous:

print("🔧 Optimisation avec GEPA (cela peut prendre quelques minutes)...\n")

# Préparer les données au format DSPy
train_examples = [
    dspy.Example(ticket=ex['ticket'], category=ex['category'], priority=ex['priority']).with_inputs('ticket')
    for ex in trainset
]

val_examples = [
    dspy.Example(ticket=ex['ticket'], category=ex['category'], priority=ex['priority']).with_inputs('ticket')
    for ex in valset
]

# Optimiser avec GEPA (DSPy 3.0+)
from dspy.teleprompt import GEPA

# GEPA 3.0+ nécessite un modèle de réflexion
reflection_lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',
    api_base='http://localhost:11434',
    temperature=1.0,
    max_tokens=8000
)

optimizer = GEPA(
    metric=evaluate_classifier,
    auto='light',  # Options: 'light', 'medium', 'heavy'
    reflection_lm=reflection_lm
)

optimized_classifier = optimizer.compile(
    classifier,
    trainset=train_examples,
    valset=val_examples
)

print("✅ Optimisation GEPA terminée!\n")

# Tester le modèle optimisé
print("📊 Test du modèle APRÈS optimisation GEPA...\n")

for i, example in enumerate(test_examples, 1):
    print(f"--- Exemple {i} (optimisé) ---")
    print(f"Ticket: {example['ticket']}")
    
    prediction = optimized_classifier(ticket=example['ticket'])
    
    print(f"✨ Prédiction: {prediction.category} | {prediction.priority}")
    print(f"✅ Attendu: {example['category']} | {example['priority']}")
    print()
"""

# =============================================================================
# ÉTAPE 7: Évaluation complète sur l'ensemble de validation
# =============================================================================

print("📈 Évaluation sur l'ensemble de validation complet...\n")

correct_both = 0
correct_one = 0
total = len(valset)

for example in valset:
    prediction = classifier(ticket=example['ticket'])
    score = evaluate_classifier(example, prediction)
    
    if score == 1.0:
        correct_both += 1
    elif score == 0.5:
        correct_one += 1

print(f"Résultats:")
print(f"  • Catégorie ET priorité correctes: {correct_both}/{total} ({correct_both/total*100:.1f}%)")
print(f"  • Une seule correcte: {correct_one}/{total} ({correct_one/total*100:.1f}%)")
print(f"  • Score moyen: {(correct_both + correct_one*0.5)/total*100:.1f}%")

# =============================================================================
# ÉTAPE 8: Exemple d'utilisation interactive
# =============================================================================

print("\n" + "="*70)
print("💡 Testez avec vos propres tickets!")
print("="*70 + "\n")

sample_tickets = [
    "Mon laptop fait un bruit bizarre et chauffe beaucoup. J'ai une démo client cet après-midi.",
    "Je voudrais accès au nouveau logiciel de gestion de projet quand c'est possible.",
    "Toutes les imprimantes du bâtiment sont hors ligne. Les rapports doivent être imprimés maintenant!"
]

for ticket in sample_tickets:
    print(f"📝 Ticket: {ticket}")
    result = classifier(ticket=ticket)
    print(f"   ➜ Catégorie: {result.category}")
    print(f"   ➜ Priorité: {result.priority}")
    print()

print("\n✨ Démo terminée! Pour optimiser avec GEPA, décommentez la section ÉTAPE 6 dans le code.")
