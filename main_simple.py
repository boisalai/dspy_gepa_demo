"""
Version simplifiée et robuste de DSPy + GEPA + Ollama
Cette version évite les problèmes de compatibilité avec les attributs

Usage: python main_simple.py
"""

import dspy
from data import trainset, valset, CATEGORIES, PRIORITIES

# =============================================================================
# Configuration du modèle
# =============================================================================

print("🚀 Configuration de DSPy avec Ollama...")

lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',
    api_base='http://localhost:11434',
    temperature=0.3
)

dspy.configure(lm=lm)
print("✅ Ollama configuré avec succès!\n")

# =============================================================================
# Définir la signature (entrées/sorties)
# =============================================================================

class TicketSignature(dspy.Signature):
    """Classifier un ticket de support IT selon sa catégorie et sa priorité."""
    ticket = dspy.InputField(desc="Description du ticket de support IT")
    category = dspy.OutputField(desc=f"Catégorie parmi: {', '.join(CATEGORIES)}")
    priority = dspy.OutputField(desc=f"Priorité parmi: {', '.join(PRIORITIES)}")

# =============================================================================
# Créer un module simple avec Predict (plus robuste)
# =============================================================================

class TicketClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        # Utilise Predict au lieu de ChainOfThought pour plus de compatibilité
        self.classifier = dspy.Predict(TicketSignature)
    
    def forward(self, ticket):
        # Appel simple et robuste
        return self.classifier(ticket=ticket)

# =============================================================================
# Tester le modèle
# =============================================================================

print("📊 Test du modèle sur quelques exemples...\n")

classifier = TicketClassifier()

# Tester sur les 3 premiers exemples
for i, example in enumerate(valset[:3], 1):
    print(f"--- Exemple {i} ---")
    print(f"📝 Ticket: {example['ticket']}")
    
    try:
        prediction = classifier(ticket=example['ticket'])
        
        print(f"✨ Prédiction: {prediction.category} | {prediction.priority}")
        print(f"✅ Attendu: {example['category']} | {example['priority']}")
        
        # Vérifier si correct
        category_ok = prediction.category.strip().lower() == example['category'].strip().lower()
        priority_ok = prediction.priority.strip().lower() == example['priority'].strip().lower()
        
        if category_ok and priority_ok:
            print("🎯 Correct!")
        elif category_ok or priority_ok:
            print("🟡 Partiellement correct")
        else:
            print("❌ Incorrect")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    print()

# =============================================================================
# Évaluation complète
# =============================================================================

print("="*70)
print("📈 Évaluation sur l'ensemble de validation complet")
print("="*70 + "\n")

correct_both = 0
correct_category = 0
correct_priority = 0
total = len(valset)

for example in valset:
    try:
        prediction = classifier(ticket=example['ticket'])
        
        category_match = prediction.category.strip().lower() == example['category'].strip().lower()
        priority_match = prediction.priority.strip().lower() == example['priority'].strip().lower()
        
        if category_match and priority_match:
            correct_both += 1
        if category_match:
            correct_category += 1
        if priority_match:
            correct_priority += 1
            
    except Exception as e:
        print(f"⚠️  Erreur sur un exemple: {e}")
        continue

print(f"Résultats sur {total} tickets:")
print(f"  • Catégorie correcte: {correct_category}/{total} ({correct_category/total*100:.1f}%)")
print(f"  • Priorité correcte: {correct_priority}/{total} ({correct_priority/total*100:.1f}%)")
print(f"  • Les deux correctes: {correct_both}/{total} ({correct_both/total*100:.1f}%)")
print()

# =============================================================================
# Exemples interactifs
# =============================================================================

print("="*70)
print("💡 Exemples de classification en temps réel")
print("="*70 + "\n")

test_tickets = [
    "Mon ordinateur portable ne s'allume plus. J'ai une présentation client dans 1 heure!",
    "Je voudrais installer Zoom quand vous aurez le temps.",
    "Le serveur principal est en panne. Toute la production est arrêtée!",
    "Ma souris sans fil ne marche plus très bien. Les piles sont peut-être faibles.",
]

for i, ticket in enumerate(test_tickets, 1):
    print(f"{i}. 📝 {ticket}")
    try:
        result = classifier(ticket=ticket)
        print(f"   ➜ Catégorie: {result.category}")
        print(f"   ➜ Priorité: {result.priority}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    print()

# =============================================================================
# Guide pour aller plus loin
# =============================================================================

print("="*70)
print("✨ Prochaines étapes")
print("="*70)
print("""
1. 🔄 Tester avec d'autres modèles :
   - ollama pull mistral:7b
   - Modifier model='ollama_chat/mistral:7b' dans le code

2. 📚 Ajouter plus de données :
   - Éditer data.py pour ajouter vos propres exemples
   - Plus de données = meilleure précision

3. 🧬 Optimiser avec GEPA :
   - Voir gepa_guide.py pour l'optimisation automatique
   - Amélioration typique : +10-20%

4. 🎯 Adapter à votre cas :
   - Modifier TicketSignature pour vos besoins
   - Changer les catégories et priorités
   - Ajuster la métrique d'évaluation

📖 Documentation complète : Voir README.md
""")

print("✅ Démo terminée avec succès!")
