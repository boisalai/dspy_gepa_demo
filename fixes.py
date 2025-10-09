"""
CORRECTIFS pour problèmes courants
===================================

Si vous rencontrez l'erreur : AttributeError: 'Prediction' object has no attribute 'rationale'
→ Les fichiers ont été corrigés automatiquement, mais voici des solutions alternatives si besoin.
"""

# =============================================================================
# Solution 1 : Module simplifié sans accès au raisonnement
# =============================================================================

import dspy

class SimpleTicketClassifier(dspy.Module):
    """Version simplifiée qui ne tente pas d'accéder au raisonnement"""
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought("ticket -> category, priority")
    
    def forward(self, ticket):
        result = self.classifier(ticket=ticket)
        # Retourner seulement category et priority
        return dspy.Prediction(
            category=result.category,
            priority=result.priority
        )

# =============================================================================
# Solution 2 : Utiliser Predict au lieu de ChainOfThought
# =============================================================================

class PredictOnlyClassifier(dspy.Module):
    """Utilise Predict qui ne génère pas de raisonnement explicite"""
    def __init__(self):
        super().__init__()
        from data import CATEGORIES, PRIORITIES
        
        class TicketSig(dspy.Signature):
            """Classifier un ticket IT"""
            ticket = dspy.InputField()
            category = dspy.OutputField(desc=f"Une catégorie parmi: {', '.join(CATEGORIES)}")
            priority = dspy.OutputField(desc=f"Une priorité parmi: {', '.join(PRIORITIES)}")
        
        self.classifier = dspy.Predict(TicketSig)
    
    def forward(self, ticket):
        return self.classifier(ticket=ticket)

# =============================================================================
# Solution 3 : Accès robuste aux attributs
# =============================================================================

def get_safe_reasoning(prediction):
    """Récupère le raisonnement de manière robuste"""
    # Essayer différents noms d'attributs possibles
    for attr in ['rationale', 'reasoning', 'chain_of_thought', 'cot']:
        if hasattr(prediction, attr):
            value = getattr(prediction, attr)
            if value:
                return str(value)
    return "Pas de raisonnement disponible"

# =============================================================================
# Comment utiliser ces solutions dans main.py
# =============================================================================

"""
Remplacez dans main.py :

# Au lieu de :
class TicketClassificationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(TicketClassifier)
    
    def forward(self, ticket):
        result = self.classifier(ticket=ticket)
        return dspy.Prediction(
            category=result.category,
            priority=result.priority,
            reasoning=result.rationale  # ← PROBLÈME ICI
        )

# Utilisez plutôt :
from fixes import SimpleTicketClassifier

classifier = SimpleTicketClassifier()

# OU pour garder le raisonnement :
class TicketClassificationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(TicketClassifier)
    
    def forward(self, ticket):
        result = self.classifier(ticket=ticket)
        reasoning = getattr(result, 'rationale', '') or getattr(result, 'reasoning', '')
        return dspy.Prediction(
            category=result.category,
            priority=result.priority,
            reasoning=reasoning  # ← SOLUTION ROBUSTE
        )
"""

# =============================================================================
# Test des solutions
# =============================================================================

if __name__ == "__main__":
    from data import valset
    
    print("🔧 Test des solutions alternatives\n")
    
    # Configuration
    lm = dspy.LM(
        model='ollama_chat/llama3.1:8b',
        api_base='http://localhost:11434',
        temperature=0.3
    )
    dspy.configure(lm=lm)
    
    # Test solution 1
    print("1️⃣  Test SimpleTicketClassifier...")
    try:
        classifier = SimpleTicketClassifier()
        result = classifier(ticket=valset[0]['ticket'])
        print(f"   ✅ Fonctionne : {result.category} | {result.priority}\n")
    except Exception as e:
        print(f"   ❌ Erreur : {e}\n")
    
    # Test solution 2
    print("2️⃣  Test PredictOnlyClassifier...")
    try:
        classifier = PredictOnlyClassifier()
        result = classifier(ticket=valset[0]['ticket'])
        print(f"   ✅ Fonctionne : {result.category} | {result.priority}\n")
    except Exception as e:
        print(f"   ❌ Erreur : {e}\n")
    
    print("✨ Si l'un des tests fonctionne, utilisez cette solution dans main.py")
