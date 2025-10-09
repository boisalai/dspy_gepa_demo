"""
Exemples avancés : Changer facilement entre différents LLMs
Démontre la flexibilité de DSPy pour switcher entre fournisseurs
"""

import dspy
from data import trainset, valset

# =============================================================================
# Configuration 1: Ollama local (GRATUIT)
# =============================================================================

def configure_ollama(model_name="llama3.1:8b"):
    """Configure DSPy pour utiliser Ollama"""
    lm = dspy.LM(
        model=f'ollama_chat/{model_name}',
        api_base='http://localhost:11434',
        temperature=0.3
    )
    dspy.configure(lm=lm)
    print(f"✅ Configuré avec Ollama: {model_name}")

# =============================================================================
# Configuration 2: OpenAI (PAYANT)
# =============================================================================

def configure_openai(model_name="gpt-4o-mini"):
    """
    Configure DSPy pour utiliser OpenAI
    Nécessite: export OPENAI_API_KEY="sk-..."
    """
    lm = dspy.LM(
        model=f'openai/{model_name}',
        temperature=0.3
    )
    dspy.configure(lm=lm)
    print(f"✅ Configuré avec OpenAI: {model_name}")

# =============================================================================
# Configuration 3: Anthropic Claude (PAYANT)
# =============================================================================

def configure_anthropic(model_name="claude-3-5-sonnet-20241022"):
    """
    Configure DSPy pour utiliser Anthropic Claude
    Nécessite: export ANTHROPIC_API_KEY="sk-ant-..."
    """
    lm = dspy.LM(
        model=f'anthropic/{model_name}',
        temperature=0.3
    )
    dspy.configure(lm=lm)
    print(f"✅ Configuré avec Anthropic: {model_name}")

# =============================================================================
# Configuration 4: Plusieurs modèles dans le même workflow
# =============================================================================

def configure_mixed_models():
    """
    Utiliser différents modèles pour différentes tâches
    Ex: Un gros modèle pour la réflexion, un petit pour l'exécution
    """
    # Modèle principal pour les tâches courantes (rapide et gratuit)
    main_lm = dspy.LM(
        model='ollama_chat/llama3.1:8b',
        api_base='http://localhost:11434'
    )
    
    # Modèle avancé pour les tâches complexes (optionnel, si disponible)
    # advanced_lm = dspy.LM(model='ollama_chat/llama3.1:70b', api_base='http://localhost:11434')
    
    dspy.configure(lm=main_lm)
    print("✅ Configuration multi-modèles activée")
    
    return main_lm  # , advanced_lm

# =============================================================================
# Exemple de workflow qui fonctionne avec N'IMPORTE QUEL LLM
# =============================================================================

class UniversalClassifier(dspy.Module):
    """
    Ce classifier fonctionne avec n'importe quel LLM configuré!
    Changez juste la configuration avant de l'utiliser.
    """
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought("ticket -> category, priority")
    
    def forward(self, ticket):
        return self.classifier(ticket=ticket)

# =============================================================================
# Démonstration du changement de modèle à la volée
# =============================================================================

def demo_model_switching():
    """Démontre comment changer de modèle facilement"""
    
    classifier = UniversalClassifier()
    test_ticket = "Mon ordinateur ne démarre plus et j'ai une réunion importante dans 1 heure."
    
    print("\n" + "="*70)
    print("Démonstration : Même code, différents modèles")
    print("="*70 + "\n")
    
    # Test avec Llama 3.1
    print("🦙 Test avec Llama 3.1 (Ollama local):")
    configure_ollama("llama3.1:8b")
    result = classifier(ticket=test_ticket)
    print(f"   Catégorie: {result.category}, Priorité: {result.priority}\n")
    
    # Test avec Mistral (si disponible)
    print("🌪️  Test avec Mistral (Ollama local):")
    try:
        configure_ollama("mistral:7b")
        result = classifier(ticket=test_ticket)
        print(f"   Catégorie: {result.category}, Priorité: {result.priority}\n")
    except Exception as e:
        print(f"   ⚠️  Mistral non disponible (faites: ollama pull mistral:7b)\n")
    
    # Test avec OpenAI (si clé API disponible)
    print("🤖 Test avec OpenAI GPT-4o-mini (si clé API configurée):")
    try:
        configure_openai("gpt-4o-mini")
        result = classifier(ticket=test_ticket)
        print(f"   Catégorie: {result.category}, Priorité: {result.priority}\n")
    except Exception as e:
        print(f"   ⚠️  OpenAI non configuré (export OPENAI_API_KEY nécessaire)\n")

# =============================================================================
# Liste des modèles Ollama populaires (gratuits)
# =============================================================================

RECOMMENDED_OLLAMA_MODELS = {
    "llama3.1:8b": "Meta Llama 3.1 - Excellent équilibre (4.7 GB)",
    "llama3.1:70b": "Meta Llama 3.1 - Très performant (40 GB, nécessite GPU)",
    "mistral:7b": "Mistral - Rapide et efficace (4.1 GB)",
    "qwen2.5:7b": "Qwen 2.5 - Très bon raisonnement (4.7 GB)",
    "deepseek-r1:7b": "DeepSeek R1 - Spécialisé raisonnement (4.7 GB)",
    "codellama:7b": "Code Llama - Pour code et technique (3.8 GB)",
    "phi3:3.8b": "Microsoft Phi-3 - Petit et rapide (2.3 GB)",
}

def print_available_models():
    """Affiche les modèles recommandés"""
    print("\n📦 Modèles Ollama recommandés (tous GRATUITS):\n")
    for model, description in RECOMMENDED_OLLAMA_MODELS.items():
        print(f"   • {model:20} - {description}")
    print("\n💡 Pour installer: ollama pull <model_name>")
    print("💡 Pour lister vos modèles: ollama list\n")

# =============================================================================
# Comparaison de performances entre modèles
# =============================================================================

def benchmark_models(models_to_test):
    """
    Compare la performance de différents modèles sur le même dataset
    """
    from data import valset, CATEGORIES, PRIORITIES
    
    print("\n" + "="*70)
    print("📊 Benchmark : Comparaison de modèles")
    print("="*70 + "\n")
    
    results = {}
    
    for model_config in models_to_test:
        model_type = model_config['type']
        model_name = model_config['name']
        
        print(f"Testing {model_name}...")
        
        try:
            # Configure le modèle
            if model_type == 'ollama':
                configure_ollama(model_name)
            elif model_type == 'openai':
                configure_openai(model_name)
            elif model_type == 'anthropic':
                configure_anthropic(model_name)
            
            # Teste sur un échantillon
            classifier = UniversalClassifier()
            correct = 0
            
            for example in valset[:5]:  # Teste sur 5 exemples
                pred = classifier(ticket=example['ticket'])
                if (pred.category.lower() == example['category'].lower() and 
                    pred.priority.lower() == example['priority'].lower()):
                    correct += 1
            
            accuracy = (correct / 5) * 100
            results[model_name] = accuracy
            print(f"   ✅ Précision: {accuracy}%\n")
            
        except Exception as e:
            print(f"   ❌ Erreur: {str(e)}\n")
            results[model_name] = None
    
    # Résumé
    print("\n📈 Résumé des performances:")
    for model, acc in results.items():
        if acc is not None:
            print(f"   {model:30} {acc}%")
        else:
            print(f"   {model:30} Non disponible")

# =============================================================================
# MAIN : Exécuter les démonstrations
# =============================================================================

if __name__ == "__main__":
    print("\n🎯 Exemples avancés : Configuration multi-LLM avec DSPy\n")
    
    # Afficher les modèles disponibles
    print_available_models()
    
    # Démonstration du changement de modèle
    demo_model_switching()
    
    # Exemple de benchmark (décommentez pour l'exécuter)
    """
    models = [
        {'type': 'ollama', 'name': 'llama3.1:8b'},
        {'type': 'ollama', 'name': 'mistral:7b'},
        # {'type': 'openai', 'name': 'gpt-4o-mini'},  # Si clé API configurée
    ]
    benchmark_models(models)
    """
    
    print("\n✨ Démonstration terminée!")
    print("\n💡 Conseil: DSPy permet de changer de LLM en 1 ligne de code!")
    print("   Développez en local avec Ollama, déployez avec OpenAI si besoin.")
