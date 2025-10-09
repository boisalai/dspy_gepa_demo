"""
Guide détaillé : Optimisation avec GEPA
Comprendre et utiliser GEPA pour améliorer automatiquement vos prompts
"""

import dspy
from data import trainset, valset, CATEGORIES, PRIORITIES

# =============================================================================
# Qu'est-ce que GEPA ?
# =============================================================================

"""
GEPA (Genetic-Pareto Algorithm) est un algorithme d'optimisation qui :

1. 🧬 ÉVOLUTION GÉNÉTIQUE
   - Génère des variantes de vos prompts
   - Teste chaque variante sur vos données
   - Garde les meilleures et mute à nouveau

2. 🤔 RÉFLEXION LLM
   - Utilise un LLM pour analyser les erreurs
   - Propose des améliorations ciblées
   - Apprend des échecs pour améliorer

3. 📊 SÉLECTION PARETO
   - Équilibre plusieurs métriques (précision, rapidité, etc.)
   - Garde les solutions non-dominées
   - Évite les optimisations locales

Résultat : Vos prompts s'améliorent automatiquement, souvent de 10-20% !
"""

# =============================================================================
# Configuration de base
# =============================================================================

def setup_for_gepa():
    """Préparer l'environnement pour GEPA"""
    
    # Configuration du modèle
    lm = dspy.LM(
        model='ollama_chat/llama3.1:8b',
        api_base='http://localhost:11434',
        temperature=0.3
    )
    dspy.configure(lm=lm)
    
    print("✅ DSPy configuré pour GEPA")

# =============================================================================
# Définir le module à optimiser
# =============================================================================

class TicketClassifier(dspy.Signature):
    """Classifier un ticket IT selon catégorie et priorité"""
    ticket = dspy.InputField(desc="Description du ticket")
    category = dspy.OutputField(desc=f"Catégorie parmi: {', '.join(CATEGORIES)}")
    priority = dspy.OutputField(desc=f"Priorité parmi: {', '.join(PRIORITIES)}")

class BasicClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(TicketClassifier)
    
    def forward(self, ticket):
        result = self.classifier(ticket=ticket)
        return dspy.Prediction(
            category=result.category,
            priority=result.priority
        )

# =============================================================================
# Définir une métrique d'évaluation
# =============================================================================

def exact_match_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Métrique simple : 1 si tout est correct, 0 sinon
    Compatible avec GEPA DSPy 3.0+
    """
    category_match = prediction.category.strip().lower() == example.category.strip().lower()
    priority_match = prediction.priority.strip().lower() == example.priority.strip().lower()

    return 1.0 if (category_match and priority_match) else 0.0

def partial_match_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Métrique plus nuancée :
    - 1.0 si les deux sont corrects
    - 0.7 si la catégorie est correcte
    - 0.5 si la priorité est correcte
    - 0.0 si rien n'est correct
    Compatible avec GEPA DSPy 3.0+
    """
    category_match = prediction.category.strip().lower() == example.category.strip().lower()
    priority_match = prediction.priority.strip().lower() == example.priority.strip().lower()

    if category_match and priority_match:
        return 1.0
    elif category_match:
        return 0.7  # La catégorie est plus importante
    elif priority_match:
        return 0.5
    else:
        return 0.0

# =============================================================================
# Exemple 1 : Optimisation basique avec GEPA
# =============================================================================

def run_basic_gepa_optimization():
    """
    Exemple simple d'optimisation GEPA
    ⚠️ Prend plusieurs minutes avec Ollama local
    """
    print("\n" + "="*70)
    print("🔧 Optimisation GEPA - Exemple basique")
    print("="*70 + "\n")
    
    setup_for_gepa()
    
    # Préparer les données au format DSPy
    train_examples = [
        dspy.Example(
            ticket=ex['ticket'],
            category=ex['category'],
            priority=ex['priority']
        ).with_inputs('ticket')
        for ex in trainset
    ]
    
    val_examples = [
        dspy.Example(
            ticket=ex['ticket'],
            category=ex['category'],
            priority=ex['priority']
        ).with_inputs('ticket')
        for ex in valset
    ]
    
    # Créer le module à optimiser
    classifier = BasicClassifier()
    
    # Évaluer AVANT optimisation
    print("📊 Évaluation AVANT optimisation...")
    score_before = evaluate_module(classifier, val_examples, exact_match_metric)
    print(f"   Score: {score_before:.2%}\n")
    
    # Optimiser avec GEPA
    print("🧬 Lancement de l'optimisation GEPA...")
    print("   (Cela peut prendre 5-10 minutes avec Ollama)\n")
    
    try:
        from dspy.teleprompt import GEPA

        # Configuration GEPA pour DSPy 3.0+
        # GEPA nécessite un modèle de réflexion pour analyser les erreurs
        reflection_lm = dspy.LM(
            model='ollama_chat/llama3.1:8b',
            api_base='http://localhost:11434',
            temperature=1.0,  # Plus de créativité pour la réflexion
            max_tokens=8000
        )

        # 'light' = optimisation rapide, 'medium' = équilibré, 'heavy' = approfondi
        optimizer = GEPA(
            metric=exact_match_metric,
            auto='light',  # Équivalent à breadth=5, depth=2 dans l'ancienne API
            reflection_lm=reflection_lm,
        )

        optimized_classifier = optimizer.compile(
            classifier,
            trainset=train_examples,
            valset=val_examples
        )
        
        print("✅ Optimisation terminée!\n")
        
        # Évaluer APRÈS optimisation
        print("📊 Évaluation APRÈS optimisation...")
        score_after = evaluate_module(optimized_classifier, val_examples, exact_match_metric)
        print(f"   Score: {score_after:.2%}\n")
        
        # Résumé
        improvement = ((score_after - score_before) / score_before) * 100
        print(f"📈 Amélioration: {improvement:+.1f}%")
        
        return optimized_classifier
        
    except ImportError:
        print("❌ GEPA non disponible dans cette version de DSPy")
        print("   Installez la dernière version: pip install -U dspy-ai")
        return None

# =============================================================================
# Fonction d'évaluation
# =============================================================================

def evaluate_module(module, examples, metric):
    """Évalue un module DSPy sur un ensemble d'exemples"""
    total_score = 0
    
    for example in examples:
        prediction = module(ticket=example.ticket)
        score = metric(example, prediction)
        total_score += score
    
    return total_score / len(examples)

# =============================================================================
# Exemple 2 : Optimisation avancée avec paramètres personnalisés
# =============================================================================

def run_advanced_gepa_optimization():
    """
    Exemple avancé avec paramètres GEPA personnalisés
    """
    print("\n" + "="*70)
    print("🚀 Optimisation GEPA - Configuration avancée")
    print("="*70 + "\n")
    
    setup_for_gepa()
    
    # Préparer les données
    train_examples = [
        dspy.Example(
            ticket=ex['ticket'],
            category=ex['category'],
            priority=ex['priority']
        ).with_inputs('ticket')
        for ex in trainset
    ]
    
    val_examples = [
        dspy.Example(
            ticket=ex['ticket'],
            category=ex['category'],
            priority=ex['priority']
        ).with_inputs('ticket')
        for ex in valset
    ]
    
    classifier = BasicClassifier()
    
    try:
        from dspy.teleprompt import GEPA

        # Configuration avancée GEPA pour DSPy 3.0+
        reflection_lm = dspy.LM(
            model='ollama_chat/llama3.1:8b',
            api_base='http://localhost:11434',
            temperature=1.0,
            max_tokens=8000
        )

        optimizer = GEPA(
            metric=partial_match_metric,  # Métrique plus nuancée
            auto='medium',                # Optimisation équilibrée (plus approfondie que 'light')
            reflection_lm=reflection_lm,
        )

        print("🧬 Optimisation avec paramètres avancés...")
        print("   - Métrique avec correspondance partielle")
        print("   - Mode 'medium' (optimisation équilibrée)\n")
        
        optimized = optimizer.compile(
            classifier,
            trainset=train_examples,
            valset=val_examples
        )
        
        print("✅ Optimisation avancée terminée!\n")
        
        return optimized
        
    except ImportError:
        print("❌ GEPA non disponible")
        return None

# =============================================================================
# Exemple 3 : Inspecter les prompts optimisés
# =============================================================================

def inspect_optimized_prompts(optimized_module):
    """
    Affiche les prompts qui ont été générés par GEPA
    Utile pour comprendre ce qui a été amélioré
    """
    print("\n" + "="*70)
    print("🔍 Inspection des prompts optimisés")
    print("="*70 + "\n")
    
    # DSPy stocke les prompts dans les predictors
    if hasattr(optimized_module, 'classifier'):
        predictor = optimized_module.classifier
        
        # Afficher le prompt système
        if hasattr(predictor, 'extended_signature'):
            print("📝 Signature optimisée:")
            print(predictor.extended_signature)
            print()
        
        # Afficher les exemples de démonstration
        if hasattr(predictor, 'demos'):
            print(f"📚 Exemples de démonstration ({len(predictor.demos)}):")
            for i, demo in enumerate(predictor.demos[:3], 1):  # Afficher les 3 premiers
                print(f"\n   Exemple {i}:")
                print(f"   Input: {demo.ticket[:100]}...")
                print(f"   Output: {demo.category} | {demo.priority}")
            print()

# =============================================================================
# Conseils pour utiliser GEPA efficacement
# =============================================================================

"""
💡 CONSEILS POUR GEPA :

1. DONNÉES
   ✅ Avoir au moins 10-20 exemples d'entraînement
   ✅ Données de validation séparées (pas dans train)
   ✅ Exemples représentatifs de la production

2. MÉTRIQUE
   ✅ Simple et claire (0 à 1)
   ✅ Capture ce qui est important pour vous
   ✅ Rapide à calculer

3. BUDGET
   ✅ Plus de breadth = plus de candidats = meilleure optimisation (mais plus lent)
   ✅ Commencer avec breadth=5, augmenter si besoin
   ✅ depth contrôle le nombre d'itérations (2-3 est généralement bon)
   ✅ Avec Ollama local : 5-10 min par run

4. ITÉRATION
   ✅ Commencer simple, complexifier progressivement
   ✅ Sauvegarder les modèles optimisés
   ✅ Tester sur nouveaux exemples pour valider

5. MODÈLES
   ✅ Task model : Le modèle que vous optimisez (peut être petit)
   ✅ Avec Ollama local : Utiliser le même modèle
   
PARAMÈTRES GEPA (DSPy 3.0+) :
- auto : Budget d'optimisation ('light', 'medium', 'heavy')
  * light : optimisation rapide (quelques minutes)
  * medium : optimisation équilibrée (10-15 min)
  * heavy : optimisation approfondie (20+ min)
- metric : Fonction d'évaluation retournant 0-1 ou {'score': float, 'feedback': str}
- max_full_evals : Alternative à auto, nombre max d'évaluations complètes
- max_metric_calls : Alternative à auto, nombre max d'appels à la métrique
"""

# =============================================================================
# MAIN : Lancer les exemples
# =============================================================================

if __name__ == "__main__":
    print("\n🎓 Guide GEPA : Optimisation automatique de prompts")
    
    # Choix de l'exemple à exécuter
    print("\n" + "="*70)
    print("Choisissez un exemple :")
    print("1. Optimisation basique (recommandé pour débuter)")
    print("2. Optimisation avancée (avec paramètres personnalisés)")
    print("3. Juste afficher les conseils")
    print("="*70 + "\n")
    
    choice = input("Votre choix (1-3) : ").strip()
    
    if choice == "1":
        optimized = run_basic_gepa_optimization()
        if optimized:
            inspect_optimized_prompts(optimized)
    
    elif choice == "2":
        optimized = run_advanced_gepa_optimization()
        if optimized:
            inspect_optimized_prompts(optimized)
    
    else:
        print("\n📚 Consultez les commentaires dans ce fichier pour les conseils GEPA!")
    
    print("\n" + "="*70)
    print("✨ Pour en savoir plus sur GEPA:")
    print("   📄 Paper: https://arxiv.org/abs/2507.19457")
    print("   💻 GitHub: https://github.com/gepa-ai/gepa")
    print("   📖 Docs: https://dspy-docs.vercel.app/")
    print("="*70 + "\n")
