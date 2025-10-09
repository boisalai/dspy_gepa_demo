"""
Script de diagnostic pour identifier l'API GEPA disponible
"""

import sys

print("🔍 Diagnostic de l'environnement DSPy et GEPA\n")

# 1. Vérifier l'installation de DSPy
try:
    import dspy
    print(f"✅ DSPy installé")
    print(f"   Version: {dspy.__version__}")
except ImportError:
    print("❌ DSPy n'est pas installé")
    sys.exit(1)

# 2. Vérifier si GEPA est disponible
try:
    from dspy.teleprompt import GEPA
    print(f"✅ GEPA disponible")
except ImportError:
    print("❌ GEPA n'est pas disponible dans cette version de DSPy")
    print("   Essayez: pip install --upgrade dspy-ai")
    sys.exit(1)

# 3. Inspecter la signature de GEPA
import inspect

print(f"\n📋 Signature de GEPA.__init__:")
try:
    sig = inspect.signature(GEPA.__init__)
    print(f"   {sig}")
    
    print(f"\n📝 Paramètres acceptés:")
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        default = param.default
        if default == inspect.Parameter.empty:
            print(f"   • {param_name} (requis)")
        else:
            print(f"   • {param_name} = {default}")
except Exception as e:
    print(f"   Erreur lors de l'inspection: {e}")

# 4. Tester différentes configurations
print(f"\n🧪 Tests de configuration GEPA:\n")

# Métrique compatible avec toutes versions
def dummy_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return 1.0

# Test avec auto (DSPy 3.0+)
print("Test 1: Avec auto='light' (DSPy 3.0+)")
try:
    # DSPy 3.0+ nécessite reflection_lm
    reflection_lm = dspy.LM(
        model='ollama_chat/llama3.1:8b',
        api_base='http://localhost:11434',
        temperature=1.0,
        max_tokens=8000
    )
    optimizer = GEPA(metric=dummy_metric, auto='light', reflection_lm=reflection_lm)
    print("   ✅ Fonctionne avec auto='light' + reflection_lm")
except Exception as e:
    print(f"   ❌ Erreur: {type(e).__name__}: {e}")

# Configuration minimale
print("\nTest 2: Configuration minimale (metric seulement)")
try:
    optimizer = GEPA(metric=dummy_metric)
    print("   ✅ Fonctionne avec juste metric")
except Exception as e:
    print(f"   ❌ Erreur: {type(e).__name__}: {e}")

# Test avec breadth/depth
print("\nTest 3: Avec breadth et depth")
try:
    optimizer = GEPA(metric=dummy_metric, breadth=5, depth=2)
    print("   ✅ Fonctionne avec breadth et depth")
except Exception as e:
    print(f"   ❌ Erreur: {type(e).__name__}: {e}")

# Test avec anciens paramètres
print("\nTest 4: Avec anciens paramètres (max_bootstrapped_demos, etc.)")
try:
    optimizer = GEPA(
        metric=dummy_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=5
    )
    print("   ✅ Fonctionne avec anciens paramètres")
except Exception as e:
    print(f"   ❌ Erreur: {type(e).__name__}: {e}")

# Test avec num_candidates
print("\nTest 5: Avec num_candidates")
try:
    optimizer = GEPA(metric=dummy_metric, num_candidates=5)
    print("   ✅ Fonctionne avec num_candidates")
except Exception as e:
    print(f"   ❌ Erreur: {type(e).__name__}: {e}")

# 5. Recommandations
print("\n" + "="*70)
print("💡 RECOMMANDATIONS")
print("="*70)

print("\nBasé sur les tests ci-dessus, voici la configuration à utiliser:")
print("\nSi Test 1 a fonctionné (DSPy 3.0+), utilisez:")
print("""
from dspy.teleprompt import GEPA

optimizer = GEPA(metric=your_metric, auto='light')  # 'light', 'medium', ou 'heavy'
optimized = optimizer.compile(module, trainset=train, valset=val)
""")

print("Si Test 2 a fonctionné, utilisez la configuration minimale:")
print("""
from dspy.teleprompt import GEPA

optimizer = GEPA(metric=your_metric)
optimized = optimizer.compile(module, trainset=train, valset=val)
""")

print("\nSi vous voyez des paramètres spécifiques acceptés ci-dessus,")
print("utilisez ceux-là dans votre code.")

print("\n" + "="*70)
print("📚 PROCHAINES ÉTAPES")
print("="*70)
print("""
1. Notez quelle configuration a fonctionné (Test 1-5)
2. Utilisez cette configuration dans gepa_guide.py
3. N'oubliez pas : votre métrique doit accepter les paramètres:
   (example, prediction, trace=None, pred_name=None, pred_trace=None)
4. Si aucun test ne fonctionne, essayez:
   pip install --upgrade dspy-ai
""")

# 6. Informations système
print("\n" + "="*70)
print("🖥️  INFORMATIONS SYSTÈME")
print("="*70)
print(f"Python: {sys.version}")
print(f"DSPy: {dspy.__version__}")

try:
    import platform
    print(f"OS: {platform.system()} {platform.release()}")
except:
    pass

print("\n✨ Diagnostic terminé!")
