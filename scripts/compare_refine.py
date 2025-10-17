#!/usr/bin/env python3
"""
Script de comparaison des performances de dspy.Refine vs autres modules.

Compare les modules suivants:
- SimpleTicketClassifier (baseline)
- ValidatedClassifier (avec validation)
- RefinedTicketClassifier (avec raffinement itératif)

Métriques mesurées:
- Accuracy (exact match)
- Temps d'exécution moyen
- Temps total
"""

import time
from typing import List, Dict
import dspy

from config import configure_ollama
from modules import SimpleTicketClassifier, ValidatedClassifier, RefinedTicketClassifier
from data import valset, CATEGORIES, PRIORITIES
from metrics import exact_match_metric, partial_match_metric


def evaluate_with_timing(
    module: dspy.Module,
    dataset: List[dict],
    module_name: str,
    verbose: bool = False
) -> Dict:
    """
    Évalue un module avec mesure du temps d'exécution.

    Args:
        module: Le module DSPy à évaluer
        dataset: Dataset de validation
        module_name: Nom du module pour l'affichage
        verbose: Afficher les détails

    Returns:
        Dictionnaire avec les métriques
    """
    correct_exact = 0
    correct_partial = 0
    total = len(dataset)
    execution_times = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Évaluation: {module_name}")
        print(f"{'='*60}")

    for i, example in enumerate(dataset, 1):
        ticket = example['ticket']
        expected_category = example['category']
        expected_priority = example['priority']

        # Mesurer le temps d'exécution
        start_time = time.time()
        prediction = module(ticket=ticket)
        execution_time = time.time() - start_time
        execution_times.append(execution_time)

        # Calculer les scores
        exact_score = exact_match_metric(example, prediction)
        partial_score = partial_match_metric(example, prediction)

        correct_exact += exact_score
        correct_partial += partial_score

        if verbose:
            print(f"\n[{i}/{total}] Ticket: {ticket[:60]}...")
            print(f"  Attendu:  {expected_category} | {expected_priority}")
            print(f"  Prédit:   {prediction.category} | {prediction.priority}")
            print(f"  Score exact: {exact_score:.0%}, partiel: {partial_score:.0%}")
            print(f"  Temps: {execution_time:.3f}s")

    # Calculer les statistiques
    accuracy_exact = correct_exact / total
    accuracy_partial = correct_partial / total
    avg_time = sum(execution_times) / len(execution_times)
    total_time = sum(execution_times)

    return {
        'module': module_name,
        'accuracy_exact': accuracy_exact,
        'accuracy_partial': accuracy_partial,
        'avg_time': avg_time,
        'total_time': total_time,
        'num_examples': total
    }


def print_comparison_table(results: List[Dict]):
    """Affiche un tableau comparatif des résultats."""

    print("\n" + "="*90)
    print(" "*30 + "COMPARAISON DES MODULES")
    print("="*90)

    # En-tête
    header = f"{'Module':<30} | {'Exact Match':<12} | {'Partial Match':<14} | {'Temps Moyen':<12} | {'Temps Total':<12}"
    print(header)
    print("-"*90)

    # Données
    for result in results:
        row = (
            f"{result['module']:<30} | "
            f"{result['accuracy_exact']:>11.1%} | "
            f"{result['accuracy_partial']:>13.1%} | "
            f"{result['avg_time']:>11.3f}s | "
            f"{result['total_time']:>11.2f}s"
        )
        print(row)

    print("="*90)

    # Analyse des meilleurs
    print("\n📊 ANALYSE DES RÉSULTATS:\n")

    best_accuracy = max(results, key=lambda x: x['accuracy_exact'])
    print(f"🎯 Meilleure précision (exact match):")
    print(f"   {best_accuracy['module']}: {best_accuracy['accuracy_exact']:.1%}")

    best_partial = max(results, key=lambda x: x['accuracy_partial'])
    print(f"\n🎯 Meilleure précision (partial match):")
    print(f"   {best_partial['module']}: {best_partial['accuracy_partial']:.1%}")

    fastest = min(results, key=lambda x: x['avg_time'])
    print(f"\n⚡ Plus rapide (temps moyen):")
    print(f"   {fastest['module']}: {fastest['avg_time']:.3f}s par prédiction")

    # Comparaison Refine vs Simple
    refine_result = next((r for r in results if 'Refined' in r['module']), None)
    simple_result = next((r for r in results if 'Simple' in r['module']), None)

    if refine_result and simple_result:
        print(f"\n🔄 Impact du raffinement (Refine vs Simple):")
        accuracy_gain = (refine_result['accuracy_exact'] - simple_result['accuracy_exact']) * 100
        time_cost = (refine_result['avg_time'] / simple_result['avg_time'] - 1) * 100

        print(f"   Gain de précision: {accuracy_gain:+.1f} points de pourcentage")
        print(f"   Coût en temps: {time_cost:+.1f}%")

        if accuracy_gain > 0:
            print(f"   💡 Le raffinement améliore la précision de {accuracy_gain:.1f}% au prix d'un temps {time_cost:.0f}% plus long")
        elif accuracy_gain == 0:
            print(f"   💡 Pas de gain de précision malgré {time_cost:.0f}% de temps supplémentaire")
        else:
            print(f"   ⚠️  La précision a diminué de {abs(accuracy_gain):.1f}% avec {time_cost:.0f}% de temps supplémentaire")


def main():
    """Fonction principale."""

    print("🚀 Comparaison des performances : Refine vs autres modules")
    print("="*90)

    # Configuration
    print("\n📋 Configuration:")
    lm = configure_ollama()
    print(f"   Dataset: validation set ({len(valset)} exemples)")
    print(f"   Catégories: {', '.join(CATEGORIES[:3])}... ({len(CATEGORIES)} total)")
    print(f"   Priorités: {', '.join(PRIORITIES[:3])}... ({len(PRIORITIES)} total)")

    # Créer les modules à comparer
    modules_to_compare = [
        ("SimpleTicketClassifier", SimpleTicketClassifier()),
        ("ValidatedClassifier", ValidatedClassifier()),
        ("RefinedTicketClassifier (N=3)", RefinedTicketClassifier(N=3, threshold=1.0)),
    ]

    # Évaluer chaque module
    results = []
    for name, module in modules_to_compare:
        print(f"\n⏳ Évaluation de {name}...")
        result = evaluate_with_timing(module, valset, name, verbose=False)
        results.append(result)
        print(f"   ✅ Terminé: {result['accuracy_exact']:.1%} exact, {result['avg_time']:.3f}s/exemple")

    # Afficher les résultats
    print_comparison_table(results)

    # Recommandations
    print("\n💡 RECOMMANDATIONS:\n")

    refine_result = next((r for r in results if 'Refined' in r['module']), None)
    simple_result = next((r for r in results if 'Simple' in r['module']), None)

    if refine_result and simple_result:
        if refine_result['accuracy_exact'] > simple_result['accuracy_exact']:
            improvement = (refine_result['accuracy_exact'] - simple_result['accuracy_exact']) * 100
            print(f"✅ Refine améliore la précision de {improvement:.1f}% points")
            print(f"   → Utilisez Refine pour les tâches critiques où la qualité prime")
            print(f"   → Acceptez le coût en temps ({refine_result['avg_time']:.2f}s vs {simple_result['avg_time']:.2f}s)")
        else:
            print(f"⚠️  Refine n'améliore pas la précision sur ce dataset")
            print(f"   → Le module Simple est suffisant et plus rapide")
            print(f"   → Gardez Refine pour des cas d'usage plus complexes/ambigus")

    print("\n" + "="*90)


if __name__ == "__main__":
    main()
