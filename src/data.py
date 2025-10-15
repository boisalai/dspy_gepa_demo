"""
Data module containing training and validation datasets for IT ticket classification.

This module defines:
- Training dataset (trainset)
- Validation dataset (valset)
- Categories and priorities constants
"""

import dspy
from typing import List, Dict

# Categories of IT tickets
CATEGORIES = [
    "Hardware",
    "Software",
    "Network",
    "Application",
    "Infrastructure",
    "Account",
    "Email",
    "Peripherals"
]

# Priority levels
PRIORITIES = [
    "Low",
    "Medium",
    "High",
    "Urgent",
    "Critical"
]

# Training dataset
trainset = [
    {"ticket": "Mon ordinateur ne démarre plus depuis ce matin. J'ai une présentation importante dans 2 heures.", "category": "Hardware", "priority": "Urgent"},
    {"ticket": "Je n'arrive pas à me connecter à l'imprimante du 3e étage. Ça peut attendre.", "category": "Peripherals", "priority": "Low"},
    {"ticket": "Le VPN ne fonctionne plus. Impossible d'accéder aux fichiers du serveur.", "category": "Network", "priority": "High"},
    {"ticket": "J'ai oublié mon mot de passe Outlook. Je peux utiliser le webmail.", "category": "Account", "priority": "Medium"},
    {"ticket": "Le site web affiche une erreur 500. Les clients ne peuvent plus commander!", "category": "Application", "priority": "Critical"},
    {"ticket": "Ma souris sans fil ne répond plus bien. Les piles sont faibles.", "category": "Peripherals", "priority": "Low"},
    {"ticket": "Le système de paie ne calcule pas les heures supplémentaires. C'est la fin du mois.", "category": "Application", "priority": "Urgent"},
    {"ticket": "J'aimerais une mise à jour de mon logiciel Adobe quand vous aurez le temps.", "category": "Software", "priority": "Low"},
    {"ticket": "Le serveur de base de données est très lent. Toute la production est impactée.", "category": "Infrastructure", "priority": "Critical"},
    {"ticket": "Je ne reçois plus les emails. J'attends des réponses de fournisseurs.", "category": "Email", "priority": "High"},
    {"ticket": "Mon écran externe ne s'affiche plus. Je peux travailler sur le laptop.", "category": "Hardware", "priority": "Medium"},
    {"ticket": "Le wifi de la salle A ne fonctionne pas. Réunion avec des externes dans 30 min.", "category": "Network", "priority": "Urgent"},
    {"ticket": "Je voudrais installer Slack pour mieux collaborer avec l'équipe.", "category": "Software", "priority": "Medium"},
    {"ticket": "Le système de sauvegarde a échoué cette nuit selon le rapport.", "category": "Infrastructure", "priority": "High"},
    {"ticket": "Mon clavier a une touche qui colle. C'est gérable mais ennuyeux.", "category": "Peripherals", "priority": "Low"}
]

# Validation dataset
valset = [
    {"ticket": "Le serveur de fichiers est inaccessible. Personne ne peut travailler.", "category": "Infrastructure", "priority": "Critical"},
    {"ticket": "J'ai besoin d'accès au dossier comptabilité pour l'audit. C'est urgent.", "category": "Account", "priority": "Urgent"},
    {"ticket": "L'écran de mon collègue en vacances clignote. On peut attendre.", "category": "Hardware", "priority": "Low"},
    {"ticket": "Le CRM plante quand j'essaie d'exporter les contacts.", "category": "Application", "priority": "High"},
    {"ticket": "Je voudrais changer ma photo de profil quand vous aurez un moment.", "category": "Account", "priority": "Low"},
    {"ticket": "La vidéoconférence ne fonctionne pas. Réunion avec New York dans 10 minutes!", "category": "Application", "priority": "Critical"},
    {"ticket": "Mon antivirus affiche un message d'expiration mais tout fonctionne.", "category": "Software", "priority": "Medium"}
]


def get_train_examples() -> List[dspy.Example]:
    """
    Convert training dataset to DSPy Example format.

    Returns:
        List of DSPy Example objects with 'ticket' as input field
    """
    return [
        dspy.Example(
            ticket=ex['ticket'],
            category=ex['category'],
            priority=ex['priority']
        ).with_inputs('ticket')
        for ex in trainset
    ]


def get_val_examples() -> List[dspy.Example]:
    """
    Convert validation dataset to DSPy Example format.

    Returns:
        List of DSPy Example objects with 'ticket' as input field
    """
    return [
        dspy.Example(
            ticket=ex['ticket'],
            category=ex['category'],
            priority=ex['priority']
        ).with_inputs('ticket')
        for ex in valset
    ]


def print_dataset_stats():
    """Print statistics about the datasets."""
    print(f"📊 Dataset statistics:")
    print(f"   Training examples: {len(trainset)}")
    print(f"   Validation examples: {len(valset)}")
    print(f"   Categories: {len(CATEGORIES)}")
    print(f"   Priorities: {len(PRIORITIES)}")
    print()
    print(f"Categories: {', '.join(CATEGORIES)}")
    print(f"Priorities: {', '.join(PRIORITIES)}")


if __name__ == "__main__":
    # Print dataset information
    print_dataset_stats()

    # Show a few examples
    print("\n📝 Sample training examples:")
    for i, ex in enumerate(trainset[:3], 1):
        print(f"\n{i}. {ex['ticket'][:60]}...")
        print(f"   Category: {ex['category']}")
        print(f"   Priority: {ex['priority']}")
