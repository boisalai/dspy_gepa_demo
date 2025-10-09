"""
Données d'entraînement et de test pour la classification de tickets IT
Cas d'usage : Classifier automatiquement les tickets selon leur catégorie et priorité
"""

# Données d'entraînement
trainset = [
    {
        "ticket": "Mon ordinateur ne démarre plus depuis ce matin. J'ai une présentation importante dans 2 heures avec le client.",
        "category": "Hardware",
        "priority": "Urgent"
    },
    {
        "ticket": "Je n'arrive pas à me connecter à l'imprimante du 3e étage. Ça peut attendre la semaine prochaine.",
        "category": "Peripherals",
        "priority": "Low"
    },
    {
        "ticket": "Le VPN ne fonctionne plus. Impossible d'accéder aux fichiers du serveur pour finaliser le rapport trimestriel.",
        "category": "Network",
        "priority": "High"
    },
    {
        "ticket": "J'ai oublié mon mot de passe Outlook. Je peux utiliser le webmail en attendant.",
        "category": "Account",
        "priority": "Medium"
    },
    {
        "ticket": "Le site web de l'entreprise affiche une erreur 500. Les clients ne peuvent plus commander en ligne!",
        "category": "Application",
        "priority": "Critical"
    },
    {
        "ticket": "Ma souris sans fil ne répond plus bien. Les piles sont peut-être faibles.",
        "category": "Peripherals",
        "priority": "Low"
    },
    {
        "ticket": "Le système de paie ne calcule pas correctement les heures supplémentaires. C'est la fin du mois.",
        "category": "Application",
        "priority": "Urgent"
    },
    {
        "ticket": "J'aimerais avoir une mise à jour de mon logiciel Adobe quand vous aurez le temps.",
        "category": "Software",
        "priority": "Low"
    },
    {
        "ticket": "Le serveur de base de données est très lent depuis 1 heure. Toute la production est impactée.",
        "category": "Infrastructure",
        "priority": "Critical"
    },
    {
        "ticket": "Je ne reçois plus les emails depuis ce matin. J'attends des réponses importantes de fournisseurs.",
        "category": "Email",
        "priority": "High"
    },
    {
        "ticket": "Mon écran externe ne s'affiche plus. Je peux travailler sur l'écran du laptop.",
        "category": "Hardware",
        "priority": "Medium"
    },
    {
        "ticket": "Le wifi de la salle de conférence A ne fonctionne pas. J'ai une réunion avec des externes dans 30 minutes.",
        "category": "Network",
        "priority": "Urgent"
    },
    {
        "ticket": "Je voudrais installer Slack sur mon poste de travail pour mieux collaborer avec l'équipe.",
        "category": "Software",
        "priority": "Medium"
    },
    {
        "ticket": "Le système de sauvegarde a échoué cette nuit selon le rapport automatique.",
        "category": "Infrastructure",
        "priority": "High"
    },
    {
        "ticket": "Mon clavier a une touche qui colle. C'est gérable mais un peu ennuyeux.",
        "category": "Peripherals",
        "priority": "Low"
    }
]

# Données de validation/test
valset = [
    {
        "ticket": "Le serveur de fichiers est inaccessible. Personne ne peut travailler sur les documents partagés.",
        "category": "Infrastructure",
        "priority": "Critical"
    },
    {
        "ticket": "J'ai besoin d'accès au dossier comptabilité pour compléter l'audit. C'est urgent.",
        "category": "Account",
        "priority": "Urgent"
    },
    {
        "ticket": "L'écran de mon collègue qui est en vacances clignote. On peut attendre son retour.",
        "category": "Hardware",
        "priority": "Low"
    },
    {
        "ticket": "Le logiciel de CRM plante à chaque fois que j'essaie d'exporter les contacts.",
        "category": "Application",
        "priority": "High"
    },
    {
        "ticket": "Je voudrais changer ma photo de profil dans l'annuaire quand vous aurez un moment.",
        "category": "Account",
        "priority": "Low"
    },
    {
        "ticket": "Le système de vidéoconférence ne fonctionne pas. J'ai une réunion avec New York dans 10 minutes!",
        "category": "Application",
        "priority": "Critical"
    },
    {
        "ticket": "Mon antivirus affiche un message d'expiration mais tout semble fonctionner normalement.",
        "category": "Software",
        "priority": "Medium"
    }
]

# Mapping des catégories et priorités possibles (pour validation)
CATEGORIES = ["Hardware", "Software", "Network", "Application", "Infrastructure", "Account", "Email", "Peripherals"]
PRIORITIES = ["Low", "Medium", "High", "Urgent", "Critical"]
