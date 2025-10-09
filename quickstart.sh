#!/bin/bash
# Script de démarrage rapide pour DSPy + GEPA + Ollama
# Usage : bash quickstart.sh

set -e

echo "🚀 Configuration de l'environnement DSPy + GEPA + Ollama"
echo "=========================================================="
echo ""

# Vérifier si Ollama est installé
echo "1️⃣  Vérification d'Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama n'est pas installé"
    echo ""
    echo "Sur macOS, installez avec :"
    echo "  brew install ollama"
    echo ""
    echo "Sur Linux, installez avec :"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    exit 1
else
    echo "✅ Ollama est installé"
fi

# Vérifier si Ollama est en cours d'exécution
echo ""
echo "2️⃣  Vérification du service Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama est en cours d'exécution"
else
    echo "⚠️  Ollama n'est pas en cours d'exécution"
    echo ""
    echo "Démarrez Ollama dans un autre terminal avec :"
    echo "  ollama serve"
    echo ""
    echo "Puis relancez ce script."
    exit 1
fi

# Vérifier si un modèle est disponible
echo ""
echo "3️⃣  Vérification des modèles disponibles..."
MODELS=$(ollama list 2>/dev/null | grep -v "NAME" | wc -l)

if [ "$MODELS" -eq 0 ]; then
    echo "⚠️  Aucun modèle disponible"
    echo ""
    echo "Téléchargez un modèle (recommandé) :"
    echo "  ollama pull llama3.1:8b"
    echo ""
    read -p "Voulez-vous télécharger llama3.1:8b maintenant ? (o/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        ollama pull llama3.1:8b
        echo "✅ Modèle téléchargé"
    else
        echo "❌ Installation annulée. Téléchargez un modèle avant de continuer."
        exit 1
    fi
else
    echo "✅ Modèles disponibles :"
    ollama list | grep -v "NAME"
fi

# Installer les dépendances Python
echo ""
echo "4️⃣  Installation des dépendances Python..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo "✅ Dépendances installées"
else
    echo "⚠️  Fichier requirements.txt non trouvé"
    echo "Installation manuelle :"
    pip install -q dspy-ai
    echo "✅ DSPy installé"
fi

# Test de l'installation
echo ""
echo "5️⃣  Test de l'installation..."
python3 -c "import dspy; print('✅ DSPy importé avec succès')" || {
    echo "❌ Erreur lors de l'import de DSPy"
    exit 1
}

# Résumé
echo ""
echo "=========================================================="
echo "✨ Configuration terminée avec succès !"
echo "=========================================================="
echo ""
echo "📚 Prochaines étapes :"
echo ""
echo "  1. Exemple basique :"
echo "     python main.py"
echo ""
echo "  2. Exemples avancés (changement de modèle) :"
echo "     python advanced_examples.py"
echo ""
echo "  3. Guide GEPA (optimisation) :"
echo "     python gepa_guide.py"
echo ""
echo "📖 Consultez README.md pour plus de détails"
echo ""
