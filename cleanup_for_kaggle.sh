#!/bin/bash
# Script de nettoyage pour préparation Kaggle
# Supprime les fichiers non nécessaires pour l'exécution locale

set -e

echo "🧹 Nettoyage du dépôt pour Kaggle..."
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour supprimer fichiers/dossiers
safe_remove() {
    if [ -e "$1" ]; then
        rm -rf "$1"
        echo -e "${GREEN}✓${NC} Supprimé: $1"
    else
        echo -e "${YELLOW}○${NC} Ignoré (n'existe pas): $1"
    fi
}

# Garder une trace de ce qui est supprimé
REMOVED_COUNT=0

echo "📁 Nettoyage des dossiers de documentation..."
# Garder uniquement KAGGLE_GUIDE_FR.md et WRITEUP.md
for doc in docs/*.md; do
    if [ "$doc" != "docs/README.md" ]; then
        safe_remove "$doc"
        ((REMOVED_COUNT++))
    fi
done

echo ""
echo "🧪 Nettoyage des tests et outils de debug..."
safe_remove "tests/"
((REMOVED_COUNT++))

echo ""
echo "📊 Nettoyage des outils d'analyse de logs..."
safe_remove "logs_parser/"
((REMOVED_COUNT++))

echo ""
echo "🔧 Nettoyage des fichiers de configuration API..."
safe_remove "config/api_keys.env.example"
((REMOVED_COUNT++))

echo ""
echo "📝 Nettoyage des fichiers temporaires..."
safe_remove "src/__pycache__"
safe_remove "src/providers/__pycache__"
safe_remove "src/providers/openai_bg/__pycache__"
safe_remove "src/parallel/__pycache__"
((REMOVED_COUNT+=4))

echo ""
echo "🗑️ Suppression des providers API non nécessaires..."
# Garder local.py mais supprimer les providers API
safe_remove "src/providers/openai.py"
safe_remove "src/providers/anthropic.py"
safe_remove "src/providers/gemini.py"
safe_remove "src/providers/openai_background.py"
safe_remove "src/providers/openai_runner.py"
safe_remove "src/providers/openai_utils.py"
safe_remove "src/providers/openai_bg/"
((REMOVED_COUNT+=6))

echo ""
echo "📦 Fichiers de requirements..."
# Garder uniquement requirements-kaggle-local.txt
safe_remove "requirements.txt"
safe_remove "requirements-kaggle.txt"
((REMOVED_COUNT+=2))

echo ""
echo "📋 Nettoyage des fichiers markdown non essentiels..."
safe_remove "Agents.md"
safe_remove "README.md"  # Garder WRITEUP.md et KAGGLE_GUIDE_FR.md
((REMOVED_COUNT+=2))

echo ""
echo "🧹 Nettoyage des modules optionnels dans src/..."
# Garder uniquement les fichiers essentiels
OPTIONAL_FILES=(
    "src/audit_prompts.py"
    "src/audit_templates_consistency.py"
    "src/audit_templates_logic.py"
    "src/augmentation.py"
    "src/image_generation.py"
    "src/hint_generation.py"
    "src/selection_advanced.py"
    "src/selection_legacy.py"
    "src/rate_limiter.py"
)

for file in "${OPTIONAL_FILES[@]}"; do
    safe_remove "$file"
    ((REMOVED_COUNT++))
done

echo ""
echo "📊 Statistiques:"
echo "   Fichiers/dossiers supprimés: $REMOVED_COUNT"
echo ""

echo "✅ Nettoyage terminé!"
echo ""
echo "📦 Structure restante minimale:"
echo "   /workspace/"
echo "   ├── run.py                          ← Point d'entrée"
echo "   ├── requirements-kaggle-local.txt   ← Dépendances"
echo "   ├── KAGGLE_GUIDE_FR.md              ← Guide complet"
echo "   ├── WRITEUP.md                      ← Méthodologie"
echo "   └── src/"
echo "       ├── models.py                   ← Gestion modèles"
echo "       ├── runner.py                   ← Orchestration"
echo "       ├── execution.py                ← Exécution"
echo "       ├── submission.py               ← Soumissions"
echo "       ├── batch_processing.py         ← Batch processing"
echo "       ├── providers/"
echo "       │   └── local.py                ← LLM local ⭐"
echo "       ├── tasks/"
echo "       │   ├── loading.py"
echo "       │   └── prompts_*.py"
echo "       ├── solver/"
echo "       │   ├── steps.py"
echo "       │   ├── state.py"
echo "       │   └── pipelines.py"
echo "       ├── parallel/"
echo "       │   └── *.py                    ← Parallélisation"
echo "       └── *.py                        ← Utils essentiels"
echo ""
echo "🚀 Prêt pour Kaggle!"
