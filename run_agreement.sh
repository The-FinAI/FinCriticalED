#!/bin/bash
# ========================================
# Fixed-parameter version: run two analyses
# ========================================

# Specify file and number of annotators
FILE="annotation/final_annotation_project.json"
ANNOTATORS=4

echo "======================================="
echo "📁 File: $FILE"
echo "👥 Number of annotators: $ANNOTATORS"
echo "======================================="

# 1️⃣ If more than 2 annotators: compute Fleiss' kappa; if exactly 2 annotators: compute Cohen’s kappa
echo ""
echo "▶️ Running kappa score (overall agreement only)"
python calculate_agreement.py --file "$FILE" --annotators "$ANNOTATORS" --pairwise

# # 2️⃣ For the same task, also compute pairwise Cohen’s kappa
# echo ""
# echo "▶️ Running Fleiss’ + pairwise Cohen’s kappa (including pairwise agreement)"
# python calculate_agreement.py --file "$FILE" --annotators "$ANNOTATORS" --pairwise

echo ""
echo "✅ Analysis completed!"
