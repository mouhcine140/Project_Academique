# Detecting and Analyzing Code-Switching Between English and Darija

Projet académique de traitement du langage naturel (NLP) portant sur la **détection et l'analyse du code-switching** entre l'anglais et le darija (dialecte arabe marocain), un phénomène linguistique fréquent dans les communautés multilingues et particulièrement complexe à traiter en raison de l'absence de système d'écriture standardisé pour le darija et des nombreux emprunts lexicaux à l'anglais.

**Auteurs :** Belfqueh Hatim, Mouhcine Riany
**Date :** Janvier 2025

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `projet-r-d-1 (2).ipynb` | Notebook Jupyter contenant l'ensemble du pipeline : nettoyage et préparation des données, construction du dataset (Darija / English / Code-Switching), entraînement des deux modèles, évaluation et visualisation des résultats (rapport de classification, matrices de confusion). |
| `Code_Switching_in_Darija_English__Challenges_and_Solutions_for_NLP.pdf` | Rapport académique complet (papier scientifique) présentant le contexte, la méthodologie, les expériences et les résultats du projet. |
| `Code-switchingdetection and analysis.pdf` | Support d'analyse détaillant le code et les modèles utilisés (extraits de code commentés, description des architectures). |

## Objectif

Classifier automatiquement des segments de texte en trois catégories :
- **Darija**
- **English**
- **Code-Switching** (mélange des deux langues)

## Données

- Corpus de phrases en darija, en anglais et en code-switching (texte transformé en minuscules, ponctuation et caractères spéciaux nettoyés via des expressions régulières).
- Split train/test : 80 % / 20 % (`random_state=42`).
- Jeu de test final : 1 432 exemples (538 Darija, 525 English, 369 Code-Switching).

## Modèles

### 1. mBERT (Multilingual BERT)
- Modèle transformer pré-entraîné sur 104 langues, avec une tête de classification personnalisée (3 classes).
- Fine-tuning avec l'optimiseur **AdamW**, learning rate `5e-5`, 3 à 5 époques.
- **Résultat : ~99 % d'accuracy**, loss décroissante de façon stable.

### 2. BiLSTM + GloVe
- Couche d'embedding initialisée avec des embeddings **GloVe pré-entraînés** (100 dimensions, vocabulaire max 10 000 mots, séquences de longueur max 100).
- 2 couches **BiLSTM** empilées (forward + backward).
- Couche dense (softmax) + dropout (0.3) pour limiter l'overfitting.
- ~1 000 000 de paramètres (embeddings non entraînables).
- Entraînement : optimiseur **Adam**, learning rate `1e-3`, 10 époques (~75 s/époque).
- **Résultat : 99,30 % d'accuracy**, loss finale 0.0285.

## Résultats

| Modèle | Accuracy | Précision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| mBERT | ~99 % | 0.99 | 0.99 | 0.99 |
| BiLSTM + GloVe | 99.30 % | 0.99–1.00 | 0.98–1.00 | 0.99–1.00 |

Les deux modèles obtiennent des performances très élevées et proches. mBERT capture mieux le contexte sémantique global, tandis que le BiLSTM se montre particulièrement efficace pour capturer les dépendances séquentielles locales, notamment dans les zones de transition entre les langues (code-switching).

Des matrices de confusion et rapports de classification détaillés (precision/recall/F1 par classe) sont disponibles dans le notebook et dans le rapport PDF.

## Stack technique

- **Python**, **Jupyter Notebook**
- **PyTorch** / **Transformers (HuggingFace)** pour mBERT
- **TensorFlow / Keras** pour le BiLSTM
- **GloVe** (embeddings pré-entraînés `glove.6B.100d`)
- **scikit-learn** (classification report, confusion matrix, train_test_split)
- **Seaborn / Matplotlib** (visualisation des matrices de confusion)
- **Pandas / re** (nettoyage et préparation des données)

## Reproduire les expériences

1. Ouvrir `projet-r-d-1 (2).ipynb` dans Jupyter ou Kaggle/Colab.
2. Adapter le chemin vers les embeddings GloVe (`glove.6B.100d.txt`) selon votre environnement.
3. Exécuter les cellules dans l'ordre : préparation des données → entraînement mBERT → entraînement BiLSTM → évaluation.

## Limites et pistes d'amélioration

- Jeu de données de taille limitée : l'élargir améliorerait la robustesse et la généralisation.
- Absence de standardisation orthographique du darija : une normalisation plus poussée (transliteration) pourrait réduire le bruit.
- Pistes futures : fine-tuning plus poussé, augmentation de données, test sur d'autres paires de langues à code-switching (ex. Darija–Français).

---
*Projet réalisé dans le cadre du cursus Ingénieur d'État Big Data & Intelligence Artificielle — Université Internationale de Rabat (UIR).*
