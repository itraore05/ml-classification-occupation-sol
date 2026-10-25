# 🛰️ Classification de l'occupation du sol — Données satellitaires multi-classes déséquilibrées

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2E7D32?style=flat)
![SMOTE](https://img.shields.io/badge/SMOTE-déséquilibre-orange?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-interprétabilité-8E24AA?style=flat)
![Master](https://img.shields.io/badge/Master%201-Big%20Data%20%26%20IA-blue?style=flat)

> Classification de pixels d'images satellitaires en **8 classes d'occupation du sol** (café, cacao, hévéa, forêt, savane, eau, sol nu, zones bâties) à partir de bandes spectrales, d'indices de végétation et de textures, avec un **déséquilibre extrême** entre les classes.
> Projet académique — Master 1 Big Data Analytics & IA (UVCI), UE Machine Learning, 2026.

---

## 🎯 Problématique

Trois difficultés principales :

- **Déséquilibre extrême** : une classe très majoritaire (hévéa et autres cultures) et des classes très rares (sol nu : 8 pixels seulement) ;
- **Redondance** entre bandes spectrales et indices de végétation ;
- **Bruit** introduit par les descripteurs de texture (GLCM, Gabor, LBP).

**Question du projet :** quelles variables et quel modèle donnent la classification la plus fiable, sans fuite de données ?

## 🧭 Méthodologie

```
Données (bandes B1-B10, indices, textures)
   │
   ├─▶ EDA : skewness / kurtosis, distribution des classes, corrélations Pearson / Spearman
   ├─▶ Séparation stratifiée 80 / 20 : le jeu de test est gelé jusqu'à l'évaluation finale
   ├─▶ 5 scénarios de variables (S1 à S5)
   ├─▶ Validation croisée stratifiée 5 plis, avec normalisation et SMOTE DANS chaque pli
   ├─▶ Comparaison Random Forest · SVM · XGBoost
   ├─▶ Évaluation finale sur le jeu de test
   └─▶ Tests statistiques (McNemar, Wilcoxon, Friedman, Nemenyi) + SHAP
```

**Point de rigueur :** la normalisation (`StandardScaler`) et le rééquilibrage (`SMOTE`) sont appliqués **uniquement sur les données d'entraînement de chaque pli**, pour éviter toute fuite de données (*data leakage*).

### Les 5 scénarios de variables

| Scénario | Variables | Accuracy (CV) |
|---|---|---|
| S1 | 10 bandes spectrales brutes | 0,486 |
| S2 | Top-k bandes (importance Random Forest) | 0,466 |
| S3 | Sélection Boruta | 0,486 |
| **S4** | **Bandes + indices spectraux (NDVI, EVI, SAVI, NDRE…)** | **0,505** |
| S5 | Bandes + indices + textures (GLCM, Gabor, LBP) | 0,488 |

➡️ **S4 retenu** : les indices spectraux apportent un vrai gain, alors que les textures ajoutent surtout du bruit.

## 📈 Résultats

**Comparaison des modèles sur S4 (validation croisée)**

| Modèle | Accuracy | Kappa |
|---|---|---|
| **XGBoost** | **0,518** | **0,258** |
| Random Forest | 0,505 | 0,253 |
| SVM | 0,318 | — |

**Validation statistique**

| Test | Résultat |
|---|---|
| McNemar (XGBoost vs SVM, jeu de test) | p = 0,006 → XGBoost significativement meilleur |
| Friedman (3 modèles) | p = 0,007 → différences significatives |
| Nemenyi (post-hoc) | XGBoost vs SVM : p = 0,004 · XGBoost vs Random Forest : non significatif |
| Wilcoxon (5 plis) | p = 0,063 → manque de puissance avec seulement 5 plis |

**Lecture des résultats**

- ✅ Très bonne reconnaissance de l'**eau** (F1 = 0,80), grâce à sa signature infrarouge nette.
- ⚠️ La classe majoritaire « attire » une partie des pixels des autres classes, malgré SMOTE.
- ❌ Le **sol nu** (8 pixels réels) n'est jamais reconnu : SMOTE ne peut pas créer une classe à partir de si peu d'exemples.
- 🔎 **SHAP** : la bande B10 et les indices **NDRE** et **SAVI** dominent la décision, ce qui est cohérent physiquement (sensibilité à la chlorophylle, correction de l'effet du sol).

> Les scores restent modestes (≈ 52 %) : le jeu de données est petit, très déséquilibré, et certaines classes ont des signatures spectrales très proches. L'intérêt du projet est la **rigueur du protocole** et l'analyse honnête des limites.

## 🧠 Perspectives

- Collecter davantage de pixels pour les classes rares
- Optimiser les hyperparamètres de XGBoost (recherche par grille dans la validation croisée)
- Tester des **Transformers tabulaires** (TabNet, FT-Transformer) ou des approches spatiales (CNN, segmentation objet)

## 📁 Structure du dépôt

```
ml-classification-occupation-sol/
├── notebook/
│   └── classification_occupation_sol.ipynb   # Pipeline complet
├── rapport/
│   └── rapport_scientifique.pdf               # Rapport de 22 pages
└── README.md
```

> ℹ️ Le jeu de données a été fourni dans le cadre du cours et n'est pas publié ici. Pour exécuter le notebook, placer le fichier de données à côté du notebook.

## ▶️ Utilisation

```bash
git clone https://github.com/itraore05/ml-classification-occupation-sol.git
pip install pandas numpy scikit-learn imbalanced-learn xgboost boruta shap scikit-posthocs statsmodels scipy matplotlib seaborn openpyxl
```

Ouvrir le notebook dans Jupyter ou Google Colab et exécuter les cellules dans l'ordre.

## 💡 Compétences démontrées

- Gestion d'un **déséquilibre de classes extrême** (SMOTE, métriques adaptées : Kappa, F1 par classe)
- **Sélection de variables** : Top-k, Boruta, comparaison de scénarios
- Protocole d'évaluation **sans fuite de données** (pipeline par pli, jeu de test gelé)
- **Tests statistiques** de comparaison de modèles
- **Interprétabilité** globale et locale avec SHAP, et lecture physique des résultats

## 👤 Auteur

**Ismaël Traoré** — Développeur Data & IA
📧 contact.traoreimael@gmail.com · 💼 [LinkedIn](https://www.linkedin.com/in/ismael-traor%C3%A9-m-4007a2225/)
