# Beruch Consulting — Mission #008 : AgriCo

 Construction d'un modèle de régression sous Python (scikit-learn) pour prédire le rendement agricole, et identifier les leviers actionnables (engrais, irrigation) les plus efficaces pour l'améliorer.
Cas fictif réalisé dans le cadre d'un défi data portfolio.


## À propos

Beruch Consulting accompagne les entreprises et organisations dans leur transition vers une culture data-driven, en transformant leurs données brutes en décisions stratégiques.
De la donnée à la décision.

Chaque projet de ce portfolio est présenté sous forme de mission client : un contexte, une problématique, des analyses et des recommandations concrètes.


## Le client : AgriCo

AgriCo (coopérative agritech fictive) souhaite fournir à ses agriculteurs un outil qui, d'une part, prédit le rendement attendu selon les conditions de culture, et d'autre part, identifie les leviers actionnables (engrais, irrigation) les plus efficaces pour l'améliorer afin d'optimiser à la fois la production et l'usage des ressources.

## Problématique

> À partir des conditions de culture (météo, sol, type de culture, pratiques), peut-on prédire le rendement agricole de façon fiable, et quels facteurs actionnables l'influencent le plus ?

La mission distingue deux familles de facteurs : les facteurs subis (météo, sol, culture), qui décrivent le contexte, et les facteurs actionnables(engrais, irrigation), qui seuls donnent lieu à des recommandations concrètes.

## Données

- Source : jeu de données public de rendements agricoles (type Kaggle), utilisé pour simuler le cas AgriCo.
- 1 000 000 de parcelles, 10 variables, aucune valeur manquante.
- Variable cible : Yield_tons_per_hectare (rendement en tonnes par hectare, moyenne ~4,65).
- Variables explicatives : région, type de sol, culture, pluviométrie (Rainfall_mm), température, condition météo, durée jusqu'à la récolte, et deux leviers booléens : usage d'engrais (Fertilizer_Used) et d'irrigation (Irrigation_Used).

## Outils & compétences mobilisées

| Domaine | Détail |
| Langage | Python (pandas, seaborn, matplotlib) |
| Machine learning | scikit-learn |
| Préparation | Conversion des leviers booléens en 0/1, pipeline de préprocessing (imputation, standardisation, encodage one-hot via ColumnTransformer) |
| Modélisation | Régression linéaire et Random Forest, séparation entraînement/test, sélection de modèle |
| Évaluation | R² et erreur absolue moyenne (MAE) |
| Livrable | Fonction de prédiction du rendement, chiffrant le gain d'activation des leviers |

## Démarche & analyses

### Exploration

La pluviométrie (Rainfall_mm) ressort comme le facteur numérique dominant (corrélation 0,77 avec le rendement), devant la température (0,09) et la durée de récolte (~0, sans effet). Mais la pluie est un facteur subi : elle décrit le contexte, pas un levier d'action.

L'analyse des deux leviers actionnables révèle des effets nets sur le rendement moyen :

| Levier | Sans | Avec | Gain |
| Engrais | 3,9 | 5,4 | +1,5 t/ha |
| Irrigation | 4,05 | 5,25 | +1,2 t/ha |

Combinés, les deux leviers font presque doubler le rendement moyen : de 3,3 t/ha (aucun des deux) à 6,0 t/ha (les deux).

### Un diagnostic déterminant

Une première version du modèle plafonnait à un R² de 0,59. En cause : les deux leviers, Fertilizer_Used et Irrigation_Used, étant de type booléen, n'étaient captés ni par la sélection des variables catégorielles ni par celle des numériques, et se trouvaient donc exclus du modèle alors qu'ils constituent le cœur de la mission. Après conversion de ces booléens en 0/1, la performance a bondi :

| Configuration | R² (test) | Erreur moyenne (MAE) |
| Sans les leviers | 0,591 | 0,89 t/ha |
| Avec les leviers | 0,913 | 0,40 t/ha |

Enseignement : la performance d'un modèle dépend d'abord de la pertinence des variables retenues, et les données booléennes demandent une attention particulière lors de la préparation.

### Modélisation et sélection

| Modèle | R² (test) | Observation |
| Régression linéaire | 0,913 | Retenue |
| Random Forest | 0,908 | Surapprend nettement (R² train 0,99) |

Le modèle retenu est la régression linéaire : performance équivalente au Random Forest mais sans surapprentissage, plus simple et plus interprétable. Le rendement dépendant des variables de façon largement linéaire, un modèle complexe n'apporte rien.

### Interprétation : facteurs subis vs leviers actionnables

Les coefficients (lus en unités réelles) confirment l'exploration :
- Pluie : coefficient faible par mm (~0,005) mais dominant par son amplitude (0 à 1 000 mm). Facteur n°1, mais subi.
- Engrais : +1,5 t/ha (levier actionnable le plus puissant).
- Irrigation : +1,2 t/ha (deuxième levier).
- Sol, culture, région, condition météo, durée : coefficients quasi nuls, aucun effet notable.

Les leviers étant codés 0/1, leur coefficient se lit directement comme le gain de rendement associé.

### Outil de prédiction

Une fonction predire_rendement() prend les conditions d'une parcelle et les choix de l'agriculteur (engrais ? irrigation ?) et renvoie le rendement estimé. Sur une parcelle type, elle chiffre l'effet des leviers : de 3,25 t/ha sans engrais ni irrigation à 5,95 t/ha avec les deux, soit un gain de +2,7 t/ha (+83 %). L'outil ne se contente pas de prédire : il quantifie le bénéfice d'activer les bons leviers.

## Recommandations pour AgriCo

1. Déployer l'outil de prédiction pour estimer, parcelle par parcelle, le rendement attendu et le gain potentiel des leviers.
2. Prioriser l'engrais (gain le plus élevé, +1,5 t/ha), puis l'irrigation (+1,2 t/ha).
3. Communiquer le gain combiné (quasi doublement du rendement) auprès des agriculteurs pour justifier l'investissement dans ces pratiques.
4. Ajuster les attentes selon la pluviométrie : la pluie domine le rendement mais reste subie ; les leviers sont ce sur quoi l'agriculteur peut réellement agir.


## Limites des données et du modèle

- Facteurs contextuels sans effet : le type de sol, la culture, la région et la condition météo n'ont aucune influence notable sur le rendement, ce qui est contre-intuitif (dans la réalité, la culture et le sol sont déterminants). Indice de données en partie synthétiques.
- Rendements négatifs : le jeu contient des valeurs de rendement négatives, physiquement impossibles  autre artefact de génération artificielle.
- Hypothèse de linéarité : la régression suppose des effets linéaires ; un système agronomique réel comporte des effets de seuil et des interactions (par exemple, l'engrais peut être moins efficace en cas de sécheresse) qu'un modèle linéaire ne capte pas.


## Contenu du dépôt

beruch-consulting-mission-008/
├── README.md                          ce fichier
├── mission_008_AgriCo.ipynb            le notebook (exploration → modèle → outil de prédiction)



## Auteur

Beruch Ibala
Projet réalisé dans le cadre d'un défi data portfolio (30 jours).


Données : jeu de données public de rendements agricoles. Cas client fictif à but pédagogique.
