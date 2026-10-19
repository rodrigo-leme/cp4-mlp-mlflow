import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Variável alvo: 1 = sobreviveu, 0 = não sobreviveu
TARGET = "Survived"

# Features selecionadas (Name, Ticket, Cabin e PassengerId foram descartadas:
# são identificadores/texto livre e Cabin tem 77% de valores ausentes)
FEATURES_NUMERICAS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
FEATURES_CATEGORICAS = ["Sex", "Embarked"]
FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS


def carregar_dados(caminho="data/titanic.csv"):
    df = pd.read_csv(caminho)
    X = df[FEATURES]
    y = df[TARGET]
    # 80% treino / 20% teste, mantendo a proporção das classes
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def criar_preprocessador():
    # Numéricas: preenche ausentes com a mediana e padroniza (média 0, desvio 1)
    numericas = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categóricas: preenche ausentes com o valor mais frequente e aplica One-Hot
    categoricas = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numericas, FEATURES_NUMERICAS),
        ("cat", categoricas, FEATURES_CATEGORICAS),
    ])
