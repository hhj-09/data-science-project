
analysis.py
100%
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TITLE = "UR3 그리퍼 파지 손실 분석"
SOURCE = "https://archive.ics.uci.edu/dataset/963/ur3+cobotops"
FEATURES = [f"{kind}_J{i}" for kind in ("Current", "Temperature", "Speed") for i in range(6)] + ["Tool_current"]
TARGET = "grip_lost"
LR = "로지스틱 회귀(확률로 답하는 모델)"
TREE = "의사결정트리(질문으로 답하는 모델)"


def load_data():
    path = Path(__file__).parent / "data" / "dataset_02052023.xlsx"
    data = pd.read_excel(path, sheet_name="data")
    data.columns = data.columns.str.strip()
    data = data.rename(columns={"Temperature_T0": "Temperature_J0"})
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    data["Timestamp"] = pd.to_datetime(data["Timestamp"].str.strip('"'), utc=True, format="mixed")
    return data


def complete_rows(data):
    return data.dropna(subset=FEATURES + [TARGET]).copy()


def split_cycles(data):
    starts = data.groupby("cycle").Timestamp.min().sort_values()
    train_cycles = starts.index[:int(len(starts) * 0.8)]
    mask = data.cycle.isin(train_cycles)
    return data.loc[mask], data.loc[~mask]


def evaluate(data):
    train, test = split_cycles(data)
    models = {
        LR: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        TREE: DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42),
    }
    scores, matrices = {}, {}
    for name, model in models.items():
        model.fit(train[FEATURES], train[TARGET])
        prediction = model.predict(test[FEATURES])
        truth = test[TARGET]
        scores[name] = {
            "정확도": accuracy_score(truth, prediction),
            "정밀도": precision_score(truth, prediction, zero_division=0),
            "재현율": recall_score(truth, prediction, zero_division=0),
            "F1": f1_score(truth, prediction, zero_division=0),
        }
        matrices[name] = confusion_matrix(truth, prediction, labels=[False, True])
    return pd.DataFrame.from_dict(scores, orient="index"), matrices
analysis.py 표시 중입니다.
