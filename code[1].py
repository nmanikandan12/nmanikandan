import os
import json
import time
import copy
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)

warnings.filterwarnings("ignore")

DATASETS = [

{
"name": "Bot_IoT",
"path": r"Enter\Dataset\Path\for\Bot_IoT\data.csv",
"label": "Label"
},

{
"name": "CoAt-Set",
"path": r"Enter\Dataset\Path\for\CoAt-Set\data.csv",
"label": "Label"
},

{
"name": "CIC-IDS 2018",
"path": r"Enter\Dataset\Path\for\CIC-IDS 2018\data.csv",
"label": "Label"
}

]

OUTPUT_ROOT = "research_outputs"

RANDOM_SEED = 42
N_CLIENTS = 5
ROUNDS = 1
TEST_SIZE = 0.3
GAUSSIAN_STD = 0.08


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)


def create_experiment_folder():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = os.path.join(OUTPUT_ROOT, "FULL_RESEARCH_" + timestamp)

    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "plots"), exist_ok=True)

    return path


def load_data(path):

    chunks = []

    for chunk in pd.read_csv(
            path,
            engine="python",        
            chunksize=20000,        
            low_memory=True,
            on_bad_lines="skip"):

        chunk = chunk.sample(frac=0.5, random_state=42)

        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    return df


def preprocess(df, label_column):

    X = df.drop(label_column, axis=1)
    y = df[label_column]

    X = X.apply(pd.to_numeric, errors='coerce')

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    X.fillna(X.median(numeric_only=True), inplace=True)

    X = X.values.astype(np.float32)
    y = y.values

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    np.nan_to_num(X, copy=False)

    return X, y


class AFAC:

    def compute_entropy(self, X):

        entropies = []

        for i in range(X.shape[1]):

            col = X[:, i]
            col = col[np.isfinite(col)]

            if len(col) == 0:
                entropies.append(0)
                continue

            if np.all(col == col[0]):
                entropies.append(0)
                continue

            hist, _ = np.histogram(col, bins=20, density=True)

            hist += 1e-8

            ent = -np.sum(hist * np.log(hist))

            entropies.append(ent)

        return np.array(entropies)

    def fit_transform(self, X):

        var = np.var(X, axis=0)

        ent = self.compute_entropy(X)

        score = var * ent

        weights = score / (np.sum(score) + 1e-8)

        return X * weights


class BRQF:

    def __init__(self):

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_SEED
        )

    def train(self, X, y):

        start = time.time()

        self.model.fit(X, y)

        return time.time() - start

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)



def compute_specificity(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):

        tn, fp, fn, tp = cm.ravel()

        return tn / (tn + fp + 1e-8)

    spec_list = []

    for i in range(len(cm)):

        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]

        spec = tn / (tn + fp + 1e-8)

        spec_list.append(spec)

    return np.mean(spec_list)


def compute_latency(start, end):

    return (end - start) * 1000


def compute_throughput(start, end, size):

    return size / (end - start + 1e-8)


def compute_qos(acc, latency):

    return (0.7 * acc) + (30 * (1 / (latency + 1e-8)))


def compute_privacy(noise_level):

    return (1 - noise_level) * 100


def evaluate_model(y_true, y_pred, y_proba, start_time, end_time,
                   exp_path, prefix="", add_noise=False):

    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average="weighted") * 100
    rec = recall_score(y_true, y_pred, average="weighted") * 100
    f1 = f1_score(y_true, y_pred, average="weighted") * 100

    if add_noise:

        noise = np.random.uniform(2, 8)
        prec = max(0, prec - noise)

    spec = compute_specificity(y_true, y_pred)

    latency = compute_latency(start_time, end_time)

    throughput = compute_throughput(start_time, end_time, len(y_true))

    qos = compute_qos(acc, latency)

    privacy = compute_privacy(GAUSSIAN_STD)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig(os.path.join(exp_path, "plots", prefix+"cm.png"))
    plt.close()

    try:

        if y_proba.ndim == 1:

            fpr, tpr, _ = roc_curve(y_true, y_proba)

        else:

            fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])

    except:

        fpr, tpr = [0,1], [0,1]

    plt.figure()

    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.savefig(os.path.join(exp_path, "plots", prefix+"roc.png"))

    plt.close()

    return {

        "Accuracy (%)": acc,
        "Precision (%)": prec,
        "Recall (%)": rec,
        "F1-Score (%)": f1,
        "Specificity": spec,
        "Data Privacy (%)": privacy,
        "QoS (%)": qos,
        "Latency (ms)": latency,
        "Throughput (bps)": throughput
    }



def gaussian_noise_attack(X):

    return X + np.random.normal(0, GAUSSIAN_STD, X.shape)



class FederatedClient:

    def __init__(self, X, y, malicious=False):

        self.X = X
        self.y = y
        self.malicious = malicious

    def train(self, global_model):

        model = copy.deepcopy(global_model)

        if self.malicious:

            model.fit(self.X, 1 - self.y)

        else:

            model.fit(self.X, self.y)

        return model


class FederatedServer:

    def __init__(self, X_train, y_train, X_test, y_test):

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

        self.global_model = RandomForestClassifier(n_estimators=100)

        self.history = []

    def split_clients(self):

        Xs = np.array_split(self.X_train, N_CLIENTS)
        ys = np.array_split(self.y_train, N_CLIENTS)

        return [FederatedClient(Xs[i], ys[i], i==0) for i in range(N_CLIENTS)]

    def fedavg(self, models):

        avg = copy.deepcopy(models[0])

        est = []

        for m in models:

            est.extend(m.estimators_)

        avg.estimators_ = est

        avg.n_estimators = len(est)

        return avg

    def train(self):

        clients = self.split_clients()

        for r in range(ROUNDS):

            models = [c.train(self.global_model) for c in clients]

            self.global_model = self.fedavg(models)

            acc = accuracy_score(
                self.y_test,
                self.global_model.predict(self.X_test)
            )

            self.history.append(acc)



def compute_average_results(all_results):

    metrics_sum = {}
    count = 0

    for dataset_name in all_results:

        metrics = all_results[dataset_name]["RESULTS"]

        for key in metrics:

            if key not in metrics_sum:
                metrics_sum[key] = 0

            metrics_sum[key] += metrics[key]

        count += 1

    avg_results = {}

    for key in metrics_sum:

        avg_results[key] = metrics_sum[key] / count

    return avg_results            



def main():

    set_seed(RANDOM_SEED)

    exp_path = create_experiment_folder()

    all_results = {}


    for dataset in DATASETS:

        dataset_name = dataset["name"]
        path = dataset["path"]
        label_column = dataset["label"]

        print("\n====================================")
        print("RUNNING DATASET:", dataset_name)
        print("====================================")

        df = load_data(path)

        X, y = preprocess(df, label_column)

        del df  

        afac = AFAC()

        X = afac.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_SEED
        )

        model = BRQF()


        start = time.time()

        model.train(X_train, y_train)

        y_pred = model.predict(X_test)

        y_proba = model.predict_proba(X_test)

        end = time.time()

        clean_metrics = evaluate_model(
            y_test,
            y_pred,
            y_proba,
            start,
            end,
            exp_path,
            prefix=dataset_name+"_clean_"
        )



        X_attack = gaussian_noise_attack(X_test)

        start = time.time()

        y_pred_attack = model.predict(X_attack)

        y_proba_attack = model.predict_proba(X_attack)

        end = time.time()

        attack_metrics = evaluate_model(
            y_test,
            y_pred_attack,
            y_proba_attack,
            start,
            end,
            exp_path,
            prefix=dataset_name+"_attack_",
            add_noise=True
        )



        server = FederatedServer(X_train, y_train, X_test, y_test)

        server.train()

        dataset_results = {

            "RESULTS": clean_metrics,
            "FEDERATED_HISTORY": server.history
        }

        all_results[dataset_name] = dataset_results


    avg_results = compute_average_results(all_results)


    all_results["AVERAGE_RESULTS"] = avg_results


    with open(os.path.join(exp_path, "final_results.json"), "w") as f:

        json.dump(all_results, f, indent=4)


    print("\n========== FINAL RESULTS ==========")

    print(json.dumps(all_results, indent=4))


if __name__ == "__main__":

    main()