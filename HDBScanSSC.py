from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import gc 

# from cuml.svm import SVC  # Versão GPU-aware do SVC
# from cuml.ensemble import RandomForestClassifier  # Random Forest na GPU



# Importação das bibliotecas necessárias
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from scipy.sparse.csgraph import laplacian
from scipy.linalg import solve
# from google.colab import drive
from scipy import stats


from collections import Counter,defaultdict

# Célula 1: Importações necessárias
import math
import numpy as np
from scipy.spatial import cKDTree
from sklearn.datasets import load_iris
import sys


module_path1 = 'D:/TCC/HDBScanSSC' # Ajuste este caminho se necessário
if module_path1 not in sys.path:
    sys.path.append(module_path1)
    print(f"Caminho '{module_path1}' adicionado ao sys.path.")
else:
    print(f"Caminho '{module_path1}' já está no sys.path.")


# Importar a classe SuperHeap do arquivo super_heap.py
from super_heap import SuperHeap ,NodeObject#

# Importar a classe HDBScanSSC do arquivo ss_hdbscan.py
from ss_hdbscan import HDBScanSSC #


import warnings



#carregamento de dados 
folder_path = "D:/TCC/Dados"
datasets = [
    "ACE.data",
    "ACHE.data",
    "AT1.data",
    "BBB.data",
    "BZR.data",
    "COX2.data",
    "DHFR.data",
    "EP2.data",
    "errba.data",
    "FONTAINE.data",
    "GPB.data",
    "GTPase.data",
    "M1.data",
    "MIC.data",
    "PPARD121.data",
    "TGFB.data",
    "THERM.data",
    "THR.data",
    "ttr.data"

]



# Definições de parâmetros do experimento
label_percents = [0.05, 0.10, 0.15]  # Percentuais de rótulos
n_repetitions = 1 # Número de repetições do experimento
n_label_selections = 20  # Número de seleções de rótulos

# Dicionário para armazenar resultados
results = {}
# Definindo o K-Fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True, random_state=42)  # Defina o número de folds e outros parâmetros




# Coloque este código no início do seu notebook ou script
warnings.filterwarnings('ignore', category=FutureWarning)
results_hdbscan_ssc = {} # Dicionário para armazenar os resultados apenas para HDBScanSSC

# Código principal
for dataset_name in datasets:
    file_path = f"{folder_path}/{dataset_name}"
    df = pd.read_csv(file_path, header=None)

    # Separação de features (X) e rótulos (y)
    X = df.iloc[:, :-1].values
    y_labels, y = np.unique(df.iloc[:, -1], return_inverse=True)

    # Normalização dos dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    dataset_results_hdbscan_ssc = {p: [] for p in label_percents}

    for rep in range(n_repetitions):
        print(f"Executando {dataset_name} - Repetição {rep + 1}/{n_repetitions}")

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            print(f"  Fold {fold + 1}/{kfold.n_splits}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for label_percent in label_percents:
                num_labeled = int(label_percent * len(y_train))

                # Listas para armazenar as acurácias do HDBScanSSC e dos classificadores pós-HDBScanSSC
                acc_hdbscan_ssc_list = []
                acc_svm_hdbscan_ssc_list = []
                acc_rf_hdbscan_ssc_list = []

                for sel in range(n_label_selections):
                    # Inicializa todos os rótulos como não rotulados (-1)
                    # O HDBScanSSC espera -1 para pontos não rotulados
                    labels_for_hdbscan_ssc = np.full(y_train.shape, -1, dtype=int)
                    unique_classes = np.unique(y_train)
                    labeled_indices = []

                    # Garante pelo menos 2 exemplos por classe para rotulagem
                    for cls in unique_classes:
                        class_indices = np.where(y_train == cls)[0]
                        if len(class_indices) >= 2:
                            selected_indices = np.random.choice(class_indices, size=2, replace=False)
                            labeled_indices.extend(selected_indices)
                        elif len(class_indices) == 1: # Se tiver apenas 1 exemplo na classe, pegue-o
                            labeled_indices.extend(class_indices)

                    # Seleciona os exemplos restantes aleatoriamente
                    remaining_indices = list(set(range(len(y_train))) - set(labeled_indices))
                    num_remaining = num_labeled - len(labeled_indices)
                    if num_remaining > 0 and len(remaining_indices) > 0:
                        # Certifica-se de não tentar selecionar mais do que o disponível
                        selected_remaining = np.random.choice(
                            remaining_indices,
                            size=min(num_remaining, len(remaining_indices)),
                            replace=False
                        )
                        labeled_indices.extend(selected_remaining)

                    # Atribui os rótulos conhecidos aos índices selecionados
                    labels_for_hdbscan_ssc[labeled_indices] = y_train[labeled_indices].astype(int)

                    # --- Execução do HDBScanSSC ---
                    hdbscan_ssc = HDBScanSSC(min_cluster_size=5) # min_cluster_size é um hiperparâmetro do HDBSCAN, pode precisar de ajuste
                    hdbscan_ssc.fit(X=X_train, y=labels_for_hdbscan_ssc)

                    # --- NOVO CÁLCULO PARA HDBScanSSC (direto) ---
                    # Calcula a acurácia da transdução no conjunto de treino
                    # Compara os rótulos transdutados (hdbscan_ssc.transduction_) com os rótulos verdadeiros (y_train)
                    # Isso mede a "qualidade" da propagação de rótulos dentro do conjunto de treino.
                    # Ignora pontos que podem ter permanecido como -1 na transdução se hdbscan não conseguiu rotulá-los
                    # Ou você pode escolher um tratamento diferente para -1 se for ruído ou classe própria
                    valid_transduced_indices = hdbscan_ssc.transduction_ != -1
                    if np.any(valid_transduced_indices):
                        acc_hdbscan_ssc = accuracy_score(
                            y_train[valid_transduced_indices],
                            hdbscan_ssc.transduction_[valid_transduced_indices]
                        )
                    else:
                        acc_hdbscan_ssc = np.nan # Se nenhum ponto foi rotulado ou todos são -1

                    # Treinar SVM e RF com os rótulos transdutados pelo HDBScanSSC
                    predicted_labels_hdbscan_ssc_train = hdbscan_ssc.transduction_

                    # Verifica se o HDBScanSSC produziu mais de uma classe para evitar erros no fit do SVM/RF
                    if len(np.unique(predicted_labels_hdbscan_ssc_train)) > 1:
                        svm_hdbscan_ssc = SVC(random_state=42)
                        rf_hdbscan_ssc = RandomForestClassifier(random_state=42)

                        svm_hdbscan_ssc.fit(X_train, predicted_labels_hdbscan_ssc_train)
                        rf_hdbscan_ssc.fit(X_train, predicted_labels_hdbscan_ssc_train)

                        acc_svm_hdbscan_ssc = accuracy_score(y_test, svm_hdbscan_ssc.predict(X_test))
                        acc_rf_hdbscan_ssc = accuracy_score(y_test, rf_hdbscan_ssc.predict(X_test))
                    else:
                        acc_svm_hdbscan_ssc = np.nan
                        acc_rf_hdbscan_ssc = np.nan

                    print(f"  Seleção {sel + 1}/{n_label_selections} - HDBScanSSC (transdução): {acc_hdbscan_ssc:.4f}, SVM pós-HDBScanSSC: {acc_svm_hdbscan_ssc:.4f}, RF pós-HDBScanSSC: {acc_rf_hdbscan_ssc:.4f}")

                    acc_hdbscan_ssc_list.append(acc_hdbscan_ssc)
                    acc_svm_hdbscan_ssc_list.append(acc_svm_hdbscan_ssc)
                    acc_rf_hdbscan_ssc_list.append(acc_rf_hdbscan_ssc)
                    
                    # NOVO: Excluir explicitamente objetos grandes e executar a coleta de lixo
                    del labels_for_hdbscan_ssc
                    del hdbscan_ssc
                    if 'svm_hdbscan_ssc' in locals():
                        del svm_hdbscan_ssc
                    if 'rf_hdbscan_ssc' in locals():
                        del rf_hdbscan_ssc
                    gc.collect() # Força a coleta de lixo

                # Adiciona os resultados médios para cada percentual de rótulos
                dataset_results_hdbscan_ssc[label_percent].append((
                    np.nanmean(acc_hdbscan_ssc_list),
                    np.nanmean(acc_svm_hdbscan_ssc_list),
                    np.nanmean(acc_rf_hdbscan_ssc_list)
                ))

    results_hdbscan_ssc[dataset_name] = {p: np.nanmean(dataset_results_hdbscan_ssc[p], axis=0) for p in label_percents}

# Exibição dos resultados finais para HDBScanSSC
print("\n--- Resultados Finais (Apenas HDBScanSSC e Modelos Pós-HDBScanSSC) ---")
for dataset, res in results_hdbscan_ssc.items():
    print(f"Dataset: {dataset}")
    for p, (acc_hdbscan_ssc_mean, acc_svm_hdbscan_ssc_mean, acc_rf_hdbscan_ssc_mean) in res.items():
        print(f"  {p*100:.0f}% rotulado -> HDBScanSSC (transdução): {acc_hdbscan_ssc_mean:.4f}, SVM pós-HDBScanSSC: {acc_svm_hdbscan_ssc_mean:.4f}, RF pós-HDBScanSSC: {acc_rf_hdbscan_ssc_mean:.4f}")
    print()