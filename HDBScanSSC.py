from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import gc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from scipy.sparse.csgraph import laplacian
from scipy.linalg import solve
from scipy import stats
from collections import Counter, defaultdict
import math
import sys
import warnings

# Importação de módulos externos (presume-se que os arquivos 'super_heap.py' e 'ss_hdbscan.py' estão disponíveis)
module_path1 = './HDBScanSSC'
if module_path1 not in sys.path:
    sys.path.append(module_path1)
    print(f"Caminho '{module_path1}' adicionado ao sys.path.")
else:
    print(f"Caminho '{module_path1}' já está no sys.path.")

from super_heap import SuperHeap, NodeObject
from ss_hdbscan import HDBScanSSC

warnings.filterwarnings('ignore', category=FutureWarning)

# Definições de parâmetros do experimento
label_percents = [0.05, 0.10, 0.15]
n_repetitions = 1
n_label_selections = 20
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Dicionários para armazenar resultados
results_hdbscan_ssc = {}

# Código principal
folder_path = "./Dados"
datasets = [
    "BBB.data", "EP2.data", "ACE.data", "ACHE.data", "AT1.data", "BZR.data",
    "COX2.data", "DHFR.data", "errba.data", "FONTAINE.data", "GPB.data",
    "GTPase.data", "M1.data", "MIC.data", "PPARD121.data", "TGFB.data",
    "THERM.data", "THR.data", "ttr.data"
]

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
                acc_hdbscan_ssc_list = []
                acc_svm_hdbscan_ssc_list = []
                acc_rf_hdbscan_ssc_list = []

                for sel in range(n_label_selections):
                    labels_for_hdbscan_ssc = np.full(y_train.shape, -1, dtype=int)
                    unique_classes = np.unique(y_train)
                    labeled_indices = []

                    for cls in unique_classes:
                        class_indices = np.where(y_train == cls)[0]
                        if len(class_indices) >= 2:
                            selected_indices = np.random.choice(class_indices, size=2, replace=False)
                            labeled_indices.extend(selected_indices)
                        elif len(class_indices) == 1:
                            labeled_indices.extend(class_indices)

                    remaining_indices = list(set(range(len(y_train))) - set(labeled_indices))
                    num_remaining = num_labeled - len(labeled_indices)
                    if num_remaining > 0 and len(remaining_indices) > 0:
                        selected_remaining = np.random.choice(
                            remaining_indices,
                            size=min(num_remaining, len(remaining_indices)),
                            replace=False
                        )
                        labeled_indices.extend(selected_remaining)

                    labels_for_hdbscan_ssc[labeled_indices] = y_train[labeled_indices].astype(int)
                    hdbscan_ssc = HDBScanSSC(min_cluster_size=5)
                    hdbscan_ssc.fit(X=X_train, y=labels_for_hdbscan_ssc)

                    valid_transduced_indices = hdbscan_ssc.transduction_ != -1
                    if np.any(valid_transduced_indices):
                        acc_hdbscan_ssc = accuracy_score(
                            y_train[valid_transduced_indices],
                            hdbscan_ssc.transduction_[valid_transduced_indices]
                        )
                    else:
                        acc_hdbscan_ssc = np.nan

                    predicted_labels_hdbscan_ssc_train = hdbscan_ssc.transduction_

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

                    # --- Limpeza de variáveis dentro do loop de seleção (sel) ---
                    vars_to_delete_inner = ['labels_for_hdbscan_ssc', 'hdbscan_ssc']

                    # Adiciona os modelos à lista se eles foram criados
                    if 'svm_hdbscan_ssc' in locals():
                        vars_to_delete_inner.append('svm_hdbscan_ssc')
                    if 'rf_hdbscan_ssc' in locals():
                        vars_to_delete_inner.append('rf_hdbscan_ssc')
                    
                    for var in vars_to_delete_inner:
                        if var in locals():
                            del locals()[var]
                    gc.collect()

                # Adiciona os resultados médios para cada percentual de rótulos
                dataset_results_hdbscan_ssc[label_percent].append((
                    np.nanmean(acc_hdbscan_ssc_list),
                    np.nanmean(acc_svm_hdbscan_ssc_list),
                    np.nanmean(acc_rf_hdbscan_ssc_list)
                ))
            
            # --- Limpeza de variáveis no final do fold ---
            vars_to_delete_fold = ['X_train', 'X_test', 'y_train', 'y_test']
            for var in vars_to_delete_fold:
                if var in locals():
                    del locals()[var]
            gc.collect()

    results_hdbscan_ssc[dataset_name] = {p: np.nanmean(dataset_results_hdbscan_ssc[p], axis=0) for p in label_percents}

    # --- Limpeza final de variáveis por dataset ---
    vars_to_delete_dataset = ['df', 'X', 'y_labels', 'y', 'scaler', 'dataset_results_hdbscan_ssc']
    for var in vars_to_delete_dataset:
        if var in locals():
            del locals()[var]
    gc.collect()


# Exibição dos resultados finais para HDBScanSSC
print("\n--- Resultados Finais (Apenas HDBScanSSC e Modelos Pós-HDBScanSSC) ---")
for dataset, res in results_hdbscan_ssc.items():
    print(f"Dataset: {dataset}")
    for p, (acc_hdbscan_ssc_mean, acc_svm_hdbscan_ssc_mean, acc_rf_hdbscan_ssc_mean) in res.items():
        print(f"  {p*100:.0f}% rotulado -> HDBScanSSC (transdução): {acc_hdbscan_ssc_mean:.4f}, SVM pós-HDBScanSSC: {acc_svm_hdbscan_ssc_mean:.4f}, RF pós-HDBScanSSC: {acc_rf_hdbscan_ssc_mean:.4f}")
    print()


# Salvando resultados
output_filename = 'experiment_results_hdbscan_ssc.txt'
with open(output_filename, 'w') as f:
    for dataset, res in results_hdbscan_ssc.items():
        print(f"Dataset: {dataset}")
        f.write(f"Dataset: {dataset}\n")
        for p, (acc_hdbscan_ssc_mean, acc_svm_hdbscan_ssc_mean, acc_rf_hdbscan_ssc_mean) in res.items():
            print(f"  {p*100:.0f}% rotulado -> HDBScanSSC (transdução): {acc_hdbscan_ssc_mean:.4f}, SVM pós-HDBScanSSC: {acc_svm_hdbscan_ssc_mean:.4f}, RF pós-HDBScanSSC: {acc_rf_hdbscan_ssc_mean:.4f}")
            f.write(f"  {p*100:.0f}% rotulado -> HDBScanSSC (transdução): {acc_hdbscan_ssc_mean:.4f}, SVM pós-HDBScanSSC: {acc_svm_hdbscan_ssc_mean:.4f}, RF pós-HDBScanSSC: {acc_rf_hdbscan_ssc_mean:.4f}\n")
        print()
        f.write("\n")
print(f"Results saved to {output_filename}")


# Salvando o plot dos resultados
plot_data = []
for dataset, res in results_hdbscan_ssc.items():
    for p, accuracies_list in dataset_results_hdbscan_ssc.items():
        for accuracies_fold in accuracies_list:
            plot_data.append({
                'Dataset': dataset,
                'Label Percentage': f'{p*100:.0f}%',
                'Metric': 'HDBScanSSC (Transdução)',
                'Accuracy': accuracies_fold[0]
            })
            plot_data.append({
                'Dataset': dataset,
                'Label Percentage': f'{p*100:.0f}%',
                'Metric': 'SVM pós-HDBScanSSC',
                'Accuracy': accuracies_fold[1]
            })
            plot_data.append({
                'Dataset': dataset,
                'Label Percentage': f'{p*100:.0f}%',
                'Metric': 'RF pós-HDBScanSSC',
                'Accuracy': accuracies_fold[2]
            })

plot_df = pd.DataFrame(plot_data)

# Criar os box plots
plt.figure(figsize=(15, 8))
sns.boxplot(data=plot_df, x='Label Percentage', y='Accuracy', hue='Metric')
plt.title('Box Plot das Acurácias por Percentual de Rótulos e Modelo (HDBScanSSC)')
plt.ylabel('Acurácia')
plt.xlabel('Percentual de Rótulos')
plt.grid(True)
plt.legend(title='Métrica')
plt.show()
