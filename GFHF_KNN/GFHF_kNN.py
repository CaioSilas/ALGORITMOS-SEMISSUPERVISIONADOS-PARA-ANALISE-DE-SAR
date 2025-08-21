# Importação das bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from scipy.sparse.csgraph import laplacian
from scipy.linalg import solve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from google.colab import drive
from sklearn.model_selection import KFold
from scipy import stats
import gc


from collections import Counter,defaultdict

# Célula 1: Importações necessárias
import math
import numpy as np
from scipy.spatial import cKDTree
from sklearn.datasets import load_iris
import seaborn as sns
from pprint import pprint

import sys
sys.path.append('./utils')

# Célula 2: Funções auxiliares
def key_with_max_val(d):
    """Retorna a chave com o valor máximo em um dicionário."""
    return max(d, key=d.get)

def key_with_min_val(d):
    """Retorna a chave com o valor mínimo em um dicionário."""
    return min(d, key=d.get)


def knn_and_rnn(X, k=7, get_distances=False):
    """
    Calcula os k-vizinhos mais próximos (kNN) e os vizinhos mais próximos reversos (RNN) para cada ponto no conjunto de dados.

    Parâmetros:
    -----------
    X : np.ndarray
        Um array 2D do numpy contendo os pontos de dados.
    k : int, opcional
        O número de vizinhos mais próximos a serem calculados. O padrão é 7.
    get_distances : bool, opcional
        Se True, retorna as distâncias para os vizinhos mais próximos. O padrão é False.

    Retorna:
    --------
    Tuple[List[List[int]], List[List[int]], cKDTree]
        Uma tupla contendo:
        - Uma lista de listas, onde cada lista interna contém os índices dos k-vizinhos mais próximos para cada ponto.
        - Uma lista de listas, onde cada lista interna contém os índices dos vizinhos mais próximos reversos para cada ponto.
        - Um objeto cKDTree usado para a busca dos vizinhos mais próximos.
    """
    ckdtree = cKDTree(X, leafsize=k + 1)
    knn_array = [0] * len(X)
    rnn_array = [[] for _ in range(len(X))]  # Inicializa o array de RNNs
    distance_array = [0] * len(X)

    for index in range(len(X)):
        distances, neighbors = ckdtree.query(X[index], k=k + 1)
        neighbors = neighbors.tolist()
        distances = distances.tolist()

        try:
            neighbors.remove(index)  # Remove o próprio ponto da lista de vizinhos
        except ValueError:
            pass  # Se o índice não estiver na lista, ignora

        knn_array[index] = neighbors
        distance_array[index] = distances

        # Atualiza o array de RNNs
        for neighbor in neighbors:
            if 0 <= neighbor < len(rnn_array):  # Verifica se o índice é válido
                rnn_array[neighbor].append(index)
            else:
                print(f"Aviso: Índice inválido {neighbor} encontrado. Ignorando.")

    if get_distances:
        output_tuple = (knn_array, rnn_array, ckdtree, distance_array)
    else:
        output_tuple = (knn_array, rnn_array, ckdtree)

    return output_tuple

# Célula 5: Implementação da SuperHeap
class SuperHeap:
    """
    Uma implementação personalizada de uma max-heap.
    """
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.size = 0
        self.Heap = [{'key': 0, 'node': None} for _ in range(self.maxsize)]
        self.index_dict = {}

    def root(self):
        return self.Heap[0]

    def parent(self, pos):
        return (pos - 1) // 2

    def leftChild(self, pos):
        return 2 * pos + 1

    def rightChild(self, pos):
        return (2 * pos) + 2

    def isLeaf(self, pos):
        return pos >= (self.size // 2) and pos <= self.size

    def swap(self, fpos, spos):
        self.Heap[fpos]['node'].heap_index = spos
        self.Heap[spos]['node'].heap_index = fpos
        self.index_dict[self.Heap[fpos]['node'].index] = spos
        self.index_dict[self.Heap[spos]['node'].index] = fpos
        self.Heap[fpos], self.Heap[spos] = self.Heap[spos], self.Heap[fpos]

    def increaseKey(self, pos, key):
        assert key > self.Heap[pos]['key'], "New key must be larger than current key"
        self.Heap[pos]['key'] = key
        while pos > 0 and self.Heap[self.parent(pos)]['key'] < self.Heap[pos]['key']:
            self.swap(pos, self.parent(pos))
            pos = self.parent(pos)

    def maxHeapify(self, pos):
        if not self.isLeaf(pos):
            if (self.Heap[pos]['key'] < self.Heap[self.leftChild(pos)]['key'] or
                self.Heap[pos]['key'] < self.Heap[self.rightChild(pos)]['key']):
                if self.Heap[self.leftChild(pos)]['key'] > self.Heap[self.rightChild(pos)]['key']:
                    self.swap(pos, self.leftChild(pos))
                    self.maxHeapify(self.leftChild(pos))
                else:
                    self.swap(pos, self.rightChild(pos))
                    self.maxHeapify(self.rightChild(pos))

    def insert(self, element):
        if self.size >= self.maxsize:
            raise Exception(IndexError)
        self.Heap[self.size] = element
        current = self.size
        element['node'].heap_index = current
        self.size += 1
        self.index_dict[element['node'].index] = current

        while self.Heap[current]['key'] > self.Heap[self.parent(current)]['key']:
            if current == 0:
                break
            self.swap(current, self.parent(current))
            current = self.parent(current)

    def extractMax(self):
        popped = self.Heap[0]
        self.Heap[0]['node'].heap_index = -2
        self.index_dict[self.Heap[0]['node'].index] = -2
        self.Heap[0] = self.Heap[self.size - 1]
        self.index_dict[self.Heap[self.size - 1]['node'].index] = 0
        self.size -= 1
        self.maxHeapify(0)
        return popped

    def __repr__(self):
        output_string = ""
        for i in range(self.size // 2):
            if 2 * i + 2 < self.size:
                output_string += f"PARENT : {self.Heap[i]['key']}  LEFT CHILD : {self.Heap[2 * i + 1]['key']} RIGHT CHILD : {self.Heap[2 * i + 2]['key']}\n"
            else:
                output_string += f"PARENT : {self.Heap[i]['key']}  LEFT CHILD : {self.Heap[2 * i + 1]['key']} No Right Child\n"
        output_string += "----------------------------"
        return output_string
    

# Célula 6: Implementação do kNN-LDP
class kNN_LDP_Node:
    def __init__(self, label_distribution, knn, index, rnn=None):
        self.label_distribution = label_distribution
        self.knn = knn
        self.rnn = rnn
        self.index = index
        self.heap_index = -1

    def __str__(self):
        return f"({self.index}, {self.label_distribution}, {self.heap_index}, {self.knn}, {self.rnn})"

    def __repr__(self):
        return self.__str__()

class kNN_LDP:
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None
        self.label_set = None
        self.classes_ = None
        self.label_distributions_ = None
        self.transduction_ = []
        self.n_iter = 0
        self.transduced = False
        self.kdtree = None
        self.max_heap = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.label_set = set(y)
        self.max_heap = SuperHeap(len(y))
        self.classes_ = list(self.label_set)
        self.label_distributions_ = [{label: 1} for label in y]
        self._transduce()

    def _transduce(self):
        knn_list, rnn_list, self.kdtree = knn_and_rnn(self.X, self.n_neighbors)
        U_indices = [index for index, label in enumerate(self.y) if label == -1]

        for index in U_indices:
            label_distribution = self.calculate_instance_label_probability_distribution(knn_list[index])
            node = kNN_LDP_Node(label_distribution=label_distribution, knn=knn_list[index], rnn=rnn_list[index], index=index)
            heap_node = {'key': self.probability_certainty(label_distribution), 'node': node}
            self.max_heap.insert(heap_node)

        while self.max_heap.size > 0:
            x = self.max_heap.extractMax()
            if x['key'] > 0:
                self.label_distributions_[x['node'].index] = self.calculate_instance_label_probability_distribution(knn_list[x['node'].index])
                for r_neighbor in x['node'].rnn:
                    if self.y[r_neighbor] == -1 and self.max_heap.index_dict[r_neighbor] != -2:
                        probability = self.calculate_instance_label_probability_distribution(knn_list[r_neighbor])
                        new_key = self.probability_certainty(probability)
                        self.max_heap.increaseKey(self.max_heap.index_dict[r_neighbor], new_key)
        self.set_transduction()
        self.transduced = True

    # def calculate_instance_label_probability_distribution(self, nearest_neighbors):
    #     label_distribution = {}
    #     for key in self.label_set:
    #         label_distribution[key] = sum([self.label_distributions_[index][key] if key in self.label_distributions_[index] else 0 for index in nearest_neighbors]) / len(nearest_neighbors)
    #     return label_distribution


    def calculate_instance_label_probability_distribution(self, nearest_neighbors):
      label_distribution = {}
      for key in self.label_set:
          total = 0
          valid_neighbors = 0
          for index in nearest_neighbors:
              # Check if the index is valid
              if 0 <= index < len(self.label_distributions_):
                  if key in self.label_distributions_[index]:
                      total += self.label_distributions_[index][key]
                  valid_neighbors += 1
          # Avoid division by zero
          if valid_neighbors > 0:
              label_distribution[key] = total / valid_neighbors
          else:
              label_distribution[key] = 0  # If no valid neighbors, probability is 0
      return label_distribution

    def probability_certainty(self, label_distribution):
        return sum([value for key, value in label_distribution.items() if key != -1])

    def set_transduction(self):
        for index in range(len(self.y)):
            if self.y[index] == -1:
                decision_dict = self.label_distributions_[index]
                self.transduction_.append(self._maximum_likelihood_prediction(decision_dict))
            else:
                self.transduction_.append(self.y[index])

    # def set_transduction(self):
    #     for index in range(len(self.y)):
    #         if self.y[index] is None:  # Modificado para usar 'is None'
    #             decision_dict = self.label_distributions_[index]
    #             self.transduction_.append(self._maximum_likelihood_prediction(decision_dict))
    #         else:
    #             self.transduction_.append(self.y[index])

    def _maximum_likelihood_prediction(self, decision_dict):
        l = {k: v for k, v in decision_dict.items() if k != -1}
        if l == {}:
            return -1
        label = key_with_max_val(l)
        if label != -1 and l[label] > 0.0:
            return label
        else:
            return -1

    # def _maximum_likelihood_prediction(self, decision_dict):
    #     l = {k: v for k, v in decision_dict.items() if k != -1}
    #     if l == {}:
    #         return None  # Retorna None se a distribuição for vazia
    #     label = key_with_max_val(l)
    #     if label != -1 and l[label] > 0.0:
    #         return label
    #     else:
    #         return None  # Retorna None se a probabilidade máxima for menor que o limiar

    def predict(self, X):
        assert self.transduced, "No basis for prediction, fit data to model"
        decision_dicts = self.predict_proba(X)
        predictions = [self._maximum_likelihood_prediction(decision_dict) for decision_dict in decision_dicts]
        return predictions

    def predict_proba(self, X):
        assert self.transduced, "No basis for prediction, fit data to model"
        neighbors_list = [self.kdtree.query(query_point, self.n_neighbors + 1)[1] for query_point in X]
        probabilities = [self.calculate_instance_label_probability_distribution(neighbors) for neighbors in neighbors_list]
        return probabilities

    def score(self, X, y, sample_weight=None):
        assert self.transduced, "The model has not yet been built, please fit before inductive predictions"
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


#carregamento de dados 
folder_path = "./Dados"
datasets = [
    "BBB.data",
    "EP2.data",
    "ACE.data",
    "ACHE.data",
    "AT1.data",
    "BZR.data",
    "COX2.data",
    "DHFR.data",
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


# Função para S3VM via self-training
def semi_supervised_svm(X, y, max_iter=15, threshold=0.3, kernel='rbf', C=1.0, gamma='scale'):
    labeled_idx = np.where(np.isfinite(y))[0]
    unlabeled_idx = np.where(np.isnan(y))[0]

    clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    clf.fit(X[labeled_idx], y[labeled_idx])

    iter_count = 0
    while len(unlabeled_idx) > 0 and iter_count < max_iter:
        iter_count += 1
        probas = clf.predict_proba(X[unlabeled_idx])
        max_proba = np.max(probas, axis=1)
        pred_labels = clf.classes_[np.argmax(probas, axis=1)]

        confident_mask = max_proba >= threshold
        if np.sum(confident_mask) == 0:
            print(f"Iteração {iter_count}: Nenhum exemplo com confiança acima do threshold.")
            break

        confident_indices = unlabeled_idx[confident_mask]
        y[confident_indices] = pred_labels[confident_mask]
        print(f"Iteração {iter_count}: Adicionados {len(confident_indices)} exemplos via S3VM.")
        # Imprimir o número de exemplos rotulados nesta iteração
        print(f"Iteração {iter_count}: Rotulados {len(confident_indices)} exemplos.")

        labeled_idx = np.where(np.isfinite(y))[0]
        unlabeled_idx = np.where(np.isnan(y))[0]

        clf.fit(X[labeled_idx], y[labeled_idx])

    # Treinar o classificador final com todos os dados (como GFHF)
    clf.fit(X, y)

    return clf, y


# Arquivo de saída (vai acumulando resultados)
output_filename = 'experiment_results_GFHF_KNN.txt'

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


    dataset_results = {p: [] for p in label_percents}



    for rep in range(n_repetitions):
        print(f"Executando {dataset_name} - Repetição {rep + 1}/{n_repetitions}")

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            print(f"  Fold {fold + 1}/{kfold.n_splits}")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for label_percent in label_percents:
                num_labeled = int(label_percent * len(y_train))

                acc_gfhf_list, acc_svm_list, acc_rf_list = [], [], []
                acc_svm_gfhf_list, acc_rf_gfhf_list = [], []
                acc_knn_ldp_list, acc_svm_knn_ldp_list, acc_rf_knn_ldp_list = [], [], []  # Listas para kNN_LDP

                for sel in range(n_label_selections):
                    labels = np.full(y_train.shape, np.nan)  # Inicialmente todos são NaN (não rotulados)
                    labelsGFHF = np.full(y_train.shape, -1)  # Inicialmente todos os rótulos são -1 (não rotulados) para o GFHF
                    unique_classes = np.unique(y_train)
                    labeled_indices = []
                    # num_labeled_per_class = num_labeled // len(unique_classes)

                    for cls in unique_classes:
                        class_indices = np.where(y_train == cls)[0]

                        # # Seleciona 'num_labeled_per_class' amostras da classe atual
                        # selected_indices = np.random.choice(class_indices, num_labeled_per_class, replace=False)
                        # labeled_indices.extend(selected_indices)  # Adiciona os índices selecionados à lista geral


                        # labeled_indices.append(np.random.choice(class_indices))

                        # # # Seleciona metade dos exemplos da classe
                        # num_to_select = min(num_labeled_per_class, len(class_indices) // 2)
                        # selected_indices = np.random.choice(class_indices, num_to_select, replace=False)
                        # labeled_indices.extend(selected_indices)

                        #pelo menos 2 exemplos por classe
                        selected_indices = np.random.choice(class_indices, size=2, replace=False)  # Seleciona 2 exemplos
                        labeled_indices.extend(selected_indices)

                        #     # Pelo menos 1 de cada classe
                        # remaining_indices = list(set(range(len(y_train))) - set(labeled_indices))
                        # labeled_indices.extend(np.random.choice(remaining_indices, num_labeled - len(unique_classes), replace=False))

                    # Seleciona os exemplos restantes aleatoriamente
                    remaining_indices = list(set(range(len(y_train))) - set(labeled_indices))
                    num_remaining = num_labeled - len(labeled_indices)  # Calcula quantos ainda faltam
                    if num_remaining > 0:
                      selected_remaining = np.random.choice(remaining_indices, size=num_remaining, replace=False)
                      labeled_indices.extend(selected_remaining)



                    labels[labeled_indices] = y_train[labeled_indices]
                    labelsGFHF[labeled_indices] = y_train[labeled_indices]

                    similarity_matrix = rbf_kernel(X_train, gamma=0.1)

                    # GFHF
                    L = laplacian(similarity_matrix, normed=True)
                    Y = np.zeros((len(y_train), len(unique_classes)))
                    for i, label in enumerate(labels):
                        if not np.isnan(label):
                            Y[i, int(label)] = 1
                    F = solve(L + np.eye(L.shape[0]) * 1e-6, Y)
                    predicted_labels_gfhf = np.argmax(F, axis=1)

                    known_mask = ~np.isnan(labels)
                    known_mask1 = labelsGFHF != -1

                    # kNN_LDP
                    knn_ldp = kNN_LDP(n_neighbors=3)
                    knn_ldp.fit(X_train[known_mask], y_train[known_mask].astype(int))
                    y_pred_knn = knn_ldp.predict(X_test)
                    acc_knn_ldp = accuracy_score(y_test, y_pred_knn)
                    acc_knn_ldp_list.append(acc_knn_ldp)  # Armazena a acurácia do kNN_LDP




                    # # Previsões do kNN_LDP para o conjunto de treinamento
                    predicted_labels_knn_ldp = knn_ldp.predict(X_train)


                    # Treinar SVM e RF com as previsões do kNN_LDP (sem verificação de classes únicas)
                    # svm_knn_ldp = SVC()
                    # rf_knn_ldp = RandomForestClassifier()
                    # svm_knn_ldp.fit(X_train, predicted_labels_knn_ldp)
                    # rf_knn_ldp.fit(X_train, predicted_labels_knn_ldp)
                    # acc_svm_knn_ldp = accuracy_score(y_test, svm_knn_ldp.predict(X_test))
                    # acc_rf_knn_ldp = accuracy_score(y_test, rf_knn_ldp.predict(X_test))

                    # Treinar SVM e RF com as previsões do kNN_LDP
                    predicted_labels_knn_ldp = knn_ldp.predict(X_train)  # Previsões do kNN_LDP para o conjunto de treinamento
                    if len(np.unique(predicted_labels_knn_ldp)) > 1:
                        svm_knn_ldp = SVC()
                        rf_knn_ldp = RandomForestClassifier()
                        svm_knn_ldp.fit(X_train, predicted_labels_knn_ldp)
                        rf_knn_ldp.fit(X_train, predicted_labels_knn_ldp)
                        acc_svm_knn_ldp = accuracy_score(y_test, svm_knn_ldp.predict(X_test))
                        acc_rf_knn_ldp = accuracy_score(y_test, rf_knn_ldp.predict(X_test))
                    else:
                        acc_svm_knn_ldp = np.nan
                        acc_rf_knn_ldp = np.nan

                    # Treinamento de SVM e RandomForest apenas com rótulos conhecidos
                    svm = SVC()
                    rf = RandomForestClassifier()
                    svm.fit(X_train[known_mask], y_train[known_mask].astype(int))
                    rf.fit(X_train[known_mask], y_train[known_mask].astype(int))

                    if len(np.unique(predicted_labels_gfhf)) > 1:
                        svm_gfhf = SVC()
                        rf_gfhf = RandomForestClassifier()
                        svm_gfhf.fit(X_train, predicted_labels_gfhf)
                        rf_gfhf.fit(X_train, predicted_labels_gfhf)
                        acc_svm_gfhf = accuracy_score(y_test, svm_gfhf.predict(X_test))
                        acc_rf_gfhf = accuracy_score(y_test, rf_gfhf.predict(X_test))
                    else:
                        acc_svm_gfhf = np.nan
                        acc_rf_gfhf = np.nan

                    acc_svm = accuracy_score(y_test, svm.predict(X_test))
                    acc_rf = accuracy_score(y_test, rf.predict(X_test))

                    print(f"  Seleção {sel + 1}/{n_label_selections} - SVM: {acc_svm:.4f}, RF: {acc_rf:.4f}, SVM GFHF: {acc_svm_gfhf:.4f}, RF GFHF: {acc_rf_gfhf:.4f}, SVM kNN-LDP: {acc_svm_knn_ldp:.4f}, RF kNN-LDP: {acc_rf_knn_ldp:.4f}")

                    acc_svm_list.append(acc_svm)
                    acc_rf_list.append(acc_rf)
                    acc_svm_gfhf_list.append(acc_svm_gfhf)
                    acc_rf_gfhf_list.append(acc_rf_gfhf)
                    acc_svm_knn_ldp_list.append(acc_svm_knn_ldp)
                    acc_rf_knn_ldp_list.append(acc_rf_knn_ldp)

                    # --- Limpeza de variáveis dentro do loop de seleção (sel) ---
                    vars_to_delete_inner = [
                        'labels', 'labelsGFHF', 'unique_classes', 'labeled_indices',
                        'remaining_indices', 'similarity_matrix', 'L', 'Y', 'F',
                        'predicted_labels_gfhf', 'predicted_labels_knn_ldp',
                        'knn_ldp', 'svm', 'rf', 'known_mask', 'known_mask1'
                    ]

                    # Variáveis que podem não existir dependendo da condição 'if'
                    if 'svm_knn_ldp' in locals():
                        vars_to_delete_inner.append('svm_knn_ldp')
                    if 'rf_knn_ldp' in locals():
                        vars_to_delete_inner.append('rf_knn_ldp')
                    if 'svm_gfhf' in locals():
                        vars_to_delete_inner.append('svm_gfhf')
                    if 'rf_gfhf' in locals():
                        vars_to_delete_inner.append('rf_gfhf')

                    for var in vars_to_delete_inner:
                        if var in locals():
                            del locals()[var]
                    gc.collect() # Força a coleta de lixo

                # Adiciona os resultados médios para cada percentual de rótulos
                dataset_results[label_percent].append((
                    np.nanmean(acc_svm_list), np.nanmean(acc_rf_list),
                    np.nanmean(acc_svm_gfhf_list), np.nanmean(acc_rf_gfhf_list),
                    np.nanmean(acc_knn_ldp_list), np.nanmean(acc_svm_knn_ldp_list), np.nanmean(acc_rf_knn_ldp_list)
                ))

            # --- Limpeza de variáveis no final do fold ---
            vars_to_delete_fold = ['X_train', 'X_test', 'y_train', 'y_test']
            for var in vars_to_delete_fold:
                if var in locals():
                    del locals()[var]
            gc.collect()

    results[dataset_name] = {p: np.nanmean(dataset_results[p], axis=0) for p in label_percents}

        # --- Salva PARCIAL após cada dataset ---
    with open(output_filename, 'a') as f:  # 'a' = append, acumula no arquivo
        f.write(f"Dataset: {dataset_name}\n")
        for p, (acc_svm, acc_rf, acc_svm_gfhf, acc_rf_gfhf,
                acc_knn_ldp, acc_svm_knn_ldp, acc_rf_knn_ldp) in results[dataset_name].items():
            f.write(
                f"  {p*100:.0f}% rotulado -> "
                f"SVM: {acc_svm:.4f}, RF: {acc_rf:.4f}, "
                f"SVM GFHF: {acc_svm_gfhf:.4f}, RF GFHF: {acc_rf_gfhf:.4f}, "
                f"kNN-LDP: {acc_knn_ldp:.4f}, SVM kNN-LDP: {acc_svm_knn_ldp:.4f}, RF kNN-LDP: {acc_rf_knn_ldp:.4f}\n"
            )
        f.write("\n")

    print(f"✅ Resultados parciais salvos para {dataset_name}")

    # --- Limpeza final de variáveis por dataset ---
    vars_to_delete_dataset = ['df', 'X', 'y_labels', 'y', 'scaler', 'dataset_results']
    for var in vars_to_delete_dataset:
        if var in locals():
            del locals()[var]
    gc.collect()


# LP: {acc_lp:.4f}
# Exibição dos resultados finais
output_filename = 'experiment_results_GFHF_KNN.txt'

with open(output_filename, 'w') as f:
    for dataset, res in results.items():
        print(f"Dataset: {dataset}")
        f.write(f"Dataset: {dataset}\n")
        for p, (acc_svm, acc_rf, acc_svm_gfhf, acc_rf_gfhf, acc_knn_ldp, acc_svm_knn_ldp, acc_rf_knn_ldp) in res.items():  # Correção aqui
            print(f"  {p*100:.0f}% rotulado ->  SVM: {acc_svm:.4f}, RF: {acc_rf:.4f}, SVM GFHF: {acc_svm_gfhf:.4f}, RF GFHF: {acc_rf_gfhf:.4f}, kNN-LDP: {acc_knn_ldp:.4f}, SVM kNN-LDP: {acc_svm_knn_ldp:.4f}, RF kNN-LDP: {acc_rf_knn_ldp:.4f}")
            f.write(f"  {p*100:.0f}% rotulado ->  SVM: {acc_svm:.4f}, RF: {acc_rf:.4f}, SVM GFHF: {acc_svm_gfhf:.4f}, RF GFHF: {acc_rf_gfhf:.4f}, kNN-LDP: {acc_knn_ldp:.4f}, SVM kNN-LDP: {acc_svm_knn_ldp:.4f}, RF kNN-LDP: {acc_rf_knn_ldp:.4f}\n")
        print()
        f.write("\n")

print(f"Results saved to {output_filename}")




# Estruturar os resultados para plotting
plot_data = []
for dataset, res in results.items(): # Corrected variable name here
    for p, accuracies_list in res.items(): # Iterate through the results for each dataset
        # accuracies_list is a tuple (acc_svm, acc_rf, acc_svm_gfhf, acc_rf_gfhf, acc_knn_ldp, acc_svm_knn_ldp, acc_rf_knn_ldp)
        # Need to structure this data correctly for plotting
        plot_data.append({
            'Dataset': dataset,
            'Label Percentage': f'{p*100:.0f}%',
            'Metric': 'SVM',
            'Accuracy': accuracies_list[0]
        })
        plot_data.append({
            'Dataset': dataset,
            'Label Percentage': f'{p*100:.0f}%',
            'Metric': 'RF',
            'Accuracy': accuracies_list[1]
        })
        plot_data.append({
            'Dataset': dataset,
            'Label Percentage': f'{p*100:.0f}%',
            'Metric': 'SVM-GFHF',
            'Accuracy': accuracies_list[2]
        })
        plot_data.append({
            'Dataset': dataset,
            'Label Percentage': f'{p*100:.0f}%',
            'Metric': 'RF-GFHF',
            'Accuracy': accuracies_list[3]
        })
        plot_data.append({
            'Dataset': dataset,
            'Label Percentage': f'{p*100:.0f}%',
            'Metric': 'kNN-LDP',
            'Accuracy': accuracies_list[4]
        })
        plot_data.append({
            'Dataset': dataset,
            'Label Percentage': f'{p*100:.0f}%',
            'Metric': 'SVM-kNN-LDP',
            'Accuracy': accuracies_list[5]
        })
        plot_data.append({
            'Dataset': dataset,
            'Label Percentage': f'{p*100:.0f}%',
            'Metric': 'RF-kNN-LDP',
            'Accuracy': accuracies_list[6]
        })


plot_df = pd.DataFrame(plot_data)


# Criar os box plots e salvar
plt.figure(figsize=(15, 8))
sns.boxplot(
    data=plot_df,
    x='Label Percentage',
    y='Accuracy',
    hue='Metric',
    width=0.6,        # deixa mais separadas
    dodge=True
)

plt.title('Box Plot das Acurácias por Percentual de Rótulos e Modelo', fontsize=16)
plt.ylabel('Acurácia')
plt.xlabel('Percentual de Rótulos')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Métrica', bbox_to_anchor=(1.05, 1), loc='upper left')

# Salvar em arquivo
plt.tight_layout()
plt.savefig('box_plots_GFHF_KNN.png', bbox_inches='tight', dpi=300)
plt.close()