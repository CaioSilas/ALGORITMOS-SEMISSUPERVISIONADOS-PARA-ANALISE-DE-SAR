# Algoritmos Semissupervisionados para Análise de Relações Estrutura-Atividade (SAR)

Este repositório contém os códigos e bases de dados utilizados no trabalho de conclusão de curso **“Algoritmos Semissupervisionados para Análise de Relações entre Estrutura Química e Atividade Biológica”**, desenvolvido por **Caio Silas de Araujo Amaro** no curso de Ciência da Computação da Universidade Federal de Ouro Preto.

---

## 📖 Descrição

O projeto investiga o uso de **algoritmos semissupervisionados** para a análise de Relações Estrutura-Atividade (SAR), com foco na predição de propriedades biológicas de compostos químicos.  

Foram implementados e avaliados:  
- **GFHF (Gaussian Field Harmonic Function)**  
- **kNN-LDP (k-Nearest Neighbor Label Distribution Propagation)**  
- **HDBScanSS*** (para detecção de ruídos e instâncias atípicas)  

Os resultados foram comparados com algoritmos supervisionados clássicos (**Random Forest** e **SVM**), evidenciando o potencial dos métodos semissupervisionados em cenários com poucos dados rotulados.

---

## 📂 Estrutura do Repositório

├── .devcontainer/ # Configuração para execução em container Docker
├── Dados/ # Bases de dados (.data e resumo em .csv)
│ ├── ACE.data
│ ├── BBB.data
│ ├── ...
│ └── resumo_bases.csv
├── GFHF_KNN/ # Implementação do GFHF e kNN-LDP
│ ├── GFHF_KNN.py
│ └── utils/
├── HDBscanSSC/ # Implementação do HDBScanSS*
│ ├── HDBscanSSC.py
│ ├── ss_hdbscan.py
│ ├── super_heap.py
│ └── HDBscanSSC.ipynb # versão em notebook (opcional)
├── Resultados/ # Resultados de execução (tabelas, métricas, gráficos)
├── gerar_boxplot.py # Script para geração de boxplots
├── README.md # Documentação do repositório
└── .gitattributes

---

## ⚙️ Tecnologias Utilizadas

- **Python 3.10+**  
- [scikit-learn](https://scikit-learn.org/)  
- [hdbscan](https://hdbscan.readthedocs.io/)  
- [numpy](https://numpy.org/)  
- [pandas](https://pandas.pydata.org/)  
- [matplotlib](https://matplotlib.org/)  
- **Docker + Dev Containers**  

---

## ▶️ Como Executar

1. Certifique-se de ter **Docker** e **Dev Containers** configurados.  

2. Clone este repositório:
   ```bash
   git clone https://github.com/CaioSilas/TCC.git
   cd TCC

3. Abra no VS Code e selecione “Reopen in Container”.
(ou rode manualmente usando Docker CLI, se preferir)

4. Para executar os experimentos:

    ## GFHF e kNN-LDP

    python GFHF_KNN/GFHF_KNN.py


    ## HDBScanSS*

    python HDBscanSSC/HDBscanSSC.py


5. Para gerar boxplots a partir dos resultados:

python gerar_boxplot.py


Os resultados serão salvos na pasta Resultados/.

## 📊 Resultados

GFHF + Random Forest apresentou boa estabilidade preditiva.

HDBScanSS* aumentou a robustez dos classificadores ao eliminar instâncias ruidosas.

Em cenários com poucos rótulos, os métodos semissupervisionados superaram Random Forest e SVM tradicionais.

Mais detalhes estão documentados na monografia anexada a este projeto.

## 👨‍💻 Autor

Caio Silas de Araujo Amaro
Curso de Ciência da Computação – Universidade Federal de Ouro Preto
Orientador: Prof. Jadson Castro Gertrudes