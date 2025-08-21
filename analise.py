import re
import pandas as pd

# Caminho do arquivo
file_path = "experiment_results.txt"

# Ler o conteúdo
with open(file_path, "r") as f:
    content = f.read()

# Regex para capturar dataset, percentual e resultados
pattern_dataset = r"Dataset: (.*?)\n(.*?)(?=(?:Dataset:|$))"
pattern_result = r"(\d+)% rotulado -> (.*)"
pattern_model = r"([^:]+): ([\d\.nan]+)"

data = []

for dataset, block in re.findall(pattern_dataset, content, re.S):
    for perc, results in re.findall(pattern_result, block):
        for model, score in re.findall(pattern_model, results):
            try:
                val = float(score)
            except:
                val = None
            data.append([dataset.strip(), int(perc), model.strip(), val])

df = pd.DataFrame(data, columns=["Dataset", "Percent", "Model", "Score"])

# Estatísticas descritivas por modelo e percentual
summary = df.groupby(["Percent", "Model"]).agg(
    mean_score=("Score", "mean"),
    median_score=("Score", "median"),
    std_score=("Score", "std"),
    count=("Score", "count")
).reset_index()

# Exibir no console
print(summary)

# (Opcional) salvar para CSV
summary.to_csv("resumo_estatistico.csv", index=False)
