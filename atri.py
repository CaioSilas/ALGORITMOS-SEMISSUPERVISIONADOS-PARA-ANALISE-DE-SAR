import pandas as pd
import os

# Caminho da pasta onde estão os arquivos .data
caminho = "./Dados"

# Descobre todos os arquivos .data da pasta
arquivos = [f for f in os.listdir(caminho) if f.endswith(".data")]

# Lista para guardar os resultados
resultados = []

for arquivo in arquivos:
    try:
        # Tenta ler como CSV separado por vírgula
        df = pd.read_csv(os.path.join(caminho, arquivo), header=None)
    except:
        # Se falhar, tenta com espaço como separador
        df = pd.read_csv(os.path.join(caminho, arquivo), sep="\s+", header=None)

    instancias, atributos = df.shape

    # Mostra no terminal
    print(f"[{arquivo}] -> Instâncias: {instancias}, Atributos: {atributos}")

    # Salva no resumo
    resultados.append({
        "Arquivo": arquivo,
        "Instancias": instancias,
        "Atributos": atributos
    })

# Converte os resultados para DataFrame
resumo = pd.DataFrame(resultados)

# Salva em um CSV
saida = os.path.join(caminho, "resumo_bases.csv")
resumo.to_csv(saida, index=False)

print("\n✅ Resumo final salvo em:", saida)

