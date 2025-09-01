import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import numpy as np # Importa numpy para lidar com 'nan'

def gerar_boxplot_de_txt(caminho_txt, saida_png):
    """
    Gera um boxplot a partir de um arquivo de texto com resultados de experimentos.

    Args:
        caminho_txt (str): O caminho para o arquivo de texto de entrada.
        saida_png (str): O caminho para salvar a imagem PNG de saída.
    """
    plot_data = []

    with open(caminho_txt, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    dataset_atual = None
    for linha in linhas:
        linha = linha.strip()

        # Pula linhas vazias
        if not linha:
            continue

        # Detecta o nome do dataset
        if linha.startswith("Dataset:"):
            dataset_atual = linha.split(":")[1].strip()
            continue

        # Detecta e processa as linhas de resultados
        match_resultados = re.match(r"\s*(\d+)%\s*rotulado\s*->\s*(.*)", linha)
        if match_resultados and dataset_atual:
            perc = match_resultados.group(1) + "%"
            resultados_str = match_resultados.group(2)

            # Divide a string de resultados em pares de métrica e valor
            pares_metricas = resultados_str.split(',')

            for par in pares_metricas:
                # Divide cada par em nome da métrica e valor
                if ':' in par:
                    partes = par.split(':', 1)
                    nome_metrica = partes[0].strip()
                    valor_str = partes[1].strip()

                    # Converte o valor para float, tratando o caso 'nan'
                    try:
                        acuracia = float(valor_str)
                    except ValueError:
                        acuracia = np.nan # Usa o NaN do numpy se a conversão falhar

                    # Adiciona os dados à lista para o plot
                    plot_data.append({
                        'Dataset': dataset_atual,
                        'Label Percentage': perc,
                        'Metric': nome_metrica,
                        'Accuracy': acuracia
                    })

    # Cria um DataFrame do Pandas com os dados coletados
    if not plot_data:
        print("Nenhum dado foi extraído do arquivo. Verifique o formato.")
        return
        
    plot_df = pd.DataFrame(plot_data)
    
    # Remove linhas com valores NaN na coluna 'Accuracy' para evitar erros no plot
    plot_df.dropna(subset=['Accuracy'], inplace=True)
    
    # Define a ordem das porcentagens no eixo X
    order = sorted(plot_df['Label Percentage'].unique(), key=lambda x: int(x.strip('%')))

    # Cria o boxplot
    plt.figure(figsize=(18, 10))
    sns.boxplot(
        data=plot_df,
        x='Label Percentage',
        y='Accuracy',
        hue='Metric',
        width=0.8,
        dodge=True,
        order=order # Garante a ordem correta no eixo X
    )

    plt.title('Acurácia dos Modelos por Percentual de Rótulos', fontsize=20, pad=20)
    plt.ylabel('Acurácia', fontsize=14)
    plt.xlabel('Percentual de Rótulos', fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Move a legenda para fora do gráfico para melhor visualização
    plt.legend(title='Métrica', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajusta o layout para caber a legenda
    plt.savefig(saida_png, bbox_inches='tight', dpi=300)
    plt.close()

# --- Execução do Script ---
# Certifique-se de que o arquivo "experiment_results.txt" está no mesmo diretório
# que este script, ou forneça o caminho completo para ele.
try:
    gerar_boxplot_de_txt("experiment_results.txt", "grafico_boxplot_resultados.png")
    print("Gráfico 'grafico_boxplot_resultados.png' gerado com sucesso!")
except FileNotFoundError:
    print("Erro: O arquivo 'experiment_results.txt' não foi encontrado.")