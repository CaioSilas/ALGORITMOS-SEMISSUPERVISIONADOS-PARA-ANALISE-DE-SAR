# Imagem base
FROM python:3.10-slim

# Evitar prompts interativos durante a instalação de pacotes do sistema
ENV DEBIAN_FRONTEND=noninteractive

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Instalar pacotes de sistema necessários
# Adicione ou remova conforme a necessidade do seu projeto
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    python3-dev \
    # Limpar o cache do APT para reduzir o tamanho da imagem
    && rm -rf /var/lib/apt/lists/*

# Copiar o arquivo de dependências do Python para aproveitar o cache do Docker
COPY requirements.txt .

# Instalar as dependências do Python a partir do arquivo requirements.txt
# Isso inclui bibliotecas como pprintpp, numpy, pandas, etc., que você listou
# O comando '--no-cache-dir' é uma boa prática para evitar cache local
RUN pip install --no-cache-dir -r requirements.txt

# Garantir o encoding UTF-8 para evitar problemas de codificação
ENV PYTHONIOENCODING=utf-8

# Copiar o restante do código do projeto para o diretório de trabalho
COPY . .

# Comando padrão ao iniciar o container
CMD ["python", "main.py"]