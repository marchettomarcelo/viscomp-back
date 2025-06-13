# VisComp - Projeto final

## Descrição

VisComp é uma aplicação backend que utiliza processamento de imagem e aprendizado de máquina para comparar imagens com suas respectivas versões processadas pelo algoritmo de Canny. A aplicação utiliza o modelo Vision Transformer (ViT) para extrair características das imagens e calcular a similaridade entre elas.

## Funcionalidades

-   Processamento de imagens enviadas pelo usuário
-   Comparação com imagens de referência processadas pelo algoritmo de Canny
-   Cálculo de similaridade usando o modelo Vision Transformer
-   Suporte para diferentes categorias de imagens (cavalo, estrela, gato, etc.)

## Requisitos

-   Python 3.x
-   FastAPI
-   PyTorch
-   Pillow
-   Uvicorn

## Instalação

1. Clone o repositório

2. Instale o uv (se ainda não tiver instalado):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Crie um ambiente virtual e instale as dependências usando uv:

```bash
uv sync
```

## Executando a Aplicação

Para iniciar o servidor:

```bash
uv run main.py
```

O servidor estará disponível em `http://localhost:8000`

## Endpoints da API

### GET /health

-   Verifica o status da API
-   Retorna: `{"status": "healthy"}`

### POST /process-image

-   Processa uma imagem e compara com sua versão Canny
-   Parâmetros:
    -   `file`: Arquivo de imagem (UploadFile)
    -   `text`: Texto identificando a categoria da imagem (string)
-   Retorna:
    -   `filename`: Nome do arquivo
    -   `text`: Categoria da imagem
    -   `cosine_similarity`: Similaridade calculada

## Categorias Suportadas

-   cavalo
-   estrela
-   gato
-   linus
-   luminaria
-   mack
-   nike
-   raposa

## Estrutura do Projeto

```
viscomp-back/
├── main.py              # Arquivo principal da aplicação
├── utils/              # Utilitários e funções auxiliares
├── fotos_canny/        # Imagens de referência processadas com Canny
└── pyproject.toml      # Configuração do projeto
```

## Tecnologias Utilizadas

-   FastAPI: Framework web para construção da API
-   Vision Transformer (ViT): Modelo de aprendizado de máquina para processamento de imagens
-   PyTorch: Framework de deep learning
-   Pillow: Processamento de imagens
