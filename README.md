# t1-introducao-a-visao-computacional

Repositório para realização do Trabalho 1 de Introdução à Visão Computacional. Professora Daniela Oliveira Ferreira do Amaral.

## Otimização de Segmentação com Filtros de Ruído e Optuna

Este repositório contém um framework experimental para automatizar a escolha dos melhores filtros de ruído e parâmetros de binarização em tarefas de segmentação de imagens. O projeto utiliza Otimização Bayesiana (Optuna) para encontrar o limiar (threshold) ideal que minimiza o erro em relação às máscaras do dataset.

### Estrutura do Projeto

* src/noise_filter.py: Módulo principal com implementações de filtros (Gaussiano, Mediana, Bilateral, etc.) e métricas de erro (MSE, Dice, IoU).
* src/experiment.py: Script que executa o experimento.
* analysis.ipynb: Notebook Jupyter para análise estatística dos resultados salvos em CSV.
* pipeline_demo.ipynb: Notebook de demonstração que visualiza o passo a passo para uma única imagem.
* data/: Pasta esperada contendo imagens/train e masks/train.

### Pré-requisitos

Certifique-se de ter o Python instalado e instale as dependências necessárias:

```bash
pip install opencv-python numpy optuna pandas matplotlib seaborn
```

### Como Utilizar

1. Executar o Experimento

Para processar o dataset e encontrar os melhores parâmetros, execute:

```bash
python src/experiment.py
```

Isso gerará o arquivo experiment_results.csv com as métricas detalhadas de cada imagem para cada tipo de filtro.

2. Analisar Resultados

Abra o arquivo analysis.ipynb para visualizar gráficos de desempenho, comparação de médias, desvios padrão e identificar qual filtro foi mais estável para o seu conjunto de dados.

3. Demonstrar o Pipeline

Para gerar imagens intermediárias e ver o efeito visual de cada etapa do processamento, utilize o pipeline_demo.ipynb.

### Métricas Implementadas

* MSE (Mean Squared Error): Utilizado como função objetivo para a otimização.
* Dice Coefficient: Medida de similaridade entre a segmentação e a máscara real (0 a 1).
* IoU (Intersection over Union): Índice de Jaccard, avalia a sobreposição das áreas.
