# 🏥 Análise de Diabetes com Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://seu-app.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Aplicação web interativa para análise e predição da progressão de diabetes usando técnicas de Machine Learning supervisionado e não supervisionado.

## 🎯 Objetivo

Desenvolver um sistema completo de análise de dados de saúde que:
- Identifique os principais fatores de risco para progressão da diabetes
- Agrupe pacientes em perfis de risco similares
- Preveja a progressão da doença com base em dados clínicos
- Forneça uma interface interativa para profissionais de saúde

## ✨ Funcionalidades

### 📊 Overview
- Visualização geral do dataset com 442 pacientes
- Dados **humanizados** (conversão de z-scores para unidades reais)
- Tabela interativa com valores clínicos interpretáveis
- Distribuição da progressão por níveis de risco

### 🔍 Análise Exploratória
- Matriz de correlação entre variáveis clínicas
- Top 5 fatores mais correlacionados com a progressão
- Identificação do **IMC como principal fator de risco**

### 🧬 Clustering (Aprendizado Não Supervisionado)
- Algoritmo **K-Means** para segmentação de pacientes
- Visualização PCA em 2D dos clusters
- Identificação de 3 perfis:
  - 🟢 **Baixo Risco** (progressão média: 109)
  - 🟡 **Risco Moderado** (progressão média: 161)
  - 🔴 **Alto Risco** (progressão média: 197)

### 🤖 Modelagem (Aprendizado Supervisionado)
- Comparação de 5 algoritmos de regressão
- **Gradient Boosting** como melhor modelo (R² = 0.43)
- Métricas: MAE = 44.73, RMSE = 55.2
- Gráfico de predições vs valores reais

### 🎯 Predição Interativa
- Interface para simular perfis de pacientes
- Sliders para ajustar idade, IMC, pressão, glicose, colesterol
- Predição em tempo real da progressão
- Classificação automática de risco

## 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **Streamlit** - Framework para aplicação web
- **Scikit-learn** - Machine Learning
- **Pandas & NumPy** - Manipulação de dados
- **Plotly** - Visualizações interativas

## 📦 Instalação e Execução Local

### Pré-requisitos
- Python 3.10 ou superior
- pip (gerenciador de pacotes)

### Passos

1. **Clone o repositório:**
```bash
git clone [https://github.com/amandarga/analise-diabetes.git](https://github.com/amandarga/analise-diabetes.git)
cd analise-diabetes```

2. **Crie e ative um ambiente virtual:**
```bash
python -m venv venv
venv\Scripts\activate```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt```

4. **Execute o Streamlit:**
```bash
streamlit run diabetes_streamlit.py```

5. **Acesse a aplicação:**
Abra o navegador e acesse `http://localhost:8501` para ver a aplicação.

## 📜 Dataset
Fonte: Scikit-learn Diabetes Dataset

442 pacientes
10 variáveis clínicas:
Age (idade)
Sex (sexo)
BMI (índice de massa corporal)
BP (pressão arterial média)
S1 (colesterol total)
S2 (LDL - colesterol ruim)
S3 (HDL - colesterol bom)
S4 (razão colesterol/HDL)
S5 (triglicerídeos)
S6 (glicose)
Variável alvo: Progressão da diabetes após um ano (escala 25-346)


