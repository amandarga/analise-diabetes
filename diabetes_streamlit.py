import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(page_title="Análise de Diabetes", page_icon="🏥", layout="wide")
reference_values = {
    'age': {'mean': 48, 'std': 13, 'unit': 'anos'},
    'bmi': {'mean': 26.4, 'std': 4.4, 'unit': 'kg/m²'},
    'bp': {'mean': 94, 'std': 14, 'unit': 'mmHg'},
    's1': {'mean': 189, 'std': 34, 'unit': 'mg/dL'},
    's6': {'mean': 91, 'std': 11, 'unit': 'mg/dL'}
}

def z_to_real(z_score, feature):
    if feature not in reference_values:
        return z_score
    ref = reference_values[feature]
    return z_score * ref['std'] + ref['mean']


@st.cache_data
def load_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    return df, diabetes

df, diabetes = load_data()

st.sidebar.title("🏥 Menu")
page = st.sidebar.radio("🧭 Navegação:", [
    "📊 Overview",
    "🔍 Análise Exploratória",
    "🧬 Clustering",
    "🤖 Modelos",
    "🎯 Predição"
])

page = page.split(' ', 1)[1] if ' ' in page else page

if page == "Overview":
    st.title("Análise de Diabetes com Machine Learning")
    st.markdown("### Visão Geral do Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total de Pacientes", f"{len(df):,}")
    col2.metric("📊 Variáveis Clínicas", "10")
    col3.metric("📈 Progressão Média", f"{df['target'].mean():.1f}")
    col4.metric("📉 Variação (std)", f"{df['target'].std():.1f}")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Dados Humanizados", "Dados Técnicos (Z-scores)", "Descrição das Variáveis"])
    
    with tab1:
        st.markdown("#### Primeiros 10 Pacientes - Valores Reais")
        
        df_display = pd.DataFrame()
        df_display['ID'] = range(1, 11)
        df_display['Idade'] = df['age'].head(10).apply(lambda x: f"{z_to_real(x, 'age'):.0f} anos")
        df_display['Sexo'] = df['sex'].head(10).apply(lambda x: 'Masculino' if x > 0 else 'Feminino')
        df_display['IMC'] = df['bmi'].head(10).apply(lambda x: f"{z_to_real(x, 'bmi'):.1f} kg/m²")
        df_display['Pressão'] = df['bp'].head(10).apply(lambda x: f"{z_to_real(x, 'bp'):.0f} mmHg")
        df_display['Colesterol'] = df['s1'].head(10).apply(lambda x: f"{z_to_real(x, 's1'):.0f} mg/dL")
        df_display['Glicose'] = df['s6'].head(10).apply(lambda x: f"{z_to_real(x, 's6'):.0f} mg/dL")
        df_display['Progressão'] = df['target'].head(10).apply(lambda x: f"{x:.1f}")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        st.info("**Valores convertidos de z-scores para unidades reais** usando médias de referência médica")
    
    with tab2:
        st.markdown("#### Dados Normalizados (Z-scores)")
        st.dataframe(df.head(10), use_container_width=True)
        
        with st.expander("O que são Z-scores?"):
            st.markdown("""
            - **Z-score** indica quantos desvios padrão um valor está da média
            - **Valor = 0:** exatamente na média da população
            - **Valor > 0:** acima da média
            - **Valor < 0:** abaixo da média
            - **|Valor| > 2:** valor extremo (muito alto ou muito baixo)
            
            **Exemplo:** Se idade tem z-score = 1.0, significa 1 desvio padrão acima da média
            """)
    
    with tab3:
        st.markdown("####Descrição das Variáveis Clínicas")
        
        features_info = pd.DataFrame({
            'Variável': ['Age', 'Sex', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
            'Nome Completo': ['Idade', 'Sexo', 'Índice de Massa Corporal', 'Pressão Arterial Média',
                             'Colesterol Total', 'LDL (Colesterol Ruim)', 'HDL (Colesterol Bom)',
                             'Razão Colesterol/HDL', 'Triglicerídeos (log)', 'Glicose'],
            'Unidade': ['anos', 'M/F', 'kg/m²', 'mmHg', 'mg/dL', 'mg/dL', 'mg/dL', 'razão', 'log', 'mg/dL'],
            'Importância': ['⭐⭐', '⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐', '⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐']
        })
        
        st.dataframe(features_info, use_container_width=True, hide_index=True)
        
        st.success("⭐⭐⭐⭐⭐ = Fator de risco muito importante | ⭐ = Impacto menor")
    
    st.markdown("---")
    
    st.markdown("### Distribuição da Progressão da Diabetes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.histogram(
            df, 
            x='target',
            nbins=30,
            title='Distribuição da Variável Alvo (Progressão da Diabetes)',
            labels={'target': 'Progressão da Diabetes', 'count': 'Número de Pacientes'},
            color_discrete_sequence=['#3b82f6']
        )
        fig.add_vline(x=df['target'].mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Média: {df['target'].mean():.1f}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Classificação")
        
        baixa = len(df[df['target'] < 100])
        moderada = len(df[(df['target'] >= 100) & (df['target'] < 200)])
        alta = len(df[df['target'] >= 200])
        
        st.metric("🟢 Baixa (<100)", f"{baixa} ({baixa/len(df)*100:.1f}%)")
        st.metric("🟡 Moderada (100-200)", f"{moderada} ({moderada/len(df)*100:.1f}%)")
        st.metric("🔴 Alta (≥200)", f"{alta} ({alta/len(df)*100:.1f}%)")
        
        st.markdown("---")
        
        fig_pie = px.pie(
            values=[baixa, moderada, alta],
            names=['Baixa', 'Moderada', 'Alta'],
            title='Distribuição por Nível',
            color_discrete_sequence=['#22c55e', '#eab308', '#ef4444'],
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

elif page == "Análise Exploratória":
    st.title("Análise Exploratória")
    
    corr = df.corr()
    fig = px.imshow(corr, text_auto='.2f', title='Matriz de Correlação')
    st.plotly_chart(fig, use_container_width=True)
    
    target_corr = corr['target'].drop('target').sort_values(ascending=False)
    fig2 = px.bar(y=target_corr.index[:5], x=target_corr.values[:5], 
                  orientation='h', title='Top 5 Correlações')
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Clustering":
    st.title("Clustering de Pacientes")
    
    n_clusters = st.slider("Clusters:", 2, 6, 3)
    
    X = df.drop('target', axis=1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], 
                     color=df['cluster'].astype(str),
                     title='Visualização PCA dos Clusters')
    st.plotly_chart(fig, use_container_width=True)
    
    for i in range(n_clusters):
        cluster_data = df[df['cluster'] == i]
        st.subheader(f"Cluster {i} - {len(cluster_data)} pacientes")
        st.write(f"Progressão média: {cluster_data['target'].mean():.1f}")

elif page == "Modelos":
    st.title("Modelagem")
    
    X = df.drop(['target', 'cluster'], axis=1, errors='ignore')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("R²", f"{r2:.4f}")
    
    fig = px.scatter(x=y_test, y=y_pred, 
                     title='Predições vs Real',
                     labels={'x': 'Real', 'y': 'Predição'})
    fig.add_trace(go.Scatter(x=[0, 350], y=[0, 350], 
                             mode='lines', name='Perfeito'))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Predição":
    st.title("Predição Interativa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        idade = st.slider("Idade:", 20, 80, 48)
        imc = st.slider("IMC:", 15.0, 45.0, 26.4)
        pa = st.slider("Pressão:", 60, 140, 94)
    
    with col2:
        colesterol = st.slider("Colesterol:", 100, 300, 189)
        glicose = st.slider("Glicose:", 60, 140, 91)
        sexo = st.radio("Sexo:", ["Feminino", "Masculino"])
    
    if st.button("PREVER", type="primary"):
        
        age_z = (idade - 48) / 13
        sex_z = 0.05 if sexo == "Masculino" else -0.04
        bmi_z = (imc - 26.4) / 4.4
        bp_z = (pa - 94) / 14
        s1_z = (colesterol - 189) / 34
        s6_z = (glicose - 91) / 11
        
        X_new = np.array([[age_z, sex_z, bmi_z, bp_z, s1_z, 0, 0, 0, 0, s6_z]])
        
        X = df.drop(['target', 'cluster'], axis=1, errors='ignore')
        y = df['target']
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)
        
        pred = model.predict(X_new)[0]
        
        st.success(f"### Progressão Prevista: {pred:.1f}")
        
        if pred < 100:
            st.success("✅ Baixo Risco")
        elif pred < 200:
            st.warning("⚠️ Risco Moderado")
        else:
            st.error("🔴 Alto Risco")

st.sidebar.markdown("---")
st.sidebar.info("Dataset: sklearn diabetes\n442 pacientes")