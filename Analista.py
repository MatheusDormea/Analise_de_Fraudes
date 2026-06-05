import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import shap

# ==========================================
# 1. CARGA E PRÉ-PROCESSAMENTO DOS DADOS
# ==========================================
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# Como o dataset é gigantesco, vamos pegar uma amostragem para o grafo rodar rápido.
# Mantemos todas as fraudes (classe 1) e uma amostra das transações normais (classe 0).
df_fraudes = df[df["Class"] == 1]
df_normais = df[df["Class"] == 0].sample(n=10000, random_state=42)
df_projeto = pd.concat([df_fraudes, df_normais]).sample(frac=1, random_state=42).reset_index(drop=True)

# Tratamento da feature de valor da transação
df_projeto["Amount_log"] = np.log1p(df_projeto["Amount"])
scaler = StandardScaler()
df_projeto["Amount_scaled"] = scaler.fit_transform(df_projeto[["Amount_log"]])

# ==========================================
# 2. ENGENHARIA DE FEATURES COM GRAFOS (CIBERSEGURANÇA)
# ==========================================
# Simulando dados de infraestrutura digital (User_ID e Device_ID) para criar o grafo
np.random.seed(42)
df_projeto['User_ID'] = [f"User_{i}" for i in np.random.randint(1, 4000, size=len(df_projeto))]
df_projeto['Device_ID'] = [f"Device_{i}" for i in np.random.randint(1, 2500, size=len(df_projeto))]

# Criando o grafo bipartido transação-infraestrutura
G = nx.Graph()

for idx, row in df_projeto.iterrows():
    G.add_node(row['User_ID'], type='user')
    G.add_node(row['Device_ID'], type='device')
    G.add_edge(row['User_ID'], row['Device_ID'])

# Calculando a Centralidade de Grau (Degree Centrality)
# Dispositivos muito conectados a vários usuários indicam comportamento suspeito (Botnets / Emuladores)
degree_centrality = nx.degree_centrality(G)

# Injetando as métricas estruturais do grafo de volta no DataFrame como novas features
df_projeto['Device_Centrality'] = df_projeto['Device_ID'].map(degree_centrality)
df_projeto['User_Centrality'] = df_projeto['User_ID'].map(degree_centrality)

# ==========================================
# 3. PREPARAÇÃO PARA O MODELO DE MACHINE LEARNING
# ==========================================
# Dropamos os IDs textuais, colunas antigas e a label para o treino
x = df_projeto.drop(["Class", "User_ID", "Device_ID", "Amount", "Amount_log"], axis=1)
y = df_projeto["Class"]

# Divisão em treino e teste mantendo a proporção de fraudes (stratify)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=42)

# ==========================================
# 4. TREINAMENTO E AJUSTE DE SENSIBILIDADE (XGBOOST)
# ==========================================
# scale_pos_weight ajuda a lidar com o desbalanceamento restante penalizando o erro na classe de fraude
xgb = XGBClassifier(
    scale_pos_weight=15, 
    eval_metric="logloss",
    random_state=42
)
xgb.fit(x_train, y_train)

# Em segurança, baixamos o threshold padrão (0.5) para não deixar fraudes passarem
y_probs = xgb.predict_proba(x_test)[:, 1]
custom_threshold = 0.3
y_pred_custom = (y_probs > custom_threshold).astype(int)

print("\n=== RELATÓRIO DE PERFORMANCE (MODELO + GRAFO) ===")
print(classification_report(y_test, y_pred_custom))

# ==========================================
# 5. VISUALIZAÇÃO DAS FEATURES E EXPLICABILIDADE
# ==========================================
# Gráfico de importância de features para provar o peso das variáveis de grafo
importancias = xgb.feature_importances_
features_nomes = x.columns

plt.figure(figsize=(10, 6))
plt.barh(features_nomes[-4:], importancias[-4:], color='darkblue') # Foco nas últimas features criadas
plt.title("Importância das Variáveis de Contexto e Rede")
plt.xlabel("Score de Importância")
plt.show()

# Análise SHAP para abrir a caixa-preta do modelo
explainer = shap.Explainer(xgb)
shap_values = explainer(x_test[:100])
shap.plots.bar(shap_values)
# ==========================================
# 6. GERAÇÃO E VISUALIZAÇÃO DO GRÁFICO DE GRAFOS
# ==========================================
# Vamos pegar uma amostra menor do grafo principal para o desenho não virar uma "mancha preta" ilegível
df_amostra_grafo = pd.concat([
    df_projeto[df_projeto["Class"] == 1].head(30), # 30 fraudes
    df_projeto[df_projeto["Class"] == 0].head(150) # 150 normais
])

G_visual = nx.Graph()
for idx, row in df_amostra_grafo.iterrows():
    G_visual.add_node(row['User_ID'], type='user', label=row['Class'])
    G_visual.add_node(row['Device_ID'], type='device')
    G_visual.add_edge(row['User_ID'], row['Device_ID'])

# Mapeia as cores: Vermelho para fraudadores, Azul para legítimos, Cinza para os dispositivos
cores_nos = []
for node, data in G_visual.nodes(data=True):
    if data.get('type') == 'device':
        cores_nos.append('lightgray')
    else:
        if data.get('label') == 1:
            cores_nos.append('red')
        else:
            cores_nos.append('skyblue')

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G_visual, k=0.18, seed=42)

# Desenha a estrutura da nossa rede de acessos
nx.draw_networkx_nodes(G_visual, pos, node_size=60, node_color=cores_nos)
nx.draw_networkx_edges(G_visual, pos, alpha=0.3, edge_color="silver")

plt.title("Rede de Transações: Vermelho (Fraude) compartilhando infraestrutura com outras contas")
plt.axis('off')

# Salva a imagem da rede na pasta de imagens
plt.savefig('images/grafico_grafo.png', bbox_inches='tight')
plt.show()