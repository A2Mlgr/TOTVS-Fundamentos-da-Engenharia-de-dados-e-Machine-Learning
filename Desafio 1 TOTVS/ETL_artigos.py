import pandas as pd

# ETAPA 1: EXTRAÇÃO 
# Leitura de arquivo csv com dados de artigos cientificos relacionas a transições sustentaveis em ambientes urbanos
# O 'encoding' garante que os acentos ou caracteres especies não gerem erro 
df = pd.read_csv('artigos.csv', sep=';', encoding='utf-8')
print("Colunas encontradas:", df.columns.tolist())
#remoção de espaços em branco nas colunas
df['Impacto_Estimado'] = df['Impacto_Estimado'].str.strip()
print("✅ Dados extraídos com sucesso. Total de artigos:", len(df))

#ETAPA 2: TRANSFORMAÇÃO
# Foi criado um critério de caracterização de relevancia de acordo com os parametros estipulador por mim de acordo com o ano de publicação
def categorizar_artigo(row):
    if row['Impacto_Estimado'] == 'Alto' and row['Ano'] >= 2022:
        return "ESSENCIAL: Prioridade na Dissertação"
    elif row['Impacto_Estimado'] == 'Alto' or row['Ano'] >= 2023:
        return "IMPORTANTE: Revisar com atenção"
    else:
        return "COMPLEMENTAR: Leitura de apoio"

# A partir da geração de caracterização da relevancia, cria-se uma naova coluna que servirá como insight para a pesquisa
df['Relevancia_Tese'] = df.apply(categorizar_artigo, axis=1)

# Padronização os nomes dos setores para maiúsculas 
df['Setor_Urbano'] = df['Setor_Urbano'].str.upper()

print("Transformação concluída.")

#ETAPA 3: CARREGAMENTO
# Criação de um novo arquivo para que seja feita a analise 
df.to_csv('artigos_processados.csv', index=False, encoding='utf-8-sig')

print("Arquivo 'artigos_processados.csv' gerado e pronto para o Load!")