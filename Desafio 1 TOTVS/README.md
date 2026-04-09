No ambiente acadêmico, pesquisadores lidam com bases de dados massivas, ainda mais quando estamos falando de revisões sistematicas. Identificar manualmente quais artigos são essenciais para a dissertação cruzando critérios de atualidade e impacto é um processo lento e sujeito a erros. Este projeto resolve esse problema através de um pipeline ETL (Extract, Transform, Load). Mesmo que atualmente as bases de dados ja tenham metodos de filtragem de artigos por data de publicação, este desafio foi apenas algo demonstrativo que é possivel utilizar dados dos artigos para caracteriza-los de acordo com a sua necessidade

Tecnologias Utilizadas
Python 3.11
Pandas: A principal biblioteca para manipulação e tratamento de dados.
VS Code: Ambiente de desenvolvimento.

O Pipeline ETL
1. Extração (Extract)
O script realiza a leitura de arquivos brutos em formato .csv contendo metadados de artigos científicos, garantindo a integridade de caracteres especiais e acentuação através do encoding utf-8.

2. Transformação (Transform)
Nesta etapa, o dado bruto é convertido em inteligência:
Data Cleaning: Remoção de espaços em branco e padronização de termos.
Lógica de Relevância: Aplicação de uma função customizada que categoriza os artigos em:
ESSENCIAL: Artigos com alto impacto publicados a partir de 2022.
IMPORTANTE: Artigos recentes ou com alto impacto.
COMPLEMENTAR: Leituras de apoio.
Padronização: Normalização dos setores urbanos para facilitar filtragens futuras.

3. Carregamento (Load)
Geração de um novo arquivo .csv enriquecido com os insights de relevância, pronto para ser importado em ferramentas de análise bibliométrica (como Biblioshiny, por exemplo, que é o que estou utilizando na minha pesquisa no momento) ou dashboards de visualização.