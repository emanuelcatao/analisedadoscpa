# Análise Exploratória dos Dados da CPA da UFT — para a descoberta do conhecimento

Pipeline reprodutível de análise exploratória dos dados da Comissão Própria de Avaliação (CPA) da Universidade Federal do Tocantins, ciclo 2025, com 2024 como termo histórico de comparação. Código aberto que acompanha o TCC homônimo de Emanuel Catão (UFT, 2026).

## O que está aqui

O repositório tem dois artefatos centrais. O arquivo `analise_dados_2025.py` é o módulo Python com todas as funções analíticas, organizado em seis estágios encadeáveis. O notebook `analise_dados_2025.ipynb` orquestra esses estágios uma célula por vez, expõe os artefatos intermediários para inspeção e gera as figuras usadas no TCC, nos slides de defesa e no kit relatório CPA.

Os dados de entrada vivem em `bases_dados/`: as planilhas brutas das respostas de 2025 e 2024 e o arquivo de mapeamento manual das 51 questões diretamente comparáveis entre os dois instrumentos. As saídas são escritas em `outputs/2025/` (CSVs descritivos, matrizes Spearman, Cronbach, tabelas auxiliares para extração no LaTeX) e `figuras/2025/` (PNGs e PDFs das figuras analíticas), com `figuras/2025/kit_cpa/` reservado para o subconjunto entregue à CPA para uso no relatório anual.

## Visão analítica

O pipeline aplica o ciclo de KDD (Fayyad, Piatetsky-Shapiro e Smyth, 1996) na sua etapa exploratória (Tukey, 1977; Behrens, 1997). A escolha é descritiva, não confirmatória, por três razões articuladas: a escala Likert é ordinal (Stevens, 1946; Jamieson, 2004), a adesão é voluntária (4,8% no ciclo 2025) e o escopo do trabalho é interpretativo. Inferência paramétrica clássica pressupõe amostragem probabilística, que não temos; o trabalho assume isso explicitamente e descreve rigorosamente o que se pode descrever.

As quatro decisões metodológicas centrais são: consolidação por regra de prioridade Docente sobre Técnico sobre Discente sobre Egresso, evitando pseudorreplicação (Hurlbert, 1984) e reduzindo a base de 657 para 649 respondentes únicos; associação por Spearman sobre postos, apropriado para Likert e nunca substituído por Pearson; tratamento do NSO (não sei opinar) como categoria informativa, não dado faltante, com a taxa por questão reportada ao lado da média; e estratificação por campus em primeiro plano, não como apêndice da agregada.

## Estágios do pipeline

O módulo expõe seis estágios chamáveis individualmente a partir do notebook. O primeiro carrega a base bruta de 2025, aplica a regra de prioridade, determina o campus a partir da coluna do segmento consolidado, normaliza colunas Likert, binárias e condicionais e adiciona os scores compostos por eixo. O segundo calcula as estatísticas descritivas em sete recortes: por questão, por eixo SINAES, taxa de NSO total e por segmento, por campus, por campus cruzado com segmento e por curso (discentes com N maior ou igual a dez). O terceiro carrega o mapeamento questão-a-questão entre 2024 e 2025, identifica as questões universais (presentes para os quatro segmentos), calcula a matriz Spearman inter-eixos, a matriz Spearman questão-a-questão agregada e estratificada por segmento, os top pares por módulo de rho e o alfa de Cronbach por eixo.

O quarto faz a análise textual da questão q70 — o único campo aberto do instrumento — produzindo estatísticas básicas do corpus, frequência de termos, bigramas e categorização por eixo SINAES via dicionário temático alinhado à análise de conteúdo de Bardin (1977). O quinto carrega a base de 2024, aplica o mesmo pipeline de consolidação (com o catálogo de eixos reconstruído a partir do mapeamento) e calcula as variações por questão e as médias por eixo de cada ano. O sexto gera todas as figuras do TCC (capítulos de Fundamentação Teórica e Resultados), as figuras dos slides de defesa e o kit relatório CPA, salvando PNG e PDF de cada uma na pasta correspondente.

Cada estágio devolve um dicionário com os artefatos nomeados (DataFrames, dicionários, matrizes) e aceita o parâmetro `salvar=False` para uso em sessões de inspeção sem escrever CSVs no disco. Há também um wrapper `pipeline_2025()` que encadeia os seis estágios na ordem canônica e devolve o conjunto completo dos artefatos; é a entrada usada quando o arquivo é executado diretamente como script.

## Dependências

Bibliotecas Python necessárias: pandas, numpy, matplotlib, seaborn, openpyxl e wordcloud. Instalação via pip a partir de qualquer ambiente Python 3.11 ou 3.12. O backend matplotlib é detectado automaticamente: quando o módulo é importado por um notebook, o backend do host é preservado (permite display inline); quando é executado como script, força Agg para operação headless.

## Como rodar

Localmente, com o `.py` e o `.ipynb` no mesmo diretório, basta abrir o notebook em Jupyter, JupyterLab ou VS Code e executar as células em ordem. Cada estágio é uma célula de chamada da função nomeada seguida de células de inspeção dos artefatos. Para uma execução end-to-end sem inspeção intermediária, há a alternativa de chamar `cpa.pipeline_2025()` em uma única célula, ou rodar `python analise_dados_2025.py` a partir do terminal, que produz exatamente os mesmos CSVs e figuras.

No Google Colab, faça upload do projeto inteiro (ou da pasta `bases_dados/` mais o `analise_dados_2025.py`) para o diretório `/content/`, instale o `wordcloud` via pip dentro do notebook, adicione `/content` ao `sys.path` e importe o módulo normalmente. A forma mais prática é zipar a raiz do projeto local, fazer upload do zip e descompactar; alternativamente, montar o Google Drive se o projeto já estiver lá. As três células iniciais do notebook que tratam disso podem ser ajustadas conforme a estratégia escolhida.

## Recarregar o módulo após editar o `.py`

Python cacheia módulos importados em `sys.modules`; uma edição no `.py` não chega ao notebook automaticamente. Para forçar o reload sem reiniciar o kernel, basta chamar `importlib.reload(analise_dados_2025)` e re-importar o alias `cpa`, e depois re-executar as células dos estágios afetados. A extensão `%autoreload` do IPython não funciona no Python 3.12 atualmente usado pelo Colab (depende do módulo `imp`, removido nessa versão), então o reload manual é o caminho seguro.

## Onde cada figura aparece

As figuras didáticas da Fundamentação Teórica (boxplot, histograma, heatmap e nuvem de palavras) são geradas sobre os próprios dados da CPA 2025 em vez de ilustrações genéricas, mantendo coerência visual com o capítulo de Resultados. As figuras analíticas dos Resultados cobrem a distribuição por eixo (média e boxplot), a variação por segmento (radar e heatmaps de eixos 2-3 e 4-5), a variação por campus (radar e heatmap por questão), o cluster Spearman (matriz inter-eixos e pares estratificados), e a comparação temporal 2024 → 2025 (radar e gap). As figuras dos slides de defesa são, em sua maioria, reaproveitadas dos Resultados — três delas (boxplot estratificado por campus, distribuição segmento × campus e top gaps Disc-Doc) ainda são geradas por scripts standalone em `slides_tcc/` e ficam registradas como dívida técnica a portar para o pipeline principal. O kit relatório CPA é um conjunto adicional voltado para entrega à comissão: tabela campus × segmento, participação por segmento 2024 versus 2025, distribuição Likert percentual por eixo e por campus em painéis, setores para questões binárias e médias por eixo cruzadas com segmento e com campus.

## Reprodutibilidade

O pipeline é determinístico: mesma base de entrada produz exatamente as mesmas saídas. Sementes aleatórias usadas em jitter de scatter plots e outras posições visuais são fixadas explicitamente nas funções de figura. A paleta cromática é Okabe-Ito, colorblind-safe e padrão de publicação científica. O tempo aproximado para rodar o pipeline completo end-to-end numa máquina razoável é da ordem de dois minutos; a maior parte é geração de figuras, com os painéis de Likert por campus dominando o orçamento de tempo.

## Limitações reconhecidas

A coleta é voluntária e a adesão de 2025 foi de 4,8% da comunidade acadêmica — os achados valem para os respondentes do ciclo, não para a comunidade da UFT como um todo. A escala Likert é ordinal e limita as operações estatísticas legítimas; médias aparecem apenas como descrição, nunca como base para inferência paramétrica. A comparação temporal opera com apenas dois pontos no tempo e por isso não distingue tendência de flutuação. A análise textual usa dicionário temático simples sobre corpus modesto (247 comentários). A ambiguidade interpretativa do campus Palmas — autoseleção de protesto versus patamar crítico real — não é resolvida pelos dados disponíveis e fica registrada como linha de continuidade.
