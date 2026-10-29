# Processamento-de-Linguagem-Natural

**Equipe:** Vitor Hugo Koehler, Daniel Melere Loes  
**Disciplina:** Processamento de Linguagem Natural (2026/2) — **Prof. Maiko Spiess** (FURB)

Este projeto consiste em um pipeline automatizado de *Web Scraping* e Processamento de Linguagem Natural (PLN) desenvolvido em Python. O objetivo é extrair documentos de licitações públicas do portal do Diário Oficial dos Municípios de Santa Catarina (DOM/SC), higienizar os textos, estruturar os dados em formatos padronizados (`.xml` e `.csv`) e aplicar representações vetoriais e modelos de PLN.

O projeto resolve o problema de dados governamentais amontoados e não estruturados, transformando-os em uma base limpa e legível para análise de dados, recuperação de informação e treinamento de modelos de linguagem.

## Arquivos do Repositório

| Arquivo | Etapa | Descrição |
| :--- | :--- | :--- |
| `extracao_licitacoes.ipynb` / `extracao_licitacoes_etapa2_aprimorado.ipynb` | **Etapas 1 e 2** | Coleta via *Web Scraping* no DOM/SC, limpeza, tokenização, normalização, remoção de *stopwords*, lematização (`spaCy`) e *stemming* (`NLTK`). |
| `licitacoes_blumenau.csv` | **Base de Dados** | Base consolidada com **66 publicações** da Prefeitura Municipal de Blumenau e **17 colunas** pré-processadas. |
| `trabalho_unidade_3_pln_licitacoes.ipynb` | **Unidade 3 (Notebook)** | Implementação completa da **Unidade 3 (Template 3)** para execução no Google Colab ou Jupyter. |
| `trabalho_unidade_3_pln_licitacoes.py` | **Unidade 3 (Script)** | Código Python da **Unidade 3** para execução via VS Code ou Terminal. |
| `Relatorio_Templates_2_e_3_PLN.docx` | **Relatório** | Relatório acadêmico com todas as respostas, tabelas de métricas e figuras incorporadas. |
| `figuras/` | **Gráficos** | Figuras geradas na Unidade 3 (`clusters_pca.png`, `silhouette_k.png`, `matriz_confusao.png` e `grafo_relacoes.png`). |

## O que foi desenvolvido na Unidade 3 (`trabalho_unidade_3_pln_licitacoes.ipynb`)

1. **Quatro Representações Textuais:**
   - **Bag of Words (BoW):** `CountVectorizer(ngram_range=(1, 2))`
   - **TF-IDF:** `TfidfVectorizer(ngram_range=(1, 2))`
   - **Word2Vec:** `Skip-gram (vector_size=100)` com *mean pooling* dos tokens
   - **BERTimbau:** `neuralmind/bert-base-portuguese-cased` com *mean pooling* ponderado pela máscara de atenção
2. **Clusterização + PCA (`K-Means`):** Avaliação para $k \in \{2, 3, 4, 5, 6\}$ com métricas intracluster e intercluster (*Inércia WCSS*, *Coeficiente de Silhueta*, *Davies-Bouldin* e *Calinski-Harabasz*).
3. **Sistema de Busca e Recomendação (`Top-3` e `Top-5`):** Recuperação por Similaridade do Cosseno comparando consultas lexicais, semânticas e administrativas nos 4 espaços vetoriais.
4. **Classificação Supervisionada e Modelo `Jev` (`typesafe-sdk`):** Classificação binária (`eh_abertura`) e multiclasse (`modalidade`), teste em frases inéditas e especificação Zero-Shot com `Noul`, `Choice` e `Score`.
5. **Análise de Sentimentos, NER e Extração de Relações com Grafo:** Avaliação de polaridade normativa, reconhecimento de entidades (`PER`, `ORG`, `LOC`) com `spaCy` e extração de triplas `(Sujeito, Relação, Objeto)` via *Dependency Parsing* (`nsubj`, `obj`, `obl`) plotadas com `networkx`.

## Como Executar a Unidade 3

### Opção 1 — Pelo Google Colab (Mais simples, igual à Etapa 1)

1. Abra o [Google Colab](https://colab.research.google.com/).
2. Vá em **Arquivo > Fazer upload de notebook** e selecione o arquivo `trabalho_unidade_3_pln_licitacoes.ipynb`.
3. Clique em **Ambiente de execução > Executar tudo** (ou pressione `Ctrl + F9`).
4. Caso você ainda não tenha colocado o arquivo `licitacoes_blumenau.csv` na aba lateral **Arquivos** do Colab, a primeira célula mostrará automaticamente um botão **Escolher arquivos** para você selecionar o `licitacoes_blumenau.csv` do seu computador.

### Opção 2 — Localmente no PC (VS Code ou Terminal)

```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
python trabalho_unidade_3_pln_licitacoes.py
```

## Tecnologias Utilizadas

- **Linguagem:** Python 3
- **Coleta e Estruturação:** `requests`, `BeautifulSoup4`, `pandas`, `xml.etree.ElementTree`
- **Pré-processamento, NER e Parsing:** `nltk`, `spacy` (`pt_core_news_sm`), `re`, `networkx`
- **Representações Vetoriais e Aprendizado de Máquina:** `scikit-learn`, `gensim` (`Word2Vec`), `transformers` & `torch` (`BERTimbau`), `typesafe-sdk` (`Jev`)
- **Visualização:** `matplotlib`
