# Processamento-de-Linguagem-Natural

Equipe: Vitor Hugo Koehler, Daniel Melere Loes

Este projeto consiste em um pipeline automatizado de Web Scraping e Processamento de Linguagem Natural (PLN) desenvolvido em Python. O objetivo é extrair documentos de licitações públicas do portal do Diário Oficial dos Municípios de Santa Catarina, higienizar os textos e estruturar os dados em um formato padronizado e acessível (XML).

O script resolve o problema de dados governamentais amontoados e não estruturados, transformando-os em uma base limpa e legível para futuras tarefas de análise de dados ou treinamento de modelos de linguagem.

Tecnologias Utilizadas Linguagem: Python Coleta de Dados: requests, bs4 (BeautifulSoup) Processamento de Linguagem Natural: spacy, nltk, re (Expressões Regulares) Estruturação: xml.etree.ElementTree, xml.dom.minidom
