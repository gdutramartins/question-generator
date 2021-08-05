 

# Gerador de Perguntas Dinâmico Utilizando T5 e PTT5 em Português/Inglês



Baseado no incrível trabalho realizado por *Suraj Patil* - [Question Generation Using Transformers](https://github.com/patil-suraj/question_generation)

Para melhorar a acurácia de buscas semânticas utilizando similaridade para um modelo de perguntas e respostas (QA), faz-se necessário treinar o modelo de similaridade (Sentence Transformer) com perguntas relacionadas ao respectivo contexto, dessa forma a representação vetorial de uma sentença ficará mais próxima das perguntas que seriam feitas e assim encontraremos mais facilmente o contexto onde está a resposta para que depois seja extraída a parte do texto onde se encontra a resposta desejada (span com a resposta).

A geração manual de perguntas é cara e trabalhosa, por isso fizemos uma prova de conceito de um modelo utilizando Transformers/BERT que monta perguntas automaticamente para um contexto informado. Treinamos um modelo para língua inglesa e outro para português, utilizando diferentes datasets. 

Os modelos pré-treinados para *fine tuning*  utilizados foram os seguintes: T5 da Google (Text To Text Transfer Transformer) e PTT5 da Unicamp (Text To Text Transfer Transformer em Português). Utilizamos o SQUAD 1.1 para treino na língua inglesa e o SQUAD 1.1 pt_br para treino em português.

O projeto é composto por três notebooks:

- Parte 1 - Preparação dos Dados - Baixar o dataset  de acordo com o idioma escolhido, adaptação do tokenizador e geração dos *tensors* que representam cada sentença, ou seja, um arquivo com o os input id's (embeddings) e attention masks.
- Parte 2 - Treino - Treino para *fine tuning* do modelo de acordo com o idioma (T5 ou PTT5), salvando o modelo para uso posterior.
- Parte 3 - Execução do modelo com exemplos de perguntas geradas em português e inglês.



O resultado final é interessante, mas eu esperava uma quantidade maior de perguntas geradas, na maioria dos casos temos uma ou duas perguntas. O framework [BEIR](https://github.com/UKPLab/beir) me parece conseguir gerar mais perguntas, contudo ele funciona com a língua inglesa e eu queria trabalhar com português também.

Para futuros estudos poderíamos utilizar um dataset maior, incorporando [MS Marco dataset](https://microsoft.github.io/msmarco/), contudo os datasets em sua maioria estão disponíveis somente na língua inglesa.



**Links que estudei para criação do modelo:**

- [Question Generation Using Transformers](https://github.com/patil-suraj/question_generation)

- [Asking the Right Questions: Training a T5 Transformer Model on a New task](https://towardsdatascience.com/asking-the-right-questions-training-a-t5-transformer-model-on-a-new-task-691ebba2d72c)

- [Generating Questions Using Transformers](https://amontgomerie.github.io/2020/07/30/question-generator.html).

-  [PTT5: Pretraining and validating the T5 model on Brazilian Portuguese data](https://github.com/unicamp-dl/PTT5)

- [Semantic Search with S-BERT is all you need](https://medium.com/mlearning-ai/semantic-search-with-s-bert-is-all-you-need-951bc710e160).

- Beir -  [A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://github.com/UKPLab/beir).

  

