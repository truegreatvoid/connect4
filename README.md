
# Connect4-AI

**Connect4-AI** é um projeto acadêmico que desenvolve uma inteligência artificial capaz de jogar o jogo Connect 4 de forma competitiva. 
O projeto inclui a simulação de partidas, geração de dados de treinamento, treinamento de uma rede neural e integração da IA no jogo, 
permitindo que a IA tome decisões inteligentes durante as partidas.

## Índice
- [Visão Geral](#visão-geral)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Geração de Dados](#geração-de-dados)
- [Treinamento da Rede Neural](#treinamento-da-rede-neural)
- [Executando o Jogo](#executando-o-jogo)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Licença](#licença)

## Visão Geral

O projeto **Connect4-AI** tem como objetivo desenvolver uma inteligência artificial que possa jogar Connect 4 de maneira competitiva contra jogadores humanos ou outras IAs. 
A IA é implementada utilizando redes neurais com TensorFlow e Keras, capazes de aprender estratégias vencedoras a partir de dados de jogos simulados.

A estrutura do jogo está implementada nos scripts fornecidos, e o papel do dev é criar o script player, que integra a IA ao jogo, permitindo que ela escolha jogadas de forma inteligente.


## Tecnologias Utilizadas

- **Python 3.12**: Linguagem de programação utilizada para desenvolver o projeto.
- **TensorFlow**: Biblioteca de aprendizado de máquina para treinamento da rede neural.
- **Keras**: API de alto nível para construção e treinamento de modelos de redes neurais.
- **NumPy**: Biblioteca para manipulação de arrays e matrizes.
- **Matplotlib**: Biblioteca para visualização de dados e gráficos.
- **CSV**: Formato utilizado para armazenamento dos dados de treinamento.
- **Virtualenv**: Ambiente virtual para gerenciamento de dependências.

## Instalação

1. **Clone este repositório:**
    ```bash
    git clone https://github.com/truegreatvoid/connect4
    ```

2. **Entre no diretório do projeto:**
    ```bash
    cd connect4
    ```

3. **Crie um ambiente virtual (opcional, mas recomendado):**
    ```bash
    python3.12 -m venv venv_cleiton
    ./venv/bin/activate  # No Windows: source venv\Scripts\activate
    ```

4. **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## Geração de Dados

Antes de treinar a rede neural, é necessário gerar os dados de treinamento. O script [`gerar.py`](./redes_neurais/gerar.py) simula partidas de Connect 4 entre dois jogadores aleatórios e salva os estados do tabuleiro e as jogadas realizadas em um arquivo CSV.

1. **Execute o script de geração de dados:**
    ```bash
    python gerar.py
    ```

    Isso criará o arquivo [`connect4_data.csv`](./connect4_data.csv) com os dados de treinamento.

## Treinamento da Rede Neural

Com os dados de treinamento gerados, você pode treinar a rede neural utilizando o script [`treinamento.py`](./rede_neurais/treinamento.py).

1. **Execute o script de treinamento:**
    ```bash
    python treinamento.py
    ```

    Este script carregará os dados de [`connect4_data.csv`](./connect4_data.csv), treinará o modelo de rede neural e salvará o modelo treinado como [`connect4_model.h5`](./connect4_model.h5).

2. **Monitoramento do Treinamento:**
    Durante o treinamento, gráficos de perda e acurácia serão exibidos para monitorar o desempenho do modelo.

## Executando o Jogo

Após treinar o modelo, você pode executar o jogo Connect 4 com a IA integrada.

1. **Certifique-se de que o modelo treinado ([`connect4_model.h5`](./connect4_model.h5)) está no diretório raiz do projeto.**

2. **Execute o mediador do jogo:**
    ```bash
    python mediador.py
    ```

    O jogo iniciará, exibindo o tabuleiro e alternando entre o jogador aleatório e a IA.
    

## Estrutura do Projeto

Abaixo está uma visão geral dos principais arquivos e diretórios no repositório:

```
connect4-ai/
├── rede_neurais/gerar.py          # Script para simular partidas e gerar dados de treinamento
├── rede_neurais/treinamento.py          # Script para treinar a rede neural com os dados gerados
├── rede_neurais/aluno.py                # Script onde a lógica da IA está implementada
├── mediador.py                          # Script principal que gerencia as partidas do jogo
├── jogador_random.py                    # Script do jogador aleatório (oponente da IA)
├── connect4_data.csv                    # Arquivo CSV contendo os dados de treinamento
├── connect4_model.h5                    # Arquivo do modelo treinado da rede neural
├── requirements.txt                     # Lista de dependências do projeto
└── README.md                            # Este arquivo
```

### Descrição dos Scripts

- **gerar.py**: Simula partidas de Connect 4 entre dois jogadores aleatórios e registra os estados do tabuleiro e as jogadas realizadas em [`connect4_data.csv`](./connect4_data.csv).

- **treinamento.py**: Carrega os dados de [`connect4_data.csv`](./connect4_data.csv), treina a rede neural utilizando TensorFlow e Keras, e salva o modelo treinado em [`connect4_model.h5`](./connect4_model.h5).

- **mediador.py**: Gerencia as partidas do jogo, alternando entre o jogador aleatório e a IA. Exibe o tabuleiro utilizando Matplotlib e declara o vencedor.

- **aluno.py**: Implementa a lógica da IA, utilizando o modelo treinado para escolher a melhor jogada com base no estado atual do tabuleiro.

- **jogador_random.py**: Implementa a lógica de um jogador que faz jogadas aleatórias, servindo como oponente da IA.

## Licença

### Licença de Uso Educacional

Este projeto é de uso exclusivo para fins educacionais e acadêmicos na disciplina de Inteligência Artificial, ministrada pelo Professor Cleiton, 
do curso de Análise e Desenvolvimento de Sistemas (4º Período, 2024.02) na Faculdade Unibra. 
É destinado ao desenvolvimento de habilidades em inteligência artificial e aprendizado de máquina.

---
