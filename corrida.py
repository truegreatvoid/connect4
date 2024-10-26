import os
import time
import random
import sys

def limpar_tela():
    if os.name == 'nt':
        os.system('cls')
    else: 
        os.system('clear')

comprimento_pista = 50 
linha_chegada = comprimento_pista
carro1 = "🐧" 
carro2 = "🐉" 

voa_carro1 = 0
voa_carro2 = 0

vencedor = None

while not vencedor:


    limpar_tela()
    
    random1 = random.randint(1, 3)
    random2 = random.randint(1, 3)
    
    print(f'anterior: {voa_carro1=}')
    print(f'passada : {voa_carro2=}')

    voa_carro1 += random1
    voa_carro2 += random2
    
    print(f'{voa_carro1=}')
    print(f'{voa_carro2=}')

    if voa_carro1 >= linha_chegada and voa_carro2 >= linha_chegada:
        vencedor = "Empate!"
    elif voa_carro1 >= linha_chegada:
        vencedor = "Jogador 1 venceu!"
    elif voa_carro2 >= linha_chegada:
        vencedor = "Jogador 2 venceu!"
    
    pista1 = "=" * voa_carro1 + carro1 + "-" * (linha_chegada - voa_carro1)
    pista2 = "." * voa_carro2 + carro2 + "-" * (linha_chegada - voa_carro2)
    
    calc1 =  "-" * voa_carro1
    calc2 =  "-" * voa_carro2

    calc1_2 = linha_chegada - voa_carro1
    calc2_2 = linha_chegada - voa_carro1

    print(f'valores pista 1: {calc1=}')
    print(f'valores pista 2: {calc2=}')
    print(f'valores pista 1: {calc1_2=}')
    print(f'valores pista 2: {calc2_2=}')

    print(f'valores pista 1: {pista1=}')
    print(f'valores pista 2: {pista2=}')


    print(f'{pista1=}')
    print(f'{pista2=}')

    print("hiago:   " + pista1)
    print("leticia: " + pista2)
    
    if vencedor:
        print("\n" + vencedor)
        break
    
    time.sleep(1)

while True:
    resposta = input("\nDeseja jogar novamente? (s/n): ").strip().lower()
    if resposta == 's':
        voa_carro1 = 0
        voa_carro2 = 0
        vencedor = None
        break
    elif resposta == 'n':
        print("Obrigado por jogar!")
        sys.exit()
    else:
        print("Por favor, digite 's' para sim ou 'n' para não.")
