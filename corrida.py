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
hiago = "🐧" 
leticia = "🐈" 

voa_hiago = 0
voa_leticia = 0

vencedor = None

while not vencedor:


    limpar_tela()
    
    random1 = random.randint(0, 1)
    random2 = random.randint(1, 2) # kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
    
    print(f'anterior: {voa_hiago=}')
    print(f'passada : {voa_leticia=}')

    voa_hiago += random1
    voa_leticia += random2
    
    print(f'{voa_hiago=}')
    print(f'{voa_leticia=}')

    if voa_hiago >= linha_chegada and voa_leticia >= linha_chegada:
        vencedor = "Empate!"
    elif voa_hiago >= linha_chegada:
        vencedor = "Hiago venceu! 🐧"
    elif voa_leticia >= linha_chegada:
        vencedor = "Leticia venceu! 🐈"
    
    pista1 = "=" * voa_hiago + hiago + "-" * (linha_chegada - voa_hiago)
    pista2 = "." * voa_leticia + leticia + "-" * (linha_chegada - voa_leticia)
    
    calc1 =  "-" * voa_hiago
    calc2 =  "-" * voa_leticia

    calc1_2 = linha_chegada - voa_hiago
    calc2_2 = linha_chegada - voa_hiago

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
