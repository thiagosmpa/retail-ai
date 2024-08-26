# TESTE GONDOLAS

Esse documento tem por finalidade, implementar todos os conceitos para o projeto das gondolas.

Ou seja, as funções totais serão:

- DETECÇÃO DE OBJETOS (pessoas)
    - PARA CADA PESSOA IDENTIFICADA:
    - FACE DETECTION
    - QUANDO DETECTA A FACE:
        - ESTIMA O GENERO E IDADE DA PESSOA
- TRACKING DAS PESSOAS IDENTIFICADAS EM CENA
- CONTA QUANTOS SEGUNDOS A PESSOA ESTÁ EM CENA
- HEATMAP PARA ANÁLISE DO FLUXO DE PESSOAS
- FAZ OS LOGS EM JSON E CSV.
    - JSON:
        - TRACKING NUMBER
        - TEMPO
        - LAST TIME SEEN
    - CSV:
        - TRACKING NUMBER
        - FRAME ATUAL
        - POSX
        - POSY
- CONTA QUANTAS PESSOAS ENTRARAM OU SAÍRAM DE UMA ÁREA


Downloads:
- [Mivolo weights / mivolo checkpoint](https://drive.google.com/drive/folders/1FagDwoq8GfayuBLEye5IolINvF-9ixDO?usp=share_link)

Repositories:
- [mivolo model link](https://github.com/WildChlamydia/MiVOLO)



