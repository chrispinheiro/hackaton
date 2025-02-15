# hackaton
Pós Tech IA para Devs - Fiap - Hackaton 

Grupo 50


**Integrantes:**
  - Ana Paula de Sa Lopes de Simone
  - Christiane Pinheiro Campelo da Silva
  - Leandro Juvenal Marques


**Apresentação:**  
  Esse projeto apresenta o MVP de uma solução de monitoramento de câmeras de segurança para a empresa FIAP VisionGuard.


**Objetivos:**  
  Identificar situações atípicas que possam colocar em risco a segurança de estabelecimentos e comércios.  
  Utilizar IA para identificar objetos cortantes (facas, tesouras e similares) e emitir alertas para a central de segurança.


**Implementação:**
  01) Criação do Dataset
    Seleção de imagens
    Uso do Roboflow para anotação das imagens
  
  02) Armazenamento do Dataset no Google Drive
  
  03) Uso do Colab para treinar o modelo YOLO11 (yolo11x.pt)
  
  04) Arquivos para testes/predição do modelo treinado  
    - yolo11x_custom.pt  
    - predict.py
  
  05) Resultados armazenados em ./runs/detect/predictN
  
  06) Retorno na tela (desenho da border box e aviso sonoro na tela)


**Utilização:**  
  Executar o script predict_02.py   
  Escolher uma das fontes de informação (imagem, vídeo ou camera)  
  Apontar caminho do arquivo escolhido na etapa anterior, se for o caso  
