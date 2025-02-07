import os

# Renomeia os arquivos do diretório  
# Ex: span_teste.img -> teste.img

PATH_REPLACE = "C:/Estudos/postech_ia_para_devs/anotar/temp/"
cont = 0
for filename in os.listdir(PATH_REPLACE):
    cont = cont + 1
    os.rename("C:/Estudos/postech_ia_para_devs/anotar/temp/"+filename, os.path.join(f"{PATH_REPLACE}imagemfaca{cont}.jpg"))


    