from dataclasses import dataclass
from gettext import npgettext
import cv2 as cv
import os
import numpy as np
from time import time

dataRuta = 'C:/Users/Leandro Mesa/Desktop/Reconocimiento Facial/data'
listaData = os.listdir(dataRuta)

#print('data', listaData)

ids = []
rostrosData = []

id = 0

tiempoInicial = time()
for fila in listaData:
    rutaCompleta = dataRuta+'/'+fila
    print('Iniciando lectura')
    for archivo in os.listdir(rutaCompleta):
        print('Imagenes: ', fila + '/' + archivo)
        ids.append(id)
        rostrosData.append(cv.imread(rutaCompleta+'/'+archivo, 0))

    id = id + 1
    tiempoLecturaFinal = time()
    tiempoLectura = tiempoLecturaFinal - tiempoInicial
    print('Tiempo total: ', tiempoLectura)




entrenamientoEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()

print('Inicio del entrenamiento')

entrenamientoEigenFaceRecognizer.train(rostrosData, np.array(ids))

tiempoFinalEntrenamiento = time()
tiempoTotalEntrenamiento = tiempoFinalEntrenamiento - tiempoLectura

print('Tiempo entrenamiento total ', tiempoTotalEntrenamiento)

entrenamientoEigenFaceRecognizer.write('EntrenamientoEigenFaceRecognizer.xml')

print('Entrenamiento concluido')