from configparser import Interpolation
from turtle import width
import cv2 as cv
import numpy as np
import os
import imutils

modelo = 'Leandro'
ruta1 = 'C:/Users/Leandro Mesa/Desktop/Reconocimiento Facial'
rutaCompleta = ruta1+ '/' + modelo

if not os.path.exists(rutaCompleta):
    os.makedirs(rutaCompleta)

camara = cv.VideoCapture(0)
ruido = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

id = 0

while True:
    respuesta,captura = camara.read()

    if respuesta == False: break

    captura = imutils.resize(captura, width=640)

    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idCaptura=captura.copy()
    cara = ruido.detectMultiScale(grises, 1.3, 5)

    for (x, y, e1, e2) in cara:
        cv.rectangle(captura, (x, y), (x+e1, y+e2), (0, 255, 0), 2)
        rostroCapturado = idCaptura[y: y + e2, x: x + e1]
        rostroCapturado = cv.resize(rostroCapturado, (160, 160), interpolation = cv.INTER_CUBIC)
        cv.imwrite(rutaCompleta+'/imagen_{}.jpg'.format(id), rostroCapturado)
        id = id + 1

    cv.imshow("Resultado rostro", captura)

    if id == 351:
        break

camara.release()
cv.destroyAllWindows()