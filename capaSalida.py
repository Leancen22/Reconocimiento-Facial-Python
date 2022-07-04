from xml.etree.ElementPath import prepare_predicate
import cv2 as cv
import os

dataRuta = 'C:/Users/Leandro Mesa/Desktop/Reconocimiento Facial/data'
listaData = os.listdir(dataRuta)

entrenamientoEigenFaceRecognizer = cv.face.EigenFaceRecognizer_create()
entrenamientoEigenFaceRecognizer.read('EntrenamientoEigenFaceRecognizer.xml')

ruido = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')

camara = cv.VideoCapture(0)

while True:
    respuesta, captura = camara.read()
    if respuesta==False: break

    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura = grises.copy()
    caras = ruido.detectMultiScale(grises, 1.3, 5)

    for (x, y, e1, e2) in caras:
        rostroCapturado = idcaptura[y: y + e2, x: x + e1]
        rostroCapturado = cv.resize(rostroCapturado, (160, 160), interpolation = cv.INTER_CUBIC)
        resultado = entrenamientoEigenFaceRecognizer.predict(rostroCapturado)
        cv.putText(captura, '{}'.format(resultado), (x, y - 5), 1, 1.3, (0, 255, 0), 1, cv.LINE_AA)

        if resultado[1] < 8200:
            cv.putText(captura, '{}'.format(listaData[resultado[0]]), (x, y - 20), 2, 1.3, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 0, 0), 2)
        else: 
            cv.putText(captura, 'No encontrado', (x, y - 20), 2, 1.3, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x, y), (x + e1, y + e2), (255, 0, 0), 2)
    
    cv.imshow("Resultados ", captura)

    if (cv.waitKey(1) == ord('s')):
        break

camara.release()
cv.destroyAllWindows()