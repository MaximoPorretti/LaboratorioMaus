import cv2
import numpy as np
import SeguimientosDeManos as sm  # Detección y seguimiento de manos
import pyautogui  # Control del mouse
import time

# Configuración de la cámara y pantalla
anchocam, altocam = 640, 480
cuadro = 50  # Área de interacción
anchopanta, altopanta = pyautogui.size()  # Resolución de pantalla
suavizado = 5  # Factor de suavizado
pubix, pubiy = 0, 0  # Posición previa del cursor
click_anterior = False  # Estado del clic
sostener_click = False  # Estado del clic sostenido
ptiempo = time.time()  # Para calcular FPS

# Inicialización de la cámara
cap = cv2.VideoCapture(0)
cap.set(3, anchocam)
cap.set(4, altocam)

# Detector de manos
detector = sm.detectormanos(maxManos=1)  # Solo una mano para mejor control

while True:
    ret, frame = cap.read()
    frame = detector.encontrarmanos(frame)
    lista, bbox, _ = detector.encontrarposicion(frame)

    # Dibujar área de interacción
    cv2.rectangle(frame, (cuadro, cuadro), (anchocam - cuadro, altocam - cuadro), (0, 255, 0), 2)

    if len(lista) != 0:
        x_indice, y_indice = lista[8][1:]

        # Mapear coordenadas a la pantalla
        x_cursor = np.interp(x_indice, (cuadro, anchocam - cuadro), (0, anchopanta))
        y_cursor = np.interp(y_indice, (cuadro, altocam - cuadro), (0, altopanta))

        # Suavizar el movimiento
        cubix = pubix + (x_cursor - pubix) / suavizado
        cubiy = pubiy + (y_cursor - pubiy) / suavizado

        # Detección de dedos arriba
        dedos = detector.dedosarriba()

        # Mueve el cursor SIEMPRE que el índice esté levantado o los 4 dedos estén arriba
        if dedos[1] == 1:
            pyautogui.moveTo(anchopanta - cubix, cubiy)
            pubix, pubiy = cubix, cubiy

        # Clic cuando el índice y el medio están levantados
        if dedos[1] == 1 and dedos[2] == 1 and dedos.count(1) == 2:
            if not click_anterior:
                pyautogui.click()
                click_anterior = True
        elif dedos.count(1) < 2:
            click_anterior = False

        # Doble clic cuando el índice, el medio y el anular están levantados
        if dedos[1] == 1 and dedos[2] == 1 and dedos[3] == 1 and dedos.count(1) == 3:
            pyautogui.doubleClick()

        # 🖌 **Dibujar en Paint mientras los cuatro dedos están levantados y mover el cursor**
        if dedos[1] == 1 and dedos[2] == 1 and dedos[3] == 1 and dedos[4] == 1:
            pyautogui.mouseDown()  # Mantiene el clic mientras te mueves
        else:
            pyautogui.mouseUp()  # Suelta el clic cuando no estén los cuatro dedos

    # Cálculo de FPS
    ctiempo = time.time()
    fps = 1 / (ctiempo - ptiempo)
    ptiempo = ctiempo
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen
    cv2.imshow("Mouse Virtual", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Salir con ESC
        break

cap.release()
cv2.destroyAllWindows()
