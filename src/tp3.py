##################################################################################################################
#   Imports necesarios                                                                                           #
##################################################################################################################
import cv2
import matplotlib.pyplot as plt
import numpy as np


##################################################################################################################
#   Funciones                                                                                                    #
##################################################################################################################
# ---- Limpieza de bordes ----------------------------------------------------------------------
def imclearborder(f):
    '''
    Esta función elimina los elementos que están en contacto con los bordes
    '''
    kernel = np.ones((70,70),np.uint8)
    marker = f.copy()
    marker[1:-1,1:-1]=0
    while True:
        tmp=marker.copy()
        marker=cv2.dilate(marker, kernel)
        marker=cv2.min(f, marker)
        difference = cv2.subtract(marker, tmp)
        if cv2.countNonZero(difference) == 0:
            break
    mask=cv2.bitwise_not(marker)
    out=cv2.bitwise_and(f, mask)
    return out

# ---- Mostrar imagen --------------------------------------------------------------------------
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    '''
    Visualización de imágenes en pantalla
    '''
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show()

# ---- Detección de elementos rojos ------------------------------------------------------------
def deteccionRojo (frame):
    '''
    La función recibe un frame y aplica una mascara que sólo muestra los elementos ROJOS
    Si el color de los dados cambia, se pueden modificar los valores HSV de la máscara
    No se incluyen los valores como parámetros para simplificar el llamado de la función
    '''
    # Convertir frame a HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Máscara de selección de color 
    # El color rojo se encuentra en ambos extremos del mapa HSV
    # Por ese motivo es necesario combinar dos máscaras

    # Rango Mask 1
    rojo11 = np.array([0, 100, 0], np.uint8)
    rojo12 = np.array([10, 255, 255], np.uint8)
    mask1 = cv2.inRange(frame_hsv, rojo11, rojo12)

    # Rango Mask 2
    rojo21 = np.array([170, 100, 20], np.uint8)
    rojo22 = np.array([180, 255, 255], np.uint8)
    mask2 = cv2.inRange(frame_hsv, rojo21, rojo22)

    # Máscara combinada
    mask = cv2.add(mask1, mask2)

    # Aplica la máscara al fotograma original
    result_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return result_frame

# ---- Procesar frame --------------------------------------------------------------------------
def procesarFrame (frame, th=150):
    '''
    La función recibe un frame y un umbral, y lo procesa para su posterior análisis morfológico
    '''
    # Pasar a escala de grises
    frame_gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Umbralar
    _, frame_th = cv2.threshold(frame_gr, th, 255, cv2.THRESH_BINARY)

    # Clausura para unificar la cara del dado
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))
    frame_cl = cv2.morphologyEx(frame_th, cv2.MORPH_CLOSE, k)

    # Eliminar bordes
    frame_ok=imclearborder(frame_cl)

    return frame_ok

# ---- Detección de movimiento -----------------------------------------------------------------
def compararFramesBin(frame_act, frame_ant, th_area=400):
    '''
    La función recibe dos frames y un umbral de área.
    Se comparan los frames y según las áreas de los componentes retorna:
    True  -> presentan diferencias con áreas mayores al umbral
    False -> presentan diferencias con áreas menores al umbral (o son iguales)
    El frame resultante de la comparación
    '''

    flag_dif = False
    frame_dif = cv2.absdiff(frame_ant, frame_act)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame_dif)
    for i in range(1, num_labels):
        if stats[i][4] > th_area:
            flag_dif = True
    
    return flag_dif, frame_dif

# ---- Detección de 5 dados en la imagen -------------------------------------------------------
def cantObjetos(frame, area_min = 3000, area_max=5500, cant=5):
    '''
    La función recibe un frame y los siguientes parámetros:
    - area_min / area_max: rango de área para los objetos a detectar
    - cant: cantidad de objetos dentro de ese rango de área
    Retorna True si encuentra exactamente la cantidad de objetos dentro del rango indicado
    '''
    count = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame)
    for i in range(1, num_labels):
        if area_min < stats[i][4] < area_max:
            count += 1
    return cant == count
    
# ---- Análisis de dados -----------------------------------------------------------------------
def analisisDados (frame, frame_out, area_min = 3000, area_max=5500):
  '''
  La función recibe un frame procesado y el frame original. 
  En ambos frames los dados están posicionados para ser analizados.
  Se utiliza el rango de área y función imclearborder() para eliminar ruido.
  - Procesa la imagen para tomar el número de la cara visible del dado
  - Identifica el número
  - Aplica el bounding box 
  - Agrega el número correspondiente a la cara visible del dado
  - Retorna un frame con la información aplicada
  '''

  # Obtener componentes conectadas
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frame)

  # Identificar número de dados y bounding box
  for i in range(1, num_labels):
    # Filtro los objetos por área
    if area_min < stats[i][4] < area_max:
        # Selecciono el objeto actual
        obj = (labels == i).astype(np.uint8)

        # Número de la cara del dado según la cantidad de huecos
        contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        numero = len(contours) - 1

        # Debug
        #print(f"Dado {i:2d} --> Número {numero}")

        # Fijar coordenadas para bounding box
        x=stats[i][0] 
        y=stats[i][1]
        w=stats[i][2]
        h=stats[i][3]

        # Dibujar bounding box de color azul
        cv2.rectangle(frame_out, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Parametrización de texto
        # Configurar la fuente y otros parámetros del texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        font_thickness = 2
        pos=(x-10,y-10)

        # Dibujar el texto sobre la imagen
        cv2.putText(frame_out, str(numero), pos, font, font_scale, font_color, font_thickness)


#################################################################################################################
#   Algoritmo principal                                                                                         #
#################################################################################################################

def procesarVideoDados(path_input, path_output, im_show=False):
    '''
    Esta función resume el procesamiento de los videos y genera el video de salida
    '''
    print("Procesando video: ", path_input)
    # Capturar video de entrada
    cap = cv2.VideoCapture(path_input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Generar video de salida
    out = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    # Leer primer frame
    ret, frame = cap.read()
    frame_num = 1

    # Aplicar filtro de color al primer frame
    frame1_rojos = deteccionRojo(frame)

    # Procesar primer frame
    prev_frame = procesarFrame(frame1_rojos)

    # Generar lista de frames
    frame_list=[prev_frame]

    # Inicializar contador de frames para indicador de dados detenidos
    contador = 0

    # Ventana de visualización
    cv2.startWindowThread()

    # Leer el resto del video
    while True:
        ret, frame = cap.read()
        
        # Fin de los frames
        if not ret:
            break

        # Tratamiento de frames
        frame_rojos = deteccionRojo(frame)
        frame_proc = procesarFrame(frame_rojos, 30)

        # Identificar frames contiguos similares
        comp_res, frame_dif = compararFramesBin(frame_proc, prev_frame, 400)
        if not comp_res:

            # Identificar al menos 4 frames contiguos que sean similares y donde se muestran los 5 dados detenidos
            if cantObjetos(frame_proc):
                contador += 1
                if contador > 3:
                    
                    # Debug
                    #print("STOP", contador-3)   # Se muestra un conteo de los frames donde los dados están detenidos
                    #break

                    # Identificación de dados
                    analisisDados(frame_proc, frame)
                    
                    # Debug
                    #break

            else:
                contador = 0
        
        # Mostrar frames en pantalla
        if im_show:
            cv2.imshow('Frame', frame_proc)
            if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
                break

        # Grabar frames en video de salida
        out.write(frame)
        # Debug
        #print(frame_num, comp_res)

        # Actualizar lista, frame anterior y contador general
        frame_list.append(frame_proc)
        prev_frame = frame_proc
        frame_num +=1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video generado: ", path_output)


#################################################################################################################
#   Procesamiento                                                                                               #
#################################################################################################################

procesarVideoDados('tirada_1.mp4','tirada_1_proc.mp4', im_show=False)

procesarVideoDados('tirada_2.mp4','tirada_2_proc.mp4', im_show=False)

procesarVideoDados('tirada_3.mp4','tirada_3_proc.mp4', im_show=False)

procesarVideoDados('tirada_4.mp4','tirada_4_proc.mp4', im_show=False)


