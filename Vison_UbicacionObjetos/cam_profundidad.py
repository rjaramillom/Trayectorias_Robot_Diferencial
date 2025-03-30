import cv2
from depth_anything.inference import DepthAnythingInference

class Localizacion_objetos:
    def __init__(self):
        video_path = 'video_prueba.mp4'
        self.cap = cv2.VideoCapture(video_path)
        
        # Se carga el video
        if not self.cap.isOpened():
            print("Error en camara o video")
            return
        
        # Resollucion de imagenes 
        self.cap.set(3, 240)
        self.cap.set(4, 240)
        
        # Inicializar el modelo de DepthAnything
        self.depth_anything = DepthAnythingInference(model_path='depth_anything_vitb14.onnx', color=True)

    def frame_processing(self):
        output_file = 'profundidad.avi'  # Nombre del archivo 
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec de video
        fps = 20.0  # Frames por segundo
        frame_width = int(self.cap.get(3))  # Ancho de los frames
        frame_height = int(self.cap.get(4))  # Alto de los frames
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

        while True:
            t = cv2.waitKey(5)
            ret, frame = self.cap.read()
            if not ret:
                break

            # Realizar inferencia de profundidad
            img_depth = self.depth_anything.frame_inference(frame)

            # Centro como posición de referencia
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2

            # Obtener distancia entre objeto y posicion de referencias
            depth_value = img_depth[center_y, center_x]  # Valor de profundidad en el centro
            print(f"Distancia al objeto en la posición de referencia ({center_x}, {center_y}): {depth_value} unidades")

            # visualizacion
            cv2.imshow('depth anything', img_depth)
            cv2.imshow('frame in real time', frame)

            # fotograma de salida
            out.write(img_depth)

            # tecla ESC para finalizar 
            if t == 27:
                break
        
        # Cerramos
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()

# Crear una instancia de la clase y ejecutar el procesamiento de los frames
robot_intelligent = Localizacion_objetos()
robot_intelligent.frame_processing()
