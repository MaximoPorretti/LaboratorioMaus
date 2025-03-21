#Resumen de lo que hace el código:
#Carga un modelo preentrenado.
#Prepara un conjunto de datos de prueba desde un directorio específico.
#Evalúa el rendimiento del modelo (pérdida y precisión) sobre el conjunto de datos de prueba.
#Define una función para predecir la clase de una imagen específica.
#Realiza una predicción sobre una imagen de prueba y muestra la clase predicha.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

# Cargar el modelo preentrenado desde un archivo .h5
model = tf.keras.models.load_model(
    'C:/Users/porre/PycharmProjects/Mouse-Virtual-con-Vision-Artificial/gesture_recognition_model.h5')

# Preparar los datos de prueba utilizando ImageDataGenerator
# Se escala el valor de los píxeles para que estén entre 0 y 1
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    'C:/Users/porre/OneDrive/Escritorio/Laboratorio/archive',  # Ruta a tu dataset de prueba
    target_size=(64, 64),  # Redimensionar las imágenes a 64x64 píxeles
    batch_size=32,  # Tamaño del lote para el generador de datos
    class_mode='categorical'  # El modelo tiene múltiples clases categóricas
)

# Evaluar el modelo con el conjunto de datos de prueba
# La función devuelve la pérdida y la precisión en el conjunto de test
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")  # Imprimir la pérdida
print(f"Test Accuracy: {accuracy}")  # Imprimir la precisión


# Función para predecir la clase de una imagen
def predict_image(img_path):
    """
    Esta función carga una imagen, la procesa y realiza una predicción sobre ella
    """
    # Cargar la imagen desde el path dado y redimensionarla a 64x64 píxeles
    img = load_img(img_path, target_size=(64, 64))

    # Convertir la imagen en un array y normalizar los valores de los píxeles
    img_array = img_to_array(img) / 255.0

    # Expandir las dimensiones para hacerla compatible con la entrada del modelo
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar la predicción sobre la imagen
    predictions = model.predict(img_array)

    # Obtener la clase con la mayor probabilidad (índice de la clase predicha)
    predicted_class = np.argmax(predictions, axis=1)

    # Imprimir la clase predicha
    print(f"Prediction: {predicted_class}")
    return predicted_class


# Ruta de una imagen de prueba
img_path = 'C:/Users/porre/OneDrive/Escritorio/Laboratorio/archive/test/test/3/905.jpg'  # Reemplazar con la ruta de tu imagen

# Realizar la predicción sobre la imagen de prueba
predict_image(img_path)
