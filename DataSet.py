#Este código crea y entrena una red neuronal convolucional (CNN)
# para el reconocimiento de gestos a partir de imágenes, utilizando un dataset
# almacenado en un directorio local. Después de entrenar el modelo,
# lo guarda para su posterior uso. El modelo está diseñado para
# clasificar imágenes de gestos en categorías y es capaz de ser
# utilizado para predicciones sobre nuevas imágenes después de su entrenamiento.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Verificar si la ruta del dataset existe
dataset_dir = 'C:/Users/porre/OneDrive/Escritorio/Laboratorio/archive/train'
if not os.path.exists(dataset_dir):
    print("La ruta no existe:", dataset_dir)
else:
    # Si la ruta existe, listar las clases en el dataset
    classes = os.listdir(dataset_dir)
    print("Clases en el dataset:", classes)

# Nuevamente verificar la ruta y las clases
dataset_dir = 'C:/Users/porre/OneDrive/Escritorio/Laboratorio/archive/train'

# Listar las clases del dataset cargado
classes = os.listdir(dataset_dir)
print("Clases en el dataset:", classes)

# Preprocesamiento de las imágenes para entrenamiento y validación
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   validation_split=0.2)  # Normalizar las imágenes y dividir en conjunto de entrenamiento y validación

# Generador de datos para el conjunto de entrenamiento
train_generator = train_datagen.flow_from_directory(
    dataset_dir,  # Directorio donde están las imágenes
    target_size=(64, 64),  # Redimensionar las imágenes a 64x64 píxeles
    batch_size=32,  # Tamaño del lote para cada iteración
    class_mode='categorical',  # Multiclase, una categoría por imagen
    subset='training'  # Subconjunto para entrenamiento
)

# Generador de datos para el conjunto de validación
validation_generator = train_datagen.flow_from_directory(
    dataset_dir,  # Directorio donde están las imágenes
    target_size=(64, 64),  # Redimensionar las imágenes a 64x64 píxeles
    batch_size=32,  # Tamaño del lote para cada iteración
    class_mode='categorical',  # Multiclase, una categoría por imagen
    subset='validation'  # Subconjunto para validación
)

# Definición del modelo de red neuronal convolucional (CNN)
model = Sequential([
    # Capa convolucional con 32 filtros de tamaño 3x3, función de activación ReLU
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),

    # Capa de max pooling para reducir la dimensionalidad (2x2)
    MaxPooling2D(pool_size=(2, 2)),

    # Segunda capa convolucional con 64 filtros
    Conv2D(64, (3, 3), activation='relu'),

    # Otra capa de max pooling para reducción de dimensionalidad
    MaxPooling2D(pool_size=(2, 2)),

    # Aplanar las salidas para pasarlas a una capa densa
    Flatten(),

    # Capa densa con 128 unidades y función de activación ReLU
    Dense(128, activation='relu'),

    # Capa de salida con tantas neuronas como clases haya en el dataset y activación softmax
    Dense(train_generator.num_classes, activation='softmax')  # Salida de múltiples clases
])

# Compilamos el modelo con optimizador Adam y función de pérdida categorical_crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamos el modelo con el generador de entrenamiento y validación
model.fit(
    train_generator,  # Generador de datos de entrenamiento
    epochs=10,  # Número de épocas (iteraciones sobre todo el dataset)
    validation_data=validation_generator  # Generador de datos de validación
)

# Guardamos el modelo entrenado en un archivo .h5
model.save('gesture_recognition_model.h5')
