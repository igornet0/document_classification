import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from prepare_image import *

def get_data(batch_size, target_size=(400, 400)):

    data_path = get_data_path()
    classes = {}
    matrix = np.zeros((5, 5), dtype=int)
    np.fill_diagonal(matrix, 1)
    for i, x in enumerate(data_path["Украина"]):
        classes[x] = matrix[i]

    for country, label in classes.items():
        print(f"{country}: {label}")

    print(f"all_files={col_files(data_path, 'dataset')}")
    images_labels = generater_images(data_path, classes, target_size=target_size, batch_size=batch_size)

    X = []
    y = []

    for image, label in images_labels:
        X.append(image)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(X[0].shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#23000 max
X_train, X_test, y_train, y_test = get_data(10000)

model = models.Sequential([
    layers.Input(shape=(*X_train[0].shape, 1)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(y_train[0]))  # Количество уникальных классов
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Настройка EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',  # Следим за изменением loss на валидации
                               patience=10,         # Количество эпох без улучшений после которых тренировка будет остановлена
                               verbose=1,           # Будет печатать сообщение при остановке
                               mode='min',          # Остановка когда значение monitor перестанет уменьшаться
                               restore_best_weights=True)  # Восстановление весов с лучшего шага

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=30,
                    callbacks=[early_stopping])

model.save('classification_model_400.keras')