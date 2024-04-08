import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from prepare_image import *
import logging

logging.getLogger('tensorflow').disabled = True

def get_data(batch_size, target_size=(250, 250), train_list=[]):

    data_path = get_data_path()
    classes = {}
    matrix = np.zeros((5, 5), dtype=int)
    np.fill_diagonal(matrix, 1)
    for i, x in enumerate(data_path["Украина"]):
        classes[x] = matrix[i]

    print(f"all_files={col_files(data_path, 'dataset')}")
    images_labels = generater_images(data_path, classes, target_size=target_size, batch_size=batch_size, train_list=train_list)
    X = []
    y = []
    for image, label in images_labels:
        X.append(image)
        y.append(label)
        if len(X) >= batch_size:
            break
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def get_data_test(target_size=(250, 250)):
    data_path = get_data_path("test")
    classes = {}
    matrix = np.zeros((5, 5), dtype=int)
    np.fill_diagonal(matrix, 1)
    for i, x in enumerate(data_path["Test"]):
        classes[x] = matrix[i]

    print(f"all_files_test={col_files(data_path, 'test')}")
    images_labels = generater_images(data_path, classes, dataset_path="test", target_size=target_size, batch_size=None)
    return images_labels

#23000 max

model = models.Sequential([
    layers.Input(shape=(250, 250, 3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64),
    layers.Dense(5, activation='softmax')  # Количество уникальных классов
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# Настройка EarlyStopping
bath_size = 10000

#model = tf.keras.models.load_model('classification_model_250-3.keras')


early_stopping = EarlyStopping(monitor='val_loss',  # Следим за изменением loss на валидации
                               patience=3,         # Количество эпох без улучшений после которых тренировка будет остановлена
                               verbose=1,           # Будет печатать сообщение при остановке
                               mode='min',          # Остановка когда значение monitor перестанет уменьшаться
                               restore_best_weights=True)  # Восстановление весов с лучшего шага

#"Миграционная карта", 'Паспорт', "Страница КПП"
X_train, X_test, y_train, y_test = get_data(bath_size, train_list=[])

history = model.fit(X_train, y_train,
                    epochs=20,
                    validation_data=(y_train, y_test),
                    callbacks=[early_stopping])

model.save('classification_model_250-3.1.keras')

data_path = get_data_path()
classes = {}
matrix = np.zeros((5, 5), dtype=int)
np.fill_diagonal(matrix, 1)
int_l = {}
for i, x in enumerate(data_path["Украина"]):
    classes[np.argmax(matrix[i])] = x
    int_l[np.argmax(matrix[i])] = matrix[i]

labels_images = {}
n = 0

for image, label in get_data_test():
    image = image[0]
    labels_images.setdefault(np.argmax(label), []).append(image)
    n += 1
    if n >= 400:
        break


for key, images in labels_images.items():
    images = np.array(images)
    labels = np.array([int_l[key] for _ in range(len(images))])
    pred = model.predict(images, verbose=0)
    data = model.evaluate(images, labels, verbose=0)
    accuracy = data[1]
    loss = data[0]
    print(f"Loss for class {classes[key]}: {round(loss, 2)}")
    print(f"Accuracy for class {classes[key]}: {round(accuracy, 2)}")