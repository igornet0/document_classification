import cv2
import os
import numpy as np
from random import shuffle

def prepare_data(image_path, target_size):
    image = cv2.imread(image_path)
    # Преобразовать изображение в черно-белое
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    resized_image = cv2.resize(image, target_size)

    # Нормализовать значения пикселей до диапазона [0, 1]
    normalized_image = resized_image / 255.0
    #print(normalized_image.shape)
    return normalized_image


def rotate_image(image, degrees):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def generater_images(data_path, classes, dataset_path="dataset", target_size=(224, 224), batch_size=None, train_list=[]):
    if not batch_size is None:
        l_batch_size = batch_size
        col_country_documents = batch_size // len(data_path.keys()) 
    while True:
        for country, documents in data_path.items():
            if not batch_size is None and batch_size < 0:
                batch_size = l_batch_size
                break
            path = os.path.join(dataset_path, country)
            for document in documents:
                if not batch_size is None and batch_size < 0:
                    break
                if document in train_list:
                    continue
                images_path = os.path.join(path, document)
                list_image = os.listdir(images_path)
                shuffle(list_image)
                for i, image in enumerate(list_image):
                    if image == ".DS_Store":
                        continue

                    if not batch_size is None and batch_size <= 0:
                        break

                    image_path = os.path.join(images_path, image)
                    image = prepare_data(image_path, target_size)
                    label = classes[document]
                    yield image,label

                    if not batch_size is None and batch_size <= 0:
                        batch_size -= 1
        if not batch_size is None and batch_size <= 0:
            break
                    
def get_data_path(dataset_path="dataset"):
    country_paths = os.listdir(dataset_path)
    data_path = {}
    for country in country_paths:
        if not os.path.isdir(os.path.join(dataset_path, country)):
            continue
        path = os.path.join(dataset_path, country)
        files = sorted([x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))])
        if len(files) > 3:
            data_path[country] = files

    sorted_tuple = sorted(data_path.items(), key=lambda x: len(x[0]))
    data_path = dict(sorted_tuple)
    return data_path


def col_files(data_path, dataset_path):
    col = 0
    for country, documents in data_path.items():
        path = os.path.join(dataset_path, country)
        document_d = {}
        for document in documents:
            images_path = os.path.join(path, document)
            list_image = os.listdir(images_path)
            col += len(list_image)
            document_d[document] = len(list_image)
        print(country)
        print(document_d)

    return col

            