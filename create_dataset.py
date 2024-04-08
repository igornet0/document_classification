import os
import shutil

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

data_path = get_data_path()
dataset_path="dataset"
dataset_path_new = "dataset_new"
n = 0
for country, documents in data_path.items():
    path = os.path.join(dataset_path, country)
    for document in documents:
        if not document in os.listdir(dataset_path_new):
            os.mkdir(os.path.join(dataset_path_new, document))
        images_path = os.path.join(path, document)
        list_image = os.listdir(images_path)
        # Копируем файл
        for image in list_image:
            source_file = os.path.join(images_path, image)
            # Создаем путь к новому файлу в папке назначения
            destination_file = os.path.join(os.path.join(dataset_path_new, document), f"{n}.png")
            shutil.copyfile(source_file, destination_file)
            n += 1


