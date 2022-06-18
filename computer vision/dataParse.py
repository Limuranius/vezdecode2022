import csv


class DataSet:
    im_count: int
    images_info: list[tuple[str, str, str]]

    def get_triplet(self, i: int) -> tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]]:
        """
        Возвращает i-ю тройку изображений
        """
        return self.images_info[i * 3], self.images_info[i * 3 + 1], self.images_info[i * 3 + 2]

    def get_by_index(self, i: int):
        return self.images_info[i]


def read_dataset_folder(path: str) -> DataSet:
    res = DataSet()
    with open(path + "/image_counter.txt") as file:
        count = int(file.read())

    with open(path + "/description.csv") as file:
        reader = csv.reader(file)
        images = [im for im in reader]
        del images[0]  # Убираем заголовок csv-таблицы

    res.im_count = count
    res.images_info = images

    return res

