# Решение задачи за 10 баллов

import cv2
import dataParse
import os


def merge_image(im_info: tuple[tuple, tuple, tuple], im_dir: str):
    """
    По информации о трёх каналах изображения создаёт одно изображение
    """
    im_b = f"{im_dir}/{im_info[0][2]}"
    im_g = f"{im_dir}/{im_info[1][2]}"
    im_r = f"{im_dir}/{im_info[2][2]}"

    im_b = cv2.imread(im_b, cv2.IMREAD_GRAYSCALE)
    im_g = cv2.imread(im_g, cv2.IMREAD_GRAYSCALE)
    im_r = cv2.imread(im_r, cv2.IMREAD_GRAYSCALE)

    return cv2.merge([im_b, im_g, im_r])


def get_merged_images(input_dir):
    """
    Возвращает изображения, соединённые по 3 каналам
    """
    dataset = dataParse.read_dataset_folder(input_dir)
    res = []
    for i in range(dataset.im_count):
        img = merge_image(dataset.get_triplet(i), f"{input_dir}/data")
        res.append(img)
    return res


def createDirIfNotExist(dir_path):
    isExist = os.path.exists(dir_path)
    if not isExist:
        os.makedirs(dir_path)


def merge_channels(input_dir, output_dir):
    createDirIfNotExist(output_dir)
    merged_images = get_merged_images(input_dir)
    for i in range(len(merged_images)):
        img = merged_images[i]
        img_name = "00%03d.jpg" % (i + 1)
        print(img_name)
        cv2.imwrite(f"{output_dir}/{img_name}", img)


if __name__ == "__main__":
    merge_channels("python_split_image_by_ch", "res")
