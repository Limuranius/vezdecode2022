# Решение задачи за 40 баллов

import carFinder
import domColorFinder
import numpy as np
from math import sqrt
import colorsys
import imageManager
import csv

colors = [("black", (0, 0, 0)), ("blue", (255, 0, 0)), ("green", (0, 255, 0)), ("red", (0, 0, 255)),
          ("white_silver", (210, 210, 210)), ("yellow", (0, 255, 255))]


def find_color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]):
    return sqrt(
        (c1[0] - c2[0]) ** 2 +
        (c1[1] - c2[1]) ** 2 +
        (c1[2] - c2[2]) ** 2
    )


def find_color_distance2(c1: tuple[int, int, int], c2: tuple[int, int, int]):
    c1 = colorsys.rgb_to_hsv(c1[2], c1[1], c1[0])
    c2 = colorsys.rgb_to_hsv(c2[2], c2[1], c2[0])
    return sqrt(
        (c1[0] - c2[0]) ** 2 * abs(c1[2] - c2[2])
    )


def get_color_name(color: tuple[int, int, int]):
    distances = []
    for named_color in colors:
        distances.append((named_color, find_color_distance(color, named_color[1])))
    distances.sort(key=lambda x: x[1])
    return distances[0][0][0]


def find_car_color(image: np.ndarray):
    x, y, w, h = carFinder.get_one_car_coord(image)
    car_color = domColorFinder.find_dominant_color(
        image[x: x + w, y: y + h]
    )
    return car_color


def find_color(input_dir, output_file="output_color.csv"):
    images = imageManager.get_merged_images(input_dir)
    out = open(output_file, "w", newline="")
    write = csv.writer(out)
    for i in range(len(images)):
        img = images[i]
        img_name = "00%03d.jpg" % (i + 1)
        print(img_name)
        write.writerow((img_name, get_color_name(find_car_color(img))))


if __name__ == "__main__":
    find_color("python_split_image_by_ch", "output_color.csv")
