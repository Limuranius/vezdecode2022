# Решение задачи за 30 баллов

import cv2
import numpy as np
from sklearn.cluster import KMeans
import imutils


def find_dominant_color(img: np.ndarray):
    clusters = 5
    img = imutils.resize(img, height=200)
    flat_img = np.reshape(img, (-1, 3))
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flat_img)
    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
    percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
    p_and_c = list(zip(percentages, dominant_colors))
    p_and_c = sorted(p_and_c, reverse=True, key=lambda x: int(x[0]))
    return tuple(p_and_c[0][1])


def calc_metric(image, x, y, w, h):
    img = cv2.imread(image)
    crop_img = img[x: x + w, y: y + h]

    p1 = (x, y)
    p2 = (x + w, y + h)
    rect_color = (255, 0, 0)
    thickness = 2
    img = cv2.rectangle(img, p1, p2, rect_color, thickness)

    height = img.shape[0]
    dom_color = find_dominant_color(crop_img)
    color_rect = np.full((height, 100, 3), dom_color)
    color_rect = np.uint8(color_rect)
    img = cv2.hconcat([img, color_rect])

    print("Доминирующий цвет:", dom_color)
    cv2.imshow("A", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    calc_metric("python_split_image_by_ch/res.jpg", 50, 100, 200, 100)
