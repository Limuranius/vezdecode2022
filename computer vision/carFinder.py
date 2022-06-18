# Решение задачи за 20 баллов

import cv2
import numpy as np
import imageManager
import csv

with open("yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]


def is_id_car(class_id: int):
    return classes[class_id] in ["car", "bus", "truck"]


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def detect_objects(image: np.ndarray):
    min_confidence = 0.5

    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > min_confidence:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = max(int(center_x - w / 2), 0)
                y = max(int(center_y - h / 2), 0)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    return class_ids, boxes


def count_cars(image: np.ndarray):
    ids = detect_objects(image)[0]
    res = 0
    for class_id in ids:
        if is_id_car(class_id):
            res += 1
    return res


def has_cars(image):
    return count_cars(image) != 0


def get_one_car_coord(image: np.ndarray):
    ids, coords = detect_objects(image)
    for i in range(len(ids)):
        if is_id_car(ids[i]):
            return coords[i]
    return [0, 0, 5, 5]


def find_car(input_dir, output_cars="output.csv"):
    images = imageManager.get_merged_images(input_dir)
    out = open(output_cars, "w", newline="")
    write = csv.writer(out)
    for i in range(len(images)):
        img = images[i]
        img_name = "00%03d.jpg" % (i + 1)
        print(img_name)
        write.writerow((img_name, has_cars(img)))


if __name__ == "__main__":
    find_car("python_split_image_by_ch", "output.csv")
