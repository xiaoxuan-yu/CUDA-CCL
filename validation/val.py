import cv2 as cv
import numpy as np


def binary_image_from_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        m, n = map(int, lines[0].strip().split())
        return np.array(
            [[int(c) for c in line.strip().split()] for line in lines[1:]]
        ).reshape(m, n)


def get_match_dict(labels_a, labels_b):
    uni_labels_a = np.unique(labels_a)
    uni_labels_b = np.unique(labels_b)
    labels_a_dict = {label: [] for label in uni_labels_a}
    labels_b_dict = {label: [] for label in uni_labels_b}
    for i in range(labels_a.shape[0]):
        for j in range(labels_a.shape[1]):
            labels_a_dict[labels_a[i][j]].append(i * labels_a.shape[1] + j)
    for i in range(labels_b.shape[0]):
        for j in range(labels_b.shape[1]):
            labels_b_dict[labels_b[i][j]].append(i * labels_b.shape[1] + j)
    match_dict = {}
    for label in uni_labels_a:
        label_index_list = labels_a_dict[label]
        for b_label in uni_labels_b:
            b_label_index_list = labels_b_dict[b_label]
            if len(label_index_list) != len(b_label_index_list):
                continue
            if all(
                [
                    label_index_list[i] == b_label_index_list[i]
                    for i in range(len(label_index_list))
                ]
            ):
                match_dict[label] = b_label
                break
    return match_dict


def map_with_dict(labels, match_dict):
    labels = labels.copy()
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            labels[i][j] = match_dict[labels[i][j]]
    return labels


def map_label(labels_a, labels_b):
    match_dict = get_match_dict(labels_a, labels_b)
    labels_a = map_with_dict(labels_a, match_dict)
    return np.all(labels_a == labels_b), labels_a


def visualize(labels, filename):
    np.random.seed(345)
    N, M = labels.shape
    color_image = np.zeros((N, M, 3), dtype=np.uint8)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        # generate random color
        color = np.random.randint(0, 255, 3)
        color_image[labels == label] = color
    cv.imwrite(f"./figure/{filename}.png", color_image)
