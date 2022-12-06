import ast
import sys
import numpy as np
import cv2
import easyocr
from collections import defaultdict
import os
from module_1 import object_detection
from tqdm import tqdm

def text_detection(img_path, coordinates):
    reader = easyocr.Reader(['en'])

    img = cv2.imread(img_path)
    image_np = np.array(img)

    def ocr(cropped_image):
        img_data = reader.readtext(cropped_image)
        results = ""
        for l in img_data:
            results += l[1] + '; '
        return results

    output = defaultdict(list)

    for i, box in enumerate(coordinates):
        ymin, xmin, ymax, xmax = box[1]
        ymin, xmin, ymax, xmax = int(
            ymin*img.shape[0]), int(xmin*img.shape[1]), int(ymax*img.shape[0]), int(xmax*img.shape[1])

        cropped_image = img[ymin:ymax, xmin:xmax]
        output[box[0]] += [ocr(cropped_image)]
        # ax.text(xmax, ymax, ocr(cropped_image))

    return output


def process_images(folder):
    output = {}
    for file in tqdm(os.listdir(folder)):
        path = os.path.join(folder, file)
        coordinates = object_detection(path)
        output[file] = text_detection(path, coordinates)
    return output


if __name__ == "__main__":
    img_folder = sys.argv[1]
    # coordinates = sys.argv[2]
    # coordinates = ast.literal_eval(coordinates)

    print(process_images(img_folder))
