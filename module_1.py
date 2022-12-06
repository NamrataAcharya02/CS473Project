import tensorflow as tf
import sys
import cv2
import numpy as np


classes_mapper = {
    1: "entity",
    2: "weak_entity",
    3: "relationship",
    4: "weak_relationship",
    5: "attribute",
}


def object_detection(img_path: str):
    model = tf.saved_model.load("model")

    img = cv2.imread(img_path)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.uint8)

    pred = model(input_tensor)

    scores, boxes, classes = (
        pred["detection_scores"].numpy(),
        pred["detection_boxes"].numpy(),
        pred["detection_classes"].numpy(),
    )

    threshold = 0.7
    boxes, classes = boxes[scores > threshold], classes[scores > threshold]

    classes = list(map(lambda x: classes_mapper[x], classes))

    output = []
    for i in range(len(classes)):
        output.append((classes[i], list(boxes[i])))

    return output


if __name__ == "__main__":
    img_path = sys.argv[1]
    output = object_detection(img_path)
    print(output)
