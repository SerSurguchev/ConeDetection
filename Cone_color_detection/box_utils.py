import cv2
import numpy as np

np.set_printoptions(suppress=True)


def yolo_to_pascal_voc(boxes, h_image, w_image):
    """
    Change bouning box annotaion from yolo format to pascal_voc format
    :param boxes: (ndarray): Numpy array containing bounding boxes are represented in the format
    normalized [x_center, y_center, width, height]
    :param h_image: (int): Height of numpy image
    :param w_image: (int): Width of numpy image
    :return: Numpy array containing bounding boxes in pascal_voc format [x_min, y_min, x_max, y_max]
    """

    orig_box = boxes.copy()

    if not isinstance(h_image, np.ndarray) and not isinstance(w_image, np.ndarray):
        h_image, w_image = map(np.array, [h_image, w_image])

    center_x, center_y = np.round(abs(boxes[:, 3] * w_image)), \
                         np.round(abs(boxes[:, 4] * h_image))

    boxes[:, 1] = np.round(orig_box[:, 1] * w_image - (center_x / 2))
    boxes[:, 2] = np.round(orig_box[:, 2] * h_image - (center_y / 2))

    boxes[:, 3] = np.round(orig_box[:, 1] * w_image + (center_x / 2))
    boxes[:, 4] = np.round(orig_box[:, 2] * h_image + (center_y / 2))

    return boxes.astype(float)


def pascal_voc_to_yolo(boxes, h_image, w_image):
    """
    Change bouning box annotaion from pascal_voc
    to yolo normalized [x_center, y_center, width, height] format

    :param labels: (ndarray) Numpy array containing cone labels
    :param boxes: (ndarray): Numpy array containing bounding boxes are represented in the format
    [x_min, y_min, x_max, y_max]

    :param h_image: (int): Height of numpy image
    :param w_image: (int): Width of numpy image
    :return: Numpy array containing bounding boxes in yolo format
    """
    orig_box = boxes.copy()
    boxes[:, 1] = np.round(((orig_box[:, 1] + orig_box[:, 3]) / 2 / float(w_image)), 6)
    boxes[:, 2] = np.round(((orig_box[:, 2] + orig_box[:, 4]) / 2 / float(h_image)), 6)
    boxes[:, 3] = np.round((orig_box[:, 3] - orig_box[:, 1]) / float(w_image), 6)
    boxes[:, 4] = np.round((orig_box[:, 4] - orig_box[:, 2]) / float(h_image), 6)

    return boxes


def draw_rect(im, cords, color=(0, 0, 0)):
    """
    Draw the rectangle on the image

    :param im: (ndarraay): Numpy image
    :param cords: (ndarray): Numpy array containing bounding boxes are represented in the
    format [x_min, y_min, x_max, y_max]

    :return (ndarray): Numpy image with bounding boxes drawn on it
    """
    cords = cords[:, :5]
    cords = cords.reshape(-1, 5)

    for cord in cords:
        pt1, pt2 = (int(cord[1]), int(cord[2])), \
                   (int(cord[3]), int(cord[4]))

        im = cv2.rectangle(im, pt1, pt2, color, 1)

    return im


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def clip_box(bbox, clip_box, alpha):
    """
    Clip the bounding boxes to the borders of an image

    :param bbox: (ndarray)
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    :param: clip_box: (ndarray)
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`

    :param: alpha: (float)
        If the fraction of a bounding box left in the image after being clipped is
        less than `alpha` the bounding box is dropped.

    return: (ndarray):
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    """

    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    delta_area = ((ar_ - bbox_area(bbox)) / ar_)

    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox[mask == 1, :]

    return bbox
