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

    center_x, center_y = np.round(abs(boxes[:, 2] * w_image)), \
                         np.round(abs(boxes[:, 3] * h_image))

    boxes[:, 0] = np.round(orig_box[:, 0] * w_image - (center_x / 2))
    boxes[:, 1] = np.round(orig_box[:, 1] * h_image - (center_y / 2))

    boxes[:, 2] = np.round(orig_box[:, 0] * w_image + (center_x / 2))
    boxes[:, 3] = np.round(orig_box[:, 1] * h_image + (center_y / 2))

    return boxes.astype(float)


def pascal_voc_to_yolo(labels, boxes, h_image, w_image):
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

    x_norm = np.round(((boxes[:, 0] + boxes[:, 2]) / 2 / float(w_image)), 6)
    y_norm = np.round(((boxes[:, 1] + boxes[:, 3]) / 2 / float(h_image)), 6)
    w_norm = np.round((boxes[:, 2] - boxes[:, 0]) / float(w_image), 6)
    h_norm = np.round((boxes[:, 3] - boxes[:, 1]) / float(h_image), 6)

    yolo_format = np.concatenate((labels,
                                  x_norm,
                                  y_norm,
                                  w_norm,
                                  h_norm))

    yolo_format = yolo_format.reshape(5, len(boxes)).T
    yolo_format[:, 0] = yolo_format[:, 0].astype(int)

    return yolo_format


def draw_rect(im, cords, color=(0, 0, 0)):
    """
    Draw the rectangle on the image

    :param image: (ndarraay): Numpy image
    :param bounding_boxes: (ndarray): Numpy array containing bounding boxes are represented in the
    format [x_min, y_min, x_max, y_max]

    :return (ndarray): Numpy image with bounding boxes drawn on it
    """
    cords = cords[:, :4]
    cords = cords.reshape(-1, 4)

    for cord in cords:
        pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])

        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])

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
