import numpy as np
import cv2
import matplotlib.pyplot as plt
from box_utils import (draw_rect,
                       yolo_to_pascal_voc,
                       pascal_voc_to_yolo,
                       clip_box)

np.set_printoptions(suppress=True)


class Horizontalflip:
    """
    Horizontally flips the Image
    """

    def __init__(self):
        pass

    def __call__(self, image, bounding_boxes):
        """
        :param image: (ndarraay): Numpy image
        :param bounding_boxes: (ndarray): Numpy array containing bounding boxes are represented in the
        format [x_min, y_min, x_max, y_max]

        returns: (ndarray): Flipped image in the numpy format
        (ndarray): Transfromed number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        """

        img_center = np.array(img.shape[:2])[::-1] / 2
        img_center = np.hstack((img_center, img_center))

        image = image[:, ::-1, :]
        bounding_boxes[:, [1, 3]] += 2 * (img_center[[0, 2]] - bounding_boxes[:, [1, 3]])

        box_w = abs(bounding_boxes[:, 1] - bounding_boxes[:, 3])

        bounding_boxes[:, 1] = bounding_boxes[:, 1] - box_w
        bounding_boxes[:, 3] = bounding_boxes[:, 3] + box_w

        return image, bounding_boxes


class ScaleImage:
    """
    Scales the image

    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained.

    :param scale_x: (float): The factor by which the image is scaled horizontally
    :param scale_y: (float): The factor by which the image is scaled vertically
    returns: (ndarray): Scaled image as numpy array
    (ndarray): Transfromed number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """

    def __init__(self, scale_x=-0.05, scale_y=-0.05):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __call__(self, image, bounding_boxes):
        """
        :param image: (ndarraay): Numpy image
        :param bounding_boxes: (ndarray): Numpy array containing bounding boxes are represented in the
        format [x_min, y_min, x_max, y_max]

        returns: (ndarray): Scaled image in the numpy format
        (ndarray): Transfromed number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        """

        img_shape = image.shape

        resize_scale_x = 1 + self.scale_x
        resize_scale_y = 1 + self.scale_y

        image = cv2.resize(image, None, fx=resize_scale_x,
                           fy=resize_scale_y, interpolation=cv2.INTER_LINEAR)

        bounding_boxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

        canvas = np.zeros(img_shape, dtype=np.uint8)

        x_lim = int(min(resize_scale_x * img_shape[1], img_shape[1]))
        y_lim = int(min(resize_scale_y * img_shape[0], img_shape[0]))

        canvas[:y_lim, :x_lim] = image[:y_lim, :x_lim]
        image = canvas

        bounding_boxes = clip_box(bounding_boxes, [0, 0, 1 + img_shape[1],
                                                   img_shape[0]], 0.25)

        return image, bounding_boxes
