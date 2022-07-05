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

    Bounding boxes which have an area of less than 50% in the remaining in the
    transformed image is dropped. The resolution is maintained.

    :param scale_x: (float): The factor by which the image is scaled horizontally
    :param scale_y: (float): The factor by which the image is scaled vertically
    returns: (ndarray): Scaled image as numpy array
    (ndarray): Transfromed number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """

    def __init__(self, scale_x=-0.05, scale_y=-0.05):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __call__(self, image, bounding_boxes, area_less=0.5):
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

        bounding_boxes[:, 1:5] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

        canvas = np.zeros(img_shape, dtype=np.uint8)

        x_lim = int(min(resize_scale_x * img_shape[1], img_shape[1]))
        y_lim = int(min(resize_scale_y * img_shape[0], img_shape[0]))

        canvas[:y_lim, :x_lim] = image[:y_lim, :x_lim]
        image = canvas

        labels = np.array([bounding_boxes[:, 0]]).reshape(-1, 1)

        bounding_boxes = clip_box(bounding_boxes[:, 1:5], [0, 0, img_shape[1],
                                                   img_shape[0]], area_less)

        bounding_boxes = np.concatenate((labels[:len(bounding_boxes)],
                                         bounding_boxes[:, :4]), axis=1)

        return image, bounding_boxes


class TranslateImage:
    """
    Randomly translate image
    """

    def __init__(self, translate_x=0.15, translate_y=0.15):
        self.translate_x = translate_x
        self.translate_y = translate_y

        assert self.translate_x > 0 \
               and self.translate_x < 1, 'Traslate factor must be between 0 and 1'

        assert self.translate_y > 0 \
               and self.translate_y < 1, 'Traslate factor must be between 0 and 1'

    def __call__(self, image, bounding_boxes, area_less=0.5):
        img_shape = image.shape

        translate_factor_x = self.translate_x
        translate_factor_y = self.translate_y

        # Get the top-left corner co-ordinates of the shifted box
        corner_x = int(img_shape[1] * translate_factor_x)
        corner_y = int(img_shape[0] * translate_factor_y)

        bounding_boxes[:, 1:5] += [corner_x, corner_y, corner_x, corner_y]

        # Change the origin to the top-left corner of the translated box
        box_cords = [max(0, corner_y), max(0, corner_x),
                     min(img_shape[0], img_shape[0] + corner_y),
                     min(img_shape[1], img_shape[1] + corner_x)]

        mask = image[max(-corner_y, 0):min(img_shape[0], -corner_y + img_shape[0]),
                     max(-corner_x, 0):min(img_shape[1], -corner_x + img_shape[1]), :]

        canvas = np.zeros(img_shape).astype('uint8')

        canvas[box_cords[0]:box_cords[2], box_cords[1]:box_cords[3], :] = mask
        image = canvas

        labels = np.array([bounding_boxes[:, 0]]).reshape(-1, 1)

        bounding_boxes = clip_box(bounding_boxes[:, 1:5], [0, 0, img_shape[1],
                                                   img_shape[0]], area_less)

        bounding_boxes = np.concatenate((labels[:len(bounding_boxes)],
                                         bounding_boxes[:, :4]), axis=1)

        return image, bounding_boxes
