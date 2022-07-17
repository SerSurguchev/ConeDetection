import glob
import numpy as np
import cv2
import pandas as pd
from box_utils import (draw_rect,
                       yolo_to_pascal_voc,
                       pascal_voc_to_yolo,
                       from_yolo_to_dataframe,
                       clip_box)
import os
import shutil

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
        (ndarray): Transfromed number of bounding boxes and 5 represents [label, x_min, y_min, x_max, y_max] of the box
        """

        img_center = np.array(image.shape[:2])[::-1] / 2
        img_center = np.hstack((img_center, img_center))

        image = image[:, ::-1, :]
        bounding_boxes[:, [1, 3]] += 2 * (img_center[[0, 2]] - bounding_boxes[:, [1, 3]])

        box_w = abs(bounding_boxes[:, 1] - bounding_boxes[:, 3])

        bounding_boxes[:, 1] = bounding_boxes[:, 1] - box_w
        bounding_boxes[:, 3] = bounding_boxes[:, 3] + box_w

        return image, bounding_boxes


class ScaleImage:
    """
    Image scaling

    Bounding boxes which have an area of less than 50% in the remaining in the
    transformed image is dropped. The resolution is maintained.

    :param scale_x: (float): The factor by which the image is scaled horizontally
    :param scale_y: (float): The factor by which the image is scaled vertically
    """

    def __init__(self, scale_x=-0.05, scale_y=-0.05):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __call__(self, image, bounding_boxes, area_less=0.5):
        """
        :param image: (ndarraay): Numpy image
        :param bounding_boxes: (ndarray): Numpy array containing bounding boxes are represented in the
        format [label, x_min, y_min, x_max, y_max]

        returns: (ndarray): Scaled image in the numpy format
        (ndarray): Transfromed number of bounding boxes and 5 represents [label, x_min, y_min, x_max, y_max] of the box
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
    Translate image

    Bounding boxes which have an area of less than 50% in the remaining in the
    transformed image is dropped. The resolution is maintained.

    :param scale_x: (float): The factor by which the image is translated horizontally
    :param scale_y: (float): The factor by which the image is translated vertically
    """

    def __init__(self, translate_x=0.15, translate_y=0.15):
        self.translate_x = translate_x
        self.translate_y = translate_y

        assert self.translate_x > 0 \
               and self.translate_x < 1, 'Traslate factor must be between 0 and 1'

        assert self.translate_y > 0 \
               and self.translate_y < 1, 'Traslate factor must be between 0 and 1'

    def __call__(self, image, bounding_boxes, area_less=0.5):
        """
        :param image: (ndarraay): Numpy image
        :param bounding_boxes: (ndarray): Numpy array containing bounding boxes are represented in the
        format [label, x_min, y_min, x_max, y_max]

        returns: (ndarray): Translated image in the numpy format
        (ndarray): Transfromed number of bounding boxes and 5 represents [label, x_min, y_min, x_max, y_max] of the box
        """

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


class ShearImage:
    """
    Shears an image in horizontal direction
    :param shear_factor: (float): Factor by which the image is sheared in the y-direction
    """

    def __init__(self, shear_factor=0.1):
        self.shear_factor = shear_factor

    def __call__(self, image, bounding_boxes):
        """
        :param image: (ndarraay): Numpy image
        :param bounding_boxes: (ndarray): Numpy array containing bounding boxes are represented in the
        format [label, x_min, y_min, x_max, y_max]

        returns: (ndarray): Scaled image in the numpy format
        (ndarray): Transfromed number of bounding boxes and 5 represents [label, x_min, y_min, x_max, y_max] of the box
        """

        shear_factor = self.shear_factor

        if shear_factor < 0:
            image, bounding_boxes = Horizontalflip()(image, bounding_boxes)

        matrix = np.array([
            [1, np.abs(shear_factor), 0],
            [0, 1, 0]
        ])

        nW = image.shape[1] \
             + abs(shear_factor * image.shape[0])

        bounding_boxes[:, [1, 3]] += ((bounding_boxes[:, [2, 4]]) * np.abs(shear_factor)).astype(int)

        image = cv2.warpAffine(image, matrix, (int(nW), image.shape[0]))

        bounding_boxes = pascal_voc_to_yolo(bounding_boxes, h_image=image.shape[0], w_image=nW)

        return image, bounding_boxes


def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])


def main(images_path, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for image_path in glob.glob(images_path + '*.jpg'):

        print(image_path)

        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        path_to_im, im_name = split(image_path, '/', 7)

        txt_file = image_path.split('.')[0] + '.txt'
        path_to_txt, txt_name = split(txt_file, '/', 7)

        boxes = np.loadtxt(txt_file, dtype=np.float64)

        if len(boxes.shape) == 1:
            boxes = np.array([boxes], dtype=np.float64)

        pascal_voc_boxes = yolo_to_pascal_voc(boxes.copy(), height, width)
        pascal_voc_boxes = pascal_voc_boxes[pascal_voc_boxes[:, 4].argsort()]

        # -----------------------------------------------------------------------
        # Scale image by various factor
        for scale_x, scale_y in np.array([
            [-0.15, -0.15],
            [-0.1, -0.1],
            [0.1, 0.1],
            [0.15, 0.15],
            [0.2, 0.2]
        ]):
            scaled_img, scaled_boxes = ScaleImage(scale_x=scale_x,
                                                  scale_y=scale_y)(img.copy(),
                                                                   pascal_voc_boxes.copy())

            scaled_boxes = pascal_voc_to_yolo(scaled_boxes,
                                              h_image=height, w_image=width)
            df_scaled_boxes = from_yolo_to_dataframe(scaled_boxes)

            # Save scaled images and their labels
            cv2.imwrite(f'{output_dir}/scaled_to_{scale_x}_{im_name}',
                        cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY))

            with open(f'{output_dir}/scaled_to_{scale_x}_{txt_name}', 'a') as f:
                df_as_string = df_scaled_boxes.to_string(header=False, index=False)
                f.write(df_as_string)

        # -----------------------------------------------------------------------
        # Translate image by various factor
        for translate_x, translate_y in np.array([
            [0.1, 0.1],
            [0.15, 0.15],
            [0.2, 0.2]
        ]):
            translated_img, translated_box = TranslateImage(translate_x=translate_x,
                                                            translate_y=translate_y)(img.copy(),
                                                                                     pascal_voc_boxes.copy())

            translated_box = pascal_voc_to_yolo(translated_box,
                                                h_image=height, w_image=width)
            df_translated_box = from_yolo_to_dataframe(translated_box)

            # Save translated images and their labels
            cv2.imwrite(f'{output_dir}/translated_to_{translate_x}_{im_name}',
                        cv2.cvtColor(translated_img, cv2.COLOR_BGR2GRAY))

            with open(f'{output_dir}/translated_to_{translate_x}_{txt_name}', 'a') as f:
                df_as_string = df_translated_box.to_string(header=False, index=False)
                f.write(df_as_string)

        # -----------------------------------------------------------------------
        # Shear image by various factor
        for shear_factor in np.array([-0.2, -0.15, -0.1, 0.1, 0.15, 0.2]):
            sheared_image, sheared_box = ShearImage(shear_factor=shear_factor)(img.copy(),
                                                                               pascal_voc_boxes.copy())

            # sheared_box = pascal_voc_to_yolo(sheared_box,
            #                                  h_image=height, w_image=width)

            df_sheared_box = from_yolo_to_dataframe(sheared_box)

            # Save translated images and their labels
            cv2.imwrite(f'{output_dir}/sheared_to_{shear_factor}_{im_name}',
                        cv2.cvtColor(sheared_image, cv2.COLOR_BGR2GRAY))

            with open(f'{output_dir}/sheared_to_{shear_factor}_{txt_name}', 'a') as f:
                df_as_string = df_sheared_box.to_string(header=False, index=False)
                f.write(df_as_string)

        # -----------------------------------------------------------------------
        # Flip image
        flipped_image, flipped_boxes = Horizontalflip()(img.copy(), pascal_voc_boxes.copy())

        flipped_boxes = pascal_voc_to_yolo(flipped_boxes,
                                           h_image=height,
                                           w_image=width)

        df_flipped_boxes = from_yolo_to_dataframe(flipped_boxes)

        # Save flipped image and labels
        cv2.imwrite(f'{output_dir}/flipped_{im_name}',
                    cv2.cvtColor(flipped_image, cv2.COLOR_BGR2GRAY))

        with open(f'{output_dir}/flipped_{txt_name}', 'a') as f:
            df_as_string = df_flipped_boxes.to_string(header=False, index=False)
            f.write(df_as_string)

        # Final save original image as grayscale
        cv2.imwrite(output_dir + '/' + im_name, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        shutil.copy2(txt_file, output_dir + '/' + txt_name)


if __name__ == '__main__':
    images_path = '/home/sergey/from_wind/programming/Computer_vision/correct_labels/'
    output_dir = '/home/sergey/from_wind/programming/Computer_vision/correct_labels_final'
    main(images_path=images_path, output_dir=output_dir)
