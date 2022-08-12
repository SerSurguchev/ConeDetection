import cv2
import numpy as np
import os
import glob
import shutil

from data_augmentation import (
    Horizontalflip,
    ScaleImage,
    TranslateImage,
    ShearImage
)

from box_utils import (
    draw_rect,
    yolo_to_pascal_voc,
    from_yolo_to_dataframe
)

np.set_printoptions(suppress=True)


def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])


def is_file_not_empty(file_path):
    """ Check if file is not empty by confirming if it's size more than 0 bytes"""
    return os.path.exists(file_path) and os.stat(file_path).st_size > 0


def main(lst_images_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_path in lst_images_path:
        print(image_path)
        image = cv2.imread(image_path)

        height, width = image.shape[:2]

        txt_file = image_path.split('.')[0] + '.txt'

        path_to_txt, txt_name = split(txt_file, '/',
                                      txt_file.count('/'))

        path_to_im, im_name = split(image_path, '/',
                                    image_path.count('/'))

        # Check if txt file is not empty (Not background image)
        if is_file_not_empty(txt_file):

            boxes = np.loadtxt(txt_file, dtype=np.float32)

            if len(boxes.shape) == 1:
                boxes = np.array([boxes], dtype=np.float32)

            pascal_voc_boxes = yolo_to_pascal_voc(boxes.copy(),
                                                  height, width)

            # Remove incomplete cones (which are not fully represented in the image)
            pascal_voc_boxes = np.delete(pascal_voc_boxes,
                                         np.where(
                                             (pascal_voc_boxes[:, 4] > height - 2) |
                                             (pascal_voc_boxes[:, 3] >= width - 1) |
                                             (pascal_voc_boxes[:, 1] == 0) |
                                             (pascal_voc_boxes[:, 1] == 1)
                                         )[0], axis=0
                                         )

            # Avoid resetting labels during bounding box
            pascal_voc_boxes = pascal_voc_boxes[pascal_voc_boxes[:, 4].argsort()]

            # ================================ #
            #   Scale image by various factor  #
            # ================================ #

            for scale_x, scale_y in np.array([
                [-0.1, -0.1],
                [0.1, 0.1],
            ]):
                scaled_img, scaled_boxes = ScaleImage(scale_x=scale_x,
                                                      scale_y=scale_y)(image.copy(),
                                                                       pascal_voc_boxes.copy())

                df_scaled_boxes = from_yolo_to_dataframe(scaled_boxes)

                # Save scaled images and their labels
                cv2.imwrite(f'{output_dir}/scaled_to_{scale_x}_{im_name}',
                            cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY))

                with open(f'{output_dir}/scaled_to_{scale_x}_{txt_name}', 'a') as f:
                    df_as_string = df_scaled_boxes.to_string(header=False, index=False)
                    f.write(df_as_string)

            # ==================================== #
            #   Translate image by various factor  #
            # ==================================== #

            for translate_x, translate_y in np.array([
                [0.1, 0.1],
                [0.15, 0.15],
            ]):
                translated_img, translated_box = TranslateImage(translate_x=translate_x,
                                                                translate_y=translate_y)(image.copy(),
                                                                                         pascal_voc_boxes.copy())

                df_translated_box = from_yolo_to_dataframe(translated_box)

                # Save translated images and their labels
                cv2.imwrite(f'{output_dir}/translated_to_{translate_x}_{im_name}',
                            cv2.cvtColor(translated_img, cv2.COLOR_BGR2GRAY))

                with open(f'{output_dir}/translated_to_{translate_x}_{txt_name}', 'a') as f:
                    df_as_string = df_translated_box.to_string(header=False, index=False)
                    f.write(df_as_string)

            # ================================ #
            #   Shear image by various factor  #
            # ================================ #

            for shear_factor in np.array([-0.1, 0.1]):
                sheared_image, sheared_box = ShearImage(shear_factor=shear_factor)(image.copy(),
                                                                                   pascal_voc_boxes.copy())

                df_sheared_box = from_yolo_to_dataframe(sheared_box)

                # Save translated images and their labels
                cv2.imwrite(f'{output_dir}/sheared_to_{shear_factor}_{im_name}',
                            cv2.cvtColor(sheared_image, cv2.COLOR_BGR2GRAY))

                with open(f'{output_dir}/sheared_to_{shear_factor}_{txt_name}', 'a') as f:
                    df_as_string = df_sheared_box.to_string(header=False, index=False)
                    f.write(df_as_string)

            # =============================== #
            #   Flip image by various factor  #
            # =============================== #

            flipped_image, flipped_boxes = Horizontalflip()(image.copy(), pascal_voc_boxes.copy())

            df_flipped_boxes = from_yolo_to_dataframe(flipped_boxes)

            # Save flipped image and labels
            cv2.imwrite(f'{output_dir}/flipped_{im_name}',
                        cv2.cvtColor(flipped_image, cv2.COLOR_BGR2GRAY))

            with open(f'{output_dir}/flipped_{txt_name}', 'a') as f:
                df_as_string = df_flipped_boxes.to_string(header=False, index=False)
                f.write(df_as_string)

            # Final save original image as grayscale
            cv2.imwrite(output_dir + '/' + im_name, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            shutil.copy2(txt_file, output_dir + '/' + txt_name)

        else:
            print('Empty image......')
            flipped_image = image.copy()[:, ::-1, :]

            # Save flipped background image and txt labels
            cv2.imwrite(f'{output_dir}/flipped_{im_name}',
                        cv2.cvtColor(flipped_image, cv2.COLOR_BGR2GRAY))

            os.mknod(f'{output_dir}/flipped_{txt_name}')

            # Save original background image
            cv2.imwrite(output_dir + '/' + im_name,
                        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

            shutil.copy2(txt_file, output_dir + '/' + txt_name)


if __name__ == '__main__':
    images_path = (img_path for img_path in
                   glob.glob('/home/sergey/from_wind/programming/Computer_vision/dataset_grayscale/' + '*.jpg'))

    output_dir = '/home/sergey/from_wind/programming/Computer_vision/final_data_full'

    main(images_path, output_dir)
    shutil.make_archive(output_dir, 'zip', output_dir)
