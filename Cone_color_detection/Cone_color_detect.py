import cv2
import numpy as np
import pandas as pd
import glob
import os
import math
from PIL import Image, ImageEnhance, ImageStat, ImageTk
import matplotlib.pyplot as plt
import ast

def calc_brightness(im_file):
    stat = ImageStat.Stat(im_file)
    r, g, b = stat.mean
    return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))

def point_into_rectangle(boxes, x, y):

    '''
    :param boxes: (list of lists): Coordinates of all boxes on image
    :param x: (int): x point coordinate
    :param y: (int): y point coordinate
    :return: (boolean): True if point into other bounding box else False
    '''

    def product(Ax, Ay,
                Bx, By,
                Px, Py):
        return (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax)

    check = False
    # x, y, w, h
    for box in boxes:
        p1 = product(box[0], box[1], box[0], box[1] + box[3], x, y)
        p2 = product(box[0], box[1] + box[3], box[0] + box[2], box[1] + box[3], x, y)
        p3 = product(box[0] + box[2], box[1] + box[3], box[0] + box[2], box[1], x, y)
        p4 = product(box[0] + box[2], box[1], box[0], box[1], x, y)

        # point into rectangle or not
        if ((p1 < 0 and p2 < 0 and p3 < 0 and p4 < 0) or
                (p1 > 0 and p2 > 0 and p3 > 0 and p4 > 0)):
            check = True
            break

    return check

def create_cone_rectangle(x, y, h, w, cone):

    '''
    Сreate cone rectangle with black pixels around the edges
    :param x: (int): x point coordinate (left down)
    :param y: (int): y point coordinate (left down)
    :param h: (int): Height of bounding box
    :param w: (int): Width of bounding box
    :param cone: (ndarray): Cropped cone image from a whole frame
    :return: (ndarray): Cone rectangle with black pixels around the edges
    '''

    cx = round((x + x + w) / 2)

    lst = []
    lst.append([cx - x, 0])
    lst.append([0, h])
    lst.append([w, h])

    mask = cv2.fillPoly(np.zeros((h, w), dtype=np.uint8), [np.asarray(lst)], (255, 255, 255))
    bitwise = cv2.bitwise_and(cone, cone, mask=mask)

    return bitwise

def get_bitwise(bitwise):

    '''
    :param bitwise: (ndarray): Cone rectangle with black pixels around the edges
    :return: (ndarray): Cone rectangle with custom color space (red, yellow, blue)
    '''

    height, width = bitwise.shape[:2]

    for i in range(0, height):
        for j in range(0, width):

            if bitwise[i, j, 1] > bitwise[i, j, 0]:
                bitwise[i, j, 1] -= bitwise[i, j, 0]
            else:
                bitwise[i, j, 1] = 0

            if bitwise[i, j, 2] > bitwise[i, j, 0]:
                bitwise[i, j, 2] -= bitwise[i, j, 0]
            else:
                bitwise[i, j, 2] = 0

            bitwise[i, j, 1] = min(bitwise[i, j, 1], bitwise[i, j, 2])
            bitwise[i, j, 2] -= bitwise[i, j, 1]

    return bitwise

def remove_white_naive(orig_bitwise, threshold):

    height, width = orig_bitwise.shape[:2]

    for row in range(height):
        for elem in range(width):
            pix_sum = int(np.sum(orig_bitwise[row][elem])/3)
            orig_bitwise[row][elem] = np.where(pix_sum > threshold, 0, orig_bitwise[row][elem])

    return orig_bitwise

def brightness_change(grad_array):

    increase, decrease = [], []

    for i in range(1, len(grad_array)):
        if grad_array[i-1] < grad_array[i]:
            increase.append(grad_array[i] - grad_array[i-1])
        else:
            decrease.append(abs(grad_array[i] - grad_array[i-1]))

    return sum(increase) >= sum(decrease)

def check_cone_gradient(orig_bitwise):

    gray = cv2.cvtColor(orig_bitwise, cv2.COLOR_BGR2GRAY)
    half = gray[round(gray.shape[0]/2):, :]
    grad_array = np.zeros([half.shape[0]])

    for ind, row in enumerate(half):
        lst = np.array(list(filter(lambda x: x != 0, row)))
        mean = np.mean(lst, axis=0)
        grad_array[ind] = mean

    return brightness_change(grad_array[:len(grad_array)//2])

def get_color(triangle, fallen, possible_big_cone):

    '''
    Detect cone class:
    0 black middle
    1 white middle
    2 unknown

    :param triangle: (ndarray): Cone rectangle with black pixels around the edges
    :param fallen: (boolean): True if cone is fallen (box width greater than height)
    :param possible_big_cone: (boolean)
    :return: (int): cone class
    '''

    gray_bitwise = cv2.cvtColor(triangle, cv2.COLOR_BGR2GRAY)
    half_b = triangle[round(triangle.shape[0]/2):, :]
    # bitwise_center = gray_bitwise.item(round(gray_bitwise.shape[0] / 2), round(gray_bitwise.shape[1] / 2))

    b = (np.sum(half_b[:, :, 0]))
    r_g = (np.sum(half_b[:, :, 2]) + np.sum(half_b[:, :, 1])) / 2
    or_sum = (b + r_g) / 100

    # Detect blue cone (first class)
    if b > r_g and (np.sum(half_b[:, :, 0])) / (np.sum(half_b[:, :, 2])) > 0.7:
        return 1

    # Detect yellow and red cone
    else:
        custom_bitwise = get_bitwise(triangle)
        yellow = np.sum(custom_bitwise[:, :, 1])
        yr = np.sum(custom_bitwise[:, :, 2])

        # Detect fallen cones
        if fallen:
            return 2 if yr/yellow > 2 else 0

        elif possible_big_cone and \
                yr/yellow > 5:
            return 2

        # Detect big orange cone (second unknown class)
        elif check_cone_gradient(orig_bitwise=triangle) and \
                (yellow/yr < 0.7 or yr/yellow > 3):
            return 2

        # Detect yellow cone (zero class)
        elif check_cone_gradient(orig_bitwise=triangle) and \
                yellow/yr >= 0.7:
            return 0

        # Detect red cone
        else:
            return 1

if __name__ == '__main__':

    data_fr_file = pd.read_csv('/home/sergey/from_wind/programming/Computer_vision/all.csv')
    dir = r'/home/sergey/from_wind/programming/Computer_vision/YOLO_Dataset/'
    output_dir = r'/home/sergey/from_wind/programming/Computer_vision/dataset_grayscale/'

    lens = len(glob.glob(dir + '*.jpg'))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for i in range(1, lens):

        j = 5
        name_of_image = data_fr_file.iloc[i, 0]
        image = cv2.imread(dir + name_of_image)
        c_height, c_width = image.shape[:2]
        name_of_file = name_of_image[:-3] + "txt"

        file1 = open(output_dir + 'gray_' + name_of_file, 'w')

        # mask = pd.DataFrame(data_fr_file.iloc[i, j:]).dropna()
        # bboxes = []
        # for index, row in mask.iterrows():
        #     h = ast.literal_eval(row[i])[2]
        #     if h > 25:
        #         bboxes.append(ast.literal_eval(row[i]))
        #
        # bboxes = sorted(bboxes, key=lambda x: x[2], reverse=True)

        while True:

            mark1 = data_fr_file.iloc[i, j]

            if pd.isna(mark1):
                break

            mark1 = str(mark1)
            mark1 = mark1[1:-1]
            mark = mark1.replace(',', '')

            x, y, h, w = mark.split()
            x, y, h, w = map(int, [x, y, h, w])

            if h > 25:

                # check_point1 = point_into_rectangle(bboxes, x=x, y=y + h)
                # check_point2 = point_into_rectangle(bboxes, x=x + w, y=y + h)

                # if check_point1 or check_point2:
                #     # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
                #     j += 1
                #     continue

                # Сreate cone rectangle with black pixels around the edges
                cone = image[y:y + h, x:x + w]
                triangle = create_cone_rectangle(x, y, h, w, cone)

                # Calculate cone brightness and if it is toо bright, pass this cone
                avr_brightness = calc_brightness(Image.fromarray(cv2.cvtColor(cone, cv2.COLOR_BGR2RGB)))

                if int(avr_brightness) < 180:

                    x_norm = round((float(x) + float(w) / 2) / float(c_width), 6)
                    y_norm = round((float(y) + float(h) / 2) / float(c_height), 6)
                    h_norm = round(float(h) / float(c_height), 6)
                    w_norm = round(float(w) / float(c_width), 6)

                    fallen = False
                    if h < w:
                        fallen = True

                    possible_big_cone = False
                    if h > 1.8 * w:
                        possible_big_cone = True

                    # Detect red cone on video contains only red cone (first class)
                    if c_height == 1080 and c_width == 1920:
                        if w > h:
                            color_ind = 1
                        else:
                            color_ind = 2 if check_cone_gradient(triangle.copy()) else 1

                    else:
                        color_ind = get_color(triangle.copy(), fallen, possible_big_cone)

                    # Write labels in .txt file
                    file1.write(f"{color_ind} {x_norm} {y_norm} {w_norm} {h_norm}\n")

                    # cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 2)
                    # cv2.putText(image, str(color_ind), (cx, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

            j += 1

        file1.close()
        cv2.imwrite(output_dir + 'gray_' + name_of_image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    #     cv2.imshow('frame', image)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
