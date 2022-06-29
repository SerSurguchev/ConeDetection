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

def pointIntoRectangle(boxes, x, y):

    def product(Ax, Ay, Bx, By, Px, Py):
        return (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax)

    check = False

    # x, y, w, h
    for box in boxes:
        p1 = product(box[0], box[1], box[0], box[1] + box[3], x, y)
        p2 = product(box[0], box[1] + box[3], box[0] + box[2], box[1] + box[3], x, y)
        p3 = product(box[0] + box[2], box[1] + box[3], box[0] + box[2], box[1], x, y)
        p4 = product(box[0] + box[2], box[1], box[0], box[1], x, y)

        # point into rectangle
        if ((p1 < 0 and p2 < 0 and p3 < 0 and p4 < 0) or
                (p1 > 0 and p2 > 0 and p3 > 0 and p4 > 0)):
            check = True
            break

    return check

# def checkOverlap(bboxes):
#
#     def pointIntoCircle(R, Xc, Yc, X1, Y1, X2, Y2):
#
#         # Find the nearest point on the
#         # rectangle to the center of
#         # the circle
#         Xn = max(X1, min(Xc, X2))
#         Yn = max(Y1, min(Yc, Y2))
#
#         Dx = Xn - Xc
#         Dy = Yn - Yc
#
#         return (Dx**2 + Dy**2) <= R**2
#
#     for x, y, h, w in bboxes:
#
#         if pointIntoCircle(w//2, (x + x + w)//2, (y + y + h)//2, x, (y + h),  (x + w), y):
#             return True
#
#     return False

def get_bitwise(bitwise):

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


def get_color(orig_bitwise, fallen, possible_big_cone):

    # 0 black middle
    # 1 white middle
    # 2 unknown

    gray_bitwise = cv2.cvtColor(orig_bitwise, cv2.COLOR_BGR2GRAY)
    bitwise_center = gray_bitwise.item(round(gray_bitwise.shape[0] / 2), round(gray_bitwise.shape[1] / 2))
    half_b = orig_bitwise[round(orig_bitwise.shape[0]/2):, :]

    b = (np.sum(half_b[:, :, 0]))
    r_g = (np.sum(half_b[:, :, 2]) + np.sum(half_b[:, :, 1])) / 2
    or_sum = (b + r_g) / 100

    # detect blue cone (class 1 white)
    if b > r_g and (np.sum(half_b[:, :, 0])) / (np.sum(half_b[:, :, 2])) > 0.7:
        return 1

    # detect yellow and red cone
    else:
        custom_bitwise = get_bitwise(bitwise)
        yellow = np.sum(custom_bitwise[:, :, 1])
        yr = np.sum(custom_bitwise[:, :, 2])

        if fallen:

        # detect fallen cones
            return 2 if yr/yellow > 2 else 0

        elif possible_big_cone and yr/yellow > 5:
            return 2

        # detect big orange cone (unknown class)
        elif check_cone_gradient(orig_bitwise=orig_bitwise)\
                and (yellow/yr < 0.7 or yr/yellow > 3):

            return 2

        # detect yellow cone (class 0)
        elif check_cone_gradient(orig_bitwise=orig_bitwise) \
                and yellow/yr >= 0.7:

            # print('black', yellow/yr)
            return 0

        # detect red cone
        else:

            # cv2.imshow(f'{str(round((yellow/yr), 4))}  {str(round((yr/yellow), 4))}', orig_bitwise)
            return 1

if __name__ == '__main__':

    data_fr_file = pd.read_csv('/home/sergey/from_wind/programming/Computer_vision/all.csv')
    dir = r'/home/sergey/from_wind/programming/Computer_vision/YOLO_Dataset/'
    output_dir = '/home/sergey/from_wind/programming/Computer_vision/dataset_grayscale/'

    lens = len(glob.glob(dir + '*.jpg'))

    if os.path.isdir(output_dir) == False:
        print(output_dir, 'created')
        os.makedirs(output_dir)

    for i in range(1, lens):

        j = 5

        name_of_image = data_fr_file.iloc[i, 0]

        image = cv2.imread(dir + name_of_image)

        c_height, c_width = image.shape[:2]

        name_of_file = name_of_image[:-3] + "txt"

        file1 = open(output_dir + 'gray_' + name_of_file, 'w')

        mask = pd.DataFrame(data_fr_file.iloc[i, j:]).dropna()
        bboxes = []
        for index, row in mask.iterrows():
            h = ast.literal_eval(row[i])[2]
            if h > 25:
                bboxes.append(ast.literal_eval(row[i]))

        bboxes = sorted(bboxes, key=lambda x: x[2], reverse=True)

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

                # check_point1 = pointIntoRectangle(bboxes, x=x, y=y + h)
                # check_point2 = pointIntoRectangle(bboxes, x=x + w, y=y + h)

                # if check_point1 or check_point2:
                #     # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
                #     j += 1
                #     continue

                # create cone rectangle with black pixels around the edges
                cx = round((x + x + w) / 2)
                lst = []
                lst.append([cx - x, 0])
                lst.append([0, h])
                lst.append([w, h])
                cone = image[y:y + h, x:x + w]
                mask = cv2.fillPoly(np.zeros((h, w), dtype=np.uint8), [np.asarray(lst)], (255, 255, 255))
                bitwise = cv2.bitwise_and(cone, cone, mask=mask)

                # leave copy of original bitwise
                orig_bitwise = bitwise.copy()

                # calculate cone brightness and if it is toо bright, pass this cone
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

                    # detect red cone on video contains only red cone (1 class)

                    if c_height == 1080 and c_width == 1920:
                        color_ind = 2 if check_cone_gradient(orig_bitwise) else 1

                    else:
                        color_ind = get_color(orig_bitwise, fallen, possible_big_cone)

                    # write labels in .txt file
                    file1.write(f"{color_ind} {x_norm} {y_norm} {w_norm} {h_norm}\n")

                    # cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 2)
                    # cv2.putText(image, str(color_ind), (cx, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

            j += 1

        file1.close()
        cv2.imwrite(output_dir + 'gray_' + name_of_image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    #     cv2.imshow('frame', image)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
