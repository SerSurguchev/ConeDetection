import cv2
import numpy as np
import pandas as pd
import glob
import os
import math
from PIL import Image, ImageEnhance, ImageStat, ImageTk
import tkinter as tk
import matplotlib.pyplot as plt

def classes_convert(data_fr_file, img_dir):

    lens = len(glob.glob(img_dir + '*.jpg'))
    final_arr = []

    for i in range(4028, lens):
        j = 5
        name_of_file = data_fr_file.iloc[i, 0]
        name_of_file = name_of_file[:-3] + "txt"
        without = name_of_file.split('.')[0] + '.jpg'

        dct = {}
        coords = []

        while True:

            mark1 = data_fr_file.iloc[i, j]
            if pd.isna(mark1):
                break

            mark1 = str(mark1)
            mark1 = mark1[1:-1]
            mark = mark1.replace(',', '')
            x, y, h, w = mark.split()
            coords.append([int(x), int(y), int(w), int(h)])
            j += 1

        dct[without] = coords
        final_arr.append(dct)

    return final_arr

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

def remove_whiteHLS(orig_bitwise, sensitivity):

    imageHLS = cv2.cvtColor(orig_bitwise, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(imageHLS, np.array([0, 255 - sensitivity, 0]), np.array([255, 255, 255]))
    mask = 255 - mask

    without_white = cv2.bitwise_and(orig_bitwise, orig_bitwise, mask=mask)

    return without_white

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

    # print(len(increase) + len(decrease))
    # print('increase: ', increase)
    # print('decrease: ', decrease)

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

def get_color(custom_bitwise, orig_bitwise, without_white_color,fallen):

    # 0 blue cone
    # 1 yellow cone
    # 2 red cone
    # 3 Unknown
    # 4 big red cone

    # detect blue cone

    half_b = without_white_color[round(without_white_color.shape[0]/2):, :]
    b = (np.sum(half_b[:, :, 0]))
    r_g = (np.sum(half_b[:, :, 2]) + np.sum(half_b[:, :, 1])) / 2
    or_sum = (b + r_g) / 100

    if (np.sum(half_b[:, :, 0])) > (np.sum(half_b[:, :, 2]) + np.sum(half_b[:, :, 1])) / 2:
        return 0

    # if the percentage is ambiguous, return 3 (unknown class)

    if abs((b/or_sum) - (r_g/or_sum)) < 10:
        return 3

    # detect yellow and red cone
    else:
        yellow = np.sum(custom_bitwise[:, :, 1])
        yr = np.sum(custom_bitwise[:, :, 2])

        a = (yellow + yr) / 100

        # detect fallen cones
        if fallen:
            return np.argmax([yellow, yr]) + 1
        
        # detect big red cone
        elif possible_big_cone and check_cone_gradient(orig_bitwise=orig_bitwise) and yr > yellow:
            return 4

        # detect yellow cone
        elif check_cone_gradient(orig_bitwise=orig_bitwise) or (check_cone_gradient(orig_bitwise=orig_bitwise) and yellow > yr):
            return 1
        
        # red cone
        else:
            # cv2.imshow(f'{str(round((b / or_sum), 4))}  {str(round((r_g / or_sum), 4))}', half_b)
            return 2

def main(final_arr, img_dir, output_dir):
    photo_ind = 0
    for dct in final_arr:
        for key, value in dct.items():
            boxes = value
            txt_file = open(output_dir + key.split('.')[0] + '.txt', 'w')
            im = cv2.imread(img_dir + key)
            c_height, c_width = im.shape[:2]

            for x, y, w, h, in value:

                # don't detect too distant cone
                if h < 25:
                    continue
                    
                point_check1 = pointIntoRectangle(boxes=boxes, x=x, y=y + h)
                point_check2 = pointIntoRectangle(boxes=boxes, x=x + w, y=y + h)
                
                if point_check1 or point_check2:
#                     cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), thickness=1)
                    continue

                # create cone rectangle with black pixels around the edges

                cx = round((x + x + w) / 2)
                lst = []
                lst.append([cx - x, 0])
                lst.append([0, h])
                lst.append([w, h])
                cone = im[y:y + h, x:x + w]
                mask = cv2.fillPoly(np.zeros((h, w), dtype=np.uint8), [np.asarray(lst)], (255, 255, 255))
                bitwise = cv2.bitwise_and(cone, cone, mask=mask)

                # leave copy of original bitwise
                orig_bitwise = bitwise.copy()

                # calculate cone brightness and if it is toо bright, pass this cone
                avr_brightness = calc_brightness(Image.fromarray(cv2.cvtColor(cone, cv2.COLOR_BGR2RGB)))
                if int(avr_brightness) > 180:
                    print('Cone not detected')
                    continue

                # create custom color bitwise
                custom_bitwise = get_bitwise(bitwise)

                # remove bright white color
                # without_white_color = remove_whiteHLS(orig_bitwise.copy(), sensitivity=15)
                without_white_color = remove_white_naive(orig_bitwise.copy(), threshold=200)

                # checking whether the cone has fallen
                fallen = False
                if h < w:
                    fallen = True
                    # cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                    
                possible_big_cone = False
                if h > w:
                    possible_big_cone = True

                # cone color detect
                color_ind = get_color(custom_bitwise, orig_bitwise, without_white_color, fallen)

#                 cv2.putText(im, str(color_ind), (cx, int(y) - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

            txt_file.close()

            # don't save txt file without labels and photo without cones
            if os.path.getsize(output_dir + key.split('.')[0] + '.txt') == 0:
                os.remove(output_dir + key.split('.')[0] + '.txt')
                os.remove(output_dir + key)

            cv2.imshow(f'{photo_ind}', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
#             cv2.imwrite(r"C:\Users/Sergey/PycharmProjects/tensorEnv/Formula/" + key, im)

        photo_ind += 1

if __name__ == '__main__':
    data_fr_file = pd.read_csv('C:\For programming\Computer vision/all.csv')
    dir = r'C:\For programming\Computer vision/YOLO_Dataset/'
    output_dir = r'C:\For programming\Computer vision/dataset_grayscale/'
    final_arr = classes_convert(data_fr_file, dir)
    main(final_arr, dir, output_dir)

    # check whether number of photos is equal to the number of labels
    print(len(glob.glob(output_dir + '*.jpg')))
    print(len(glob.glob(output_dir + '*.txt')))

