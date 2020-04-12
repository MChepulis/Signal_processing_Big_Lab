import argparse
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

from chair_finder_ORB import find_best_template
from chair_finder_bin import get_chair_width
from door_finder import find_door_width

DATASET_DIR = "dataset"
TESTS_DER = "tests"
BORDER_PERCENT = 1 / 9


def chair_ORB_finder_test(output_flag=False):
    dir = "chair_test"
    dir_path = os.path.join(TESTS_DER, dir)
    files = os.listdir(dir_path)
    images = filter(lambda x: x.endswith('.jpg'), files)
    for img_name in images:
        print(img_name)
        path = os.path.join(dir_path, img_name)
        image = cv2.imread(path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        find_best_template(image_gray, output_flag)


def find_door_width_near_chair(chair_border, line_params):
    # y = chair_border.get_y() + chair_border.get_height()
    y = chair_border.get_y()
    x = []
    for param in line_params:
        if param == {}:
            return -1
        angle = param["angle"]
        dist = param["dist"]
        tmp_x = (dist - y * np.sin(angle))/np.cos(angle)
        x.append(tmp_x)
    x_max = np.max(x)
    x_min = np.min(x)
    left_x = x_min + (x_max - x_min) * BORDER_PERCENT
    right_x = x_max - (x_max - x_min) * BORDER_PERCENT
    line = [[left_x, right_x], [y, y]]
    width = right_x - left_x
    return width, line


def test(mode="common", dir_path=None, output_detail_flag=False, output_flag=False):
    if mode == "common":
        if dir_path is None:
            dir = "common_test"
            dir_path = os.path.join(TESTS_DER, dir)
        chair_proc_flag = True
        door_proc_flag = True
    elif mode == "door":
        if dir_path is None:
            dir = "door_test"
            dir_path = os.path.join(TESTS_DER, dir)
        chair_proc_flag = False
        door_proc_flag = True
    elif mode == "chair":
        if dir_path is None:
            dir = "chair_test"
            dir_path = os.path.join(TESTS_DER, dir)
        chair_proc_flag = True
        door_proc_flag = False
    else:
        print("Wrong mode \"%s\"" % mode)
        return
    correct_num = 0
    all_num = 0

    files = os.listdir(dir_path)
    images = filter(lambda x: x.endswith('.jpg'), files)
    for img_name in images:
        print(img_name)
        path = os.path.join(dir_path, img_name)
        image = cv2.imread(path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if door_proc_flag:
            door_width, lines, param = find_door_width(image_gray, output_detail_flag)
        if chair_proc_flag:
            chair_width, border = get_chair_width(image_gray, output_detail_flag)

        if output_flag:
            fig, ax = plt.subplots(1, 2)
            fig.suptitle(img_name)
            ax[0].set_title("\ninput")
            ax[1].set_title("\nresult")
            ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax[1].imshow(image_gray, cmap='gray')
            if chair_proc_flag:
                ax[1].add_patch(border)
            if door_proc_flag:
                for line in lines:
                    if not line:
                        continue
                    ax[1].plot(line[0], line[1], "r")
                ax[1].set_xlim((0, image.shape[1]))
                ax[1].set_ylim((image.shape[0], 0))

        if door_proc_flag and mode == "door":
            print("door`s width  = %i" % door_width)
        if chair_proc_flag:
            print("chair`s width = %i" % chair_width)
        if mode == "common":
            alt_door_width, alt_line = find_door_width_near_chair(chair_border=border, line_params=param)
            print("alt door width = %i" % alt_door_width)
            if output_flag and alt_line:
                ax[1].plot(alt_line[0], alt_line[1], "g.-")

            if alt_door_width >= chair_width:
                ans = "yes"
            else:
                ans = "no"
            print("Answer is %s" % ans)
            all_num += 1
            if img_name.startswith(ans):
                correct_num += 1
                print("Correct\n")
            else:
                print("Incorrect\n")

        if output_flag:
            fig.show()
            plt.close(fig)
    if mode == "common":
        print("Correct percent = %s" % (correct_num / all_num))


def parse_arguments():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-p', '--dir', default=None, help='"path to the input images dir')
    parser.add_argument('-m', '--mode', default="common",
                        help='mode can be "door", "chair", "common" (default: "common")')
    parser.add_argument('-o', '--output_img', default=False,
                        help='output result images')
    parser.add_argument('-d', '--details', default=False,
                        help='output details of algorithm')

    return parser.parse_args()


def main():
    args = parse_arguments()
    img_dir = args.dir
    mode = args.mode
    output_flag = args.output_img
    output_details_flag = args.details
    test(mode=mode, dir_path=img_dir, output_flag=output_flag, output_detail_flag=output_details_flag)
    # chair_ORB_finder_test(True)


if __name__ == "__main__":
    main()










































