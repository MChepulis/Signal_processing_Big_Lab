from matplotlib import pyplot as plt
from imageio import imread, imsave
from skimage.color import rgb2gray
import numpy as np
from skimage.feature import canny
from skimage.filters import try_all_threshold
from skimage.filters import gaussian
from skimage.filters import threshold_otsu, rank
from skimage.filters import threshold_triangle
from skimage.morphology import disk
from skimage.morphology import binary_closing
from skimage.morphology import binary_opening
from skimage.util import img_as_ubyte
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
import cv2
from scipy.ndimage.morphology import binary_fill_holes
from matplotlib.patches import Rectangle


dataset_path = "..\\dataset"


def get_border(mask):
    vertical_indices = np.where(np.any(mask, axis=1))[0]
    top, bottom = vertical_indices[0], vertical_indices[-1]
    horizontal_indices = np.where(np.any(mask, axis=0))[0]
    left, right = horizontal_indices[0], horizontal_indices[-1]
    corner = (left, top)
    height = bottom - top
    width = right - left
    rect_border = Rectangle(corner, width, height, linewidth=5, edgecolor='b', facecolor='none')
    return rect_border


def crop_by_border(img, border):
    return img[border[0]:border[1], border[2]:border[3]]


def select_chair():
    chair_name = "yes_ok_1.jpg"
    chair_img = imread("%s\\%s" % (dataset_path, chair_name))
    '''
    plt.imshow(chair_img)
    plt.show()
    '''
    first_border = [2000, 3500, 200, 1500]
    chair_img_crop = crop_by_border(chair_img, first_border)
    '''
    plt.imshow(chair_img_crop)
    plt.show()
    '''
    chair_img_crop_gray = rgb2gray(chair_img_crop)
    thresh = threshold_triangle(chair_img_crop_gray)
    chair_img_crop_gray_binary = chair_img_crop_gray > thresh
    chair_img_crop_gray_binary = binary_opening(chair_img_crop_gray_binary, selem=np.ones((5, 5)))
    chair_img_crop_gray_binary = binary_closing(chair_img_crop_gray_binary, selem=np.ones((10, 10)))
    '''
    plt.imshow(chair_img_crop_gray_binary, cmap="gray")
    plt.show()
    '''
    border = get_border(1 - chair_img_crop_gray_binary)
    '''
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax.imshow(chair_img_crop_gray_binary, cmap='gray')
    ax.add_patch(border)
    plt.show()
    '''
    chair_border = [first_border[0] + border.get_y(), first_border[0] + border.get_y() + border.get_height(),
                    first_border[2] + border.get_x(), first_border[2] + border.get_x() + border.get_width()]
    selected_chair = crop_by_border(chair_img, chair_border)
    '''
    plt.imshow(selected_chair)
    plt.show()
    '''
    return selected_chair, chair_border


def select_door():
    door_name = "door_only.jpg"
    door_img = imread("%s\\%s" % (dataset_path, door_name))
    door_border = [70, 870, 150, 480]
    door_img_crop = crop_by_border(door_img, door_border)

    '''
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].imshow(door_img)
    ax[1].imshow(door_img_crop)
    plt.show()
    '''
    selected_door = door_img_crop
    return selected_door, door_border


def my_find_homography(img_1_gray, img_2_gray):
    '''
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].set_title('box')
    ax[0].imshow(img_1_gray, cmap='gray')
    ax[1].set_title('compos')
    ax[1].imshow(img_2_gray, cmap='gray')
    fig.show()
    '''
    orb_detector = cv2.ORB_create(5000)

    kp1, d1 = orb_detector.detectAndCompute(img_1_gray, None)
    kp2, d2 = orb_detector.detectAndCompute(img_2_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches_length = len(matches)
    p1 = np.zeros((matches_length, 2))
    p2 = np.zeros((matches_length, 2))
    for i in range(matches_length):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
    '''
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].set_title('box')
    ax[0].imshow(img_1_gray, cmap='gray')
    ax[0].plot(p1[:, 0], p1[:, 1], "o")

    ax[1].set_title('compos')
    ax[1].imshow(img_2_gray, cmap='gray')
    ax[1].plot(p2[:, 0], p2[:, 1], "o")
    fig.show()
    '''
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    return homography, mask


def highlight_obj_by_template(img_gray, template_img_gray):
    height, width = img_gray.shape
    homography, _ = my_find_homography(template_img_gray, img)
    mask = np.ones(template_img_gray.shape)
    transformed_mask = cv2.warpPerspective(mask, homography, (width, height))
    transformed_template = cv2.warpPerspective(template_img_gray, homography, (width, height))
    '''
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].set_title('box')
    ax[0].axis('off')
    ax[0].imshow(template_img_gray, cmap='gray')
    border = get_border(mask)
    ax[0].add_patch(border)

    ax[1].set_title('compos')
    ax[1].axis('off')
    ax[1].imshow(transformed_template, cmap='gray')
    border = get_border(transformed_mask)
    ax[1].add_patch(border)
    fig.show()
    '''
    return transformed_mask, transformed_template


def perspective_offset(img, door_template, chair_template):
    height, width = img.shape

    chair_transformed_mask, chair_transformed_template = highlight_obj_by_template(img, chair_template)
    door_transformed_mask, door_transformed_template = highlight_obj_by_template(img, door_template)

    homography, mask = my_find_homography(door_transformed_template, door_template)
    height, width = door_template.shape
    door_non_perspective_mask = cv2.warpPerspective(door_transformed_mask, homography, (width, height))
    non_perspective_img = cv2.warpPerspective(img, homography, (width, height))
    chair_non_perspective_mask = cv2.warpPerspective(chair_transformed_mask, homography, (width, height))

    return non_perspective_img, door_non_perspective_mask, chair_non_perspective_mask


# TODO скорее всего, матрицу обратного преобразования мужно найти при помощи матричных операций
# TODO скорее всего, вместо преобразования маски можно преобразовывать прямогольники, однако, если маска будет более
#  сложной формы, то прямоугольником уже не обойтись
if __name__ == "__main__":
    door_template, door_template_border = select_door()
    chair_template, chair_template_border = select_chair()
    door_template_gray = cv2.cvtColor(door_template, cv2.COLOR_BGR2GRAY)
    chair_template_gray = cv2.cvtColor(chair_template, cv2.COLOR_BGR2GRAY)

    img_dataset_name = ["yes_ok_4.jpg", "yes_top_spin_1.jpg", "yes_scale_2.jpg", "yes_scale_1.jpg", "yes_ok_3.jpg", "yes_ok_2.jpg", "yes_ok_1.jpg", "yes_blur_1.jpg", "yes_blur_2.jpg", "yes_blur_3.jpg", "yes_blur_4.jpg", "yes_2chair_1.jpg"]
    for name in img_dataset_name:
        img = cv2.imread("%s\\%s" % (dataset_path, name))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        non_perspective_img, non_perspective_door_mask, non_perspective_chair_mask = perspective_offset(img_gray, door_template_gray, chair_template_gray)

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].set_title('%s' % name)
        ax[0].imshow(img_gray, cmap='gray')
        ax[0].axis('off')

        ax[1].set_title('non perspective %s' % name)
        ax[1].axis('off')
        ax[1].imshow(non_perspective_img, cmap='gray')
        chair_non_perspective_border = get_border(non_perspective_chair_mask)
        ax[1].add_patch(chair_non_perspective_border)
        door_non_perspective_border = get_border(non_perspective_door_mask)
        ax[1].add_patch(door_non_perspective_border)
        plt.show()
        if (door_non_perspective_border.get_width() > chair_non_perspective_border.get_width()):
            print("Yes : %s" % name)
        else:
            print("No  : %s" % name)










































