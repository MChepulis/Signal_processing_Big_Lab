
from matplotlib import pyplot as plt
from skimage import img_as_ubyte
from skimage.morphology import disk
from skimage.transform import hough_line, hough_line_peaks
from skimage.filters import gaussian, rank
from skimage.morphology import binary_closing, binary_opening
import numpy as np


def try_all_threshold_for_hough(image, output_flag=False):
    '''
    # canny
    img_canny = canny(image)
    percent = 0.5
    border = [0, round(img_canny.shape[0]*percent), 0, img_canny.shape[1]]
    tmp_img = img_canny[border[0]:border[1], border[2]:border[3]]
    draw_all_hough_line(image, tmp_img)
    '''
    # local_otsu + morphology
    tmp_img = local_otsu_preproc(image, output_flag)
    draw_all_hough_line(image, tmp_img)


def draw_all_hough_line(origin, image, output_flag=False):

    h, theta, d = hough_line(image)
    if output_flag:
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))

        ax[0].imshow(origin, cmap="gray")
        ax[0].set_title('original')
        ax[0].set_axis_off()

        ax[1].imshow(image, cmap='gray')
        ax[1].set_title('input')

        ax[2].imshow(origin, cmap="gray")
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        if output_flag:
            ax[2].plot((0, image.shape[1]), (y0, y1), '-r')

    if output_flag:
        ax[2].set_xlim((0, image.shape[1]))
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')
        plt.tight_layout()
        fig.show()
        plt.close(fig)


def local_otsu_preproc(image, output_flag=False):
    img = img_as_ubyte(image)
    radius = 15
    selem = disk(radius)
    local_otsu = rank.otsu(img, selem)
    img_local_otsu = img >= local_otsu

    img_local_otsu_blur = gaussian(img_local_otsu, sigma=0.9, multichannel=True)
    tmp_res_otsu_open = binary_closing(img_local_otsu_blur, selem=np.ones((5, 5)))
    tmp_res_otsu_close = binary_opening(tmp_res_otsu_open, selem=np.ones((10, 10)))

    if output_flag:
        plt.imshow(tmp_res_otsu_close, cmap="gray")
        plt.show()
    percent = 0.5
    border = [0, round(tmp_res_otsu_close.shape[0] * percent), 0, tmp_res_otsu_close.shape[1]]
    tmp_img = tmp_res_otsu_close[border[0]:border[1], border[2]:border[3]]
    res_img = tmp_img == 0
    return res_img


# FIXME прямые - наклонные, на какой именно высоте стоит искать ширину? на высоте стула (верхняя, нижняя, середина)
def find_door_width(image, output_flag=False, vertical_limiter=0.3):
    x_min = image.shape[1]
    x_max = 0.0
    tmp_image = local_otsu_preproc(image, output_flag)
    h, theta, d = hough_line(tmp_image)

    if output_flag:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()
        ax[1].imshow(image, cmap='gray')

    left_line = []
    right_line = []
    left_line_param = {}
    right_line_param = {}
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        if np.abs(angle) <= vertical_limiter:
            if x_min > dist:
                left_line = [[0, image.shape[1]], [y0, y1]]
                left_line_param["angle"] = angle
                left_line_param["dist"] = dist
                x_min = dist
            if x_max < dist:
                right_line = [[0, image.shape[1]], [y0, y1]]
                right_line_param["angle"] = angle
                right_line_param["dist"] = dist
                x_max = dist
            if output_flag:
                ax[1].plot((0, image.shape[1]), (y0, y1), '-r')
    if output_flag:
        ax[1].set_xlim((0, image.shape[1]))
        ax[1].set_ylim((image.shape[0], 0))
        # ax[1].set_axis_off()
        ax[1].set_title('Detected lines')
        plt.tight_layout()
        fig.show()
        plt.close(fig)

    return x_max - x_min, [left_line, right_line], [left_line_param, right_line_param]





