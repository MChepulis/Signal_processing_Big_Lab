from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.morphology import binary_closing, binary_opening
import numpy as np
from matplotlib.patches import Rectangle
from skimage.color import rgb2gray
from skimage.filters import threshold_minimum
from skimage.measure import label, regionprops


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


def get_largest_component(mask, background=0):
    # разбиение маски на компоненты связности
    labels = label(mask, background=background)
    # нахождение свойств каждой области (положение центра, площадь, bbox, интервал интенсивностей и т.д.)
    props = regionprops(labels)
    # нас интересуют площади компонент связности
    areas = [prop.area for prop in props]

    # print("Значения площади для каждой компоненты связности: {}".format(areas))
    largest_comp_id = np.array(areas).argmax() # находим номер компоненты с максимальной площадью
    # print("labels - матрица, заполненная индексами компонент связности со значениями из множества: {}".format(np.unique(labels)))
    # print("максимальная компонента:", largest_comp_id + 1)
    # области нумеруются с 1, поэтому надо прибавить 1 к индексу
    return labels == (largest_comp_id + 1)


def get_chair_width(image, output_flag=False):
    image_blur = gaussian(image, sigma=2, multichannel=True)
    largest_component = binarization(image_blur, output_flag)
    border = get_border(largest_component)

    if output_flag:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, cmap='gray')
        ax.add_patch(border)
        fig.show()
        plt.close(fig)
    return border.get_width(), border


def binarization(image, output_flag=False):
    image_gray = rgb2gray(image)
    thresh = threshold_minimum(image_gray)
    image_binary = image_gray > thresh

    image_binary = binary_opening(image_binary, selem=np.ones((5, 5)))
    image_binary = binary_closing(image_binary, selem=np.ones((10, 10)))
    largest_component = get_largest_component(image_binary, background=1)
    if output_flag:
        fig, ax = plt.subplots(1, 3)
        ax[0].set_title('input')
        ax[1].set_title('binarization')
        ax[2].set_title('largest component')
        ax[0].imshow(image, cmap='gray')
        ax[1].imshow(image_binary, cmap='gray')
        ax[2].imshow(largest_component, cmap='gray')
        fig.show()
        plt.close(fig)

    return largest_component



