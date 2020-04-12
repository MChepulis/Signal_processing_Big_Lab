from matplotlib import pyplot as plt
import os
import numpy as np
import cv2


class Match:
    def __init__(self):
        self.name = ""
        self.length = 0
        self.kp = []
        self.d = []
        self.matches = []
        self.image = None


def get_bottom(image, percent=0.5):
    border = [round(image.shape[0] * (1 - percent)), image.shape[0], 0, image.shape[1]]
    return image[border[0]:border[1], border[2]:border[3]]


def find_best_template(image, output_flag=False):
    templates_dir = "chair_templates/without_background"
    image = get_bottom(image)
    if output_flag:
        plt.imshow(image, cmap="gray")
        plt.show()

    orb_detector = cv2.ORB_create(50000)
    kp1, d1 = orb_detector.detectAndCompute(image, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match = Match()

    files = os.listdir(templates_dir)
    templates = filter(lambda x: x.endswith('.jpg'), files)
    if not templates:
        print("wrowg dir")
    threshold = 50
    for templ_name in templates:

        path = os.path.join(templates_dir, templ_name)
        template = cv2.imread(path)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        kp2, d2 = orb_detector.detectAndCompute(template_gray, None)

        matches = matcher.match(d1, d2)
        matches_length = len(matches)

        if output_flag:
            print("templ: %s, %i" % (templ_name, matches_length))
            p1 = np.zeros((matches_length, 2))
            p2 = np.zeros((matches_length, 2))
            for i in range(matches_length):
                p1[i, :] = kp1[matches[i].queryIdx].pt
                p2[i, :] = kp2[matches[i].trainIdx].pt

            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            ax[0].set_title('image')
            ax[0].imshow(image, cmap='gray')
            ax[0].plot(p1[:, 0], p1[:, 1], "o")

            ax[1].set_title('template')
            ax[1].imshow(template_gray, cmap='gray')
            ax[1].plot(p2[:, 0], p2[:, 1], "o")
            plt.show()

            img_matches = cv2.drawMatches(image, kp1, template_gray, kp2, matches,
                                          None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img_matches)
            plt.show()

        if output_flag:
            good_matches = []
            good_matches_total = 0
            for m in matches:
                if m.distance <= threshold:
                    good_matches.append(m)
                    good_matches_total += 1

            print("templ: %s, %i" % (templ_name, good_matches_total))
            p1 = np.zeros((good_matches_total, 2))
            p2 = np.zeros((good_matches_total, 2))
            for i in range(good_matches_total):
                p1[i, :] = kp1[good_matches[i].queryIdx].pt
                p2[i, :] = kp2[good_matches[i].trainIdx].pt

            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
            ax[0].set_title('image')
            ax[0].imshow(image, cmap='gray')
            ax[0].plot(p1[:, 0], p1[:, 1], "o")

            ax[1].set_title('template')
            ax[1].imshow(template_gray, cmap='gray')
            ax[1].plot(p2[:, 0], p2[:, 1], "o")
            plt.show()

            img_matches = cv2.drawMatches(image, kp1, template_gray, kp2, good_matches,
                                          None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img_matches)
            plt.show()


        if matches_length >= best_match.length:
            best_match.length = matches_length
            best_match.kp = kp2
            best_match.d = d2
            best_match.name = templ_name
            best_match.matches = matches
            best_match.image = template_gray

    # как понять что совпадение не найдено? каков порог?
    if output_flag:
        print("best_match: %s, %i" % (best_match.name, best_match.length))
        plt.imshow(best_match.image, cmap="gray")
        plt.show()

    # теперь, зная шаблон и особые точки нужно как-то определить шарину стула.
    # тривиальный: min, max по координате x
    p1 = np.zeros((best_match.length, 2))
    p2 = np.zeros((best_match.length, 2))
    for i in range(best_match.length):
        p1[i, :] = kp1[best_match.matches[i].queryIdx].pt
        p2[i, :] = best_match.kp[best_match.matches[i].trainIdx].pt

    if output_flag:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].set_title('image')
        ax[0].imshow(image, cmap='gray')
        ax[0].plot(p1[:, 0], p1[:, 1], "o")

        ax[1].set_title('template')
        ax[1].imshow(best_match.image, cmap='gray')
        ax[1].plot(p2[:, 0], p2[:, 1], "o")
        plt.show()

    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)




