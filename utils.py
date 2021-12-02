import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2


def mean_frame_changing(p0, p1, st):
    """
    finding the mean do all feathers differences
    :param p0: the best feathers in the old frame
    :param p1: the best feathers in the new frame
    :param st: check if the feather exist in p0 and p1
    :return: the mean different between frames
    """
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    changing_inside_frames = []
    for (new, old) in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        tamp = ((int(a) - int(c)) ** 2 + (int(b) - int(d)) ** 2) ** 0.5
        changing_inside_frames.append(tamp)
    changing_inside_frames = np.array(changing_inside_frames).mean()
    return changing_inside_frames


def show_frame_diff(old_gray, gray, p0, p1, st):
    """
    print two frames and show the differences between them
    :param old_gray: the older frame
    :param gray: the current frame
    :param p0: the best feathers in the old frame
    :param p1: the best feathers in the new frame
    :param st: check if the feather exist in p0 and p1
    :return: None
    """
    if p1 is not None:
        good_new = p1[st == 1]  # find only existing points
        good_old = p0[st == 1]
        X_shift = np.squeeze(good_old[:, 0] - good_new[:, 0]).mean()
        Y_shift = np.squeeze(good_old[:, 1] - good_new[:, 1]).mean()
        X_shift = int(X_shift)  # find the mean different in the X label
        Y_shift = int(Y_shift)  # find the mean different in the Y label
        num_rows, num_cols = old_gray.shape[:2]
        translation_matrix = np.float32([[1, 0, X_shift], [0, 1, Y_shift]])
        new_pic = cv2.warpAffine(old_gray, translation_matrix,
                                 (num_cols, num_rows))  # moving the old pic to look like the new
        print(f"X: {X_shift}")
        print(f"Y:{Y_shift}")
        ## plot the required picther
        plt.subplot(1, 3, 1), plt.imshow(old_gray, 'gray')
        plt.subplot(1, 3, 2), plt.imshow(gray, 'gray')
        plt.subplot(1, 3, 3), plt.imshow(new_pic, 'gray')
        plt.show()
        ##
