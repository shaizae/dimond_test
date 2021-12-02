import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2


def mean_frame_changing(p0, p1, st):
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
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        X_shift = np.squeeze(good_old[:, 0] - good_new[:, 0]).mean()
        Y_shift = np.squeeze(good_old[:, 1] - good_new[:, 1]).mean()
        X_shift = int(X_shift)
        Y_shift = int(Y_shift)
        num_rows, num_cols = old_gray.shape[:2]
        translation_matrix = np.float32([[1, 0, X_shift], [0, 1, Y_shift]])
        new_pic = cv2.warpAffine(old_gray, translation_matrix, (num_cols, num_rows))
        print(f"X: {X_shift}")
        print(f"Y:{Y_shift}")


    plt.subplot(1, 3, 1), plt.imshow(old_gray, 'gray')
    plt.subplot(1, 3, 2), plt.imshow(gray, 'gray')
    plt.subplot(1, 3, 3), plt.imshow(new_pic, 'gray')
    plt.show()
