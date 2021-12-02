import numpy as np
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