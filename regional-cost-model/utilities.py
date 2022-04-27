import numpy as np
import cv2

# https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
def simplest_cb(img, percent=1):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)

def equalise(img):
    # r = cv2.equalizeHist(img[:, :, 0])
    # g = cv2.equalizeHist(img[:, :, 1])
    # b = cv2.equalizeHist(img[:, :, 2])
    # img[:, :, 0] = r
    # img[:, :, 1] = g
    # img[:, :, 2] = b
    # return img

    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 0] = l
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img