import cv2
import numpy as np

class Filters:
    @staticmethod
    def apply_brightness_contrast(img, brightness, contrast):
        return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

    @staticmethod
    def apply_gamma(img, gamma):
        gamma = max(gamma, 0.01)

        table = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ]).astype("uint8")

        return cv2.LUT(img, table)

    @staticmethod
    def apply_gaussian_blur(img, k):
        if k % 2 == 0: k += 1
        return cv2.GaussianBlur(img, (k, k), 0) if k > 1 else img

    @staticmethod
    def apply_rotation(img, angle):
        if angle == 0: return img
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h))

    @staticmethod
    def apply_hist_eq(img):
        if len(img.shape) == 2:
            return cv2.equalizeHist(img)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)

    @staticmethod
    def apply_sobel(img):
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))

    @staticmethod
    def apply_canny(img):
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)

    @staticmethod
    def apply_laplacian(img):
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)

    @staticmethod
    def apply_averaging(img, k):
        if k < 1: k = 1
        return cv2.blur(img, (k, k))

    @staticmethod
    def apply_median(img, k):
        if k % 2 == 0: k += 1
        if k < 1: k = 1
        return cv2.medianBlur(img, k)

    @staticmethod
    def apply_bilateral(img, d=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    @staticmethod
    def apply_zoom_nearest(img, scale):
        h, w = img.shape[:2]
        new_w, new_h = max(1, int(w / scale)), max(1, int(h / scale))
        x1, y1 = (w - new_w) // 2, (h - new_h) // 2
        return cv2.resize(img[y1:y1 + new_h, x1:x1 + new_w], (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def apply_zoom_bilinear(img, scale):
        h, w = img.shape[:2]
        new_w, new_h = max(1, int(w / scale)), max(1, int(h / scale))
        x1, y1 = (w - new_w) // 2, (h - new_h) // 2
        return cv2.resize(img[y1:y1 + new_h, x1:x1 + new_w], (w, h), interpolation=cv2.INTER_LINEAR)