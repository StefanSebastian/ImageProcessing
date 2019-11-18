import cv2
import numpy as np 
import sys

maxval = 255
cross = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
square = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
xshape = [(0, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]
diamond = [(-1, 0), (1, 0), (0, 1), (0, -1)]

def threshold(img, threshold, max_val):
    new_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > threshold:
                new_img[i][j] = 0
            else:
                new_img[i][j] = max_val
    return new_img

def erosion(img, kernel):
    new_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            contained = True
            for (kx, ky) in kernel:
                px = i + kx 
                py = j + ky
                if img[px][py] == 0:
                    contained = False
            new_img[i][j] = maxval if contained else 0

    return new_img

def dilation(img, kernel):
    new_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i][j] == maxval:
                for (kx, ky) in kernel:
                    px = i + kx 
                    py = j + ky
                    new_img[px][py] = maxval
    return new_img

def absdiff(img1, img2):
    new_img = np.zeros((img1.shape[0], img1.shape[1], 1), np.uint8)
    for i in range(img1.shape[0]):
        for j in range(img2.shape[1]):
            new_img[i][j] = abs(img1[i][j] - img2[i][j])
    return new_img

def build_result(img, corners):
    new_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = 255 if corners[i][j] == 255 else img[i][j]
    return new_img

def corner_detection(img_path, threshold_val):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = threshold(img_gray, threshold_val, maxval)

    R1 = dilation(img_bin, cross)
    R1 = erosion(R1, diamond)
    R2 = dilation(img_bin, xshape)
    R2 = erosion(R2, square)
    R = absdiff(R1, R2)
    cv2.imshow(img_path, R)

    result = build_result(img_gray, R)
    cv2.imshow(img_path + "res", result)
    
corner_detection('./MorpologicalCornerDetection.png', 40)
corner_detection('./square-rectangle.png', 200)


while(True):
    if cv2.waitKey(1) == ord('q'):
        break 
cv2.destroyAllWindows()