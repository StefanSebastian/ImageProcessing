import cv2
import numpy as np 
import sys

img1 = cv2.imread('./MorpologicalCornerDetection.png')
img2 = cv2.imread('./square-rectangle.png')

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cross = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
square = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
xshape = [(0, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]
diamond = [(-1, 0), (1, 0), (0, 1), (0, -1)]

def erosion(img, kernel):
    new_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            minval = sys.maxsize
            for (kx, ky) in kernel:
                px = i + kx 
                py = j + ky

                if minval > img[px][py]:
                    minval = img[px][py]

            new_img[i][j] = minval
    return new_img

def dilation(img, kernel):
    new_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            maxval = -sys.maxsize
            for (kx, ky) in kernel:
                px = i + kx 
                py = j + ky

                if maxval < img[px][py]:
                    maxval = img[px][py]

            new_img[i][j] = maxval
    return new_img

def absdiff(img1, img2):
    new_img = np.zeros((img1.shape[0], img1.shape[1], 1), np.uint8)
    for i in range(img1.shape[0]):
        for j in range(img2.shape[1]):
            new_img[i][j] = abs(img1[i][j] - img2[i][j])
    return new_img


def corner_detection(img, name):
    R1 = dilation(img, cross)
    R1 = erosion(R1, diamond)
    R2 = dilation(img, xshape)
    R2 = erosion(R2, square)
    R = absdiff(R1, R2)
    cv2.imshow(name, R)

corner_detection(img1_gray, "building")
corner_detection(img2_gray, "square")


while(True):
    if cv2.waitKey(1) == ord('q'):
        break 
cv2.destroyAllWindows()