import cv2
import numpy as np 
import math

img = cv2.imread("./test.jpg")
edges = cv2.Canny(img, 100, 200)

cv2.imshow("canny", edges)

h, w = edges.shape
d = np.ceil(np.sqrt(w * w + h * h))
ro = 180

# range is from -diag to diag ; and from 0 to 180
H = np.zeros((int(d * 2), ro + 1), np.uint8)

for i in range(h):
    for j in range(w):
        if edges[i][j] != 0:
            for ro_c in range(181):
                d_c = i * math.sin(ro_c) + j * math.cos(ro_c) + d # add diag value to avoid negative values
                d_c = int(d_c)
                H[d_c][ro_c] += 1

# find the max values in H matrix 
acc = []
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        if H[i][j] > 0:
            acc.append(H[i][j])
acc.sort(reverse = True)
acc = acc[:5]

# draw lines on image
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        if H[i][j] in acc:
            rho = i - d 
            theta = j
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
cv2.imshow("lines", img)

cv2.waitKey(0)