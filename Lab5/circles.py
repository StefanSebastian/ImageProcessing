import cv2
import numpy as np 
import math


img = cv2.imread("./circles5.jpg")
print('Original Dimensions : ',img.shape)
 
scale_percent = 60 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

edges = cv2.Canny(resized, 100, 200)

cv2.imshow("canny", edges)

h, w = edges.shape
H = {}
theta_st = 1
for i in range(h):
    for j in range(w):
        if edges[i][j] != 0:
            for r in range(10, 100):
                for theta in range(0, 360, theta_st):
                    a = int(i - r * math.cos(theta * math.pi / 180))
                    b = int(j + r * math.sin(theta * math.pi / 180))
                    t = (a, b, r)
                    if t in H:
                        H[t] += 1
                    else:
                        H[t] = 0

vals = list(H.values())
vals.sort(reverse = True)
vals = vals[:10]
print(vals)

for key in H:
    if H[key] in vals:
        a, b, r = key
        cv2.circle(resized, (a, b), r, (255, 0, 0), 5)
cv2.imshow("res", resized)

cv2.waitKey(0)