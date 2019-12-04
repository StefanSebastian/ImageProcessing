import cv2 
import numpy as np

# read img
img = cv2.imread("star.png", 0)
ret,A = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY_INV)
cv2.imshow("initial", img)

p = [66, 51]
rows, cols = A.shape
print(rows, cols, p)

# build complementary mat
Ac = [[0 for x in range(cols)] for y in range(rows)]
for i in range(rows):
    for j in range(cols):
        Ac[i][j] = 1 - A[i][j]

B = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]]

X0 = set()
X0.add((p[0], p[1]))
X1 = set()
step = 0
while True:
    step += 1
    # intersect Ac with Xk-1 + B
    for point in X0:
        for dir in B:
            candidate = (point[0] + dir[0], point[1] + dir[1])
            if Ac[candidate[0]][candidate[1]] == 1:
                X1.add(candidate)
        
    # build solution as intersection
    imgclone = img.copy()
    #sol = [[0 for x in range(cols)] for y in range(rows)]
    for i in range(rows):
        for j in range(cols):
            imgclone[i][j] = img[i][j]
            if (i, j) in X0 or A[i][j] == 1:
                imgclone[i][j] = 0
    cv2.imshow("res", imgclone)
    cv2.waitKey(0)

    if X1 == X0:
        break

    X0 = X1 
    X1 = set()

# build solution as intersection
sol = [[0 for x in range(cols)] for y in range(rows)]
for i in range(rows):
    for j in range(cols):
        if (i, j) in X1 or A[i][j] == 1:
            sol[i][j] = 1

# write to file
with open('fill_out.txt', 'w') as outfile:
    for line in sol:
        for item in line:
            outfile.write("%s " % item)
        outfile.write("\n")

for i in range(rows):
    for j in range(cols):
        if sol[i][j] == 1:
            img[i][j] = 0
cv2.imshow("result", img)
cv2.waitKey(0)