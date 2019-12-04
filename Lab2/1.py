import cv2
logo = cv2.imread('./logo_opencv.png')
scene = cv2.imread('./lena.jpg')
logo = cv2.resize(logo, (0, 0), fx=1, fy=1)
logogray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
retval, logogray = cv2.threshold(logogray, 200, 255, cv2.THRESH_BINARY_INV)

roi = scene[0:logogray.shape[0], scene.shape[1]-logogray.shape[1]:scene.shape[1]]
cv2.add(logo, roi, roi, logogray)
cv2.imshow("scene", scene)

cv2.waitKey(0)