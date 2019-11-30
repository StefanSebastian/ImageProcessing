import glob
import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np 

images = [cv2.imread(file) for file in glob.glob("./dataset/*.jpg")]

test_im = cv2.imread('./test_im/4.jpg')
cv2.imshow("test_image", test_im)

color_space_size = 16

# reduce color space of given image
def reduce_color_space(image, color_space_size):
    (h, w) = image.shape[:2]
    '''
    Use L*a*b* color space, because K-Means uses Euclidean distance;
    CIELAB was designed so that the same amount of numerical change in these values
    corresponds to roughly the same amount of visually perceived change.
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # reshape into feature vector
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = MiniBatchKMeans(n_clusters=color_space_size)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape feature vectors into images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    return quant

reduce_color_space(test_im, color_space_size)
cv2.waitKey(0)
