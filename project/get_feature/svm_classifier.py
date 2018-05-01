'''
linear classification for image
---created by Z.Zhang 3/21/2018
'''
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import imutils
import cv2
import os
from sklearn.decomposition import RandomizedPCA

from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern

dataPath = "/Users/mouqingjin/Desktop/get_feature/original"

# color histogram
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()

# hog feature
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

bin_n = 16# Number of bins

# LBP
def LBP(img):
    radius = 3
    n_points = 8 * radius
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image, n_points, radius)
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    # return the histogram of Local Binary Patterns
    return hist

#pca
n_components= 100
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)












# initialize the data matrix and labels list
data_train = []
labels_train = []
data_test = []
labels_test = []

# loop over the input images
for stage in ['train', 'test']:
    classList = os.listdir(os.path.join(dataPath, stage))
    # for className in classList:
    for k in range(len(classList)-1):
        className = classList[k+1]
        imagePaths = os.path.join(dataPath, stage, className)
        for imageName in list(paths.list_images(imagePaths)):
            imagePath = os.path.join(dataPath, stage, className, imageName)
            image = cv2.imread(imagePath)
            label = className
            #hist = extract_color_histogram(image)
            hist = LBP(image)
            if stage is 'train':
                data_train.append(hist)
                labels_train.append(label)
            elif stage is 'test':
                data_test.append(hist)
                labels_test.append(label)

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels_train = le.fit_transform(labels_train)
labels_test = le.fit_transform(labels_test)

# train the linear regression clasifier
print("[INFO] training Linear SVM classifier...")
model = LinearSVC()
model.fit(data_train, labels_train)

# evaluate the classifier
print("[INFO] evaluating classifier...")
predictions = model.predict(data_test)
print(classification_report(labels_test, predictions,
                            target_names=le.classes_))