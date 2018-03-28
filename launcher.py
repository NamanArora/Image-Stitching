# add the images in left to right order
# make sure no image is greater than 100kb

import sys
from matplotlib import pyplot as plt
import cv2

# initialize empty lists here

imageFilesPath = []
images = []
left = []
right = []
center = None
BF = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
count = 0
centerIdx = 0
surf = cv2.xfeatures2d.SURF_create()    

#Get keypoints and features
def getSURFFeatures(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = surf.detectAndCompute(gray, None)
    x1 = [p.pt[0] for p in kp]
    y1 = [p.pt[1] for p in kp]
    plt.gray()
    plt.figure(1)
    plt.imshow(image)
    plt.plot(x1, y1, 'r.')
    plt.title('KeyPoints')
    plt.show()
    ret = []
    ret.append(des)
    ret.append(kp)
    return ret




def stitchLeft():
    img1 = left[0]
    for i in range(1, len(left)):
        # Feature extraction
        featureSet1 = getSURFFeatures(img1)
        featureSet2 = getSURFFeatures(left[i])
        desc1 = featureSet1[0]
        desc2 = featureSet2[0]
        kp1 = featureSet1[1]
        kp2 = featureSet2[1]

        # Image Matching
        matches = BF.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        img3 = cv2.drawMatches(img1, kp1, left[i], kp2, matches[:10], None, flags=2)
        plt.imshow(img3), plt.show()

        # Finding homography


def populate_data():
    centerIdx = count / 2
    center_im = images[centerIdx]
    for i in range(count):
        if i <= centerIdx:
            left.append(images[i])
        else:
            right.append(images[i])
    print "Image lists prepared"


if __name__ == '__main__':
    args = sys.argv[1]
    print "File loaded : " + args
    pathToImagesFile = open(args, 'r')

    for path in pathToImagesFile.readlines():
        imageFilesPath.append(path.rstrip('\r\n'))

    pathToImagesFile.close()

    print imageFilesPath

    for imageFilePath in imageFilesPath:
        images.append(cv2.resize(cv2.imread(imageFilePath), (480, 320)))

    print images

    count = len(images)
    populate_data()
    stitchLeft()
