# add the images in left to right order
# make sure no image is greater than 100kb

import sys
import cv2

# initialize empty lists here

imageFilesPath = []
images = []
left = []
right = []
flann = cv2.FlannBasedMatcher()


def stitchLeft():
    img1 = left[0]
    for i in range(1, len(left)):
        # Feature extraction
        featureSet1 = getFeatures(img1)
        featureSet2 = getFeatures(left[i])
        desc1 = featureSet1[0]
        desc2 = featureSet2[0]
        kp1 = featureSet1[1]
        kp2 = featureSet2[1]

        # Image Matching
        matcher = flann


        # Finding homography


if __name__ == '__main__':
    args = sys.argv[1]
    print "File loaded : " + args
    pathToImagesFile = open(args, 'r')

    for path in pathToImagesFile.readlines():
        imageFilesPath.append(path.rstrip('\r\n'))

    print imageFilesPath

    for imageFilePath in imageFilesPath:
        images.append(cv2.resize(cv2.imread(imageFilePath), (480, 320)))

    print images
