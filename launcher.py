# add the images in left to right order
# make sure no image is greater than 100kb

import sys
from matplotlib import pyplot as plt
import cv2
import numpy as np

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


# Get keypoints and features
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


def match(i1, i2, direction=None):
    imageSet1 = getSURFFeatures(i1)
    imageSet2 = getSURFFeatures(i2)
    print "Direction : ", direction
    matches = BF.knnMatch(
        imageSet2[0],
        imageSet1[0],
        k=2
    )
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append((m.trainIdx, m.queryIdx))

    if len(good) > 4:
        pointsCurrent = imageSet2[1]
        pointsPrevious = imageSet1[1]

        matchedPointsCurrent = np.float32(
            [pointsCurrent[i].pt for (__, i) in good]
        )
        matchedPointsPrev = np.float32(
            [pointsPrevious[i].pt for (i, __) in good]
        )

        H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
        return H
    return None


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

        # Finding homography Index
        HI = match(img1, left[i], 'left')
        print "homography is \n"
        print HI
        iHI = np.linalg.inv(HI)
        print "Inverse Homography :", iHI

        # Find final points
        final = np.dot(iHI , np.array([img1.shape[1], img1.shape[0], 1]))
        # Normalize the last index to 1
        final = final/final[-1]

        # Find coordinate shift
        shift = np.dot(iHI,np.array([0,0,1]))
        # Normalize
        shift = shift/shift[-1]

        # Get the offsets 
        offsetX = abs(int(shift[0]))
        offsetY = abs(int(shift[1]))

        # Modfy the H to incorporate origin shift
        iHI[0][-1] += abs(shift[0])
        iHI[1][-1] += abs(shift[1])

        #Find final points after H shift
        final = np.dot(iHI , np.array([img1.shape[1], img1.shape[0], 1]))

        # Final image size
        imgSize = (int(final[0]) + offsetX, int(final[1]) + offsetY)

        # Warp the first img wrt to second
        tmp = cv2.warpPerspective(img1, iHI, imgSize)
        cv2.imshow("warped",tmp)
        cv2.waitKey()
        tmp[offsetY:left[i].shape[0]+offsetY, offsetX:left[i].shape[1]+offsetX] = left[i]
        img1 = tmp;

    return tmp


def stitchRight(leftImage):
        for rImage in right:
            H = match(leftImage, rImage)
            #cv2.imshow("right Image", rImage)
            #print "Homography :", H
            newCord = np.dot(H, np.array([rImage.shape[1], rImage.shape[0], 1]))
            #print " shapes"
            #print H.shape, newCord.shape
            #print newCord
            newCord = newCord / newCord[-1]
            newSize = (int(newCord[0]) + leftImage.shape[1], int(newCord[1]) + leftImage.shape[0])
            tmp = cv2.warpPerspective(rImage, H, newSize)
            cv2.imshow("tpright", tmp)
            cv2.waitKey()
            tmp = mix_and_match(leftImage, tmp)
            #print "tmp shape", tmp.shape
            #print "self.leftimage shape=", self.leftImage.shape
            leftImage = tmp

        return leftImage

def populate_data():
    centerIdx = count / 2
    center_im = images[centerIdx]
    for i in range(count):
        if i <= centerIdx:
            left.append(images[i])
        else:
            right.append(images[i])
    print "Image lists prepared"
    # print left
    # print right


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

    # print images

    count = len(images)
    populate_data()
    leftImage = stitchLeft()
    final = stitchRight(leftImage)
    cv2.imshow("final",final)
    cv2.waitKey()
    cv2.destroyAllWindows()