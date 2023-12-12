import numpy as np
import cv2
from grade_paper import ProcessPage
from scipy.spatial import distance as dist

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-coordinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left points
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

cv2.namedWindow('Original Image')
cv2.namedWindow('Scanned Paper')

# Load the image
image = cv2.imread("img_12.png")
ratio = image.shape[1] / 500.0 # used for resizing the image
original_image = image.copy() # make a copy of the original image

# Resize the image for faster processing
image = cv2.resize(image, (500, int(image.shape[0] / ratio)))
cv2.imshow("image", image)
cv2.waitKey(0)
# Gray and filter the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 250, 300)

# Find the contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Find the biggest contour
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        biggestContour = approx
        break
else:
    biggestContour = None
# Proceed if a contour was found
if biggestContour is not None:
    # Sort the contour points
    points = order_points(biggestContour.reshape(4, 2) * ratio)

    # Desired points for the perspective transform
    desired_points = np.float32([[0, 0], [425, 0], [425, 550], [0, 550]])

    # Perspective transform
    M = cv2.getPerspectiveTransform(points, desired_points)
    paper = cv2.warpPerspective(original_image, M, (425, 550))


    answers, paper = ProcessPage(paper)

    # Display the processed paper
    cv2.imshow("Scanned Paper", paper)

    # Draw the contour on the original image
    cv2.drawContours(image, [biggestContour], -1, (0, 255, 0), 3)
    print(answers)
    # if codes is not None:
    #     print(codes)

# Show the original image
cv2.imshow("Original Image", cv2.resize(image, (500, int(image.shape[0] / ratio))))

cv2.waitKey(0)
cv2.destroyAllWindows()
