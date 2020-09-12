import cv2


image = cv2.imread('cat.jpg')

cv2.imshow("Original image", image)

imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV image", imageHSV)

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale image", imageGray)

ret, imageBinary = cv2.threshold(imageGray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary image", imageBinary)

cv2.circle(image, (245, 210), 30, (255, 0, 0), thickness=5, lineType=8, shift=0)
cv2.imshow("Image with circle", image)

cv2.waitKey(0)