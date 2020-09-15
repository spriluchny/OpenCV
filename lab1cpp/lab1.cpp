#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

int main() {
    Mat image = imread("cat.jpg", IMREAD_COLOR);
    imshow("Original image", image);

    Mat imageHSV;
    cvtColor(image, imageHSV, COLOR_BGR2HSV);
    imshow("HSV image", imageHSV);

    Mat imageGray;
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    imshow("Grayscale image", imageGray);

    Mat imageBinary;
    threshold(imageGray, imageBinary, 127, 255, THRESH_BINARY);
    imshow("Binary image", imageBinary);

    circle(image, Point(245, 210), 30, Scalar(255, 0, 0), 5);
    imshow("Image with circle", image);

    waitKey(0);
    return 0;
}
