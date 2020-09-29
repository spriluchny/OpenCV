#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0);

    if(!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    while(1) {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        imshow( "Original Stream", frame);

        Mat gaussBlur;
        GaussianBlur(frame, gaussBlur, Size(21, 21), 0, 0);
        imshow("Gaussian Blur Stream", gaussBlur);

        Mat frameHSV, redColor;
        cvtColor(frame, frameHSV, COLOR_BGR2HSV);
        inRange(frameHSV, Scalar(0, 50, 180), Scalar(5, 255, 255), redColor);
        imshow("Red Color Stream", redColor);

        Mat edges;
        cvtColor(frame, edges, COLOR_BGR2GRAY);
        GaussianBlur(edges, edges, Size(7, 7), 0, 0);
        Canny(edges, edges, 30, 100);
        imshow("Canny Edge Stream", edges);

        char c=(char)waitKey(25);
        if(c==27)
            break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
