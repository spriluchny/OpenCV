#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap("solidYellowLeft.mp4");

    if(!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    int topline = 120;
    int baseline = 410;
    int height = 170;

    int toplineMaxValue = 470;
    int baselineMaxValue = 465;
    int heightMaxValue = 540;

    namedWindow("Original video");
    createTrackbar("Topline", "Original video", &topline, toplineMaxValue);
    createTrackbar("Baseline", "Original video", &baseline, baselineMaxValue);
    createTrackbar("Height", "Original video", &height, heightMaxValue);

    while(1) {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        Point2f src_vertices[4];
        src_vertices[0] = Point(490 - topline, heightMaxValue - height);
        src_vertices[1] = Point(490 + topline, heightMaxValue - height);
        src_vertices[2] = Point(500 + baseline, heightMaxValue);
        src_vertices[3] = Point(500 - baseline, heightMaxValue);

        Point2f dst_vertices[4];
        dst_vertices[0] = Point(0, 0);
        dst_vertices[1] = Point(640, 0);
        dst_vertices[2] = Point(640, 480);
        dst_vertices[3] = Point(0, 480);

        Mat M = getPerspectiveTransform(src_vertices, dst_vertices);
        Mat dst(480, 640, CV_8UC3);
        warpPerspective(frame, dst, M, dst.size());

        circle(frame, src_vertices[0], 4, Scalar(0, 255, 0), FILLED);
        circle(frame, src_vertices[1], 4, Scalar(0, 255, 0), FILLED);
        circle(frame, src_vertices[2], 4, Scalar(0, 255, 0), FILLED);
        circle(frame, src_vertices[3], 4, Scalar(0, 255, 0), FILLED);

        line(frame, src_vertices[0], src_vertices[1], Scalar(0, 255, 0), 2);
        line(frame, src_vertices[1], src_vertices[2], Scalar(0, 255, 0), 2);
        line(frame, src_vertices[2], src_vertices[3], Scalar(0, 255, 0), 2);
        line(frame, src_vertices[3], src_vertices[0], Scalar(0, 255, 0), 2);

        char fps[20];
        char time[20];

        sprintf(fps, "FPS: %.0f", cap.get(CAP_PROP_FPS));
        putText(frame, fps, Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(200, 70, 20), 2);

        sprintf(time, "TIME: %.0f ms", cap.get(CAP_PROP_POS_MSEC));
        putText(frame, time, Point(20, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(200, 70, 20), 2);

        imshow("Original video", frame);
        imshow("IPM", dst);

        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
            break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
