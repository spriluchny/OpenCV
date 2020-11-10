#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void showInfo(Mat frame, VideoCapture cap) {
	char fps[20];
    char time[20];

    sprintf(fps, "FPS: %.0f", cap.get(CAP_PROP_FPS));
    putText(frame, fps, Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(200, 70, 20), 2);

    sprintf(time, "TIME: %.0f ms", cap.get(CAP_PROP_POS_MSEC));
    putText(frame, time, Point(20, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(200, 70, 20), 2);
}

vector<Point2f> detectLanes(Mat frame) {
    Rect rect(0, 0, 10, 10);
    vector<Point2f> nonZeroPoints, points, regionOfInterest;

    for (rect.y = 0; rect.y < frame.rows; rect.y += rect.height) {
        for (rect.x = 0; rect.x < frame.cols; rect.x += rect.width) {
            findNonZero(frame(rect), regionOfInterest);
            if (!regionOfInterest.empty()) {
                for (int i = 0; i < regionOfInterest.size(); i++) {
                    regionOfInterest[i].x += rect.x;
                }
                nonZeroPoints.insert(nonZeroPoints.end(), regionOfInterest.begin(), regionOfInterest.end());

            } else if (!nonZeroPoints.empty()) {
                points.push_back(Point((nonZeroPoints.front().x + nonZeroPoints.back().x) / 2,
                                    rect.y + (nonZeroPoints.front().y + nonZeroPoints.back().y) / 2));
                nonZeroPoints.clear();
            }
        }
    }
    return points;
}

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

    namedWindow("Original Video");
    createTrackbar("Topline", "Original Video", &topline, toplineMaxValue);
    createTrackbar("Baseline", "Original Video", &baseline, baselineMaxValue);
    createTrackbar("Height", "Original Video", &height, heightMaxValue);

    while(1) {
        Mat frame;
        cap >> frame;

        if (frame.empty())
            break;

        Mat frameWithDetection;
		frame.copyTo(frameWithDetection);

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

        Mat binaryFrame;
		vector<Point2f> points;
		cvtColor(dst, binaryFrame, COLOR_BGR2GRAY);
        blur(binaryFrame, binaryFrame, Size(9, 9));
		threshold(binaryFrame, binaryFrame, 180, 255, THRESH_BINARY);
        points = detectLanes(binaryFrame);

		Mat invM = getPerspectiveTransform(dst_vertices, src_vertices);
		perspectiveTransform(points, points, invM);

        for(int i = 0; i < points.size(); i++) {
            circle(frameWithDetection, points[i], 4, Scalar(200, 70, 20), FILLED);
        }

        showInfo(frame, cap);
        showInfo(frameWithDetection, cap);

        imshow("Original Video", frame);
        imshow("IPM", dst);
        imshow("Binary", binaryFrame);
        imshow("Lane Detecton", frameWithDetection);

        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
            break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
