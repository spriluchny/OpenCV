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

vector<double> approximation(vector<Point2f> p) {
  int pts = p.size();
  double sums[pts][pts], b[pts];
  vector<double> a(pts);
  int K = 2;
  double x[pts];
  double y[pts];
  for (int i = 0; i < pts; ++i) {
    x[i] = p[i].y;
    y[i] = p[i].x;
  }

  int i, j, k;
  long double s, t, M;

  for (i = 0; i < pts; i++) {
    for (int j = i; j >= 1; j--)
      if (x[j] < x[j - 1]) {
        t = x[j - 1];
        x[j - 1] = x[j];
        x[j] = t;
        t = y[j - 1];
        y[j - 1] = y[j];
        y[j] = t;
      }
  }

  for (i = 0; i < K + 1; i++) {
    for (j = 0; j < K + 1; j++) {
      sums[i][j] = 0;
      for (k = 0; k < pts; k++) sums[i][j] += pow(x[k], i + j);
    }
  }

  for (i = 0; i < K + 1; i++) {
    b[i] = 0;
    for (k = 0; k < pts; k++) b[i] += pow(x[k], i) * y[k];
  }

  for (k = 0; k < K + 1; k++) {
    for (i = k + 1; i < K + 1; i++) {
      M = sums[i][k] / sums[k][k];
      for (j = k; j < K + 1; j++) sums[i][j] -= M * sums[k][j];
      b[i] -= M * b[k];
    }
  }

  for (i = K; i >= 0; i--) {
    s = 0;
    for (j = i; j < K + 1; j++) s += sums[i][j] * a[j];
    a[i] = (b[i] - s) / sums[i][i];
  }

  return a;
}

void detectLanes(Mat &frame, vector<vector<Point2f>> &centers) {
    Rect rect(0, 0, 10, 10);
    vector<Point2f> nonZeroPoints, regionOfInterest;

    for (rect.y = 0; rect.y + 10 < frame.rows; rect.y += rect.height) {
        for (rect.x = 0; rect.x + 10 < frame.cols; rect.x += rect.width) {
            findNonZero(frame(rect), regionOfInterest);
            if (!regionOfInterest.empty()) {
                for (int i = 0; i < regionOfInterest.size(); i++) {
                    regionOfInterest[i].x += rect.x;
                }
                nonZeroPoints.insert(nonZeroPoints.end(), regionOfInterest.begin(), regionOfInterest.end());

            } else if (!nonZeroPoints.empty()) {
                if (rect.x < frame.cols / 2)
                    centers[0].push_back(Point((nonZeroPoints.front().x + nonZeroPoints.back().x) / 2,
                        rect.y + (nonZeroPoints.front().y + nonZeroPoints.back().y) / 2));
                else
                    centers[1].push_back(Point((nonZeroPoints.front().x + nonZeroPoints.back().x) / 2,
                        rect.y + (nonZeroPoints.front().y + nonZeroPoints.back().y) / 2));
                nonZeroPoints.clear();
            }
        }
        if (!nonZeroPoints.empty()) {
              centers[1].push_back(
                  Point((nonZeroPoints.front().x + nonZeroPoints.back().x) / 2,
                        rect.y + (nonZeroPoints.front().y + nonZeroPoints.back().y) / 2));
              nonZeroPoints.clear();
        }
    }
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

        Mat polygonFrame;
        frame.copyTo(polygonFrame);

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

        vector<vector<Point2f> > centers(2);
        cvtColor(dst, binaryFrame, COLOR_BGR2GRAY);
        blur(binaryFrame, binaryFrame, Size(9, 9));
        threshold(binaryFrame, binaryFrame, 180, 255, THRESH_BINARY);
        detectLanes(binaryFrame, centers);

        Mat invM = getPerspectiveTransform(dst_vertices, src_vertices);
        perspectiveTransform(centers[0], centers[0], invM);
        perspectiveTransform(centers[1], centers[1], invM);

        vector<Point> leftPoints;
        vector<Point> rightPoints;
        vector<double> leftPolynom = approximation(centers[0]);
        vector<double> rightPolynom = approximation(centers[1]);

        for (auto y = centers[0][0].y; y < centers[0].back().y; y += 0.1) {
            double x = 0;
            for (unsigned i = 0; i < leftPolynom.size(); ++i) {
                x += leftPolynom[i] * pow(y, i);
          }
          leftPoints.push_back(Point(x, y));

          x = 0;
          for (unsigned i = 0; i < rightPolynom.size(); ++i) {
              x += rightPolynom[i] * pow(y, i);
          }
          rightPoints.push_back(Point(x, y));
        }

        vector<vector<Point> > points;
        points.push_back(leftPoints);
        points[0].insert(points[0].end(), rightPoints.rbegin(), rightPoints.rend());
        fillPoly(polygonFrame, points, Scalar(60, 20, 220));

        for (int i = 0; i < centers[0].size(); i++) {
            circle(frameWithDetection, centers[0][i], 4, Scalar(200, 70, 20), FILLED);
        }

        for (int i = 0; i < centers[1].size(); i++) {
            circle(frameWithDetection, centers[1][i], 4, Scalar(200, 70, 20), FILLED);
        }

        showInfo(frame, cap);
        showInfo(frameWithDetection, cap);
        showInfo(polygonFrame, cap);

        imshow("Original Video", frame);
        imshow("IPM", dst);
        imshow("Binary", binaryFrame);
        imshow("Lane Detecton", frameWithDetection);
        imshow("Polygon", polygonFrame);

        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
            break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
