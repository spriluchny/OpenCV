#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

uchar convolution(const Mat& imageOrig, const vector<vector<double> >& kernel, int i, int j) {
    int newPixel = 0;
    int div = static_cast<int>(kernel.size() * kernel.size());
    int border = static_cast<int>(kernel.size() / 2);

    for (unsigned k = 0; k < kernel.size(); ++k) {
        for (unsigned l = 0; l < kernel.size(); ++l) {
            newPixel += imageOrig.at<uchar>(i + static_cast<int>(k) - border, j + static_cast<int>(l) - border) * kernel[k][l];
        }
    }

    return saturate_cast<uchar>(newPixel);
}

uchar convolution(const Mat& imageOrig, const vector<vector<double> >& kernel,int i, int j, int p) {
    int newPixel = 0;
    int div = static_cast<int>(kernel.size() * kernel.size());
    int border = static_cast<int>(kernel.size() / 2);

    for (unsigned k = 0; k < kernel.size(); ++k) {
        for (unsigned l = 0; l < kernel.size(); ++l) {
            newPixel += imageOrig.at<Vec3b>(i + static_cast<int>(k) - border, j + static_cast<int>(l) - border)[p] * kernel[k][l];
        }
    }

    newPixel /= div;
    return saturate_cast<uchar>(newPixel);
}

void filter(const Mat& imageOrig, Mat& imageResult, const vector<vector<double> >& kernel) {
    int rows = imageOrig.rows;
    int cols = imageOrig.cols;
    int border = static_cast<int>(kernel.size() / 2);
    if (imageOrig.type() == 0) {
        for (int i = border; i < rows - border + 1; ++i) {
            for (int j = border; j < cols - border + 1; ++j) {
                imageResult.at<uchar>(i, j) = convolution(imageOrig, kernel, i, j);
            }
        }
    } else {
        for (int i = border; i < rows - border + 1; ++i) {
            for (int j = border; j < cols - border + 1; ++j) {
                for (int p = 0; p < 3; ++p) {
                    imageResult.at<Vec3b>(i, j)[p] = convolution(imageOrig, kernel, i, j, p);
                }
            }
        }
    }
}

void customBlur(const Mat& imageOrig, Mat& imageResult, unsigned size) {
    imageOrig.copyTo(imageResult);
    vector<vector<double> > kernel(size, vector<double>(size, 1.));
    filter(imageOrig, imageResult, kernel);
}

void customGradient(const Mat& imageOrig, Mat& imageResult) {
    imageOrig.copyTo(imageResult);
    vector<vector<double> > kernel = {{3, 0, -3}, {10, 0, -10}, {3, 0, -3}};
    filter(imageOrig, imageResult, kernel);
}

int main() {

    Mat image = imread("cat.jpg", IMREAD_COLOR);
    imshow("Original image", image);

    Mat imageGaussBlur;
    GaussianBlur(image, imageGaussBlur, Size(15, 15), 0, 0);
    imshow("Gaussian Blur", imageGaussBlur);

    Mat customBlurImage;
    customBlur(image, customBlurImage, 5);
    imshow("Custom Blur", customBlurImage);

    Mat customGradientImage;
    cvtColor(image, image, COLOR_BGR2GRAY);
    customGradient(image, customGradientImage);
    imshow("Custom Gradient", customGradientImage);

    waitKey(0);
    return 0;
}