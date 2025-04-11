#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>

int main() {
    // parametre kamery, premenne
    const double fx = 335.970824 *0.2;
    const double fy = 336.411475 *0.2;
    const double cx = 313.369979 *0.2;
    const double cy = 201.104536 *0.2;         
    const double depth = 1.0;         // konstantna vyska  (Z = 1 meter), dáta z lasera, lidaru?

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat prevFrame, currFrame, flow;
    std::vector<cv::Point2f> prevPoints, nextPoints;
    
    // spracovanie videa
    std::string filename = "/home/rootroot/cpp_test/test_video_calc_velocity.mp4";
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video capture.\n";
        return -1;
    }

    // prvy obraz videa
    cap >> prevFrame;
    if (prevFrame.empty()) {
        std::cerr << "Error: Could not capture first frame.\n";
        return -1;
    }

    // znizenie rozlisenia kamery a graysacaling pre optimalizaciu
    cv::resize(prevFrame, prevFrame, cv::Size(), 0.2, 0.2, cv::INTER_LINEAR);
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const double delta_t = 1.0 / fps; // Time interval between frames
    double scaleFactor = depth / delta_t; 

while (true) {
    // ziskavanie a optimalizacia dalsich obrazov videa
    cap >> currFrame;
    if (currFrame.empty()) break;
    cv::Mat og_frame = currFrame;

    // Resize and preprocess the rest of frames in loop
    cv::resize(currFrame, currFrame, cv::Size(), 0.2, 0.2, cv::INTER_LINEAR);
    cv::cvtColor(currFrame, currFrame, cv::COLOR_BGR2GRAY);

    // vypocet optickeho toku
    cv::calcOpticalFlowFarneback(prevFrame, currFrame, flow, 0.4, 2, 5, 4, 7, 1.5, 0);

    // Prepare points for essential matrix calculation
        prevPoints.clear();
        nextPoints.clear();
        for (int y = 0; y < flow.rows; y += 10) {
            for (int x = 0; x < flow.cols; x += 10) {
                cv::Point2f flowAtXY = flow.at<cv::Point2f>(y, x);
                prevPoints.emplace_back(x, y);
                nextPoints.emplace_back(x + flowAtXY.x, y + flowAtXY.y);
            }
        }

    // Compute essential matrix
    cv::Mat essentialMatrix = cv::findEssentialMat(prevPoints, nextPoints, cameraMatrix, cv::RANSAC, 0.999, 1.0);

    // Recover pose (rotation and translation)
    cv::Mat R, T;
    cv::recoverPose(essentialMatrix, prevPoints, nextPoints, cameraMatrix, R, T);

    double ds_speed = cv::norm(T) * scaleFactor;
    std::cout << "DS velocity: " + std::to_string(ds_speed) + '\n';

    // Calculate translational velocity
    cv::Mat translationalVelocity = T * scaleFactor;
    //cv::Mat translationalVelocity = T * fps;

    // Display velocities on the frame
    //std::string transVelText = "Translational Velocity: " + std::to_string(cv::norm(translationalVelocity));
    //cv::putText(currFrame, transVelText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    std::cout << "Translational Velocity: " + std::to_string(cv::norm(translationalVelocity)) + '\n';


    // Update the previous frame
    prevFrame = currFrame.clone();

    // Exit on 'q' key press
    if (cv::waitKey(1) == 'q') break;
}
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
