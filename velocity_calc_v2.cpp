#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>
#include <fstream>

int main() {
    // parametre kamery, premenne
    const double fx = 335.970824 ;
    const double fy = 336.411475 ;
    const double cx = 313.369979 ;
    const double cy = 201.104536 ;         
    const double depth = 5.0;         // konstantna vyska 

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat K_inv = cameraMatrix.inv();
    cv::Mat prevFrame, currFrame, flow;

    // spracovanie videa
    std::string filename = "/home/rootroot/cpp_test/gray_output_1.mkv";
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video capture.\n";
        return -1;
    }

     // Open an output file for writing the velocity data
    std::ofstream outfile("velocity.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open the file to write velocity data.\n";
        return -1;
    }

    // prvy obraz videa
    cap >> prevFrame;
    if (prevFrame.empty()) {
        std::cerr << "Error: Could not capture first frame.\n";
        return -1;
    }

    // znizenie rozlisenia kamery a graysacaling pre optimalizaciu
    //cv::resize(prevFrame, prevFrame, cv::Size(), 0.2, 0.2, cv::INTER_LINEAR);
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const double delta_t = 1.0 / fps; // Time interval between frames
    cv::namedWindow("Grayscale Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Grayscale Video", 850, 450);

    while (true) {
        // ziskavanie a optimalizacia dalsich obrazov videa
        cap >> currFrame;
        cv::Mat displayFrame;
        if (currFrame.empty()) break;
        cv::Mat og_frame = currFrame;

        // Resize and preprocess the rest of frames in loop
        // cv::resize(currFrame, currFrame, cv::Size(), 0.2, 0.2, cv::INTER_LINEAR);
        cv::cvtColor(currFrame, currFrame, cv::COLOR_BGR2GRAY);
        currFrame.copyTo(displayFrame); 

        // vypocet optickeho toku
        cv::calcOpticalFlowFarneback(prevFrame, currFrame, flow, 0.4, 2, 5, 4, 7, 1.5, 0);

        cv::Scalar avgFlow = cv::mean(flow);
        double du = avgFlow[0];
        double dv = avgFlow[1];
        // Create a 3x1 vector for the pixel flow (third component is 0)
        cv::Mat pixelFlow = (cv::Mat_<double>(3, 1) << du, dv, 0);

        // Back-project pixel flow to camera coordinates
        cv::Mat flowCam = K_inv * pixelFlow;

        // Scale by depth Z and FPS to get velocity in m/s
        double vx = depth * flowCam.at<double>(0, 0) * fps;
        double vy = depth * flowCam.at<double>(1, 0) * fps;
        double velocity = sqrt(vx * vx + vy * vy); // Magnitude of velocity

        // show motion vectors as arrows
        int stepSize = 20;
        int thickness = 1;
        double scaleFactor = 2.0;
        for (int y = 0; y < flow.rows; y += stepSize) {
            for (int x = 0; x < flow.cols; x += stepSize) {
                cv::Point2f flow_at_point = flow.at<cv::Point2f>(y, x);
                cv::Point start(x, y);
                cv::Point end(x + static_cast<int>(flow_at_point.x * scaleFactor), 
                              y + static_cast<int>(flow_at_point.y * scaleFactor));
                cv::arrowedLine(displayFrame, start, end, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
            }
        }

        // Write the velocity to the file
        outfile << velocity << std::endl;

        std::string speedText = "Speed: " + std::to_string(velocity).substr(0, 4) + " m/s";

        // Draw the text in the top-left corner
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        thickness = 2;
        cv::Point textOrg(10, 30);
        cv::putText(displayFrame, speedText, textOrg, fontFace, fontScale, cv::Scalar(0), thickness);
        cv::imshow("Grayscale Video", displayFrame);

        // Update the previous frame
        prevFrame = currFrame.clone();

        // Exit on 'q' key press
        if (cv::waitKey(1) == 'q') break;
    }

    // Release video capture and close windows
    outfile.close();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}