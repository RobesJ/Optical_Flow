#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <math.h>
#include <fstream>

int main() {
    // camera's intrinsic parameters
    const double fx = 335.970824 ;
    const double fy = 336.411475 ;
    const double cx = 313.369979 ;
    const double cy = 201.104536 ;         
    const double depth = 5.0;         // const. depth, later laser data

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat prevFrame, currFrame, flow;
    std::vector<cv::Point2f> prevPoints, nextPoints;

    // get video capture
    std::string filename = "/home/rootroot/cpp_test/gray_output_1.mkv";
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video capture.\n";
        return -1;
    }

    // Open an output file for writing the velocity data
    std::ofstream outfile("velocity1.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open the file to write velocity data.\n";
        return -1;
    }

    // get first frame of capture
    cap >> prevFrame;
    if (prevFrame.empty()) {
        std::cerr << "Error: Could not capture first frame.\n";
        return -1;
    }
    
    // Resize and preprocess the rest of frames in loop
    //cv::resize(prevFrame, prevFrame, cv::Size(), 0.2, 0.2, cv::INTER_LINEAR);
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);

    const double fps = cap.get(cv::CAP_PROP_FPS);
    const double delta_t = 1.0 / fps; // Time interval between frames

    cv::namedWindow("Grayscale Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Grayscale Video", 850, 450);

    while (true) {
        cap >> currFrame;
        cv::Mat displayFrame;
        if (currFrame.empty()) break;

        // Resize and preprocess the rest of frames in loop
        // cv::resize(currFrame, currFrame, cv::Size(), 0.2, 0.2, cv::INTER_LINEAR);
        cv::cvtColor(currFrame, currFrame, cv::COLOR_BGR2GRAY);

        currFrame.copyTo(displayFrame); 

        cv::calcOpticalFlowFarneback(prevFrame, currFrame, flow, 0.4, 2, 5, 4, 7, 1.5, 0);

        // Calculate average motion in real-world units
        double sumVx = 0.0, sumVy = 0.0;
        int count = 0;
        int thickness = 1;
        double scaleFactor = 2.0;

        for (int y = 0; y < flow.rows; y +=10) {
            for (int x = 0; x < flow.cols; x +=10) {
                cv::Point2f flowAtXY = flow.at<cv::Point2f>(y, x);
                double dx = flowAtXY.x;
                double dy = flowAtXY.y;
                
                // create motion vectors as arrows for visualization
                cv::Point start(x*2, y*2);
                cv::Point end(x*2 + static_cast<int>(flowAtXY.x * scaleFactor), 
                              y*2 + static_cast<int>(flowAtXY.y * scaleFactor));
                cv::arrowedLine(displayFrame, start, end, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);

                // Convert pixel motion to real-world motion
                double Vx = (dx * depth * fps) / fx;
                double Vy = (dy * depth * fps) / fy;

                //double Vx = (dx * depth) / (fx * delta_t);
                //double Vy = (dy * depth) / (fy * delta_t);
                sumVx += Vx;
                sumVy += Vy;
                count++;
            }
        }

        // Compute average velocity
        double avgVx = sumVx / count;
        double avgVy = sumVy / count;
        double velocity = sqrt(avgVx * avgVx + avgVy * avgVy); // Magnitude of velocity

        outfile << velocity << std::endl;
        
         // Draw the text in the top-left corner
        std::string speedText = "Speed: " + std::to_string(velocity).substr(0, 4) + " m/s";
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.0;
        thickness = 2;
        cv::Point textOrg(10, 30);
        cv::putText(displayFrame, speedText, textOrg, fontFace, fontScale, cv::Scalar(0), thickness);

        // Show the result
        cv::imshow("Grayscale Video", displayFrame);

        // Update the previous frame
        prevFrame = currFrame.clone();

        // Exit on 'q' key press
        if (cv::waitKey(1) == 'q') break;
    }

    outfile.close();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}