//to compile use: g++ main2.cpp -o [OUTPUT_FILE_NAME] `pkg-config --cflags --libs opencv4`

#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <string>

int main(int argc, char** argv) {
    std::string filename = "/home/rootroot/cpp_test/gray_output_1.mkv";
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video capture.\n";
        return -1;
    }

    // variables
    //Mat is matrix variable return type 
    cv::Mat prevFrame, currFrame, flow;

    // flowparts - matrix of 2 matrices, one for vertical flow (v) and one for horizontal flow (u)
    // magnitude - stores magnitude of vector of each pixel sqrt(v^2 + u^2)
    // angle - stores directions/angles of each flow vector [rad or °] arctan(u/v) 
    cv::Mat flowParts[2], magnitude, angle, hsv, hsv8, bgr, laplacianFrame;

    // capture the first frame and convert it to grayscale
    cap >> prevFrame;
    if (prevFrame.empty()) {
        std::cerr << "Error: Could not capture first frame.\n";
        return -1;
    }
    //cv::resize(prevFrame, prevFrame, cv::Size(), 0.2, 0.2, cv::INTER_LINEAR);
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);

    while (true) {
        // capture the current frame and convert it to grayscale
        cap >> currFrame;

        if (currFrame.empty()) break;
        //cv::resize(currFrame, currFrame, cv::Size(), 0.2, 0.2, cv::INTER_LINEAR);
        cv::cvtColor(currFrame, currFrame, cv::COLOR_BGR2GRAY);

        // thresholding filter
        //double thresholdValue = 15;  // threshold value
        //double maxValue = 250;        // Maximum value to assign to pixels above the threshold
        //int thresholdType = cv::THRESH_BINARY; // Binary thresholding type
        //cv::threshold(currFrame, currFrame, thresholdValue, maxValue, thresholdType);

        // laplacian filter
        //int kernelSize = 3; //must be odd number in range 1-31
        //cv::Laplacian(currFrame, laplacianFrame, CV_16S, kernelSize);
        //cv::Mat laplacianAbs;
        //cv::convertScaleAbs(laplacianFrame, laplacianAbs);


        // calculate optical flow
        //cv2.calcOpticalFlowFarneback(prev, next,              pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow])
        cv::calcOpticalFlowFarneback(prevFrame, currFrame, flow,      0.4,      2,       5,          4,      7,         1.5,            0);        

        // split the flow into horizontal and vertical components
        cv::split(flow, flowParts);

        // calculate the magnitude and angle of the flow
        cv::cartToPolar(flowParts[0], flowParts[1], magnitude, angle, true);
    
        // normalize magnitude to [0,1] for visualization
        cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);

        // convert the angle and magnitude to an HSV image for visualization
        cv::Mat hsvChannels[3];
        hsvChannels[0] = angle;                // Hue represents the flow direction
        hsvChannels[1] = cv::Mat::ones(angle.size(), CV_32F); // Saturation set to maximum
        hsvChannels[2] = magnitude;            // Value represents the flow magnitude

        // merge HSV channels and convert to 8-bit
        cv::merge(hsvChannels, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);

        // convert HSV to BGR for display
        cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

        // display outpout
        cv::namedWindow("Optical Flow - Farneback",cv::WINDOW_NORMAL);
        cv::resizeWindow("Optical Flow - Farneback", 850, 450); 
        cv::moveWindow("Optical Flow - Farneback", 800, 100);
        cv::imshow("Optical Flow - Farneback", bgr);

        /*cv::namedWindow("Original Video",cv::WINDOW_NORMAL);
        cv::resizeWindow("Original Video", 850, 450); 
        cv::moveWindow("Original Video", 100, 100);
        cv::imshow("Original Video", currFrame);
       ;*/
        // update frame
        prevFrame = currFrame.clone();

        // break on 'ESC' key
        if (cv::waitKey(5) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}