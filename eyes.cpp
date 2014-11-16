#include <cstdio>
#include <algorithm>

#include <armadillo>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "findEyeCenter.h"
#include "cvplot.h"

#define PATH "lib/opencv/share/OpenCV/haarcascades"

int main() {

    // Load object descriptions
    cv::CascadeClassifier face_cascade(PATH "/haarcascade_frontalface_alt2.xml");
    cv::CascadeClassifier eye_cascade(PATH "/haarcascade_eye.xml");

    // Open webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }

    // Set video to 1080x720
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1080);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

    while (true) {

        // Load frame
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Invert the source image and convert to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, CV_BGR2GRAY);

        // Detect faces
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(120 ,120));

        printf("-----\n");
        printf("Detected %ld faces\n", faces.size());

        for (auto face: faces) {

            printf("\tFace size %d x %d\n", face.width, face.height);

            cv::Point face_corner(face.x, face.y);

            std::vector<cv::Point> centers;

            // Draw a green rectangle around face
            // cv::rectangle(frame, face, CV_RGB(0, 255, 0));

            // Get just the face
            cv::Mat gray_face = gray(face);

            // Detect eyes
            std::vector<cv::Rect> eyes;
            eye_cascade.detectMultiScale(gray_face, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

            printf("\tDetected %ld eyes\n", eyes.size());

            for (auto eye: eyes) {

                // Discard eyes in lower half of the face
                if (2 * eye.y + eye.height > face.height) {
                    continue;
                }

                printf("\t\tEye size %d x %d\n", eye.width, eye.height);

                // Draw a red rectangle around eye
                // cv::rectangle(frame, eye + face_corner, CV_RGB(255, 0, 0));

                // Find eye center
                cv::Point center = findEyeCenter(gray_face, eye, "eye") + cv::Point(eye.x, eye.y);
                centers.push_back(center);
            }

            if (centers.size() == 2) {

                // Sort centers & eyes
                if (centers[0].x > centers[1].x) {
                    std::swap(centers[0], centers[1]);
                    std::swap(eyes[0], eyes[1]);
                }

                // Trimm line
                float delta = 1.0 / 12;
                cv::Point left  = centers[0] * (1 - delta) + centers[1] * delta;
                cv::Point right = centers[0] * delta + centers[1] * (1 - delta);

                // Draw green line between eye centers
                // cv::line(frame, centers[0] + face_corner, centers[1] + face_corner, CV_RGB(0, 255, 0), 2);

                // Draw red line after trimming
                cv::line(frame, left + face_corner, right + face_corner, CV_RGB(255, 0, 0), 2);

                // Reduce the noise so we avoid false circle detection
                cv::GaussianBlur(gray_face, gray_face, cv::Size(9, 9), 2, 2);

                // Sample image along line
                cv::LineIterator iter(gray_face, left, right);
                std::vector<double> line(iter.count);
                for (int i = 0; i < iter.count; ++i, ++iter) {
                    line[i] = **iter;
                }

                printf("\tLine length %ld\n", line.size());

                // Discard eyes if too far away
                if (line.size() < 50) {
                    continue;
                }

                printf("\tCorrectly detected one pair of eyes\n");

                arma::rowvec vec(line);

                // TODO:
                // * Normalize values using mean+-2*std
                // * Downsample to 50 elements
                // * Derivative / FFT

                // Make integral again
                std::vector<int> normalized(vec.size());
                for (int i = 0; i < vec.size(); i++) {
                    normalized[i] = vec[i];
                }

                // Plot line sample
                CvPlot::clear("line");
                CvPlot::plot("line", normalized.data(), normalized.size());
            }
        }

        // Show frame
        cv::imshow("video", frame);

        // Wait for key or 60 ms
        int key = key = cv::waitKey(60);
        switch(key) {
            case 'q': return 0;
        }
    }

    return 0;
}
