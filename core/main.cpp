#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

class ImageEnhancer {
private:
    cv::Ptr<cv::FaceDetectorYN> face_detector;

public:
    ImageEnhancer(const std::string& yunet_model_path) {
        face_detector = cv::FaceDetectorYN::create(
            yunet_model_path, "", cv::Size(320, 320), 0.6f, 0.3f, 100
        );
        if (face_detector.empty()) {
            throw std::runtime_error("Failed to load YuNet model at " + yunet_model_path);
        }
    }

    cv::Mat enhance(const cv::Mat& img) {
        if (img.empty()) throw std::invalid_argument("Empty image provided.");

        int h = img.rows;
        int w = img.cols;
        face_detector->setInputSize(cv::Size(w, h));

        cv::Mat ycrcb;
        cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);
        
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        cv::Mat Y = channels[0];
        
        cv::Mat Y_float;
        Y.convertTo(Y_float, CV_32F);

        cv::Mat small_blur, mean, stddev;
        cv::GaussianBlur(Y_float, small_blur, cv::Size(3, 3), 0);
        cv::Mat diff = Y_float - small_blur;
        cv::meanStdDev(diff, mean, stddev);
        float noise_level = static_cast<float>(stddev.at<double>(0, 0));

        cv::Mat smooth;
        int diag = std::sqrt(h * h + w * w);
        int d = std::max(5, std::min(15, diag / 300));
        int sigmaColor = std::max(20, std::min(80, static_cast<int>(noise_level * 1.5f)));
        int sigmaSpace = std::max(20, std::min(80, diag / 200));
        
        cv::bilateralFilter(Y, smooth, d, sigmaColor, sigmaSpace);
        
        cv::Mat smooth_float;
        smooth.convertTo(smooth_float, CV_32F);

        cv::Mat detail = Y_float - smooth_float;
        float noise_threshold = std::max(2.0f, noise_level * 0.5f);
        
        // Multi-threaded Coring
        cv::parallel_for_(cv::Range(0, detail.rows), [&](const cv::Range& range) {
            for (int r = range.start; r < range.end; r++) {
                float* row_ptr = detail.ptr<float>(r);
                for (int c = 0; c < detail.cols; c++) {
                    if (std::abs(row_ptr[c]) <= noise_threshold) {
                        row_ptr[c] = 0.0f;
                    }
                }
            }
        });

        cv::Mat alpha(h, w, CV_32F, cv::Scalar(1.0f)); 
        cv::Mat faces;
        face_detector->detect(img, faces);
        cv::Mat face_mask(h, w, CV_32F, cv::Scalar(1.0f));

        if (!faces.empty()) {
            for (int i = 0; i < faces.rows; i++) {
                int fx = static_cast<int>(faces.at<float>(i, 0));
                int fy = static_cast<int>(faces.at<float>(i, 1));
                int fw = static_cast<int>(faces.at<float>(i, 2));
                int fh = static_cast<int>(faces.at<float>(i, 3));

                cv::Point center(fx + fw / 2, fy + fh / 2);
                cv::Size axes(fw / 2, fh / 2);
                cv::ellipse(face_mask, center, axes, 0, 0, 360, cv::Scalar(0.4f), -1);
            }
            int blur_size = static_cast<int>(std::max(h, w) * 0.04);
            if (blur_size % 2 == 0) blur_size++; 
            if (blur_size > 0) cv::GaussianBlur(face_mask, face_mask, cv::Size(blur_size, blur_size), 0);
        }

        cv::multiply(alpha, face_mask, alpha);
        cv::Mat Y_enhanced;
        cv::multiply(detail, alpha, detail);
        cv::add(Y_float, detail, Y_enhanced);
        
        Y_enhanced.convertTo(channels[0], CV_8U);

        cv::Mat final_ycrcb, result;
        cv::merge(channels, final_ycrcb);
        cv::cvtColor(final_ycrcb, result, cv::COLOR_YCrCb2BGR);

        return result;
    }
};

const char* keys =
    "{help h usage ? |      | print this message   }"
    "{@input         |<none>| path to input image  }"
    "{@output        |<none>| path to output image }"
    "{model m        |face_detection_yunet_2023mar.onnx| path to YuNet ONNX model }";

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("help") || argc < 3) {
        parser.printMessage();
        return 0;
    }

    std::string input_path = parser.get<std::string>("@input");
    std::string output_path = parser.get<std::string>("@output");
    std::string model_path = parser.get<std::string>("model");

    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }

    try {
        ImageEnhancer enhancer(model_path);
        
        cv::Mat img = cv::imread(input_path);
        if (img.empty()) {
            std::cerr << "Error: Could not read image at " << input_path << std::endl;
            return 1;
        }

        cv::Mat enhanced_img = enhancer.enhance(img);
        
        if (!cv::imwrite(output_path, enhanced_img)) {
            std::cerr << "Error: Could not write to " << output_path << std::endl;
            return 1;
        }
        
        std::cout << "Success: " << output_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}