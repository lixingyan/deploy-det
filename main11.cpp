#include <iostream>
#include "TrtModel.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <deque>
#include <array>


int main()
{

    cv::VideoCapture cap("./media/lence222.mp4"); //sence3 24 30

    // std::string rtsp1 = "rtsp://admin:xs180301@192.168.254.2:554/Streaming/Channels/101";
    // cv::VideoCapture cap = cv::VideoCapture(rtsp1, cv::CAP_FFMPEG);

    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file.\n";
        return -1;
    }

    cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 获取帧率
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "width: " << frameSize.width << " height: " << frameSize.height << " fps: " << video_fps << std::endl;

    //初始化模型
    TrtModel trtmodel("./weights/yolov5s.onnx",true,1);

    // 创建一个窗口来显示视频
    cv::namedWindow("Processed Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Processed Video", frameSize.width, frameSize.height);

    cv::Mat frame;
    int frame_nums = 0;

    // 读取和显示视频帧，直到视频结束
    while (cap.read(frame)) {

        auto start = std::chrono::high_resolution_clock::now();

        auto results = trtmodel.doInference(frame);

        // std::cout<<"检测目标个数："<<results.size()<<std::endl;


        trtmodel.draw_bbox(frame,results);

        // 显示处理后的帧
        cv::imshow("Processed Video", frame);

// 获取程序结束时间点
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (cv::waitKey(25) == 27) {
            break;
        }
    }

    // 释放视频捕获对象和关闭所有窗口
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
