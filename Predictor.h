//
// Created by Zy on 24-12-7.
//

#ifndef PREDICT_H
#define PREDICT_H
#ifndef PREDICTOR_H
#define PREDICTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class Predictor {
public:
    Predictor();
    ~Predictor() = default;

    // 预测目标中心点
    cv::Point predict();

    // 更新测量值
    void updateMeasurement(const cv::Point& point);

    // 重置滤波器
    void reset();

private:
    cv::KalmanFilter KF; // 卡尔曼滤波器
    cv::Mat state;       // 状态矩阵
    cv::Mat meas;        // 测量矩阵
    bool initialized;    // 是否已初始化
};

#endif // PREDICTOR_H

#endif //PREDICT_H
