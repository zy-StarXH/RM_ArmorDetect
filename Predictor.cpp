//
// Created by Zy on 24-12-7.
//
#include "predictor.h"

// 构造函数：初始化卡尔曼滤波器
Predictor::Predictor()
    : KF(4, 2, 0),       // 4个状态变量（位置+速度），2个测量值（x, y）
      state(4, 1, CV_32F, cv::Scalar(0)),
      meas(2, 1, CV_32F, cv::Scalar(0)),
      initialized(false)
{
    // 转移矩阵
    KF.transitionMatrix = (cv::Mat_<float>(4, 4) <<
        1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1);

    // 测量矩阵
    KF.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);

    // 过程噪声协方差矩阵
    KF.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-4;

    // 测量噪声协方差矩阵
    KF.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * 1e-2;

    // 初始误差协方差
    KF.errorCovPost = cv::Mat::eye(4, 4, CV_32F);
}

// 预测目标中心点
cv::Point Predictor::predict() {
    if (!initialized) {
        // 如果未初始化，返回零点
        return cv::Point(0, 0);
    }

    // 预测
    cv::Mat prediction = KF.predict();
    return cv::Point(prediction.at<float>(0), prediction.at<float>(1));
}

// 更新测量值
void Predictor::updateMeasurement(const cv::Point& point) {
    meas.at<float>(0) = point.x;
    meas.at<float>(1) = point.y;

    if (!initialized) {
        // 初始化状态
        KF.statePost.at<float>(0) = meas.at<float>(0);
        KF.statePost.at<float>(1) = meas.at<float>(1);
        initialized = true;
    }

    // 更新卡尔曼滤波器
    KF.correct(meas);
}

// 重置滤波器
void Predictor::reset() {
    KF.statePost = cv::Mat::zeros(4, 1, CV_32F);
    KF.errorCovPost = cv::Mat::eye(4, 4, CV_32F);
    initialized = false;
}
