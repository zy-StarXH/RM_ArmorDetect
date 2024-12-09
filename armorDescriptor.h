//
// Created by HP on 24-8-15.
//

#ifndef ARMORDESCRIPTOR_H
#define ARMORDESCRIPTOR_H

#include <LightDescriptor.h>
#include <armorParam.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
#define BIG_ARMOR 1
#define SMALL_ARMOR 0
#define UNKNOWN_ARMOR -1
class ArmorDescriptor
{
public:

    ArmorDescriptor();

    ArmorDescriptor(const LightDescriptor& lLight, const LightDescriptor& rLight, const int armorType, const cv::Mat& srcImg, const float rotationScore, ArmorParam param);

    // 清除所有信息，包括顶点信息
    void clear()
    {
        rotationScore = 0;
        sizeScore = 0;
        distScore = 0;
        finalScore = 0;
        for(int i = 0; i < 4; i++)
        {
            vertex[i] = cv::Point2f(0, 0);
        }
        type = UNKNOWN_ARMOR;
    }

    void getFrontImg(const cv::Mat& grayImg);

    // bool isArmorPattern() const;

public:
    std::array<cv::RotatedRect, 2> lightPairs;  // 创建一个含有两个旋转矩形的数组，用于存放一对灯条
    float sizeScore;
    float distScore;
    float rotationScore;
    float finalScore;
    Point2f center;
    std::vector<cv::Point2f> vertex;
    cv::Mat frontImg;
    int type;
};

#endif
