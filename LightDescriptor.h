//
// Created by HP on 24-8-15.
//

#ifndef LIGHTDESCRIPTOR_H
#define LIGHTDESCRIPTOR_H
#include<opencv2/opencv.hpp>
#include<vector>

class LightDescriptor
{
public:
    LightDescriptor() {};  // 默认构造函数，能够使得在创建类的对象的时候不需要立即填入参数
    LightDescriptor(const cv::RotatedRect& light) // 解释作为旋转矩形框出的灯条
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
    const LightDescriptor& operator = (const LightDescriptor& ld)
    {
        this->width = ld.width;
        this->length = ld.length;
        this->center = ld.center;
        this->angle = ld.angle;
        this->area = ld.area;
        return *this;  // 返回当前对象的引用，便于链式赋值
    }

    cv::RotatedRect rec() const
    {
        return cv::RotatedRect(center, cv::Size2f(width, length), angle);
    }

public:
    float width;
    float length;
    cv::Point2f center;
    float angle;
    float area;
};

#endif
