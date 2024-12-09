#include "ArmorDescriptor.h"


ArmorDescriptor::ArmorDescriptor()// 默认构造函数，初始化armorDescriptor的一些基本属性
{
    rotationScore = 0;
    sizeScore = 0;
    vertex.resize(4);
    for(int i = 0; i < 4; i++)
    {
        vertex[i] = cv::Point2f(0, 0);
    }
    type = UNKNOWN_ARMOR;
}

ArmorDescriptor::ArmorDescriptor(const LightDescriptor & lLight, const LightDescriptor & rLight, const int armorType, const cv::Mat & grayImg, float rotaScore, ArmorParam _param)
{
    // 将左右灯条的旋转矩形的信息储存在lightPairs中
    lightPairs[0] = lLight.rec();
    lightPairs[1] = rLight.rec();
    // 创建更大的矩形区域，方便后续处理
    cv::Size exLSize(int(lightPairs[0].size.width), int(lightPairs[0].size.height * 2));
    cv::Size exRSize(int(lightPairs[1].size.width), int(lightPairs[1].size.height * 2));
    cv::RotatedRect exLLight(lightPairs[0].center, exLSize, lightPairs[0].angle);
    cv::RotatedRect exRLight(lightPairs[1].center, exRSize, lightPairs[1].angle);

    cv::Point2f pts_l[4];
    exLLight.points(pts_l);// 将左灯条的四个顶点赋值给pts_l
    cv::Point2f upper_l = pts_l[2];
    cv::Point2f lower_l = pts_l[3];

    cv::Point2f pts_r[4];
    exRLight.points(pts_r);
    cv::Point2f upper_r = pts_r[1];
    cv::Point2f lower_r = pts_r[0];

    vertex.resize(4);
    vertex[0] = upper_l;
    vertex[1] = upper_r;
    vertex[2] = lower_r;
    vertex[3] = lower_l;

    for (const auto& point : vertex) {
        center += point;
    }
    center.x /= 4;
    center.y /= 4;

    type = armorType;
    getFrontImg(grayImg);

    rotationScore = rotaScore;


    float normalized_area = contourArea(vertex) / _param.area_normalized_base; // 指定多边形面积/1000，归一化面积计算
    sizeScore = exp(normalized_area); // 计算normalized_area评分 exp函数能将较小的面积比例映射到较大的评分值，从而强调较小装甲板的重要性


}

// 提取roi区域，方便后续使用SVM或者模板匹配
void ArmorDescriptor::getFrontImg(const Mat& grayImg)
{

}



