#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <armorParam.h>
#include <LightDescriptor.h>
#include <armorDescriptor.h>

#include "predictor.h"

using namespace std;
using namespace cv;

// 创建模板函数distance，计算两个point之间的欧几里得距离
// 接受的参数可以是point<int>或者point<float>
template<typename T>
float distance(const cv::Point_<T>& pt1, const cv::Point_<T>& pt2) // &表示直接对点的引用而不是复制
{
    return std::sqrt(std::pow((pt1.x - pt2.x), 2) + std::pow((pt1.y - pt2.y), 2)); // 勾股定理
}

// 定义ArmorDetector类
class ArmorDetector {
public:
    Point2i center;
    // 初始化敌人颜色
    void init(int selfColor) {
        if (selfColor == RED) { // RED = 0
            _enemy_color = BLUE; // BLUE = 1
            _self_color = RED;
        }
        else {
            _enemy_color = RED;
            _self_color = BLUE;
        }
        cout << "initialization complete." << endl;
    }

    void loadImg(Mat& img) {
        _srcImg = img;
        cvtColor(img, hsvImg, COLOR_BGR2HSV);
        Rect imgBound = Rect(cv::Point(0,0), Point(_srcImg.cols, _srcImg.rows));// rect类为cv中表示矩形的区域
        _roi = imgBound;
        _roiImg = _srcImg(_roi).clone();
    }

    // 此函数不起作用
    Mat CaptureVideo(VideoCapture& video) {
        cv::Mat src_img, resized_img, _grayimg, hsv_img, gray_img, debug_img;
        vector<Mat> channels;
        while (video.isOpened()) {
            Mat frame;
            video >> frame;
            if (frame.empty()) {
                cout << "something is wrong." << endl;
                break;
            }
            cvtColor(frame, hsv_img, COLOR_BGR2HSV);
            cvtColor(frame, gray_img, COLOR_BGR2GRAY, 1);
            split(frame, channels);
            _grayimg=channels.at(0)-channels.at(2);
            return _grayimg;
        }
    }

    // 检测主程序
    int detect() {
        _grayImg = separateColors_2();
        imshow("grayImg", _grayImg);

        // 二值化图像，便于查找轮廓
        int brightness_threshold = 205;// 设置阈值下限,根据实际情况（相机、场地明亮度）更改
        Mat binBrightImg;
        threshold(_grayImg, binBrightImg, brightness_threshold, 255, cv::THRESH_BINARY);
        // imshow("thresh", binBrightImg);

        // 膨胀处理，防止部分灯条中断连接造成的多轮廓
        Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
        dilate(binBrightImg, binBrightImg, element);
        imshow("dilate", binBrightImg);

        // 寻找经过膨胀处理的二值化图像的轮廓
        vector<vector<Point>> lightContours; // 二维数组，每行代表一个轮廓
        findContours(binBrightImg.clone(), lightContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // debug
        _debugImg = _roiImg.clone();
        for(size_t i = 0; i < lightContours.size(); i++) {
            // drawContours(_debugImg, lightContours, i, Scalar(0,0,255), 1, 8);
        }
        // imshow("contours", _debugImg);

        // 将轮廓信息添加到LightDescriptor类的lightInfos对象中
        vector<LightDescriptor> lightInfos;
        filterContours(lightContours, lightInfos);
        if(lightInfos.empty()) {
            cout << "No lights have been detected" << endl;
            return -1;
        }

        // 绘制灯条轮廓
        drawLightInfo(lightInfos);

        _armors = matchArmor(lightInfos);
        if(_armors.empty()) {
            //cout << "armor empty" << endl;
            // return -1;
        }

        // 遍历每个装甲板，绘制装甲板区域
        for(size_t i = 0; i < _armors.size(); i++) {
            vector<Point2i> points;
            for (int j = 0; j < 4; j++) {
                points.push_back(Point(static_cast<int>(_armors[i].vertex[j].x), // 第一个装甲板的顶点数组
                    static_cast<int>(_armors[i].vertex[j].y)));
            }
            polylines(_debugImg, points, true, Scalar(0,255,0),
                    1, 8, 0);//绘制两个不填充的多边形
            // 计算装甲板中心
            Point2i center(0, 0);
            for (const auto& point : points) {
                center += point;
            }
            center.x /= 4;
            center.y /= 4;
            circle(_debugImg, center, 3, Scalar(0,255,0), -1);
            cout << center << endl;
        }
        // imshow("armors", _debugImg);
    }

    //分离颜色，提取敌人颜色，返回灰度图
    Mat separateColors() {
        vector<Mat> channels;
        split(_roiImg, channels);
        Mat grayImg;
        // 剔除不想要的颜色
        // 将不想要的颜色减去（我方颜色）
        if (_enemy_color==RED) {
            grayImg = channels.at(2)-channels.at(0);//R-B
        }
        else {
            grayImg = channels.at(0)-channels.at(2);//B-R
        }
        return grayImg;
    }

    // 第二种处理方式，适用摄像头拍摄泛色情况
    Mat separateColors_2() {
        vector<Mat> channels;
        split(_roiImg, channels);
        Mat blueImg = channels[0];
        Mat grayImg;
        // GaussianBlur(blueImg, grayImg, Size(5,5), 0);
        return blueImg;
    }

    Mat separateColors_3() {
        Mat mask;
        if (_enemy_color == RED) {
            // 红色范围分割（红色有两个色调范围）
            Mat lowerRedMask, upperRedMask;
            inRange(hsvImg, Scalar(0, 43, 46), Scalar(10, 255, 255), lowerRedMask);  // 红色低范围
            inRange(hsvImg, Scalar(170, 43, 46), Scalar(180, 255, 255), upperRedMask); // 红色高范围

            // 合并两个红色范围的掩膜
            mask = lowerRedMask | upperRedMask;
        } else if (_enemy_color == BLUE) {
            // 蓝色范围分割
            inRange(hsvImg, Scalar(100, 43, 120), Scalar(124, 255, 255), mask);
        }

        // 提取白色部分
        Mat whiteMask;
        inRange(hsvImg, Scalar(0, 0, 221), Scalar(180, 50, 255), whiteMask);

        // 合并目标颜色和白色部分
        mask |= whiteMask;

        return mask;
    }

    // 筛选符合条件的轮廓
    // 输入存储轮廓的矩阵，返回存储灯条信息的外接矩形
    void filterContours(vector<vector<Point>>& lightContours, vector<LightDescriptor>& lightInfos) {
        for (const auto& contour : lightContours) { // 由auto自动推导lightcontour类型，每次循环以contour进行，冒号定义循环变量的类型
            // 得到面积
            float lightContourArea = contourArea(contour);
            // 剔除面积小的轮廓
            if (lightContourArea < _param.light_min_area) continue;
            // 椭圆拟合区域得到外接矩形
            RotatedRect lightRec = fitEllipse(contour);
            // 矫正灯条角度，约束到—45到45，使得长轴和短轴始终为正确的一边
            adjustRec(lightRec);
            // 宽高比、凸度来筛选灯条 凸度=轮廓面积/外接矩形面积
            // 正常的宽高比不超过0.4 凸度不低于0.5 太低的凸度代表是一个接近直线没有宽度的灯条
            if (lightRec.size.width / lightRec.size.height > _param.light_max_ratio ||
                lightContourArea / lightRec.size.area() < _param.light_contour_min_solidity) continue;
            // 对灯条范围适当扩大
            lightRec.size.width *= _param.light_color_detect_extend_ratio;
            lightRec.size.height *= _param.light_color_detect_extend_ratio;
            // 己方灯条已经被separate，可以直接保存灯条
            lightInfos.push_back(LightDescriptor(lightRec));
        }
    }

    // 绘制旋转矩形
    void drawLightInfo(vector<LightDescriptor>& LD) {
        _debugImg = _roiImg.clone();
        vector<vector<Point>> cons;
        int i = 0;
        for (auto &lightinfo: LD) {
            RotatedRect rotate = lightinfo.rec();
            auto vertices = new Point2f[4];// 创建一个包含四个point2f的数组
            rotate.points(vertices);// 将旋转矩形的四个顶点给到上面的数组中
            vector<Point> con;
            for (int i = 0; i < 4; i++) {
                con.push_back((vertices[i]));// 将vertices中的四个顶点返回给con
            }
            cons.push_back(con);// 第一排被填充
            drawContours(_debugImg, cons, i, Scalar(0, 255, 255), 1, 8);
            // imshow("rotateRec", _debugImg);
            i++;
            delete vertices;
        }
    }

    // 匹配灯条，筛选出装甲板
    vector<ArmorDescriptor> matchArmor(vector<LightDescriptor>& lightInfos) {
        vector<ArmorDescriptor> armors;
        // 按灯条中心x从小到大排序
        sort(lightInfos.begin(), lightInfos.end(), [](const LightDescriptor& ld1, const LightDescriptor& ld2){
            return ld1.center.x < ld2.center.x;
        });
        // 遍历所有灯条进行匹配
        for (size_t i = 0; i < lightInfos.size(); i++) {
            // 尝试多次排列左右灯条作比较
            for (size_t j = i + 1; (j < lightInfos.size()); j++) {
                const LightDescriptor& leftLight = lightInfos[i];
                const LightDescriptor& rightLight = lightInfos[j];

                // 角差，装甲板两灯条角差相近
                float angleDiff = abs(leftLight.angle - rightLight.angle);
                // 长度差比率，比率越小说明两灯条的长度越相近，相似度越高
                float lenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
                // 筛选
                if (angleDiff > _param.light_max_angle_diff_ || lenDiff_ratio > _param.light_max_height_diff_ratio_) {
                    continue;
                }

                // 左右灯条相距距离
                float dis = distance(leftLight.center, rightLight.center);
                // 左右灯条长度的平均值
                float meanLen = (leftLight.length + rightLight.length) / 2;
                // 左右灯条中心点y的差值
                float yDiff = abs(leftLight.center.y - rightLight.center.y);
                // y差的比率
                float yDiff_ratio = yDiff / meanLen;
                // 左右灯条中心点x的差值
                float xDiff = abs(leftLight.center.x - rightLight.center.x);
                // x差的比率
                float xDiff_ratio = xDiff / meanLen;
                // 相距距离与灯条长度比值
                float ratio = dis / meanLen;
                // 筛选
                if (yDiff_ratio > _param.light_max_y_diff_ratio_ || // 确保左右灯条在垂直方向上的位置相对接近，保证他们是在一条水平线上
                    xDiff_ratio < _param.light_min_x_diff_ratio_ || // 确保左右灯条的水平方向上有足够的距离，这是装甲板灯条的基本条件
                    ratio > _param.armor_max_aspect_ratio_ || // 确保灯条之间的距离和灯条本身成合理比例，以确保灯条之间距离合适
                    ratio < _param.armor_min_aspect_ratio_) {
                    continue;
                }

                // 按照比值来确定大小装甲
                int armorType = ratio > _param.armor_big_armor_ratio ? BIG_ARMOR : SMALL_ARMOR; // 距离长的为大装甲板，反之为小
                // 计算旋转积分
                float ratiOff = (armorType == BIG_ARMOR) ? max(_param.armor_big_armor_ratio-ratio, float(0)) :
                max(_param.armor_small_armor_ratio-ratio, float(0));
                float yOff = yDiff / meanLen;
                float rotationScore = -(ratiOff * ratiOff + yOff * yOff);
                // 得到匹配的装甲板
                ArmorDescriptor armor(leftLight, rightLight, armorType,
                    _grayImg, rotationScore, _param);
                armors.emplace_back(armor);
                break;
            }
        }
        return armors;
    }

    void showRoi() {

    }

    void adjustRec(cv::RotatedRect& rec)
    {
        using std::swap; // 标准库交换函数
        // 加上&使得中间变量能直接修改旋转矩形的值
        float& width = rec.size.width;
        float& height = rec.size.height;
        float& angle = rec.angle;
        // 角度为0，表示长边和水平线平行 正角度，表示矩形顺时针旋转
        // 将角度限制在 -90 到 90 度范围内。这是因为旋转矩形的角度通常在 -180 到 180 度之间，规范化到 -90 到 90 度更有利于处理。
        // 保证长短轴的一致性
        while(angle >= 90.0) angle -= 180.0;
        while(angle < -90.0) angle += 180.0;

        if(angle >= 45.0)
        {
            swap(width, height);
            angle -= 90.0;
        }
        else if(angle < -45.0)
        {
            swap(width, height);
            angle += 90.0;
        }
    }
    cv::Mat _debugImg;

    vector<ArmorDescriptor> getArmors() const {
        return _armors;
    }

private:
    int _enemy_color;
    int _self_color;
    cv::Rect _roi;
    cv::Mat _srcImg;
    Mat _grayImg;
    Mat hsvImg;
    Mat _roiImg;
    ArmorParam _param;
    vector<ArmorDescriptor> _armors;
};



int main() {
    ArmorDetector detector;
    Predictor predictor;
    detector.init(RED);

    // Mat img = imread("D:/OneDrive/桌面/大创/vision_lib/test.jpeg");
    VideoCapture capture("D:/rm_-armor-detect/live.avi");
    if (!capture.isOpened()) {
        cout << "Error epening video file" << endl;
        return -1;
    }

    double fps = capture.get(CAP_PROP_FPS);
    cout << "Frames fps:" << fps << endl;

    while (true) {
        Mat frame, adjustImg;
        capture >> frame;//读取下一帧给frame
        if (frame.empty()) break;//没有帧了，退出循环
        imshow("capture", frame);

        // 降低曝光度
        frame.convertTo(adjustImg, -1, 1, 0); // -1表示与输入图像相同的深度, 1是对比度因子, -50是亮度调整值
        detector.loadImg(adjustImg);
        // imshow("adjustImg", adjustImg);

        detector.detect();//处理每一帧
        auto armors = detector.getArmors();
        if (!armors.empty()) {
            Point center = armors[0].center;
            cout << "Center: " << center << endl;
            predictor.updateMeasurement(center);
            Point predictcenter = predictor.predict();
            circle(detector._debugImg, predictcenter, 5, cv::Scalar(0, 0, 255), 2);
        }
        double current_fps = capture.get(CAP_PROP_FPS);
        string FPS = "fps:" + to_string(current_fps);

        // 类中函数处理的图像 能够通过对象.图像名在主程序中调用
        putText(detector._debugImg, FPS, Point(3, 20), FONT_HERSHEY_TRIPLEX, 0.5,
            Scalar(255, 255, 255), 1);
        imshow("armors", detector._debugImg);

        int key = waitKey(10);
        if (key == 27 || key == 'q') break; // 按esc或者q返回
    }

    capture.release();
    destroyAllWindows();
    return 0;
}