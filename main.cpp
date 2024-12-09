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

// ����ģ�庯��distance����������point֮���ŷ����þ���
// ���ܵĲ���������point<int>����point<float>
template<typename T>
float distance(const cv::Point_<T>& pt1, const cv::Point_<T>& pt2) // &��ʾֱ�ӶԵ�����ö����Ǹ���
{
    return std::sqrt(std::pow((pt1.x - pt2.x), 2) + std::pow((pt1.y - pt2.y), 2)); // ���ɶ���
}

// ����ArmorDetector��
class ArmorDetector {
public:
    Point2i center;
    // ��ʼ��������ɫ
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
        Rect imgBound = Rect(cv::Point(0,0), Point(_srcImg.cols, _srcImg.rows));// rect��Ϊcv�б�ʾ���ε�����
        _roi = imgBound;
        _roiImg = _srcImg(_roi).clone();
    }

    // �˺�����������
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

    // ���������
    int detect() {
        _grayImg = separateColors_2();
        imshow("grayImg", _grayImg);

        // ��ֵ��ͼ�񣬱��ڲ�������
        int brightness_threshold = 205;// ������ֵ����,����ʵ���������������������ȣ�����
        Mat binBrightImg;
        threshold(_grayImg, binBrightImg, brightness_threshold, 255, cv::THRESH_BINARY);
        // imshow("thresh", binBrightImg);

        // ���ʹ�����ֹ���ֵ����ж�������ɵĶ�����
        Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
        dilate(binBrightImg, binBrightImg, element);
        imshow("dilate", binBrightImg);

        // Ѱ�Ҿ������ʹ���Ķ�ֵ��ͼ�������
        vector<vector<Point>> lightContours; // ��ά���飬ÿ�д���һ������
        findContours(binBrightImg.clone(), lightContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // debug
        _debugImg = _roiImg.clone();
        for(size_t i = 0; i < lightContours.size(); i++) {
            // drawContours(_debugImg, lightContours, i, Scalar(0,0,255), 1, 8);
        }
        // imshow("contours", _debugImg);

        // ��������Ϣ��ӵ�LightDescriptor���lightInfos������
        vector<LightDescriptor> lightInfos;
        filterContours(lightContours, lightInfos);
        if(lightInfos.empty()) {
            cout << "No lights have been detected" << endl;
            return -1;
        }

        // ���Ƶ�������
        drawLightInfo(lightInfos);

        _armors = matchArmor(lightInfos);
        if(_armors.empty()) {
            //cout << "armor empty" << endl;
            // return -1;
        }

        // ����ÿ��װ�װ壬����װ�װ�����
        for(size_t i = 0; i < _armors.size(); i++) {
            vector<Point2i> points;
            for (int j = 0; j < 4; j++) {
                points.push_back(Point(static_cast<int>(_armors[i].vertex[j].x), // ��һ��װ�װ�Ķ�������
                    static_cast<int>(_armors[i].vertex[j].y)));
            }
            polylines(_debugImg, points, true, Scalar(0,255,0),
                    1, 8, 0);//�������������Ķ����
            // ����װ�װ�����
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

    //������ɫ����ȡ������ɫ�����ػҶ�ͼ
    Mat separateColors() {
        vector<Mat> channels;
        split(_roiImg, channels);
        Mat grayImg;
        // �޳�����Ҫ����ɫ
        // ������Ҫ����ɫ��ȥ���ҷ���ɫ��
        if (_enemy_color==RED) {
            grayImg = channels.at(2)-channels.at(0);//R-B
        }
        else {
            grayImg = channels.at(0)-channels.at(2);//B-R
        }
        return grayImg;
    }

    // �ڶ��ִ���ʽ����������ͷ���㷺ɫ���
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
            // ��ɫ��Χ�ָ��ɫ������ɫ����Χ��
            Mat lowerRedMask, upperRedMask;
            inRange(hsvImg, Scalar(0, 43, 46), Scalar(10, 255, 255), lowerRedMask);  // ��ɫ�ͷ�Χ
            inRange(hsvImg, Scalar(170, 43, 46), Scalar(180, 255, 255), upperRedMask); // ��ɫ�߷�Χ

            // �ϲ�������ɫ��Χ����Ĥ
            mask = lowerRedMask | upperRedMask;
        } else if (_enemy_color == BLUE) {
            // ��ɫ��Χ�ָ�
            inRange(hsvImg, Scalar(100, 43, 120), Scalar(124, 255, 255), mask);
        }

        // ��ȡ��ɫ����
        Mat whiteMask;
        inRange(hsvImg, Scalar(0, 0, 221), Scalar(180, 50, 255), whiteMask);

        // �ϲ�Ŀ����ɫ�Ͱ�ɫ����
        mask |= whiteMask;

        return mask;
    }

    // ɸѡ��������������
    // ����洢�����ľ��󣬷��ش洢������Ϣ����Ӿ���
    void filterContours(vector<vector<Point>>& lightContours, vector<LightDescriptor>& lightInfos) {
        for (const auto& contour : lightContours) { // ��auto�Զ��Ƶ�lightcontour���ͣ�ÿ��ѭ����contour���У�ð�Ŷ���ѭ������������
            // �õ����
            float lightContourArea = contourArea(contour);
            // �޳����С������
            if (lightContourArea < _param.light_min_area) continue;
            // ��Բ�������õ���Ӿ���
            RotatedRect lightRec = fitEllipse(contour);
            // ���������Ƕȣ�Լ������45��45��ʹ�ó���Ͷ���ʼ��Ϊ��ȷ��һ��
            adjustRec(lightRec);
            // ��߱ȡ�͹����ɸѡ���� ͹��=�������/��Ӿ������
            // �����Ŀ�߱Ȳ�����0.4 ͹�Ȳ�����0.5 ̫�͵�͹�ȴ�����һ���ӽ�ֱ��û�п�ȵĵ���
            if (lightRec.size.width / lightRec.size.height > _param.light_max_ratio ||
                lightContourArea / lightRec.size.area() < _param.light_contour_min_solidity) continue;
            // �Ե�����Χ�ʵ�����
            lightRec.size.width *= _param.light_color_detect_extend_ratio;
            lightRec.size.height *= _param.light_color_detect_extend_ratio;
            // ���������Ѿ���separate������ֱ�ӱ������
            lightInfos.push_back(LightDescriptor(lightRec));
        }
    }

    // ������ת����
    void drawLightInfo(vector<LightDescriptor>& LD) {
        _debugImg = _roiImg.clone();
        vector<vector<Point>> cons;
        int i = 0;
        for (auto &lightinfo: LD) {
            RotatedRect rotate = lightinfo.rec();
            auto vertices = new Point2f[4];// ����һ�������ĸ�point2f������
            rotate.points(vertices);// ����ת���ε��ĸ�������������������
            vector<Point> con;
            for (int i = 0; i < 4; i++) {
                con.push_back((vertices[i]));// ��vertices�е��ĸ����㷵�ظ�con
            }
            cons.push_back(con);// ��һ�ű����
            drawContours(_debugImg, cons, i, Scalar(0, 255, 255), 1, 8);
            // imshow("rotateRec", _debugImg);
            i++;
            delete vertices;
        }
    }

    // ƥ�������ɸѡ��װ�װ�
    vector<ArmorDescriptor> matchArmor(vector<LightDescriptor>& lightInfos) {
        vector<ArmorDescriptor> armors;
        // ����������x��С��������
        sort(lightInfos.begin(), lightInfos.end(), [](const LightDescriptor& ld1, const LightDescriptor& ld2){
            return ld1.center.x < ld2.center.x;
        });
        // �������е�������ƥ��
        for (size_t i = 0; i < lightInfos.size(); i++) {
            // ���Զ���������ҵ������Ƚ�
            for (size_t j = i + 1; (j < lightInfos.size()); j++) {
                const LightDescriptor& leftLight = lightInfos[i];
                const LightDescriptor& rightLight = lightInfos[j];

                // �ǲװ�װ��������ǲ����
                float angleDiff = abs(leftLight.angle - rightLight.angle);
                // ���Ȳ���ʣ�����ԽС˵���������ĳ���Խ��������ƶ�Խ��
                float lenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
                // ɸѡ
                if (angleDiff > _param.light_max_angle_diff_ || lenDiff_ratio > _param.light_max_height_diff_ratio_) {
                    continue;
                }

                // ���ҵ���������
                float dis = distance(leftLight.center, rightLight.center);
                // ���ҵ������ȵ�ƽ��ֵ
                float meanLen = (leftLight.length + rightLight.length) / 2;
                // ���ҵ������ĵ�y�Ĳ�ֵ
                float yDiff = abs(leftLight.center.y - rightLight.center.y);
                // y��ı���
                float yDiff_ratio = yDiff / meanLen;
                // ���ҵ������ĵ�x�Ĳ�ֵ
                float xDiff = abs(leftLight.center.x - rightLight.center.x);
                // x��ı���
                float xDiff_ratio = xDiff / meanLen;
                // ��������������ȱ�ֵ
                float ratio = dis / meanLen;
                // ɸѡ
                if (yDiff_ratio > _param.light_max_y_diff_ratio_ || // ȷ�����ҵ����ڴ�ֱ�����ϵ�λ����Խӽ�����֤��������һ��ˮƽ����
                    xDiff_ratio < _param.light_min_x_diff_ratio_ || // ȷ�����ҵ�����ˮƽ���������㹻�ľ��룬����װ�װ�����Ļ�������
                    ratio > _param.armor_max_aspect_ratio_ || // ȷ������֮��ľ���͵�������ɺ����������ȷ������֮��������
                    ratio < _param.armor_min_aspect_ratio_) {
                    continue;
                }

                // ���ձ�ֵ��ȷ����Сװ��
                int armorType = ratio > _param.armor_big_armor_ratio ? BIG_ARMOR : SMALL_ARMOR; // ���볤��Ϊ��װ�װ壬��֮ΪС
                // ������ת����
                float ratiOff = (armorType == BIG_ARMOR) ? max(_param.armor_big_armor_ratio-ratio, float(0)) :
                max(_param.armor_small_armor_ratio-ratio, float(0));
                float yOff = yDiff / meanLen;
                float rotationScore = -(ratiOff * ratiOff + yOff * yOff);
                // �õ�ƥ���װ�װ�
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
        using std::swap; // ��׼�⽻������
        // ����&ʹ���м������ֱ���޸���ת���ε�ֵ
        float& width = rec.size.width;
        float& height = rec.size.height;
        float& angle = rec.angle;
        // �Ƕ�Ϊ0����ʾ���ߺ�ˮƽ��ƽ�� ���Ƕȣ���ʾ����˳ʱ����ת
        // ���Ƕ������� -90 �� 90 �ȷ�Χ�ڡ�������Ϊ��ת���εĽǶ�ͨ���� -180 �� 180 ��֮�䣬�淶���� -90 �� 90 �ȸ������ڴ���
        // ��֤�������һ����
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

    // Mat img = imread("D:/OneDrive/����/��/vision_lib/test.jpeg");
    VideoCapture capture("D:/rm_-armor-detect/live.avi");
    if (!capture.isOpened()) {
        cout << "Error epening video file" << endl;
        return -1;
    }

    double fps = capture.get(CAP_PROP_FPS);
    cout << "Frames fps:" << fps << endl;

    while (true) {
        Mat frame, adjustImg;
        capture >> frame;//��ȡ��һ֡��frame
        if (frame.empty()) break;//û��֡�ˣ��˳�ѭ��
        imshow("capture", frame);

        // �����ع��
        frame.convertTo(adjustImg, -1, 1, 0); // -1��ʾ������ͼ����ͬ�����, 1�ǶԱȶ�����, -50�����ȵ���ֵ
        detector.loadImg(adjustImg);
        // imshow("adjustImg", adjustImg);

        detector.detect();//����ÿһ֡
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

        // ���к��������ͼ�� �ܹ�ͨ������.ͼ�������������е���
        putText(detector._debugImg, FPS, Point(3, 20), FONT_HERSHEY_TRIPLEX, 0.5,
            Scalar(255, 255, 255), 1);
        imshow("armors", detector._debugImg);

        int key = waitKey(10);
        if (key == 27 || key == 'q') break; // ��esc����q����
    }

    capture.release();
    destroyAllWindows();
    return 0;
}