#ifndef ARMORPARAM_H
#define ARMORPARAM_H

//
#define RED 0
#define BLUE 1
class ArmorParam
{
public:
    ArmorParam();
public:
    int brightness_threshold;
    int color_threshold;
    float light_color_detect_extend_ratio;

    float light_min_area;
    float light_max_angle;
    float light_min_size;
    float light_contour_min_solidity;
    float light_max_ratio;

    float light_max_angle_diff_;
    float light_max_height_diff_ratio_;
    float light_max_y_diff_ratio_;
    float light_min_x_diff_ratio_;

    float armor_big_armor_ratio;
    float armor_small_armor_ratio;
    float armor_min_aspect_ratio_;
    float armor_max_aspect_ratio_;

    float sight_offset_normalized_base;
    float area_normalized_base;
    int enemy_color;
    int max_track_num = 3000;


};

#endif

