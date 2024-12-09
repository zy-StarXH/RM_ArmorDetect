#include "ArmorParam.h"

ArmorParam::ArmorParam()
{
    brightness_threshold = 210;
    color_threshold = 40;
    light_color_detect_extend_ratio = 1.1;

    light_min_area = 100;
    light_max_angle = 45.0;
    light_min_size = 5.0;
    light_contour_min_solidity = 0.5;
    light_max_ratio = 0.4;

    light_max_angle_diff_ = 7.0;
    light_max_height_diff_ratio_ = 0.2;
    light_max_y_diff_ratio_ = 2.0;
    light_min_x_diff_ratio_ = 0.5;

    armor_big_armor_ratio = 3.2;
    armor_small_armor_ratio = 2;

    armor_min_aspect_ratio_ = 1.0;
    armor_max_aspect_ratio_ = 5.0;

    sight_offset_normalized_base = 200;
    area_normalized_base = 1000;
    enemy_color = BLUE;

}

