#include "opticalFlowTracker.h"

opticalFlowTracker::opticalFlowTracker()
{
    // 设置迭代算法的终止条件
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.05);

    if (lk_optical_flow_kernel == nullptr)
        lk_optical_flow_kernel = std::make_shared<LKOpticalFlowKernel>(cv::Size(21, 21), 3, criteria, cv_OPTFLOW_LK_GET_MIN_EIGENVALS);

    // 跟踪过程中最大允许跟踪的特征点的数量
    maximum_tracked_points = 300;
}

// 更新和追加跟踪点
void opticalFlowTracker::updateAndAppendTrackPoints(cloudFrame *p_frame, rgbMapTracker *map_tracker, double mini_distance, int minimum_frame_diff)
{
    // 投影到图像上的坐标
    double u_d, v_d;
    // 四舍五入取整
    int u_i, v_i;

    // 允许的最大重投影误差
    double max_allow_reproject_error = 2.0 * p_frame->image_cols / 320.0;

    // 一个哈希映射，用于存储二维点是否已被占用
    Hash_map_2d<int, float> map_2d_points_occupied;

    // 遍历上一个图像姿态中的rgb点
    for (auto it = map_rgb_points_in_last_image_pose.begin(); it != map_rgb_points_in_last_image_pose.end();)
    {
        // 当前RGB 点
        rgbPoint *rgb_point = ((rgbPoint*)it->first);
        // 当前 RGB 点的三维坐标
        Eigen::Vector3d point_3d = ((rgbPoint*)it->first)->getPosition();

        // 尝试将3D点投影到当前图像上
        bool res = p_frame->project3dPointInThisImage(point_3d, u_d, v_d, nullptr, 1.0);

        // 四舍五入取整
        u_i = std::round(u_d / mini_distance) * mini_distance;
        v_i = std::round(v_d / mini_distance) * mini_distance;

        // 计算重投影误差
        double error = Eigen::Vector2d(u_d - it->second.x, v_d - it->second.y).norm();

        // 如果误差大于允许的最大误差，则标记为离群点
        if (error > max_allow_reproject_error)
        {
            // 离群点计数增加
            rgb_point->is_out_lier_count++;

            // 如果连续多次离群或误差非常大，则从跟踪列表中移除
            if ((rgb_point->is_out_lier_count > 1) || (error > max_allow_reproject_error * 2))
            {
                rgb_point->is_out_lier_count = 0;
                it = map_rgb_points_in_last_image_pose.erase(it);   // 从映射中移除点
                continue;
            }
        }
        else
        {
            // 重置离群点计数
            rgb_point->is_out_lier_count = 0;
        }

        // 如果点成功投影
        if (res)
        {
            double depth = (point_3d - p_frame->p_state->t_world_camera).norm();

            // 如果当前点在哈希映射中不存在，则插入
            if (map_2d_points_occupied.if_exist(u_i, v_i) == false)
            {
                map_2d_points_occupied.insert(u_i, v_i, depth);
            }
        }

        it++;
    }

    // 如果存在用于投影的rgb点向量
    if (map_tracker->points_rgb_vec_for_projection != nullptr)
    {
        // 获取用于投影的 RGB 点的数量
        int point_size = map_tracker->points_rgb_vec_for_projection->size();

        // 遍历所有点
        for (int i = 0; i < point_size; i++)
        {
            // 如果点在映射中则跳过
            if (map_rgb_points_in_last_image_pose.find((*(map_tracker->points_rgb_vec_for_projection))[i]) !=
                 map_rgb_points_in_last_image_pose.end())
            {
                continue;
            }

            // 获取点的三维坐标
            Eigen::Vector3d point_3d = (*(map_tracker->points_rgb_vec_for_projection))[i]->getPosition();

            // 将三维点投影在图像上
            bool res = p_frame->project3dPointInThisImage(point_3d, u_d, v_d, nullptr, 1.0);

            // 四舍五入取整
            u_i = std::round(u_d / mini_distance) * mini_distance;
            v_i = std::round(v_d / mini_distance) * mini_distance;

            // 如果点成功投影
            if (res)
            {
                double depth = (point_3d - p_frame->p_state->t_world_camera).norm();

                // 如果当前点在哈希映射中不存在，则插入
                if (map_2d_points_occupied.if_exist(u_i, v_i) == false)
                {
                    map_2d_points_occupied.insert(u_i, v_i, depth);

                    map_rgb_points_in_last_image_pose[(*(map_tracker->points_rgb_vec_for_projection))[i]] = cv::Point2f(u_d, v_d);
                }
            }

            // 超过最大跟踪点数，结束循环
            if (map_rgb_points_in_last_image_pose.size() >= maximum_tracked_points)
            {
                break;
            }
        }
    }

    // 更新上一次跟踪向量和索引
    updateLastTrackingVectorAndIds();
}

// 设置相机的内参和畸变
void opticalFlowTracker::setIntrinsic(Eigen::Matrix3d intrinsic_, Eigen::Matrix<double, 5, 1> dist_coeffs_, cv::Size image_size_)
{
    cv::eigen2cv(intrinsic_, intrinsic);
    cv::eigen2cv(dist_coeffs_, dist_coeffs);
    initUndistortRectifyMap(intrinsic, dist_coeffs, cv::Mat(), intrinsic, image_size_, CV_16SC2, m_ud_map1, m_ud_map2);
}

// 在给定的图像帧上跟踪特征点
void opticalFlowTracker::trackImage(cloudFrame *p_frame, double distance)
{
    // 当前帧彩色图像
    cur_image = p_frame->rgb_image;
    // 当前帧时间戳
    current_image_time = p_frame->time_sweep_end;
    // 清除当前图像姿态中的rgb点映射
    map_rgb_points_in_cur_image_pose.clear();

    if (cur_image.empty()) return;

    // 当前帧灰度图
    cv::Mat gray_image = p_frame->gray_image;

    // 用于存储光流跟踪的状态（成功或失败）
    std::vector<uchar> status;
    // 用于存储光流跟踪的误差
    std::vector<float> error;

    // 将上一帧跟踪的点赋值给当前帧
    cur_tracked_points = last_tracked_points;

    // 上一帧跟踪点的数量
    int before_track = last_tracked_points.size();

    // 如果上一帧的跟踪点数少于 30 个，则不跟踪
    if (last_tracked_points.size() < 30)
    {
        last_image_time = current_image_time;
        return;
    }

    // 用 lk 光流法跟踪点
    lk_optical_flow_kernel->trackImage(gray_image, last_tracked_points, cur_tracked_points, status, 2);

    // 根据跟踪状态减少点向量
    reduce_vector(last_tracked_points, status);
    // 根据跟踪状态减少ID向量
    reduce_vector(old_ids, status);
    // 根据跟踪状态减少当前跟踪点向量
    reduce_vector(cur_tracked_points, status);

    // 跟踪后的点数
    int after_track = last_tracked_points.size();
    // 基础矩阵
    cv::Mat mat_F;

    // 基础矩阵计算前的点数
    unsigned int points_before_F = last_tracked_points.size();
    // 使用 ransac 算法计算基础矩阵
    mat_F = cv::findFundamentalMat(last_tracked_points, cur_tracked_points, cv::FM_RANSAC, 1.0, 0.997, status);

    // 当前帧跟踪点的数量
    unsigned int size_a = cur_tracked_points.size();
    // 根据基础矩阵计算结果，进一步筛选跟踪点
    reduce_vector(last_tracked_points, status);
    reduce_vector(old_ids, status);
    reduce_vector(cur_tracked_points, status);

    // 清除当前图像姿态中的rgb点映射
    map_rgb_points_in_cur_image_pose.clear();

    // 当前帧与上一帧的时间差
    double frame_time_diff = (current_image_time - last_image_time);

    // 遍历所有的跟踪点
    for (uint i = 0; i < last_tracked_points.size(); i++)
    {
        // 对每个跟踪点进行处理，包括更新点的图像速度和映射
        if (p_frame->if2dPointsAvailable(cur_tracked_points[i].x, cur_tracked_points[i].y, 1.0, 0.05))
        {
            rgbPoint *rgb_point_ptr = ((rgbPoint*)rgb_points_ptr_vec_in_last_image[old_ids[i]]);
            map_rgb_points_in_cur_image_pose[rgb_point_ptr] = cur_tracked_points[i];

            cv::Point2f point_image_velocity;

            if (frame_time_diff < 1e-5)
                point_image_velocity = cv::Point2f(1e-3, 1e-3);
            else
                point_image_velocity = (cur_tracked_points[i] - last_tracked_points[i]) / frame_time_diff;

            rgb_point_ptr->image_velocity = Eigen::Vector2d(point_image_velocity.x, point_image_velocity.y);
        }
    }

    // 排除误差较大的跟踪点
    if (distance > 0)
        rejectErrorTrackingPoints(p_frame, distance);

    // 更新用于下一帧跟踪的变量
    old_gray = gray_image.clone();
    old_image = cur_image;
    
    std::vector<cv::Point2f>().swap(last_tracked_points);
    last_tracked_points = cur_tracked_points;

    updateLastTrackingVectorAndIds();

    // 更新时间戳和帧索引
    image_idx++;
    last_image_time = current_image_time;
}

// 初始化光流跟踪
void opticalFlowTracker::init(cloudFrame *p_frame, std::vector<rgbPoint*> &rgb_points_vec, std::vector<cv::Point2f> &points_2d_vec)
{
    // 设置跟踪点，传入当前帧的彩色图像以及RGB点和2D点的向量。这个函数可能用于初始化跟踪点，即选择图像中的一些特征点进行跟踪
    setTrackPoints(p_frame->rgb_image, rgb_points_vec, points_2d_vec);

    // 当前帧时间戳
    current_image_time = p_frame->time_sweep_end;
    last_image_time = current_image_time;

    std::vector<uchar> status;
    // lk 光流跟踪
    lk_optical_flow_kernel->trackImage(p_frame->gray_image, last_tracked_points, cur_tracked_points, status);
}

// 设置光流跟踪的初始跟踪点
void opticalFlowTracker::setTrackPoints(cv::Mat &image, std::vector<rgbPoint*> &rgb_points_vec, std::vector<cv::Point2f> &points_2d_vec)
{
    // 克隆当前帧彩色图像
    old_image = image.clone();
    // 转换为灰度图
    cv::cvtColor(old_image, old_gray, cv::COLOR_BGR2GRAY);
    // 清空上一帧图像姿态中的rgb点映射
    map_rgb_points_in_last_image_pose.clear();

    // 将 2D 点映射到 RGB 点
    for (unsigned int i = 0; i < rgb_points_vec.size(); i++)
    {
        map_rgb_points_in_last_image_pose[(void*)rgb_points_vec[i]] = points_2d_vec[i];
    }

    // 更新上一帧的跟踪向量和ID向量
    updateLastTrackingVectorAndIds();
}

// 排除跟踪过程中误差较大的点
void opticalFlowTracker::rejectErrorTrackingPoints(cloudFrame *p_frame, double distance)
{
    // 投影到图像平面的点的坐标
    double u, v;

    // 记录被移除的跟踪点的数量
    int remove_count = 0;
    // 存储当前图像姿态中rgb点映射的总数
    int total_count = map_rgb_points_in_cur_image_pose.size();

    // 设置输出颜色
    scope_color(ANSI_COLOR_BLUE_BOLD);

    // 遍历所有 RGB 点映射
    for (auto it = map_rgb_points_in_cur_image_pose.begin(); it != map_rgb_points_in_cur_image_pose.end(); it++)
    {
        // RGB 点对应的 2D 坐标
        cv::Point2f predicted_point = it->second;

        // RGB 点的 3D 坐标
        Eigen::Vector3d point_position = ((rgbPoint*)it->first)->getPosition();

        // 投影到图像平面
        int res = p_frame->project3dPointInThisImage(point_position, u, v, nullptr, 1.0);

        if (res)
        {
            // 判断误差是否超过阈值
            if ((fabs(u - predicted_point.x ) > distance) || (fabs(v - predicted_point.y) > distance))
            {
                // Remove tracking pts
                map_rgb_points_in_cur_image_pose.erase(it);
                remove_count++;
            }
        }
        else
        {
            map_rgb_points_in_cur_image_pose.erase(it);
            remove_count++;
        }
    }
}

// 更新上一帧的跟踪点向量以及与之对应的ID向量
void opticalFlowTracker::updateLastTrackingVectorAndIds()
{
    // 初始化索引变量
    int idx = 0;

    // 清空上一帧跟踪点向量与上一帧RGB点指针向量
    last_tracked_points.clear();
    rgb_points_ptr_vec_in_last_image.clear();

    // 清除old_ids向量，该向量存储了每个跟踪点的ID
    old_ids.clear();

    // 遍历当前图像姿态中的rgb点映射
    for (auto it = map_rgb_points_in_last_image_pose.begin(); it != map_rgb_points_in_last_image_pose.end(); it++)
    {
        rgb_points_ptr_vec_in_last_image.push_back(it->first);
        last_tracked_points.push_back(it->second);

        old_ids.push_back(idx);

        idx++;
    }
}

// 使用RANSAC算法和PnP问题求解来移除跟踪中的异常值
bool opticalFlowTracker::removeOutlierUsingRansacPnp(cloudFrame *p_frame, int if_remove_ourlier)
{
    // 存储OpenCV格式的旋转和平移
    cv::Mat cv_so3, cv_trans;
    // 存储Eigen格式的旋转和平移
    Eigen::Vector3d eigen_so3, eigen_trans;

    // 初始化3D和2D点向量
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;

    // 初始化指向rgb点的指针向量
    std::vector<void *> map_ptr;

    // 遍历当前图像姿态中的rgb点映射
    for (auto it = map_rgb_points_in_cur_image_pose.begin(); it != map_rgb_points_in_cur_image_pose.end(); it++)
    {
        // 将指向rgbPoint的指针添加到map_ptr向量中
        map_ptr.push_back(it->first);
        // 填充 3D 坐标和 2D 坐标
        Eigen::Vector3d point_3d = ((rgbPoint*)it->first)->getPosition();

        points_3d.push_back(cv::Point3f(point_3d(0), point_3d(1), point_3d(2)));
        points_2d.push_back(it->second);
    }

    // 检查点数是否足够
    if (points_3d.size() < 10)
    {
        return false;
    }

    std::vector<int> status;

    try
    {
        // 使用OpenCV的solvePnPRansac函数来估计相机姿态
        cv::solvePnPRansac(points_3d, points_2d, intrinsic, cv::Mat(), cv_so3, cv_trans, false, 200, 1.5, 0.99, status); // SOLVEPNP_ITERATIVE
    }
    catch (cv::Exception &e)
    {
        // 捕获并处理可能的异常
        scope_color(ANSI_COLOR_RED_BOLD);
        std::cout << "Catching a cv exception: " << e.msg << std::endl;
        return 0;
    }

    if (if_remove_ourlier)
    {
        // 如果if_remove_outlier为真，则清空映射，并仅重新添加被RANSAC标记为内点的点
        // Remove outlier
        map_rgb_points_in_last_image_pose.clear();
        map_rgb_points_in_cur_image_pose.clear();

        for (unsigned int i = 0; i < status.size(); i++)
        {
            int inlier_idx = status[i];
            {
                map_rgb_points_in_last_image_pose[map_ptr[inlier_idx]] = points_2d[inlier_idx];
                map_rgb_points_in_cur_image_pose[map_ptr[inlier_idx]] = points_2d[inlier_idx];
            }
        }
    }

    // 更新跟踪点向量和索引
    updateLastTrackingVectorAndIds();

    return true;
}