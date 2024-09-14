#include "rgbMapTracker.h"

rgbMapTracker::rgbMapTracker()
{
    // 储存 3D 点
	std::vector<point3D> v_point_temp;
    // 创建状态对象并初始化点云帧
	state *p_state_ = new state();
	p_cloud_frame = new cloudFrame(v_point_temp, p_state_);

    // 设置投影到图像平面的最小和最大深度阈值
	minimum_depth_for_projection = 0.1;
    maximum_depth_for_projection = 200;

    // 设置最近访问的体素激活的时间阈值
	recent_visited_voxel_activated_time = 1.0;

    // 初始化新访问体素的数量为0
	number_of_new_visited_voxel = 0;

    // 初始化更新的帧索引为-1
	updated_frame_index = -1;

    // 初始化是否在添加点的标志为false
	in_appending_points = false;

    // 设置用于投影的 RGB 向量
	points_rgb_vec_for_projection = nullptr;

    // 创建互斥锁
    mutex_rgb_points_vec = std::make_shared<std::mutex>();
    mutex_frame_index = std::make_shared<std::mutex>();
}

// 新用于投影的 RGB 点向量
void rgbMapTracker::refreshPointsForProjection(voxelHashMap &map)
{
    // 获取当前点云帧
	cloudFrame *p_frame = p_cloud_frame;

    // 检查图像的尺寸
	if (p_frame->image_cols == 0 || p_frame->image_rows == 0) return;

    // 检查帧的索引，如果与上一帧更新的帧相同，则不更新，直接返回
	if (p_frame->frame_id == updated_frame_index) return;

    // 创建一个临时RGB点向量，用于存储投影到当前帧的RGB点
	std::vector<rgbPoint*> *points_rgb_vec_for_projection_temp = new std::vector<rgbPoint*>();

    // 从体素映射中选择适合投影的点
	selectPointsForProjection(map, p_frame, points_rgb_vec_for_projection_temp, nullptr, 10.0, 1);

    // 赋值给 points_rgb_vec_for_projection
	points_rgb_vec_for_projection = points_rgb_vec_for_projection_temp;

    // 更新帧索引
    mutex_frame_index->lock();
	updated_frame_index = p_frame->frame_id;
    mutex_frame_index->unlock();
}

// 从体素映射中选择用于投影的 RGB 点
void rgbMapTracker::selectPointsForProjection(voxelHashMap &map, cloudFrame *p_frame, std::vector<rgbPoint*> *pc_out_vec, 
	std::vector<cv::Point2f> *pc_2d_out_vec, double minimum_dis, int skip_step, bool use_all_points)
{
    // 清空输出向量指针，避免重叠
	if (pc_out_vec != nullptr)
    {
        pc_out_vec->clear();
    }

    if (pc_2d_out_vec != nullptr)
    {
        pc_2d_out_vec->clear();
    }

    // 创建两个哈希映射，用于储存投影点的索引和深度
    Hash_map_2d<int, int> mask_index;
    Hash_map_2d<int, float> mask_depth;

    // 创建两个映射，分别用于存储投影点的索引和原始2D位置创建两个映射，分别用于存储投影点的索引和原始2D位置
    std::map<int, cv::Point2f> map_idx_draw_center;     // 投影点四舍五入后的像素坐标
    std::map<int, cv::Point2f> map_idx_draw_center_raw_pose;    // 投影点的原始像素坐标

    int u, v;
    double u_f, v_f;

    int acc = 0;
    int blk_rej = 0;

    // 储存所有的 RGB 点
    std::vector<rgbPoint*> points_for_projection;

    std::vector<voxelId> boxes_recent_hitted = voxels_recent_visited;

    if ((!use_all_points) && boxes_recent_hitted.size())
    {
        // 从最近访问的体素中选择点
    	for(std::vector<voxelId>::iterator it = boxes_recent_hitted.begin(); it != boxes_recent_hitted.end(); it++)
        {
            if (map[voxel((*it).kx, (*it).ky, (*it).kz)].NumPoints() > 0)
            {
                // 把映射中该点所在体素的最后一个点加入
                points_for_projection.push_back(&(map[voxel((*it).kx, (*it).ky, (*it).kz)].points.back()));
            }
        }
    }
    else
    {
        // 从全局 rgb 点向量中选择点
        mutex_rgb_points_vec->lock();
        points_for_projection = rgb_points_vec;
        mutex_rgb_points_vec->unlock();
    }

    // 用于投影的 RGB 点的数量
    int point_size = points_for_projection.size();

    // 遍历所有用于投影的点
    for (int point_index = 0; point_index < point_size; point_index += skip_step)
    {
        // 用于投影的 RGB 点的 3D 坐标
        Eigen::Vector3d point_world = points_for_projection[point_index]->getPosition();

        // 点的深度
        double depth = (point_world - p_frame->p_state->t_world_camera).norm();

        // 检查深度是否在范围内
        if (depth > maximum_depth_for_projection)
        {
            continue;
        }

        if (depth < minimum_depth_for_projection)
        {
            continue;
        }

        // 投影 3D 点到图像平面上
        bool res = p_frame->project3dPointInThisImage(point_world, u_f, v_f, nullptr, 1.0);

        if (res == false)
        {
            continue;
        }

        // 四舍五入的图像坐标
        u = std::round(u_f / minimum_dis) * minimum_dis;
        v = std::round(v_f / minimum_dis) * minimum_dis;

        // 如果当前点的深度小于已有的深度或当前点在掩码中不存在，则更新掩码索引和深度
        if ((!mask_depth.if_exist(u, v)) || mask_depth.m_map_2d_hash_map[u][v] > depth)
        {
            acc++;

            // 如果当前点在掩码中存在，则删除旧的索引，并更新新的索引和深度
            if (mask_index.if_exist(u, v))
            {
                int old_idx = mask_index.m_map_2d_hash_map[u][v];

                blk_rej++;

                map_idx_draw_center.erase(map_idx_draw_center.find(old_idx));
                map_idx_draw_center_raw_pose.erase(map_idx_draw_center_raw_pose.find(old_idx));
            }

            mask_index.m_map_2d_hash_map[u][v] = (int)point_index;
            mask_depth.m_map_2d_hash_map[u][v] = (float)depth;

            map_idx_draw_center[point_index] = cv::Point2f(v, u);
            map_idx_draw_center_raw_pose[point_index] = cv::Point2f(u_f, v_f);
        }
    }

    // 将选中的点及其 2D 投影坐标添加到输出向量中
    if (pc_out_vec != nullptr)
    {
        for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++)
            pc_out_vec->push_back(points_for_projection[it->first]);
    }

    if (pc_2d_out_vec != nullptr)
    {
        for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++)
            pc_2d_out_vec->push_back(map_idx_draw_center_raw_pose[it->first]);
    }
}

// 更新当前云帧的状态对象，以准备进行投影操作
void rgbMapTracker::updatePoseForProjection(cloudFrame *p_frame, double fov_margin)
{
    // 更新相机的内参
	p_cloud_frame->p_state->fx = p_frame->p_state->fx;
	p_cloud_frame->p_state->fy = p_frame->p_state->fy;
	p_cloud_frame->p_state->cx = p_frame->p_state->cx;
	p_cloud_frame->p_state->cy = p_frame->p_state->cy;

    // 更新图像的行列
	p_cloud_frame->image_cols = p_frame->image_cols;
	p_cloud_frame->image_rows = p_frame->image_rows;

    // 设置视野边距
	p_cloud_frame->p_state->fov_margin = fov_margin;
    // 设置帧 ID
	p_cloud_frame->frame_id = p_frame->frame_id;

    // 更新相机的姿态，旋转和平移
	p_cloud_frame->p_state->q_world_camera = p_frame->p_state->q_world_camera;
	p_cloud_frame->p_state->t_world_camera = p_frame->p_state->t_world_camera;

    // 更新彩色图像和灰度图像
	p_cloud_frame->rgb_image = p_frame->rgb_image;
	p_cloud_frame->gray_image = p_frame->gray_image;

    // 根据当前的相机姿态和图像尺寸更新投影姿态
    p_cloud_frame->refreshPoseForProjection();
}

// 图像观测噪声协方差
const double image_obs_cov = 15;
// 过程标准差
const double process_noise_sigma = 0.1;

// 原子，储存渲染点的计数
std::atomic<long> render_point_count;

// 在体素映射中渲染点
void rgbMapTracker::threadRenderPointsInVoxel(voxelHashMap &map,        // 3D 空间的体素映射
                                              const int &voxel_start,   // 开始渲染的体素索引
                                              const int &voxel_end,     // 结束渲染的体素索引
                                              cloudFrame *p_frame,      // 当前帧的信息
                                              const std::vector<voxelId> *voxels_for_render,    // 需要渲染的体素 ID
                                              const double obs_time)    // 观测时间，用于更新点的颜色信息
{
    // 三维坐标
	Eigen::Vector3d point_world;
	Eigen::Vector3d point_color;

    // 像素坐标
	double u, v;
	double point_camera_norm;

    // 遍历指定范围内的体素
	for (int voxel_index = voxel_start; voxel_index < voxel_end; voxel_index++)
	{
        // 获取对应的体素块
        voxelBlock &voxel_block = map[voxel((*voxels_for_render)[voxel_index].kx, (*voxels_for_render)[voxel_index].ky, (*voxels_for_render)[voxel_index].kz)];

        // 遍历体素块中的点
        for (int point_index = 0; point_index < voxel_block.NumPoints(); point_index++)
        {
        	auto &point = voxel_block.points[point_index];
            // 点的三维坐标
        	point_world = point.getPosition();

            // 投影到当前帧
        	if (p_frame->project3dPointInThisImage(point_world, u, v, nullptr, 1.0) == false) continue;

            // 点到相机的距离
        	point_camera_norm = (point_world - p_frame->p_state->t_world_camera).norm();

            // 获取点在当前帧的颜色信息
        	point_color = p_frame->getRgb(u, v, 0);

            // 更新点的颜色信息
            mutex_rgb_points_vec->lock();
        	if (voxel_block.points[point_index].updateRgb(point_color, point_camera_norm, 
        		Eigen::Vector3d(image_obs_cov, image_obs_cov, image_obs_cov), obs_time))
        	{
        		render_point_count++;
        	}
            mutex_rgb_points_vec->unlock();
        }
	}
}

// 全局体素 ID 向量
std::vector<voxelId> g_voxel_for_render;

// 在体素映射中渲染点
void rgbMapTracker::renderPointsInRecentVoxel(voxelHashMap &map,        // 3D 空间体素映射
                                              cloudFrame *p_frame,      // 当前帧
                                              std::vector<voxelId> *voxels_for_render,      // 需要渲染的体素 ID
                                              const double &obs_time)   // 观测时间，可能用于更新点的颜色信息
{
    // 清空全局体素 ID 向量
	g_voxel_for_render.clear();
    std::vector<voxelId>().swap(g_voxel_for_render);

    // 将需要渲染的体素 ID 加入到全局体素 ID
	for (std::vector<voxelId>::iterator it = (*voxels_for_render).begin(); it != (*voxels_for_render).end(); it++)
	{
		g_voxel_for_render.push_back(*it);
	}

    // 创建一个std::vector，用于存储异步任务std::future的引用。这些任务将在后续的cv::parallel_for_循环中执行
	std::vector<std::future<double>> results;

    // 需要渲染的体素数量
	int number_of_voxels = g_voxel_for_render.size();

	render_point_count = 0;

    // 使用OpenCV的并行循环渲染体素
	cv::parallel_for_(cv::Range(0, number_of_voxels), [&](const cv::Range &r)
					{ threadRenderPointsInVoxel(map, r.start, r.end, p_frame, &g_voxel_for_render, obs_time); });
}