// ROS includes.
#define _USE_MATH_DEFINES
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include <signal.h>
#include <chrono>
#include <memory>

// Messages
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
//#include "rosgraph_msgs/msg/Clock.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include "nav_msgs/msg/occupancy_grid.hpp"

// Service
#include "environment_controller/srv/use_key.hpp"
using UseKey = environment_controller::srv::UseKey;
//using UseKey = environment_controller.srv.UseKey;

#include <sstream>
#include <vector>
#include <iostream>
#include <tuple>
#include <random>
#include <queue>
#include <map>
#include <cmath>
#include <algorithm>

const int BLOCKED = 1;
const int OPEN = 0;
const int DOOR = -1;
const int GOAL = 2;
const int OPEN_DOOR = -2;

const double DOOR_THRESH = 1.5;  // small buffer for distance

double drone_x = std::numeric_limits<double>::infinity();
double drone_y = std::numeric_limits<double>::infinity();
double drone_yaw = 0;
std::vector<std::vector<int>> map;
int key_count;
int map_width, map_height, seed, goal_x, goal_y;
unsigned int door_count;
bool crashed = false;
std::vector<std::tuple<int, int>> final_path;
std::vector<std::tuple<int, int>> shortest_path;


void printMap(int sx, int sy) {
    std::cout << std::endl;
    char output = ' ';
    for (int j = map_height - 1; j >= 0; j--) {
        for (int i = 0; i < map_width; i++) {
            output = ' ';
            if (i == sx && j == sy) {
                output = '*';
            } else if (i == goal_x && j == goal_y) {
                output = 'G';
            } else if (map[i][j] == BLOCKED) {
                output = '#';
            } else if (map[i][j] == DOOR) {
                output = 'D';
            } else if (std::find(shortest_path.begin(), shortest_path.end(), std::tuple<int, int>(i, j)) != shortest_path.end()) {
                output = '.';
            }
            std::cout << output;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void get_neighbors(std::vector<std::vector<int>>& map,
                   std::vector<std::tuple<int, int>>& neighbors,
                   int map_width, int map_height,
                   int cur_x, int cur_y, int value) {
        if (cur_x > 2 && map[cur_x - 2][cur_y] == value) {
            neighbors.push_back(std::tuple<int, int>(cur_x - 2, cur_y));
        }
        if (cur_y > 2 && map[cur_x][cur_y - 2] == value) {
            neighbors.push_back(std::tuple<int, int>(cur_x, cur_y - 2));
        }
        if (cur_x < map_width - 2 && map[cur_x + 2][cur_y] == value) {
            neighbors.push_back(std::tuple<int, int>(cur_x + 2, cur_y));
        }
        if (cur_y < map_height - 2 && map[cur_x][cur_y + 2] == value) {
            neighbors.push_back(std::tuple<int, int>(cur_x, cur_y + 2));
        }
}

void gen_map(std::default_random_engine& generator, std::vector<std::vector<int>> &map,
             int map_width, int map_height, int &goal_x, int &goal_y,
             std::vector<std::tuple<int, int>> &shortest_path, unsigned int door_count) {
    // always start at the center
    int start_x = map_width / 2;
    int start_y = map_height / 2;
    int cur_x = start_x;
    int cur_y = start_y;
    std::vector<std::tuple<int, int>> frontier;
    // set frontier to open (it may already be open)
    map[cur_x][cur_y] = OPEN;
    // initialize the frontier
    get_neighbors(map, frontier, map_width, map_height, cur_x, cur_y, BLOCKED);
    while(frontier.size() > 0) { // loop until we run out of frontier
        // pick a random point on the frontier
        std::uniform_int_distribution<int> frontier_picker(0, frontier.size() - 1);
        int index = frontier_picker(generator);
        std::tuple<int, int> next_cell = frontier[index];
        frontier.erase(frontier.begin() + index);
        cur_x = std::get<0>(next_cell);
        cur_y = std::get<1>(next_cell);
        // if the frontier position is already open, we should not add another bypass
        if (map[cur_x][cur_y] == OPEN) {
            continue;
        }
        // get neighborhood
        std::vector<std::tuple<int, int>> neighbors;
        get_neighbors(map, neighbors, map_width, map_height, cur_x, cur_y, OPEN);
        if (neighbors.size() == 0) {
            continue;
        }
        std::uniform_int_distribution<int> neighbor_picker(0, neighbors.size() - 1);
        index = neighbor_picker(generator);
        next_cell = neighbors[index];
        int orig_x = std::get<0>(next_cell);
        int orig_y = std::get<1>(next_cell);
        // set the cur cell to open (this is the one on the frontier
        map[cur_x][cur_y] = OPEN;
        // remove the wall between them
        int between_x = (cur_x + orig_x) / 2;
        int between_y = (cur_y + orig_y) / 2;
        map[between_x][between_y] = OPEN;
        get_neighbors(map, frontier, map_width, map_height, cur_x, cur_y, BLOCKED);
    }
    // generate goal
    std::uniform_int_distribution<int> goal_picker(0, 3);
    int which_goal = goal_picker(generator);
    goal_x = (which_goal % 2 == 0) ? 1 : map_width - 2;
    goal_y = (which_goal < 2) ? 1 : map_height - 2;
    // run A-star to find a path to the goal
    std::priority_queue<std::tuple<int, int, int>> a_star_frontier;
    std::map<std::tuple<int, int>, std::tuple<int, int>> came_from;
    std::map<std::tuple<int, int>, int> cost_so_far;
    std::tuple<int, int> start_tuple = std::tuple<int, int>(start_x, start_y);
    cost_so_far[start_tuple] = 0;
    a_star_frontier.push(std::tuple<int, int, int>(0, start_x, start_y));
    while (!a_star_frontier.empty()) {
        std::tuple<int, int, int> frontier_pop = a_star_frontier.top();
        std::tuple<int, int> frontier_point = std::tuple<int, int>(std::get<1>(frontier_pop), std::get<2>(frontier_pop));
        a_star_frontier.pop();
        cur_x = std::get<0>(frontier_point);
        cur_y = std::get<1>(frontier_point);
        if (cur_x == goal_x && cur_y == goal_y) {
            break;
        }
        int current_cost = cost_so_far[frontier_point];
        int neighbor_x, neighbor_y, priority;
        std::tuple<int, int> neighbor;
        for (int which = 0; which < 4; which++) {
            int i = ((which % 2 == 0) ? 1 : 0) * ((which < 2) ? -1 : 1);
            int j = ((which % 2 == 0) ? 0 : 1) * ((which < 2) ? -1 : 1);
            if (map[cur_x + i][cur_y + j] != OPEN) {
                // if the map is not open in this direction, this is not a neighbor
                continue;
            }
            // we actually move by 2 spaces
            neighbor_x = cur_x + i*2;
            neighbor_y = cur_y + j*2;
            neighbor = std::tuple<int, int>(neighbor_x, cur_y + j*2);
            int new_cost = current_cost + 1; // all actions are cost 1 in the grid
            if (cost_so_far.count(neighbor) == 0 || new_cost < cost_so_far[neighbor]) {
                cost_so_far[neighbor] = new_cost;
                priority = new_cost + std::abs(goal_x - neighbor_x) + std::abs(goal_y - neighbor_y);
                // priority is negative below because the
                a_star_frontier.push(std::tuple<int, int, int>(-priority, neighbor_x, neighbor_y));
                came_from[neighbor] = frontier_point;
            }
        }
    }
    // build shortest path
    std::tuple<int, int> working_point = std::tuple<int, int>(goal_x, goal_y);
    shortest_path.clear();
    while (working_point != start_tuple) {
        shortest_path.insert(shortest_path.begin(), working_point);
        working_point = came_from[working_point];
    }
    shortest_path.insert(shortest_path.begin(), start_tuple);
    // add doors to the shortest path
    std::vector<int> possible_door_locations;
    for (unsigned int i = 0; i < shortest_path.size() - 1; i++) {
//        std::cout << "Shortest Path: " << std::get<0>(shortest_path[i]) << ", " << std::get<1>(shortest_path[i]) << std::endl;
        possible_door_locations.push_back(i);
    }
    std::shuffle(std::begin(possible_door_locations), std::end(possible_door_locations), generator);
    door_count = (door_count < possible_door_locations.size()) ? door_count : possible_door_locations.size();
    for (unsigned int i = 0; i < door_count; i++) {
        int door_index = possible_door_locations[i];
        std::tuple<int, int> point1 = shortest_path[door_index];
        std::tuple<int, int> point2 = shortest_path[door_index + 1];
        int door_x = (std::get<0>(point1) + std::get<0>(point2)) / 2;
        int door_y = (std::get<1>(point1) + std::get<1>(point2)) / 2;
//        std::cout << "New Door: " << door_x << ", " << door_y << std::endl;
        map[door_x][door_y] = DOOR;
    }
    map[goal_x][goal_y] = GOAL;
}

void finalPathCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg) {
    final_path.clear();
    for(std::vector<int>::const_iterator it = msg->data.begin(); it != msg->data.end();) {
        int x = *it + map_width / 2.0;
        it++; // move to next element
        int y = *it + map_height / 2.0;
        it++; // move to next element
        final_path.push_back(std::tuple<int, int>(x, y));
    }
}

void gpsCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    drone_x = msg->pose.position.x;
    drone_y = msg->pose.position.y;
    tf2::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y,
                     msg->pose.orientation.z, msg->pose.orientation.w);
    tf2::Matrix3x3 m(q);
//    tf2::Quaternion q_tf;
//    tf2::fromMsg(msg->pose.orientation, q_tf);        // convert ROS msg -> tf2 quaternion
    double roll, pitch, yaw;
//    tf2::Matrix3x3(q_tf).getRPY(roll, pitch, yaw);
    m.getRPY(roll, pitch, yaw);
    drone_yaw = yaw;
//    std::cout << "Got GPS: " << drone_x << ", " << drone_y << std::endl;
}

void use_key_func(  const std::shared_ptr<rmw_request_id_t> request_header,
               const std::shared_ptr<UseKey::Request> req,
               const std::shared_ptr<UseKey::Response> res) {
//(environment_controller::use_key::Request  &req,
//         environment_controller::use_key::Response &res) {
    res->success = false;
    std::cout << "Door request: " << req->door_loc.x << ", " << req->door_loc.y << std::endl;
    std::cout << "   Drone loc: " << (drone_x) << ", " << (drone_y) << std::endl;
    std::cout << "Keys remaining: " << key_count << std::endl;
    if (key_count > 0 && !crashed) {
      key_count -= 1;
      double dist_to_request = std::hypot(req->door_loc.x - drone_x, req->door_loc.y - drone_y);
//      std::cout << "Dist to request: " << dist_to_request << std::endl;
      if (dist_to_request < DOOR_THRESH) {
          double map_door_x = req->door_loc.x + map_width / 2.0;
          double map_door_y = req->door_loc.y + map_height / 2.0;
//          std::cout << "map door: " << map_door_x << ", " << map_door_y << std::endl;
//          std::cout << "map: " << map[(int) map_door_x][(int) map_door_y] << std::endl;
          if (map[(int) map_door_x][(int) map_door_y] == DOOR) {
            res->success = true;
            map[(int) map_door_x][(int) map_door_y] = OPEN_DOOR;
          } else {
            res->success = (request_header->sequence_number == 0) && false;
          }
      }
    }
}

void shutdown_hook() {
    std::cout << "environment_controller_node shutting down. Printing full map to console:" << std::endl;
    printMap(-1, -1);
    rclcpp::shutdown();
}

void generate_map(std::default_random_engine& generator, int seed) {
    generator.seed(seed);
    map.clear();
    for (int i = 0; i < map_width; i++) {
        map.push_back(std::vector<int>(map_height, BLOCKED));
    }
    gen_map(generator, map, map_width, map_height, goal_x, goal_y, shortest_path, door_count);
}


//class EnvironmentController {
//    public:
//        EnvironmentController(std::shared_ptr<rclcpp::Node> nh);
//        std::shared_ptr<rclcpp::Node> node_;

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    rclcpp::on_shutdown(&shutdown_hook);

    std::shared_ptr<rclcpp::Node> nh = rclcpp::Node::make_shared("flightgoggles_uav_dynamics_node");
    std::shared_ptr<rclcpp::Node> node_(nh);
    const unsigned int lidar_count = 16;
    nh->declare_parameter("map_width", map_width);
    map_width = nh->get_parameter("map_width").get_value<int>();
    nh->declare_parameter("map_height", map_height);
    map_height = nh->get_parameter("map_height").get_value<int>();
    if (map_width % 4 != 3 || map_height % 4 != 3) {
        RCLCPP_ERROR(node_->get_logger(), "Map width and height must both be 1 less than a multiple of 4, received parameters of %d by %d", map_width, map_height);
        rclcpp::shutdown();
        return 1;
    }
    if (map_width < 11 || map_height < 11) {
        RCLCPP_ERROR(node_->get_logger(), "Minimum map size is 11 by 11, received parameters of %d by %d", map_width, map_height);
        rclcpp::shutdown();
        return 1;
    }
    nh->declare_parameter("seed", seed);
    seed = nh->get_parameter("seed").get_value<int>();
    door_count = 3;
    key_count = 4;
    double lidar_resolution = 0.01;
    double rate = 20;
    double lidar_min_range = 0.0;
    double lidar_max_range = 5.0; // 5
    double lidar_angle_min = M_PI / 7;  // something completely off of anything you will hit accidentally
    int lidar_max_range_count = (lidar_max_range / lidar_resolution);
    std::default_random_engine generator;
    generate_map(generator, seed);

    rclcpp::Rate loop_rate(rate);
    auto service = node_->create_service<UseKey>("use_key", use_key_func);

    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::LaserScan>> lidar_pub = node_->create_publisher<sensor_msgs::msg::LaserScan>("/uav/sensors/lidar", 1);
    std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::Point>> goal_pub = node_->create_publisher<geometry_msgs::msg::Point>("/cell_tower/position", 1);
    std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Bool>> crash_pub = node_->create_publisher<std_msgs::msg::Bool>("/uav/sensors/crash_detector", 1);
    std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Int32>> keys_remaining_pub = node_->create_publisher<std_msgs::msg::Int32>("/keys_remaining", 1);

    std::shared_ptr<rclcpp::Subscription<geometry_msgs::msg::PoseStamped>> gps_sub = node_->create_subscription<geometry_msgs::msg::PoseStamped>("/uav/sensors/gps", 1, gpsCallback);
    std::shared_ptr<rclcpp::Subscription<std_msgs::msg::Int32MultiArray>> final_path_sub = node_->create_subscription<std_msgs::msg::Int32MultiArray>("/uav/final_path", 1, finalPathCallback);

    tf2_ros::TransformBroadcaster br(
                                    nh->get_node_parameters_interface(),
                                    nh->get_node_topics_interface(),
                                    rclcpp::QoS(rclcpp::KeepLast(100)),
                                    rclcpp::PublisherOptionsWithAllocator<std::allocator<void>>()
                                    );
    tf2::Transform celltower_transform;
    std::uniform_real_distribution<double> pos_dist(-1000, 1000);
    std::uniform_real_distribution<double> angle_dist(0, 2 * M_PI);
    celltower_transform.setOrigin(tf2::Vector3(pos_dist(generator), pos_dist(generator), pos_dist(generator)) );
    tf2::Quaternion q;
    q.setRPY(0, 0, angle_dist(generator));
    celltower_transform.setRotation(q);
    geometry_msgs::msg::TransformStamped stamped_transform;
    stamped_transform.header.stamp = node_->get_clock()->now();
    stamped_transform.header.frame_id = "world";
    stamped_transform.child_frame_id = "cell_tower";
    stamped_transform.transform.translation.x = (double) celltower_transform.getOrigin().x();
    stamped_transform.transform.translation.y = (double) celltower_transform.getOrigin().y();
    stamped_transform.transform.translation.z = (double) celltower_transform.getOrigin().z();
    stamped_transform.transform.rotation.x = (double) celltower_transform.getRotation().x();
    stamped_transform.transform.rotation.y = (double) celltower_transform.getRotation().y();
    stamped_transform.transform.rotation.z = (double) celltower_transform.getRotation().z();
    stamped_transform.transform.rotation.w = (double) celltower_transform.getRotation().w();
    geometry_msgs::msg::Point goal_in_world, goal_in_cell;
    goal_in_world.x = goal_x - map_width / 2.0;
    goal_in_world.y = goal_y - map_height / 2.0;
    goal_in_world.z = 3.0;
    tf2::doTransform(goal_in_world, goal_in_cell, stamped_transform);

    double ranges[lidar_count];
    double intensities[lidar_count];
    double angles[lidar_count];
    double cosines[lidar_count];
    double sines[lidar_count];
    for(unsigned int i = 0; i < lidar_count; i++){
        angles[i] = lidar_angle_min + i * 2 * M_PI / lidar_count;// + M_PI / 2;
        cosines[i] = std::cos(angles[i]) * lidar_resolution;
        sines[i] = std::sin(angles[i]) * lidar_resolution;
    }
    std::normal_distribution<double> lidar_noise_distribution(0, lidar_resolution / 2);
    rclcpp::Time start_time = node_->get_clock()->now();
    long tick = 0;
    while (rclcpp::ok()) {
        std_msgs::msg::Int32 key_msg;
        key_msg.data = key_count;
        keys_remaining_pub->publish(key_msg);
        tick += 1;
        rclcpp::Time loop_start_time = node_->get_clock()->now();

        stamped_transform.header.stamp = node_->get_clock()->now();
        br.sendTransform(stamped_transform);
        if (!std::isinf(drone_x)) {
            double map_drone_x = drone_x + map_width / 2.0;
            double map_drone_y = drone_y + map_height / 2.0;
            int map_drone_round_x = (int) (map_drone_x);
            int map_drone_round_y = (int) (map_drone_y);
            int map_val = map[map_drone_round_x][map_drone_round_y];
            if (map_val == GOAL) {
                if (final_path.size() > 0) {
                    for (unsigned int i = 1; i < final_path.size() - 1; i++) {
                        int avg_x = (std::get<0>(final_path[i - 1]) + std::get<0>(final_path[i + 1])) / 2;
                        int avg_y = (std::get<1>(final_path[i - 1]) + std::get<1>(final_path[i + 1])) / 2;
                        if ((avg_x == std::get<0>(final_path[i]) && avg_y == std::get<1>(final_path[i])) &&
                             (avg_x % 2 == 0 || avg_y % 2 == 0)) {
                            final_path.erase(final_path.begin() + i);
                        }
                    }
                    bool good_path = shortest_path.size() == final_path.size();
                    if (good_path) {

                        for (unsigned int i = 0; i < shortest_path.size(); i++) {
                            good_path &= std::get<0>(shortest_path[i]) == std::get<0>(final_path[i]);
                            good_path &= std::get<1>(shortest_path[i]) == std::get<1>(final_path[i]);
                        }
                    }
                    if (good_path) {
                        RCLCPP_INFO(node_->get_logger(), "GOAL REACHED!!! Final path was correct. Time taken: %0.3f seconds", (node_->get_clock()->now() - start_time).seconds());
                    } else {
                        RCLCPP_INFO(node_->get_logger(), "Goal reached in %0.3f seconds, but final path was incorrect.", (node_->get_clock()->now() - start_time).seconds());
                    }
                    rclcpp::shutdown();
                } else {
                    if (tick % 10 == 0) {
                        RCLCPP_INFO(node_->get_logger(), "Goal reached, waiting for final path on /uav/final_path");
                    }
                }
            }
            if (map_val != OPEN &&
                map_val != GOAL &&
                map_val != OPEN_DOOR) {
                crashed = true;
            }
            if (crashed) {
                std_msgs::msg::Bool crash_msg;
                crash_msg.data = crashed;
                crash_pub->publish(crash_msg);
                std::cout << "Crashed!" << std::endl;
            } else {
                // send goal
                goal_pub->publish(goal_in_cell);
                // Generate LIDAR Data
                for(unsigned int i = 0; i < lidar_count; i++){
                    double sim_x = map_drone_x;
                    double sim_y = map_drone_y;
                    double dist_count = 0;
                    while (map[(int) sim_x][(int) sim_y] != DOOR && map[(int) sim_x][(int) sim_y] != BLOCKED && dist_count <= lidar_max_range_count) {
                        sim_x += cosines[i];
                        sim_y += sines[i];
                        dist_count += 1;
                    }
                    double dist = dist_count * lidar_resolution;
                    if (dist < lidar_min_range) {
                        ranges[i] = 0;
                    } else if (dist > lidar_max_range) {
                        ranges[i] = std::numeric_limits<double>::infinity();
                    } else {
                        double noise = lidar_noise_distribution(generator);
                        if (map[(int) sim_x][(int) sim_y] == DOOR) {
                            noise *= 10;  // doors have additional noise
                        }
                        ranges[i] = dist + noise;
                    }
                    intensities[i] = 1.0;
                }

                // Publish LIDAR
                rclcpp::Time scan_time = node_->get_clock()->now(); //TODO convert me
                sensor_msgs::msg::LaserScan scan;
                scan.header.stamp = scan_time;
                scan.header.frame_id = "drone";
                scan.angle_increment = 2 * M_PI / lidar_count;
                scan.angle_min = lidar_angle_min;
                scan.angle_max = lidar_angle_min + 2 * M_PI - scan.angle_increment;
                scan.time_increment = (1 / rate);
                scan.range_min = lidar_min_range;
                scan.range_max = lidar_max_range;

                scan.ranges.resize(lidar_count);
                scan.intensities.resize(lidar_count);
                for(unsigned int i = 0; i < lidar_count; ++i){
                    scan.ranges[i] = ranges[i];
                    scan.intensities[i] = intensities[i];
                }
                lidar_pub->publish(scan);
            }
        }
        rclcpp::spin_all(node_, std::chrono::seconds(0));
        loop_rate.sleep();
    }
    return 0;
}
