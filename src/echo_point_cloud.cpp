#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>


using namespace std::placeholders;

class EchoPointCloud : public rclcpp::Node
{
public:
    EchoPointCloud() : Node("echo_point_cloud")
    {
        raw_subcriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zedx/zed_node/point_cloud/cloud_registered", 10, std::bind(&EchoPointCloud::callbackRepubPointCloud, this, _1));
        processed_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("processed_point_cloud", 10);
        RCLCPP_INFO(this->get_logger(), "Echo Point Cloud node has started!");
    }

private:
    void callbackRepubPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr message){
        // Converting PointCloud2 interface to pcl pcl::PointCloud's PointXYZ
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        try{
            pcl::fromROSMsg(*message, *cloud);
        } catch (const std::exception & e){
            RCLCPP_ERROR(this->get_logger(), "Failed to convert PointCloud2 to PCL: %s", e.what());
            return;
        }
        RCLCPP_DEBUG(this->get_logger(), "Received cloud with %zu points", cloud->size());

        // Voxel Downsampling (voxel leaf size = 1cm)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsample = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(0.01f, 0.01f, 0.01f);
        vg.filter(*cloud_downsample);

        // Plane Segmentation - RANSAC Model Configuration
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(cloud_downsample);
        seg.segment(*inliers, *coefficients);

        // Optional verification and logging, {Begin
        if(inliers->indices.empty()){
            RCLCPP_WARN(this->get_logger(), "No inliers found, publishing downsampled cloud.");
            sensor_msgs::msg::PointCloud2 empty_msg;
            pcl::toROSMsg(*cloud_downsample, empty_msg);
            empty_msg.header = message->header;
            processed_publisher_->publish(empty_msg);
        }

        RCLCPP_INFO(this->get_logger(), "Plane found with %zu inliers.", inliers->indices.size());
        RCLCPP_DEBUG(this->get_logger(), "Plane coefficients: [%f, %f, %f, %f]",
                 coefficients->values.size() >= 4 ? coefficients->values[0] : 0.0,
                 coefficients->values.size() >= 4 ? coefficients->values[1] : 0.0,
                 coefficients->values.size() >= 4 ? coefficients->values[2] : 0.0,
                 coefficients->values.size() >= 4 ? coefficients->values[3] : 0.0);
        // End}
        
        // Plane Segmentation - Extracting Inliers
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_downsample);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*cloud_plane);

        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*cloud_plane, msg);
        msg.header = message->header;
        processed_publisher_->publish(msg);
        // auto msg = sensor_msgs::msg::PointCloud2();
        // msg.header = message->header;
        // msg.width = message->width;
        // msg.height = message->height;
        // msg.fields = message->fields;
        // msg.is_bigendian = message->is_bigendian;
        // msg.point_step = message->point_step;
        // msg.row_step = message->row_step;
        // msg.data = message->data;
        // msg.is_dense = message->is_dense;
        // RCLCPP_INFO(this->get_logger(), "Received cloud of width %u, height=%u", msg.width, msg.height);
        // processed_publisher_->publish(msg);
    }
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr raw_subcriber_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr processed_publisher_;

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EchoPointCloud>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
