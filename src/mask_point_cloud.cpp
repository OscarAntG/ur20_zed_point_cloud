#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

using namespace std::placeholders;

class MaskPointCloud : public rclcpp::Node 
{
public:
    MaskPointCloud() : Node("mask_point_cloud"){
        rclcpp::QoS mask_qos_profile(1);
        mask_qos_profile.transient_local();
        mask_qos_profile.reliable();
        mask_qos_profile.keep_last(1);

        mask_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("mask_point_cloud",10);
        isolated_mask_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("isolated_mask_cloud",10);
        mask_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/active_sam3_mask", mask_qos_profile, std::bind(&MaskPointCloud::callbackMaskImage, this, _1));
        cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zedx/zed_node/point_cloud/cloud_registered", 10, std::bind(&MaskPointCloud::callbackMaskPointCloud, this, _1));
    }
private:
    void callbackMaskImage(const sensor_msgs::msg::Image::SharedPtr message){
        try{
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(message, "mono8");
            latest_mask_ = cv_ptr->image;
            RCLCPP_INFO(this->get_logger(),"Received and saved segmentation mask.");
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(),"cv_bridge exception: %s", e.what());
        }
    }

    void callbackMaskPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr message){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr raw_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        try{
            pcl::fromROSMsg(*message,*raw_cloud);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to convert to PCL point cloud: %s", e.what());
            return;
        }

        // Safety checks
        if(latest_mask_.empty()){
            RCLCPP_WARN_THROTTLE(this->get_logger(),*this->get_clock(),2000, "Waiting for segmentation mask...");
            return;
        }
        if(!raw_cloud->isOrganized()){
            RCLCPP_ERROR_THROTTLE(this->get_logger(),*this->get_clock(),2000,"Cloud is not organized!");
            return;
        }
        cv::Mat aligned_mask;
        if(static_cast<uint32_t>(latest_mask_.cols) != raw_cloud->width || static_cast<uint32_t>(latest_mask_.rows) != raw_cloud->height){
            RCLCPP_INFO_ONCE(this->get_logger(), "Dimension mismatch! Mask = %dx%d | Cloud = %dx%d", latest_mask_.cols, latest_mask_.rows, raw_cloud->width, raw_cloud->height);
            cv::resize(latest_mask_, aligned_mask, cv::Size(raw_cloud->width, raw_cloud->height), 0,0, cv::INTER_NEAREST);
            RCLCPP_INFO_ONCE(this->get_logger(), "Updated Dimensions: Mask = %dx%d | Cloud = %dx%d", aligned_mask.cols, aligned_mask.rows, raw_cloud->width, raw_cloud->height);
        } else {
            aligned_mask = latest_mask_;
        }

        // Mask mapping
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr masked_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_mask_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

        for (int v = 0; v < aligned_mask.rows; ++v){
            for(int u = 0; u < aligned_mask.cols; ++u){
                if (aligned_mask.at<uchar>(v,u) > 127){
                    pcl::PointXYZRGB& pt = raw_cloud->at(u,v);
                    if(std::isfinite(pt.x) && std::isfinite(pt.y) && isfinite(pt.z)){
                        pt.r = 255;
                        pt.g = 0;
                        pt.b = 0;
                        // To publish mask only, uncomment line below.
                        masked_cloud->push_back(pt);
                        isolated_mask_cloud->push_back(pt);
                    }
                }
            }
        }

        *masked_cloud = *raw_cloud;
        masked_cloud->header = raw_cloud->header;
        isolated_mask_cloud->header = raw_cloud->header;
        
        // Optional PassThrough Filter: isolates region of cloud
        // pcl::PassThrough<pcl::PointXYZRGB> pass;
        // pass.setInputCloud(masked_cloud);
        // pass.setFilterFieldName("x");
        // pass.setFilterLimits(0.0, 1.25); // whiteboard = 0.0, 1.25
        // pass.filter(*masked_cloud);
        
        // Outlier removal
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(masked_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.75);
        sor.filter(*masked_cloud);

        // Voxel downsampling
        pcl::VoxelGrid<pcl::PointXYZRGB> vg;
        vg.setInputCloud(masked_cloud);
        vg.setLeafSize(0.01f, 0.01f, 0.01f);
        vg.filter (*masked_cloud);

        // Publish clouds
        sensor_msgs::msg::PointCloud2 mask_msg;
        pcl::toROSMsg(*masked_cloud, mask_msg);
        mask_msg.header = message->header;
        // mask_publisher_->publish(mask_msg);

        sensor_msgs::msg::PointCloud2 isolated_mask_msg;
        pcl::toROSMsg(*isolated_mask_cloud, isolated_mask_msg);
        isolated_mask_msg.header = message->header;
        isolated_mask_publisher_->publish(isolated_mask_msg);
    }

    cv::Mat latest_mask_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mask_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr isolated_mask_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mask_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscriber_;
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MaskPointCloud>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}