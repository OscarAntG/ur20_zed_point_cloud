#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/voxel_grid.h>

using namespace std::placeholders;

class StatsPointCloud : public rclcpp::Node
{
public:
    StatsPointCloud() : Node("stats_point_cloud")
    {
        downsample_subcriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/processed_point_cloud", 10, std::bind(&StatsPointCloud::callbackPointCloud, this, _1));
        mean_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("mean_point_cloud", 10);
        median_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("median_point_cloud", 10);
        std_dev_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("std_dev_cloud", 10);
        RCLCPP_INFO(this->get_logger(), "Point Cloud stats node has started!");
    }

private:
    void callbackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr message){
        if (sample_count_ >= MAX_SAMPLES){
            return;
        }
        
        // Converting PointCloud2 interface to pcl pcl::PointCloud's PointXYZ
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        try{
            pcl::fromROSMsg(*message, *pcl_cloud);
        } catch (const std::exception & e){
            RCLCPP_ERROR(this->get_logger(), "Failed to convert PointCloud2 to PCL: %s", e.what());
            return;
        }
        RCLCPP_DEBUG(this->get_logger(), "Received cloud with %zu points", pcl_cloud->size());

        // Store clouds into samples vector and calculate mean cloud.
        samples_.push_back(pcl_cloud);
        sample_count_++;

        RCLCPP_INFO(this->get_logger(), "Received cloud #%zu/%zu", sample_count_, MAX_SAMPLES);
        
        if (sample_count_ == MAX_SAMPLES){
            RCLCPP_INFO(this->get_logger(), "Received %zu point clouds, calculating mean...", MAX_SAMPLES);
            
            size_t N = samples_.size();
            size_t M = samples_[0]->points.size();

            mean_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            mean_cloud_->points.resize(M);

            median_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            median_cloud_->points.resize(M);

            std_dev_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            std_dev_cloud_->points.resize(M);

            std::vector<float> sigma_values(M);

            for(size_t i = 0; i < M; i++){
                //  Mean cloud
                float sxM = 0, syM = 0, szM = 0;
                for(size_t k = 0; k < N; k++){
                    const auto& p = samples_[k]->points[i];
                    sxM += p.x;
                    syM += p.y;
                    szM += p.z;
                }
                mean_cloud_->points[i].x = sxM/N;
                mean_cloud_->points[i].y = syM/N;
                mean_cloud_->points[i].z = szM/N;

                //  Median cloud
                std::vector<float> sxMd(N), syMd(N), szMd(N);
                for(size_t k = 0; k < N; k++){
                    sxMd[k] = samples_[k]->points[i].x;
                    syMd[k] = samples_[k]->points[i].y;
                    szMd[k] = samples_[k]->points[i].z;
                }

                std::nth_element(sxMd.begin(), sxMd.begin() + N/2, sxMd.end());
                std::nth_element(syMd.begin(), syMd.begin() + N/2, syMd.end());
                std::nth_element(szMd.begin(), szMd.begin() + N/2, szMd.end());
                // float median_x = sxMd[N/2];
                // float median_y = syMd[N/2];
                float median_z = szMd[N/2];
                median_cloud_->points[i].x = sxMd[N/2];
                median_cloud_->points[i].y = syMd[N/2];
                median_cloud_->points[i].z = szMd[N/2];

                //  Standard Deviation Cloud
                float accum = 0.0f;
                for(size_t k = 0; k < N; k++){
                    float z_k = samples_[k]->points[i].z;
                    float diff = z_k - median_z;
                    accum += diff * diff;
                }
                float variance = accum / static_cast<float>(N);
                float sigma = std::sqrt(variance);
                sigma_values[i] = sigma;                
            }

            float min_sigma = *std::min_element(sigma_values.begin(), sigma_values.end());
            float max_sigma = *std::max_element(sigma_values.begin(), sigma_values.end());
            for (size_t i = 0; i < M; i++){
                
                float sigma = sigma_values[i];

                std_dev_cloud_->points[i].x = median_cloud_->points[i].x;
                std_dev_cloud_->points[i].y = median_cloud_->points[i].y;
                std_dev_cloud_->points[i].z = median_cloud_->points[i].z;

                uint32_t rgb = jetColorMap(sigma, min_sigma, max_sigma);
                std_dev_cloud_->points[i].rgb = *reinterpret_cast<float*>(&rgb);
                
            }
            mean_cloud_->width = M;
            mean_cloud_->height = 1;
            mean_cloud_->is_dense = true;

            median_cloud_->width = M;
            median_cloud_->height = 1;
            median_cloud_->is_dense = true;

            std_dev_cloud_->width = M;
            std_dev_cloud_->height = 1;
            std_dev_cloud_->is_dense = true;
            
            RCLCPP_INFO(this->get_logger(), "Mean point cloud calculated!");
            RCLCPP_INFO(this->get_logger(), "Median point cloud calculated!");
            RCLCPP_INFO(this->get_logger(), "Standard Deviation Cloud has been generated.");
        }
  
        // Publish
        if(!mean_cloud_){
            return;
        }
        if(!median_cloud_){
            return;
        }
        
        sensor_msgs::msg::PointCloud2 msgMean;
        pcl::toROSMsg(*mean_cloud_, msgMean);
        msgMean.header = message->header;
        mean_publisher_->publish(msgMean);

        sensor_msgs::msg::PointCloud2 msgMedian;
        pcl::toROSMsg(*median_cloud_, msgMedian);
        msgMedian.header = message->header;
        median_publisher_->publish(msgMedian);

        sensor_msgs::msg::PointCloud2 msgStdDev;
        msgStdDev.fields.clear();
        sensor_msgs::PointCloud2Modifier modifier(msgStdDev);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        modifier.resize(std_dev_cloud_->points.size());
        pcl::toROSMsg(*std_dev_cloud_, msgStdDev);
        msgStdDev.header = message->header;
        std_dev_publisher_->publish(msgStdDev);
    }

    uint32_t jetColorMap(float value, float min_val, float max_val){
        float ratio = 2 * (value - min_val) / (max_val - min_val);
        uint8_t r = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, 255 * (ratio - 1))));
        uint8_t g = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, 255 * (1 - std::abs(ratio - 1)))));
        uint8_t b = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, 255 * (1 - ratio))));

        return (r << 16) | (g << 8) | b;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr downsample_subcriber_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr mean_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr median_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr std_dev_publisher_;
    size_t sample_count_ = 0;
    const size_t MAX_SAMPLES = 200;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> samples_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mean_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr median_cloud_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr std_dev_cloud_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StatsPointCloud>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
