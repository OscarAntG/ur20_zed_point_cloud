#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <vector>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/filters/project_inliers.h>

#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/msac.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing_rgb.h>

#include <pcl/surface/mls.h>
#include <pcl/surface/concave_hull.h>

using namespace std::placeholders;

class PCLDemo : public rclcpp::Node 
{
public:
    PCLDemo() : Node("pcl_demo"){
        raw_cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zedx/zed_node/point_cloud/cloud_registered", 10, std::bind(&PCLDemo::callbackFilterPointCloud, this, _1));
        filtered_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_point_cloud", 10);
        
    }
private:
        void callbackFilterPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr message){
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            try{
                pcl::fromROSMsg(*message, *pcl_cloud);
            } catch (const std::exception & e){
                RCLCPP_ERROR(this->get_logger(), "Failed to convert PointCloud2 to pcl::PointCloud: %s", e.what());
                return;
            }
            RCLCPP_DEBUG(this->get_logger(), "Received cloud %zu points.", pcl_cloud->size());

            // ================================ FILTERS ==========================================//
            // PassThrough Filter
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr passthrough_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            pcl::PassThrough<pcl::PointXYZRGB> pass;
            pass.setInputCloud(pcl_cloud);
            pass.setFilterFieldName("x");
            pass.setFilterLimits(0.0, 1.25); // 1.25
            pass.filter(*passthrough_cloud);

            // OutlierRemoval
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr outlier_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
            sor.setInputCloud(passthrough_cloud);
            sor.setMeanK(50);
            sor.setStddevMulThresh(1.75);
            sor.filter(*outlier_cloud);
            
            // VoxelGrid
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxel_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            pcl::VoxelGrid<pcl::PointXYZRGB> vg;
            vg.setInputCloud(outlier_cloud);
            vg.setLeafSize(0.01f, 0.01f, 0.01f);
            vg.filter(*voxel_cloud);

            // =============================== SURFACE PROCESSING =======================================//
            // MLS Polynomial - Smoothing/Normal Estimation
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr smooth_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            pcl::search::Search<pcl::PointXYZRGB>::Ptr tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZRGB>>();
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mls_points = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
            pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGBNormal> mls;

            mls.setComputeNormals(true);
            mls.setInputCloud(voxel_cloud);
            mls.setPolynomialOrder(2);
            mls.setSearchMethod(tree);
            mls.setSearchRadius(0.04);
            mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointXYZRGBNormal>::NONE);
            mls.process(*mls_points);

            // =============================== SEGMENTATION =======================================//
            // SAC Segmentation
            pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZRGB> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_MSAC);
            seg.setMaxIterations(1000);
            seg.setDistanceThreshold(0.005);
            seg.setInputCloud (voxel_cloud);
            seg.segment(*inliers, *coefficients);          

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr sac_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            pcl::ExtractIndices<pcl::PointXYZRGB> extract;
            extract.setInputCloud(voxel_cloud);
            extract.setIndices(inliers);
            // Extract board, remove waste
            extract.setNegative(false);
            extract.filter(*sac_cloud);
            // Extract waste, remove board
            extract.setNegative(true);
            extract.filter(*cluster_cloud);

            // =============================== CONCAVE HULL ===============================//
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            pcl::ProjectInliers<pcl::PointXYZRGB> proj;
            proj.setModelType(pcl::SACMODEL_PLANE);
            proj.setInputCloud(sac_cloud);
            proj.setModelCoefficients(coefficients);
            proj.filter(*projected_cloud);

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr hull_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
            pcl::ConcaveHull<pcl::PointXYZRGB> chull;
            chull.setInputCloud(projected_cloud);
            chull.setAlpha(0.1);
            chull.reconstruct(*hull_cloud);

            // =============================== Region Growing ===============================
            pcl::PointCloud<pcl::Normal>::Ptr normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
            pcl::search::Search<pcl::PointXYZRGBNormal>::Ptr region_tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZRGBNormal>>();

            pcl::IndicesPtr indices (new std::vector <int>);
            pcl::removeNaNFromPointCloud(*mls_points, *indices);

            pcl::NormalEstimation<pcl::PointXYZRGBNormal, pcl::Normal> normal_estimator;
            normal_estimator.setSearchMethod(region_tree);
            normal_estimator.setInputCloud(mls_points);
            normal_estimator.setKSearch(20);    // tried: 50, 20, 10
            normal_estimator.compute(*normals);
            
            pcl::RegionGrowing<pcl::PointXYZRGBNormal, pcl::Normal> reg;
            reg.setMinClusterSize(50);
            reg.setMaxClusterSize(10000);
            reg.setSearchMethod(region_tree);
            reg.setNumberOfNeighbours(10); // tried 30, 20, 10
            reg.setInputCloud(mls_points);
            reg.setIndices(indices);
            reg.setInputNormals(normals);
            reg.setSmoothnessThreshold(1.5/180.0 * M_PI); // tried 3.0, 1.5, 1.3, 1.0
            reg.setCurvatureThreshold(1.0); // tried 1.0, 0.7, 0.2

            std::vector <pcl::PointIndices> clusters;
            reg.extract (clusters);
            pcl::PointCloud <pcl::PointXYZRGBNormal>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>(*mls_points));
            int largest_cluster_idx = -1;
            size_t max_size = 0;
            for (size_t i = 0; i < clusters.size(); ++i){
                if (clusters[i].indices.size() > max_size){       
                    max_size = clusters[i].indices.size();
                    largest_cluster_idx = static_cast<int>(i);
                }
            }   
            for (auto& pt : colored_cloud->points){
                pt.r = 255;
                pt.g = 0;
                pt.b = 0;
            }
            if (largest_cluster_idx >= 0){
                for (int idx : clusters[largest_cluster_idx].indices){
                    colored_cloud->points[idx].r = 0;
                    colored_cloud->points[idx].g = 255;
                    colored_cloud->points[idx].b = 0;
                }
            }

            // Publish filtered cloud
            sensor_msgs::msg::PointCloud2 msg;
            pcl::toROSMsg(*colored_cloud, msg);
            msg.header = message->header;
            filtered_cloud_publisher_->publish(msg);            
        }

        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr raw_cloud_subscriber_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_publisher_;      
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PCLDemo>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}