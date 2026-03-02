#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <map>
#include <vector>

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/features/normal_3d.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>

#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/lccp_segmentation.h>

using namespace std::placeholders;

class FilterPointCloud : public rclcpp::Node 
{
public:
    FilterPointCloud() : Node("filter_point_cloud"){
        raw_cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zedx/zed_node/point_cloud/cloud_registered", 10, std::bind(&FilterPointCloud::callbackFilterPointCloud, this,_1));
        filtered_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_point_cloud",10);
    }
private:
    void callbackFilterPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr message){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        try{
            pcl::fromROSMsg(*message, *pcl_cloud);
        }catch (const std::exception & e){
            RCLCPP_ERROR(this->get_logger(), "Failed to convert PointCloud2 to pcl::PointCloud: %s", e.what());
            return;
        }
        
        // Optional PassThrough Filter: isolates region of cloud
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud(pcl_cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(0.0, 1.25); // whiteboard = 0.0, 1.25
        pass.filter(*pcl_cloud);
        
        // Outlier removal
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(pcl_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.75);
        sor.filter(*pcl_cloud);

        // Supervoxel cluster initialization
        float voxel_res = 0.005f;   // 0.005, 0.008, 0.01 | larger->fails to segment
        float seed_res = 0.03f;    // 0.01, 0.08 | smaller->holes in point cloud
        float color_importance = 0.0f;
        float spatial_importance = 1.0f;
        float normal_importance = 4.0f;
        pcl::SupervoxelClustering<pcl::PointXYZRGB> super (voxel_res, seed_res);
        super.setUseSingleCameraTransform (false);
        super.setInputCloud(pcl_cloud);
        super.setColorImportance(color_importance);
        super.setSpatialImportance(spatial_importance);
        super.setNormalImportance(normal_importance);

        // Supervoxel extraction
        RCLCPP_INFO(this->get_logger(), "Extracting supervoxels...");
        std::map<std::uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr> supervoxel_clusters;
        super.extract(supervoxel_clusters);
        RCLCPP_INFO(this->get_logger(), "Found %ld supervoxels", supervoxel_clusters.size());
        
        // Loading cloud - Point type changed to XYZL for labeled cloud, color information is lost
        pcl::PointCloud<pcl::PointXYZL>::Ptr sv_labeled_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZL>>();
        sv_labeled_cloud = super.getLabeledCloud();
        std::multimap<std::uint32_t, std::uint32_t> supervoxel_adjacency;
        super.getSupervoxelAdjacency(supervoxel_adjacency);
        
        pcl::PointCloud<pcl::PointNormal>::Ptr sv_centroid_normal_cloud = pcl::SupervoxelClustering<pcl::PointXYZRGB>::makeSupervoxelNormalCloud(supervoxel_clusters);

        // LCCP Segmentation
        float concavity_tolerance_thresh = 6;   // 10, 7, 5, 2 | too low starts to segment flat surface, 7 is best for Roy
        float smoothness_thresh = 5.7/180.0 * M_PI; //5.7
        std::uint32_t min_segment_size = 3;
        bool use_extended_convexity = true;
        bool use_sanity_criterion = true;
        pcl::LCCPSegmentation<pcl::PointXYZRGB> lccp;
        lccp.setConcavityToleranceThreshold(concavity_tolerance_thresh);
        lccp.setSanityCheck(use_sanity_criterion);
        lccp.setSmoothnessCheck(true, voxel_res, seed_res, smoothness_thresh);
        if (use_extended_convexity)
            lccp.setKFactor(1);
        else
            lccp.setKFactor(0);
        lccp.setInputSupervoxels(supervoxel_clusters,supervoxel_adjacency);
        lccp.setMinSegmentSize(min_segment_size);
        lccp.segment();

        pcl::PointCloud<pcl::PointXYZL>::Ptr lccp_labeled_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZL>>();
        pcl::copyPointCloud(*sv_labeled_cloud, *lccp_labeled_cloud);
        lccp.relabelCloud(*lccp_labeled_cloud);
        
        // Process cloud
        pcl::PointCloud<pcl::PointXYZL>::Ptr display_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZL>>();
        display_cloud = lccp_labeled_cloud;
        
        // Publish cloud
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*display_cloud, msg);
        msg.header = message->header;
        filtered_cloud_publisher_->publish(msg);
    }

    // REGION GROWING
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr RegionGrowing(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud){
        // RegionGrowing - Search + Normal Estimation
        pcl::search::Search<pcl::PointXYZRGB>::Ptr tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZRGB>>();
        pcl::PointCloud<pcl::Normal>::Ptr normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
        
        pcl::IndicesPtr indices = std::make_shared<std::vector<int>>();
        pcl::removeNaNFromPointCloud(*input_cloud, *indices);

        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
        normal_estimator.setSearchMethod(tree);
        normal_estimator.setInputCloud(input_cloud);
        normal_estimator.setKSearch(50);
        normal_estimator.compute(*normals);

        // RegionGrowing - filter declaration
        pcl::RegionGrowing<pcl::PointXYZRGB, pcl::Normal> reg;
        reg.setMinClusterSize(50);
        reg.setMaxClusterSize(10000);
        reg.setSearchMethod(tree);
        reg.setNumberOfNeighbours(10);
        reg.setInputCloud(input_cloud);
        reg.setIndices(indices);
        reg.setInputNormals(normals);
        reg.setSmoothnessThreshold(1.5/180.0 * M_PI);
        reg.setCurvatureThreshold(1.0);

        // RegionGrowing - cluster processing
        std::vector <pcl::PointIndices> clusters;
        reg.extract(clusters);
        RCLCPP_INFO(this->get_logger(), "Found %ld clusters.", clusters.size());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr reg_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        reg_cloud = reg.getColoredCloud();
        return reg_cloud;
    }
    // VOXEL GRID
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr VoxelGrid(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr vg_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        pcl::VoxelGrid<pcl::PointXYZRGB> vg;
        vg.setInputCloud(input_cloud);
        vg.setLeafSize(0.01f, 0.01f, 0.01f);
        vg.filter (*vg_cloud);
        return vg_cloud;
    }
    // SAC SEGMENTATION
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr SacPlaneFitting(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud){
        // MSAC Plane fitting
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_MSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.005);
        seg.setInputCloud (input_cloud);
        seg.segment(*inliers, *coefficients);          

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sac_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(input_cloud);
        extract.setIndices(inliers);
        // Extract board, remove waste
        extract.setNegative(false);
        extract.filter(*sac_cloud);
        // Extract waste, remove board
        extract.setNegative(true);
        extract.filter(*cluster_cloud);
        return sac_cloud;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr raw_cloud_subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_publisher_;
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FilterPointCloud>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}