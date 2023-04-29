#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/impl/centroid.hpp>
#include <pcl/common/transforms.h>


// #include <pcl/features/impl/centroid.hpp>

#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <string>
#include <algorithm>
#include <json/json.h>

#include <pcl/pcl_config.h>


int main (int argc, char** argv)
{
  if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
      return EXIT_FAILURE;
  }

  const std::string filename = argv[1];
    
  // Construct the new filename by appending "_transformed" to the filename
  const std::size_t dot_pos = filename.find_last_of(".");
  const std::string output_filename = filename.substr(0, dot_pos) + "_transformed" + filename.substr(dot_pos);

  // Open the JSON file
  std::ifstream file(filename);
  if (!file) {
      std::cerr << "Failed to open file" << std::endl;
      return -1;
  }

  // Load the JSON data
  Json::Value root;
  file >> root;


  int num_timesteps = root.size();
  // Create a JSON array to hold the point cloud data
  Json::Value output;

  // Declare variables for running average
  int count = 0;
  float avg_a = 0.0, avg_b = 0.0, avg_c = 0.0, avg_d = 0.0;

  for (int t = 0; t < num_timesteps; t++) { 

    auto timestep = root[t];
    int num_points = timestep.size();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(num_points);
    
    // Parse the points of the first timestep
    for (int i = 0; i < num_points; i++) {
      auto point = timestep[i];
      pcl::PointXYZ p;
      p.x = point[0].asFloat();
      p.y = point[1].asFloat();
      p.z = point[2].asFloat();
      cloud->push_back(p);
    }
    
    // Perform plane segmentation
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);

    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
      PCL_ERROR ("Could not estimate a planar model for the given dataset.");
      return (-1);
    }

    std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                        << coefficients->values[1] << " "
                                        << coefficients->values[2] << " " 
                                        << coefficients->values[3] << std::endl;

    float percent_inliers = 100.0 * inliers->indices.size() / cloud->size();
    std::cerr << cloud->size () << " points" << ", model inliers: " << inliers->indices.size () << " points (" << percent_inliers << "%)" << std::endl << std::endl;

    if (percent_inliers > 10.0) {
      count++;
      avg_a += (coefficients->values[0] - avg_a) / count;
      avg_b += (coefficients->values[1] - avg_b) / count;
      avg_c += (coefficients->values[2] - avg_c) / count;
      avg_d += (coefficients->values[3] - avg_d) / count;
    }

    // // Compute the centroid of the point cloud
    // Eigen::Vector4f centroid;
    // pcl::compute3DCentroid(*cloud, centroid);

    // // Demean the point cloud by subtracting the centroid from each point
    // for (pcl::PointXYZ& point : cloud->points) {
    //   point.x -= centroid[0];
    //   point.y -= centroid[1];
    //   point.z -= centroid[2];
    // }

    // // Transform the point cloud
    // pcl::PointCloud<pcl::PointXYZ>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZ>());
    // Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    // pcl::transformPointCloud(*cloud, *transformed, transform);


    // // Create a JSON array to hold the point cloud data
    // Json::Value timestep_output;
    // timestep_output.resize(transformed->size());
    // for (int i = 0; i < transformed->size(); i++) {
    //   const pcl::PointXYZ& point = transformed->points[i];
    //   Json::Value& coords = timestep_output[i];
    //   coords.append(point.x);
    //   coords.append(point.y);
    //   coords.append(point.z);
    // }

    // output.append(timestep_output);
  }

  // // Write the JSON array to disk
  // std::ofstream output_file(output_filename);
  // Json::StreamWriterBuilder builder;
  // builder["indentation"] = "";
  // std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
  // writer->write(output, &output_file);

  // Print the final running average
  std::cerr << "Final running average of model coefficients: " << avg_a << " " 
                                                            << avg_b << " "
                                                            << avg_c << " " 
                                                            << avg_d << ", count " << count << "/" << num_timesteps << std::endl;

  return (0);
}
