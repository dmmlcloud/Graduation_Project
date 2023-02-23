#include <stdio.h>
#include <vector>
#include <unordered_map>
#include "base/reconstruction.h"

namespace colmap{

class DepthEstimation {
    void estimateDepth(const std::vector<Image>& images);
    std::unordered_map<image_t, std::vector<std::vector<double>>> depthMaps;
    std::string checkpointPath;
    std::string configPath;
    std::string showDir;
    std::shared_ptr<std::string> database_path;
public:
    DepthEstimation();
    DepthEstimation(std::string configPath, std::string checkpointPath, std::string showDir, std::shared_ptr<std::string> database_path);
    void EstimateDepth(const std::vector<Image>& images);
    double GetDepthInfo(image_t image_id, double x, double y);
};

}