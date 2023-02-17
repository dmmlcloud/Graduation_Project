#include <stdio.h>
#include <vector>
#include <unordered_map>
#include "base/reconstruction.h"

namespace colmap{

class DepthEstimation {
    std::unordered_map<image_t, std::vector<std::vector<float>>> depthMaps;
    std::string checkpointPath;
    std::string configPath;
    bool show;
    std::string showDir;
public:
    DepthEstimation();
    DepthEstimation(std::string configPath, std::string checkpointPath,
    bool show, std::string showDir);
    void EstimateDepth();
    void GetDepthInfo(image_t image_id, double x, double y);
};

}