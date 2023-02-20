#include <tools/depth_estimation.h>
#include <fstream>
#include <Python.h>

namespace colmap{
DepthEstimation::DepthEstimation() {
    checkpointPath = "checkpoints/depthformer_swinl_w7_22k_nyu.pth";
    configPath = "configs/depthformer/depthformer_swinl_22k_w7_nyu.py";
    showDir = "results";
}

DepthEstimation::DepthEstimation(std::string configPath, std::string checkpointPath,
                                std::string showDir) : configPath(configPath), 
                                checkpointPath(checkpointPath), showDir(showDir) {

}

void stringSplit(std::string s, char symbol, std::vector<std::string>& result) {
    std::istringstream iss(s);
    std::string token;
    while(getline(iss, token, symbol)) {
        result.emplace_back(token);
    }
}

void DepthEstimation::EstimateDepth() {

    std::string szCommand, shellName, shellParam;
    shellName = "zsh ./depth_estimate.sh";
    shellParam = configPath + " " + checkpointPath + " " + showDir;
    std::cout << shellParam << std::endl;
    szCommand = shellName + " " + shellParam;     //取得执行shell脚本的命令szCommand
	system(szCommand.c_str());
    std::cout << "\n Depth Estimation Complete !" << std::endl;

    // read depth info
    std::ifstream ifs;
    std::string depthFile = "./depth_results/depth_info.txt";
    ifs.open(depthFile, std::ios::in);
    if(!ifs.is_open()) {
        std::cout << "open file failed" << std::endl;
        return;
    }

    std::string buffer;
    while(getline(ifs, buffer)) {
        std::vector<double> tempDepth;
        std::vector<std::string> splitResult;
        stringSplit(buffer, ' ', splitResult);
        for(auto& s : splitResult) {
            tempDepth.push_back(std::stod(s));
        }
        depthMaps[0].push_back(tempDepth);
    }
    std::cout << "Load Depth Data Complete !" << std::endl;
}

void DepthEstimation::GetDepthInfo(image_t image_id, double x, double y) {

}

}