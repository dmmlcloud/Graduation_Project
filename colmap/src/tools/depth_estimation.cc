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
                                std::string showDir, std::shared_ptr<std::string> database_path) :
                                configPath(configPath), checkpointPath(checkpointPath),
                                showDir(showDir), database_path(database_path){

}

void stringSplit(std::string s, char symbol, std::vector<std::string>& result) {
    std::istringstream iss(s);
    std::string token;
    while(getline(iss, token, symbol)) {
        result.emplace_back(token);
    }
}

void DepthEstimation::estimateDepth(const std::vector<Image>& images) {

    std::string szCommand, shellName, shellParam;
    shellName = "zsh ./depth_estimate.sh";
    shellParam = configPath + " " + checkpointPath + " " + showDir;
    std::cout << shellParam << std::endl;
    szCommand = shellName + " " + shellParam;     //取得执行shell脚本的命令szCommand
	system(szCommand.c_str());
    std::cout << "\n Depth Estimation Complete !" << std::endl;

    // read depth info
    for(const auto& image : images) {
        std::ifstream ifs;
        std::string depthFile = "./depth_results/" + image.Name() + ".txt";
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
            depthMaps[image.ImageId()].push_back(tempDepth);
        }
        ifs.close();
    }
    std::cout << "Load Depth Data Complete !" << std::endl;
}

double DepthEstimation::GetDepthInfo(image_t image_id, double x, double y) {
    int left = x;
    int right = x+1;
    int up = y;
    int down = y+1;
    auto& depthInfo = depthMaps[image_id];
    double depth = depthInfo[left][up];
    double slopWidth = (depthInfo[up][right] - depthInfo[up][left]);
    double slopHeight = (depthInfo[down][left] - depthInfo[up][left]);
    double result = depth + (x - (double)left) * slopWidth + (y - (double)up) * slopHeight;
    return result;
}

void DepthEstimation::EstimateDepth(const std::vector<Image>& images) {
    Database database(*this->database_path);
    std::ofstream ofs;
    std::string depthFile = "./data/nyu/nyu_test.txt";
    ofs.open(depthFile, std::ios::out);
    if(!ofs.is_open()) {
        std::cout << "open file failed" << std::endl;
        return;
    }
    for(const auto& image : images) {
        // const auto& camera = database.ReadCamera(image.CameraId());
        // double focal = camera.FocalLength();
        double nyuFocal = 518.8579;
        std::string testPath = "/home/lzy/my_project/Graduation_Project/test/images/";
        std::string writeImage = testPath + image.Name() + " " + testPath + image.Name() + " " + std::to_string(nyuFocal) + "\n";
        std::cout << writeImage << std::endl;
        ofs << writeImage;
    }
    ofs.close();
    
}

}