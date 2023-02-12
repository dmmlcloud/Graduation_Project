// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 VLOC_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// VLOC_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
#ifdef VLOC_EXPORTS
#define VLOC_API __declspec(dllexport)
#else
#define VLOC_API __declspec(dllimport)
#endif

#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <Eigen\Core>
#include <opencv2\opencv.hpp>
#include "controllers/incremental_mapper.h"
#include "util/option_manager.h"
#include "util/string.h"
#include "util/misc.h"
#include "controllers/incremental_mapper.h"
#include "feature/extraction.h"
#include "feature/matching.h"
#include "feature/utils.h"
#include "retrieval/visual_index.h"
#include "estimators/pose.h"

#define CREATE_USER 'c'
#define RESTORE_OBJ 'r'

enum class Platform{ANDROID, IOS, UNITY, UNSUPPORTED};

// 此类是从 VLoc.dll 导出的
class VLOC_API User {
public:
	User(int id, int scene_id, Eigen::Matrix4d T);
	User(const User& u);
	User &User::operator =(const User& u);
	int id_ = -1;
	int scene_id_ =  -1;
	Eigen::Transform<double, 3, Eigen::Affine> T_ = Eigen::Transform<double, 3, Eigen::Affine>::Identity();
	double scale_ = 1;
	std::shared_ptr<std::vector<std::thread>> match_thread_;
	std::shared_ptr<std::vector<int>> match_scores_;
	std::shared_ptr<std::mutex> match_mutex_;
	std::shared_ptr<std::condition_variable> match_cv_;
	bool match_down_;
	//void SelectHighScoreModel();
}; 

class VLOC_API VirtualObj {
public:
	Eigen::Matrix3d r_ = Eigen::Matrix3d::Identity();
	Eigen::Vector3d t_ = Eigen::Vector3d::Identity();
	float scaleFactor = 1;
	VirtualObj(Eigen::Matrix3d r = Eigen::Matrix3d::Identity(), Eigen::Vector3d t = Eigen::Vector3d::Identity(), float s = 1);
};

class VLOC_API Scene {
public:
	int id_ = -1;
	int image_nums_;
	std::string location_;
	std::string model_;
	std::string database_path_;
	std::string reconstruction_path_;
	std::string vi_path_;
	std::string data_path_;
	std::vector<VirtualObj> objs_;
	colmap::retrieval::VisualIndex<> visual_index_;
	colmap::Database database_;
	colmap::Reconstruction reconstruction_;
	Scene(int id, std::string data_path);
	Scene(const Scene& s);
	Scene &Scene::operator =(const Scene &s);
	~Scene();


	void CreateVocabTree();
	[[deprecated("Use Lifetime::Locate instead")]]
	bool Locate(std::vector<std::string>& imageNames, std::vector<FIBITMAP*>& fibitmaps,
		double focal_length_a, Eigen::Vector4d& q1, Eigen::Vector4d& q2,
		Eigen::Vector3d& t1, Eigen::Vector3d& t2);
	void ReadReconstruction();
	void ReadDatabase();
	void ReadVI();
};

class VLOC_API Lifetime {
public:
	std::vector<Scene> scenes;
	std::vector<User> users;
	std::string temp_image_path;
	int scene_now_id;
	int user_now_id;

	Lifetime();
	~Lifetime() {}

	void Init(char** argv);

	// Locate step functions
	MyMultiDatabaseSiftFeatureExtractor* GetMyFeaturesExtractor(const std::vector<std::string>& imageNames,
		const std::vector<FIBITMAP*>& fibitmaps, OptionManager& options);
	VocabTreeFeatureMatcher* GetVocabTreeFeatureMatcher(const Scene& s, OptionManager& options, const std::vector<std::string>& imageNames,
		std::vector<image_t>& imageIds);
	bool SolvePnP(Scene& s, const IncrementalMapper::Options& mapper_options, const std::string& name,
		Image& image, Camera& camera);


	std::string CreateUser(std::string &encoded_data);
	std::string RestoreObjs(int &user_id);
	std::string RestoreAllObjsOfOther(int& restore_id, int& user_id);
	bool CreateObj(std::string &encoded_data);
	std::string CreateObjWithOtherUser(int &restore_id, int &user_id);

	// @brief: Locate function
	bool Locate(Scene& s, std::vector<std::string>& imageNames, std::vector<FIBITMAP*>& fibitmaps,
		double focal_length_a, Eigen::Vector4d& q1, Eigen::Vector4d& q2,
		Eigen::Vector3d& t1, Eigen::Vector3d& t2);

	void ThreadLocate(User& u, Scene& s, OptionManager& options, std::vector<std::string> imageNames, std::vector<FIBITMAP*> fibitmaps,
		const double focal_length_a, Eigen::Vector4d& q1, Eigen::Vector4d& q2,
		Eigen::Vector3d& t1, Eigen::Vector3d& t2, bool& match_suceess, int& query_down);

	// @brief: for one determined scene locating
	bool SaveImageAndComputeTMono(std::vector<BYTE>& str_decoded_byte, std::string& xyz,
		std::string f_a, Eigen::Transform<double, 3, Eigen::Affine>& T, Platform platform, const int scene_id);
	bool SaveImageAndComputeTStereo(std::vector<BYTE>& str_decoded_byte1, std::vector<BYTE>& str_decoded_byte2,
		std::string& xyz_a, std::string& xyz_b, std::string f_a, Eigen::Transform<double, 3, Eigen::Affine>& T,
		double& scale, Platform platform, const int scene_id);

	// @brief: for many scenes locating
	bool FindMatchedSceneAndComputeTMono(int user_id, std::vector<BYTE>& str_decoded_byte, std::string& xyz,
		std::string f_a, Eigen::Transform<double, 3, Eigen::Affine>& T, Platform platform, std::string sceneLocation, std::string model);
	bool FindMatchedSceneAndComputeTStereo(int user_id, std::vector<BYTE>& str_decoded_byte, std::string& xyz,
		std::string f_a, Eigen::Transform<double, 3, Eigen::Affine>& T, Platform platform, std::string sceneLocation, std::string model);
	
	// @brief: two different function corresponse multi or single scene
	int ComputeT(std::string& encoded_data, const bool singleShot, const std::vector<std::string>& locations);
	bool ComputeT(std::string& encoded_data, const bool singleShot, const int scene_id);
};

std::vector<std::string> VLOC_API split(std::string &Input, const char* Regex);

