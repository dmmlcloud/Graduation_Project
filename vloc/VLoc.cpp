// VLoc.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "VLoc.h"

using namespace colmap;

std::mutex query_mutex;
std::mutex count_mutex;

namespace util {
	static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	static std::vector<std::string> split(std::string& Input, const char* Regex) {
		std::vector<std::string> Result;
		int pos = 0;
		int npos = 0;
		int regexlen = strlen(Regex);
		while ((npos = Input.find(Regex, pos)) != -1) {
			std::string tmp = Input.substr(pos, npos - pos);
			Result.push_back(tmp);
			pos = npos + regexlen;
		}
		Result.push_back(Input.substr(pos, Input.length() - pos));
		return Result;
	}

	static std::string remove_surplus_spaces(const std::string& src) {
		std::string result = "";
		for (int i = 0; src[i] != '\0'; i++) {
			if (src[i] != ' ')
				result.append(1, src[i]);
			else
				if (src[i + 1] != ' ')
					result.append(1, src[i]);
		}
		return result;
	}

	static inline bool is_base64(BYTE c) {
		return (isalnum(c) || (c == '+') || (c == '/'));
	}

	static std::string base64_encode(BYTE const* buf, unsigned int bufLen) {
		std::string ret;
		int i = 0;
		int j = 0;
		BYTE char_array_3[3];
		BYTE char_array_4[4];

		while (bufLen--) {
			char_array_3[i++] = *(buf++);
			if (i == 3) {
				char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
				char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
				char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
				char_array_4[3] = char_array_3[2] & 0x3f;

				for (i = 0; (i < 4); i++)
					ret += base64_chars[char_array_4[i]];
				i = 0;
			}
		}

		if (i)
		{
			for (j = i; j < 3; j++)
				char_array_3[j] = '\0';
			char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
			char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
			char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
			char_array_4[3] = char_array_3[2] & 0x3f;
			for (j = 0; (j < i + 1); j++)
				ret += base64_chars[char_array_4[j]];
			while ((i++ < 3))
				ret += '=';
		}
		return ret;
	}

	static std::vector<BYTE> base64_decode(std::string const& encoded_string) {
		int in_len = encoded_string.size();
		int i = 0;
		int j = 0;
		int in_ = 0;
		BYTE char_array_4[4], char_array_3[3];
		std::vector<BYTE> ret;

		while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
			char_array_4[i++] = encoded_string[in_]; in_++;
			if (i == 4) {
				for (i = 0; i < 4; i++)
					char_array_4[i] = base64_chars.find(char_array_4[i]);

				char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
				char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
				char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

				for (i = 0; (i < 3); i++)
					ret.push_back(char_array_3[i]);
				i = 0;
			}
		}

		if (i) {
			for (j = i; j < 4; j++)
				char_array_4[j] = 0;

			for (j = 0; j < 4; j++)
				char_array_4[j] = base64_chars.find(char_array_4[j]);

			char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
			char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
			char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

			for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
		}
		return ret;
	}

	static std::string getNowTime() {
		struct tm* ptr;
		time_t lt;
		lt = time(NULL);
		return std::to_string(lt);
	}

	static std::string Trim(std::string& s) {
		if (s.empty()) return s;
		s.erase(0, s.find_first_not_of(" "));
		s.erase(s.find_last_not_of(" ") + 1);
		return s;
	}

	static HBITMAP ConvertCVMatToBMP(cv::Mat frame) {
		auto convertOpenCVBitDepthToBits = [](const int32_t value) {
			auto regular = 0u;

			switch (value)
			{
			case CV_8U:
			case CV_8S:
				regular = 8u;
				break;

			case CV_16U:
			case CV_16S:
				regular = 16u;
				break;

			case CV_32S:
			case CV_32F:
				regular = 32u;
				break;

			case CV_64F:
				regular = 64u;
				break;

			default:
				regular = 0u;
				break;
			}

			return regular;
		};

		auto imageSize = frame.size();
		assert(imageSize.width && "invalid size provided by frame");
		assert(imageSize.height && "invalid size provided by frame");

		if (imageSize.width && imageSize.height) {
			auto headerInfo = BITMAPINFOHEADER{};
			ZeroMemory(&headerInfo, sizeof(headerInfo));

			headerInfo.biSize = sizeof(headerInfo);
			headerInfo.biWidth = imageSize.width;
			headerInfo.biHeight = -(imageSize.height); // negative otherwise it will be upsidedown
			headerInfo.biPlanes = 1;// must be set to 1 as per documentation frame.channels();

			const auto bits = convertOpenCVBitDepthToBits(frame.depth()); // depth() gets bits of every pixel
			headerInfo.biBitCount = frame.channels() * bits; // channels() gets value num of element in the matrix

			auto bitmapInfo = BITMAPINFO{};
			ZeroMemory(&bitmapInfo, sizeof(bitmapInfo));

			bitmapInfo.bmiHeader = headerInfo;
			bitmapInfo.bmiColors->rgbBlue = 0;
			bitmapInfo.bmiColors->rgbGreen = 0;
			bitmapInfo.bmiColors->rgbRed = 0;
			bitmapInfo.bmiColors->rgbReserved = 0;

			auto dc = GetDC(nullptr);
			assert(dc != nullptr && "Failure to get DC");
			auto bmp = CreateDIBitmap(dc,
				&headerInfo,
				CBM_INIT,
				frame.data,
				&bitmapInfo,
				DIB_RGB_COLORS);
			assert(bmp != nullptr && "Failure creating bitmap from captured frame");

			return bmp;
		}
		else {
			return nullptr;
		}
	}

	static FIBITMAP* ConvertHBitmapToFiBitmap(HBITMAP hbmp) {
		FIBITMAP* dib = NULL;
		if (hbmp) {
			BITMAP bm = { 0 };
			int ret = GetObject(hbmp, sizeof(BITMAP), (LPSTR)&bm);
			//std::cout << bm.bmWidth << " " << bm.bmHeight << std::endl;
			if (ret == 0 || ret > sizeof(BITMAP)) {
				std::cout << "nullptr error 1" << std::endl;
				return nullptr;
			}
			dib = FreeImage_AllocateT(FIT_BITMAP, bm.bmWidth, bm.bmHeight, bm.bmBitsPixel);
			if (dib == nullptr) {
				std::cout << "nullptr error 2" << std::endl;
				return nullptr;
			}
			int nColors = FreeImage_GetColorsUsed(dib);
			HDC dc = GetDC(NULL);
			int Success = GetDIBits(dc, hbmp, 0, FreeImage_GetHeight(dib),
				FreeImage_GetBits(dib), FreeImage_GetInfo(dib), DIB_RGB_COLORS);
			ReleaseDC(NULL, dc);
			FreeImage_GetInfoHeader(dib)->biClrUsed = nColors;
			FreeImage_GetInfoHeader(dib)->biClrImportant = nColors;
		}
		dib = FreeImage_ConvertToGreyscale(dib);

		return dib;
	}

	Eigen::Transform<double, 3, Eigen::Affine> CalFinalT(Eigen::Vector4d& q1, Eigen::Vector4d& q2,
		Eigen::Vector3d& t1, Eigen::Vector3d& t2, Eigen::Quaterniond& qarcore_a, Eigen::Vector3d& tarcore_a) {
		Eigen::Transform<double, 3, Eigen::Affine> T1, T2, T2T, TA;
		Eigen::Quaterniond qcolmap_a(q1[0], q1[1], q1[2], q1[3]);
		Eigen::Vector3d tcolmap_a(t1[0], t1[1], t1[2]);

		Eigen::Matrix4d T_arcore_world2Cam = Eigen::Matrix4d::Identity();
		Eigen::Matrix4d T_colmap_world2Cam = Eigen::Matrix4d::Identity();
		T_arcore_world2Cam.block(0, 0, 3, 3) = qarcore_a.matrix();
		T_arcore_world2Cam.block(0, 3, 3, 1) = tarcore_a;
		T_colmap_world2Cam.block(0, 0, 3, 3) = qcolmap_a.matrix();
		T_colmap_world2Cam.block(0, 3, 3, 1) = tcolmap_a;
		T1 = T_arcore_world2Cam;
		T2 = T_colmap_world2Cam;
		T2T = T2.inverse();
		TA = T2T * T1;
		return TA;
	}

	bool ParseRAndT(Platform platform, std::string& xyz, Eigen::Quaterniond& qarcore, Eigen::Vector3d& tarcore) {
		// q:wxyz
		xyz = util::Trim(xyz);
		std::vector<std::string> xyz_s = util::split(xyz, " ");
		Eigen::Quaterniond qarcore_temp(stod(xyz_s[0]), stod(xyz_s[1]), stod(xyz_s[2]), stod(xyz_s[3]));
		Eigen::Vector3d tarcore_temp(stod(xyz_s[4]), stod(xyz_s[5]), stod(xyz_s[6]));

		if (platform == Platform::ANDROID) {
			Eigen::Matrix3d Rwc(qarcore_temp);
			Eigen::Matrix3d Rcw = Rwc.inverse();//Rwc to Rcw
			Eigen::Matrix3d RX;
			RX << 1, 0, 0,  // 绕x旋转180度
				0, -1, 0,
				0, 0, -1;
			Eigen::Matrix3d Rcw_xr = RX * Rcw;
			Eigen::Quaterniond Qcw(Rcw_xr);
			qarcore = Qcw;
			tarcore = -Rcw_xr * tarcore_temp;

			return true;
		}
		else if (platform == Platform::UNITY) {
			// z轴反向
			qarcore_temp.x() *= -1;
			qarcore_temp.y() *= -1;
			tarcore_temp.z() *= -1;

			Eigen::Matrix3d Rwc(qarcore_temp);
			Eigen::Matrix3d Rcw = Rwc.inverse();

			Eigen::Matrix3d RX;
			RX << 1, 0, 0,  // 绕x旋转180度
				0, -1, 0,
				0, 0, -1;
			Eigen::Matrix3d Rcw_xr = RX * Rcw;
			Eigen::Quaterniond Qcw(Rcw_xr);
			qarcore = Qcw;
			tarcore = -Rcw_xr * tarcore_temp;

			return true;
		}
		else if (platform == Platform::IOS) {
			// TODO
			return true;
		}
		else {
			std::cout << "NOT SUPPORT PLATFORM" << std::endl;
			return false;
		}
	}
}

// Class User
User::User(int id, int scene_id, Eigen::Matrix4d T) {
	id_ = id;
	T_ = T;
	scene_id_ = scene_id;
	match_thread_ = std::make_shared<std::vector<std::thread>>();
	match_scores_ = std::make_shared<std::vector<int>>();
	match_mutex_ = std::make_shared<std::mutex>();
	match_cv_ = std::make_shared<std::condition_variable>();
	match_down_ = false;
}

User::User(const User& u) {
	this->id_ = u.id_;
	this->T_ = u.T_;
	this->scene_id_ = u.scene_id_;
	this->scale_ = u.scale_;
	this->match_thread_ = u.match_thread_;
	this->match_scores_ = u.match_scores_;
	this->match_mutex_ = u.match_mutex_;
	this->match_cv_ = u.match_cv_;
	this->match_down_ = u.match_down_;
}

User& User::operator =(const User& u) {
	this->id_ = u.id_;
	this->T_ = u.T_;
	this->scene_id_ = u.scene_id_;
	this->scale_ = u.scale_;
	this->match_thread_ = u.match_thread_;
	this->match_scores_ = u.match_scores_;
	this->match_mutex_ = u.match_mutex_;
	this->match_cv_ = u.match_cv_;
	this->match_down_ = u.match_down_;
	return *this;
}

// Class VirtualObj
VirtualObj::VirtualObj(Eigen::Matrix3d r, Eigen::Vector3d t, float s) {
	r_ = r;
	t_ = t;
	scaleFactor = s;
}


// Class Scene
Scene::Scene(int id, std::string data_path) {
	id_ = id;
	data_path_ = data_path;
	reconstruction_path_ = data_path;
	database_path_ = data_path + "\\database.db";
	vi_path_ = data_path + "\\vi.bin";
}

Scene::Scene(const Scene& s) {
	id_ = s.id_;
	location_ = s.location_;
	model_ = s.model_;
	image_nums_ = s.image_nums_;
	data_path_ = s.data_path_;
	reconstruction_path_ = s.data_path_;
	database_path_ = s.database_path_;
	objs_.assign(s.objs_.begin(), s.objs_.end());
	vi_path_ = s.vi_path_;
	reconstruction_ = s.reconstruction_;
	visual_index_.Read(vi_path_);
	database_.Open(database_path_);
}

Scene& Scene::operator =(const Scene& s) {
	id_ = s.id_;
	location_ = s.location_;
	model_ = s.model_;
	image_nums_ = s.image_nums_;
	data_path_ = s.data_path_;
	reconstruction_path_ = s.data_path_;
	database_path_ = s.database_path_;
	objs_.assign(s.objs_.begin(), s.objs_.end());
	vi_path_ = s.vi_path_;
	reconstruction_ = s.reconstruction_;
	visual_index_.Read(vi_path_);
	database_.Open(database_path_);
	return *this;
}

Scene::~Scene() {
	database_.Close();
}

void Scene::CreateVocabTree() {
	OptionManager options;
	std::string file_path, vt_path;
	file_path = vi_path_;
	vt_path = "..//VT32K.bin";
	options.AddVocabTreeMatchingOptions();
	options.vocab_tree_matching->vocab_tree_path = vt_path;

	retrieval::VisualIndex<> visual_index;

	VocabTreeFeatureMatcher matcher(*options.vocab_tree_matching, *options.sift_matching, database_path_);
	matcher.IndexImages(&visual_index);

	visual_index.Write(file_path);
}



void Scene::ReadReconstruction() {
	reconstruction_.Read(reconstruction_path_);
}

void Scene::ReadDatabase() {
	database_.Open(database_path_);
	image_nums_ = database_.ReadAllImages().size();
}

void Scene::ReadVI() {
	if (!ExistsFile(vi_path_)) {
		CreateVocabTree();
	}

	visual_index_.Read(vi_path_);
}

bool Scene::Locate(std::vector<std::string>& imageNames, std::vector<FIBITMAP*>& fibitmaps,
	double focal_length_a, Eigen::Vector4d& q1, Eigen::Vector4d& q2,
	Eigen::Vector3d& t1, Eigen::Vector3d& t2
) {
	Timer timer;
	timer.Start();
	//////////////////////////////////////////////
	// 定位部分
	//////////////////////////////////////////////
	// Options and checks
	OptionManager options;
	options.AddMapperOptions();
	options.AddExtractionOptions();
	options.AddVocabTreeMatchingOptions();

	options.vocab_tree_matching->vocab_tree_path = vi_path_;

	ImageReaderOptions reader_options = *options.image_reader;
	reader_options.database_path = database_path_;
	reader_options.image_list = imageNames;
	reader_options.camera_model = "OPENCV";

	//////////////////////////////////////////////
	// Feature Extracting...
	//////////////////////////////////////////////

	// Options
	options.sift_extraction->max_num_features = 8000;
	options.sift_extraction->use_gpu = true;
	options.sift_extraction->gpu_index = "0";
	reader_options.single_camera = true;
	MySiftFeatureExtractor feature_extractor(reader_options, *options.sift_extraction, fibitmaps);
	feature_extractor.Start();
	feature_extractor.Wait();

	std::cout << "Extract Done" << std::endl;
	timer.PrintSeconds();

	//////////////////////////////////////////////
	// Feature Matching
	//////////////////////////////////////////////

	// Options
	int min_match_num_inliners = 15;

	//options.vocab_tree_matching->num_nearest_neighbors = 10;
	options.vocab_tree_matching->num_images = 10;
	//options.vocab_tree_matching->max_num_features = -1;
	//options.vocab_tree_matching->num_images_after_verification = 60;
	options.sift_matching->use_gpu = true;
	options.sift_matching->gpu_index = "0";
	options.sift_matching->multiple_models = true;
	options.sift_matching->guided_matching = true;
	options.sift_matching->min_num_inliers = min_match_num_inliners;

	std::vector<image_t> imageIds;
	for (std::string imageName : imageNames) {
		Image image = database_.ReadImageWithName(imageName);
		imageIds.push_back(image.ImageId());
	}
	std::vector<std::vector<retrieval::ImageScore>> images_scores;
	//MyVocabTreeFeaturePatialMatcher myMatcher(*options.vocab_tree_matching, *options.sift_matching, database_path_);
	//myMatcher.IndexAndMatch(imageIds, visual_index_, image_nums_);

	VocabTreeFeatureMatcher matcher(*options.vocab_tree_matching, *options.sift_matching, database_path_);
	matcher.IndexAndMatch(imageIds, visual_index_, image_nums_, nullptr);
	std::cout << "Match Done" << std::endl;
	timer.PrintSeconds();

	Camera cam1 = database_.ReadCamera(database_.ReadImage(imageIds[0]).CameraId());
	cam1.SetFocalLengthX(focal_length_a);
	cam1.SetFocalLengthY(focal_length_a);
	database_.UpdateCamera(cam1);

	//////////////////////////////////////////////
	// PNP
	//////////////////////////////////////////////

	auto mapper_options = options.mapper->Mapper();
	mapper_options.abs_pose_max_error = 0.5; //1.0
	mapper_options.abs_pose_min_num_inliers = 10; // 5
	mapper_options.abs_pose_min_inlier_ratio = 0.3; // 0.3
	mapper_options.min_focal_length_ratio = 0.3;
	mapper_options.max_focal_length_ratio = 3;

	for (auto name : imageNames) {
		Image& image = database_.ReadImageWithName(name);
		Camera& camera = database_.ReadCamera(image.CameraId());

		//////////////////////////////////////////////////////////////////////////////
		// Search for 2D-3D correspondences
		//////////////////////////////////////////////////////////////////////////////
		std::vector<Eigen::Vector2d> tri_points2D;
		std::vector<Eigen::Vector3d> tri_points3D;

		std::unordered_set<point2D_t> point2D_ids;
		std::unordered_set<point3D_t> point3D_ids;
		point2D_ids.clear();
		point3D_ids.clear();

		point2D_ids.reserve(image.NumPoints2D());
		point3D_ids.reserve(image.NumPoints2D());

		// 对数据库中每一张图片遍历找匹配
		for (auto im : database_.ReadAllImages()) {
			if (im.ImageId() == image.ImageId()) continue;
			if (!reconstruction_.ExistsImage(im.ImageId())) continue;
			// 如果存在匹配
			if (database_.ExistsMatches(image.ImageId(), im.ImageId())) {
				// 得到这些匹配
				auto matches = database_.ReadMatches(image.ImageId(), im.ImageId());
				if (matches.size() > 0) {
					// 对每一对匹配的2D点
					for (auto match : matches) {
						// 如果定位帧中的2D点已经在集合中则跳过
						if (point2D_ids.count(match.point2D_idx1) > 0)
							continue;
						if (image.NumPoints2D() == 0) {
							const FeatureKeypoints keypoints =
								database_.ReadKeypoints(image.ImageId());
							const std::vector<Eigen::Vector2d> points =
								FeatureKeypointsToPointsVector(keypoints);
							image.SetPoints2D(points);
						}
						Point2D& p2d = image.Point2D(match.point2D_idx1);
						// 从已经建好的模型中取对应点，数据库中没有3D点信息的
						Image& corr_image = reconstruction_.Image(im.ImageId());
						if (!corr_image.IsRegistered())
							continue;

						Point2D& p2d_corr = corr_image.Point2D(match.point2D_idx2);
						// 如果参考帧中的2D点没有对应的3D点则跳过
						if (!p2d_corr.HasPoint3D())
							continue;
						// 如果参考帧中对应的3D点在集合中则跳过
						if (point3D_ids.count(p2d_corr.Point3DId()) > 0)
							continue;
						Point3D& p3d = reconstruction_.Point3D(p2d_corr.Point3DId());
						// 将2D点和对应的3D点加入集合
						point2D_ids.emplace(match.point2D_idx1);
						point3D_ids.emplace(p2d_corr.Point3DId());
						tri_points2D.emplace_back(p2d.XY());
						tri_points3D.emplace_back(p3d.XYZ());
					}
				}
			}
		}

		std::cout << tri_points2D.size() << std::endl;

		if (tri_points2D.size() <
			static_cast<size_t>(mapper_options.abs_pose_min_num_inliers)) {
			std::cout << "not enough points for pnp, maybe because of the lack of match points" << std::endl;
			return false;
		}

		std::cout << "Has " << tri_points2D.size() << " pair of points to pnp." << std::endl;
		//////////////////////////////////////////////////////////////////////////////
		// 2D-3D estimation
		//////////////////////////////////////////////////////////////////////////////

		AbsolutePoseEstimationOptions abs_pose_options;
		abs_pose_options.num_threads = mapper_options.num_threads;
		abs_pose_options.num_focal_length_samples = 30;
		abs_pose_options.min_focal_length_ratio = mapper_options.min_focal_length_ratio;
		abs_pose_options.max_focal_length_ratio = mapper_options.max_focal_length_ratio;
		abs_pose_options.ransac_options.max_error = mapper_options.abs_pose_max_error;
		abs_pose_options.ransac_options.min_inlier_ratio = mapper_options.abs_pose_min_inlier_ratio;
		abs_pose_options.ransac_options.min_num_trials = 50;
		abs_pose_options.ransac_options.confidence = 0.9999;
		abs_pose_options.estimate_focal_length = true;

		AbsolutePoseRefinementOptions abs_pose_refinement_options;
		abs_pose_refinement_options.refine_focal_length = false;
		abs_pose_refinement_options.refine_extra_params = true;

		size_t num_inliers;
		std::vector<char> inlier_mask;

		if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D,
			&image.Qvec(), &image.Tvec(), &camera, &num_inliers,
			&inlier_mask)) {
			std::cout << "estimate failure" << std::endl;
			return false;
		}
		std::cout << camera.FocalLengthX() << " " << focal_length_a << std::endl;
		std::cout << num_inliers << std::endl;
		if (num_inliers < static_cast<size_t>(mapper_options.abs_pose_min_num_inliers)) {
			std::cout << "pnp has not enough inliners" << std::endl;
			return false;
		}

		//////////////////////////////////////////////////////////////////////////////
		// Pose refinement
		//////////////////////////////////////////////////////////////////////////////

		if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask,
			tri_points2D, tri_points3D, &image.Qvec(),
			&image.Tvec(), &camera)) {
			std::cout << "refine failure" << std::endl;
			return false;
		}

		std::cout << "Locate Done" << std::endl;
		timer.PrintSeconds();

		std::cout << camera.FocalLengthX() << " " << focal_length_a << std::endl;

		if (name.compare(imageNames[0]) == 0) {
			q1 = image.Qvec();
			t1 = image.Tvec();
			std::cout << q1[0] << " " << q1[1] << " " << q1[2] << " " << q1[3] << " " << t1[0] << " " << t1[1] << " " << t1[2] << std::endl;
		}
		else if (imageNames.size() == 2 & name.compare(imageNames[1]) == 0) {
			q2 = image.Qvec();
			t2 = image.Tvec();
			std::cout << q2[0] << " " << q2[1] << " " << q2[2] << " " << q2[3] << " " << t2[0] << " " << t2[1] << " " << t2[2] << std::endl;
		}
		else {
			std::cout << "奇怪的一个错误啊！" << std::endl;
			return false;
		}
	}
	if (imageNames.size() == 1) {
		if (q1[0] == 1) {
			std::cout << "locate failed, please try again..." << std::endl;
			return false;
		}
	}
	else {
		if (q1[0] == 1 || q2[0] == 1) {
			std::cout << "locate failed, please try again..." << std::endl;
			return false;
		}
	}
	return true;
}


// Class Lifetime
Lifetime::Lifetime() {
	scene_now_id = 0;
	user_now_id = 0;
	scenes.clear();
	users.clear();
}

void Lifetime::Init(char** argv) {
	colmap::InitializeGlog(argv);
}

MyMultiDatabaseSiftFeatureExtractor* Lifetime::GetMyFeaturesExtractor(const std::vector<std::string>& imageNames,
	const std::vector<FIBITMAP*>& fibitmaps, OptionManager& options) {
	//////////////////////////////////////////////
	// Feature Extracting
	//////////////////////////////////////////////
	std::vector<Database*> databases;
	for (auto& s : scenes) {
		databases.emplace_back(&s.database_);
	}
	options.AddMapperOptions();
	options.AddExtractionOptions();
	ImageReaderOptions reader_options = *options.image_reader;
	reader_options.image_list = imageNames;
	reader_options.camera_model = "OPENCV";

	// Options
	options.sift_extraction->num_threads = -1; // when threads num = -1, using max num thread hardware can support to extract features.
	options.sift_extraction->max_num_features = 5000;
	options.sift_extraction->use_gpu = true;
	options.sift_extraction->gpu_index = "0"; // when gpu_index = "-1", use all cuda devices.
	reader_options.single_camera = true;
	MyMultiDatabaseSiftFeatureExtractor* feature_extractor = new MyMultiDatabaseSiftFeatureExtractor(reader_options,
		*options.sift_extraction, fibitmaps, databases);
	return feature_extractor;
}

VocabTreeFeatureMatcher* Lifetime::GetVocabTreeFeatureMatcher(const Scene& s, OptionManager& options, const std::vector<std::string>& imageNames,
	std::vector<image_t>& imageIds) {
	// Options
	options.AddVocabTreeMatchingOptions();


	//////////////////////////////////////////////
	// Feature Matching
	//////////////////////////////////////////////

	// Options
	int min_match_num_inliners = 15;

	//options.vocab_tree_matching->num_nearest_neighbors = 10;
	options.vocab_tree_matching->num_images = 10;
	options.vocab_tree_matching->vocab_tree_path = s.vi_path_;
	//options.vocab_tree_matching->max_num_features = -1;
	//options.vocab_tree_matching->num_images_after_verification = 60;
	options.sift_matching->use_gpu = true;
	options.sift_matching->gpu_index = "0";
	options.sift_matching->multiple_models = true;
	options.sift_matching->guided_matching = true;
	options.sift_matching->min_num_inliers = min_match_num_inliners;

	for (std::string imageName : imageNames) {
		Image image = s.database_.ReadImageWithName(imageName);
		imageIds.push_back(image.ImageId());
	}

	VocabTreeFeatureMatcher* matcher = new VocabTreeFeatureMatcher(*options.vocab_tree_matching, *options.sift_matching, s.database_path_);

}


bool Lifetime::SolvePnP(Scene& s, const IncrementalMapper::Options& mapper_options, const std::string& name,
	Image& image, Camera& camera) {
	//////////////////////////////////////////////////////////////////////////////
	// Search for 2D-3D correspondences
	//////////////////////////////////////////////////////////////////////////////
	std::vector<Eigen::Vector2d> tri_points2D;
	std::vector<Eigen::Vector3d> tri_points3D;

	std::unordered_set<point2D_t> point2D_ids;
	std::unordered_set<point3D_t> point3D_ids;
	point2D_ids.clear();
	point3D_ids.clear();

	point2D_ids.reserve(image.NumPoints2D());
	point3D_ids.reserve(image.NumPoints2D());

	// 对数据库中每一张图片遍历找匹配
	for (auto im : s.database_.ReadAllImages()) {
		if (im.ImageId() == image.ImageId()) continue;
		if (!s.reconstruction_.ExistsImage(im.ImageId())) continue;
		// 如果存在匹配
		if (s.database_.ExistsMatches(image.ImageId(), im.ImageId())) {
			// 得到这些匹配
			auto matches = s.database_.ReadMatches(image.ImageId(), im.ImageId());
			if (matches.size() > 0) {
				// 对每一对匹配的2D点
				for (auto match : matches) {
					// 如果定位帧中的2D点已经在集合中则跳过
					if (point2D_ids.count(match.point2D_idx1) > 0)
						continue;
					if (image.NumPoints2D() == 0) {
						const FeatureKeypoints keypoints =
							s.database_.ReadKeypoints(image.ImageId());
						const std::vector<Eigen::Vector2d> points =
							FeatureKeypointsToPointsVector(keypoints);
						image.SetPoints2D(points);
					}
					Point2D& p2d = image.Point2D(match.point2D_idx1);
					// 从已经建好的模型中取对应点，数据库中没有3D点信息的
					Image& corr_image = s.reconstruction_.Image(im.ImageId());
					if (!corr_image.IsRegistered())
						continue;

					Point2D& p2d_corr = corr_image.Point2D(match.point2D_idx2);
					// 如果参考帧中的2D点没有对应的3D点则跳过
					if (!p2d_corr.HasPoint3D())
						continue;
					// 如果参考帧中对应的3D点在集合中则跳过
					if (point3D_ids.count(p2d_corr.Point3DId()) > 0)
						continue;
					Point3D& p3d = s.reconstruction_.Point3D(p2d_corr.Point3DId());
					// 将2D点和对应的3D点加入集合
					point2D_ids.emplace(match.point2D_idx1);
					point3D_ids.emplace(p2d_corr.Point3DId());
					tri_points2D.emplace_back(p2d.XY());
					tri_points3D.emplace_back(p3d.XYZ());
				}
			}
		}
	}

	std::cout << tri_points2D.size() << std::endl;

	if (tri_points2D.size() <
		static_cast<size_t>(mapper_options.abs_pose_min_num_inliers)) {
		std::cout << "not enough points for pnp, maybe because of the lack of match points" << std::endl;
		return false;
	}

	std::cout << "Has " << tri_points2D.size() << " pair of points to pnp." << std::endl;
	//////////////////////////////////////////////////////////////////////////////
	// 2D-3D estimation
	//////////////////////////////////////////////////////////////////////////////

	AbsolutePoseEstimationOptions abs_pose_options;
	abs_pose_options.num_threads = mapper_options.num_threads;
	abs_pose_options.num_focal_length_samples = 30;
	abs_pose_options.min_focal_length_ratio = mapper_options.min_focal_length_ratio;
	abs_pose_options.max_focal_length_ratio = mapper_options.max_focal_length_ratio;
	abs_pose_options.ransac_options.max_error = mapper_options.abs_pose_max_error;
	abs_pose_options.ransac_options.min_inlier_ratio = mapper_options.abs_pose_min_inlier_ratio;
	abs_pose_options.ransac_options.min_num_trials = 50;
	abs_pose_options.ransac_options.confidence = 0.9999;
	abs_pose_options.estimate_focal_length = true;

	AbsolutePoseRefinementOptions abs_pose_refinement_options;
	abs_pose_refinement_options.refine_focal_length = false;
	abs_pose_refinement_options.refine_extra_params = true;

	size_t num_inliers;
	std::vector<char> inlier_mask;

	if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D,
		&image.Qvec(), &image.Tvec(), &camera, &num_inliers,
		&inlier_mask)) {
		std::cout << "estimate failure" << std::endl;
		return false;
	}
	std::cout << "Num inliers: " << num_inliers << std::endl;
	if (num_inliers < static_cast<size_t>(mapper_options.abs_pose_min_num_inliers)) {
		std::cout << "pnp has not enough inliners" << std::endl;
		return false;
	}

	//////////////////////////////////////////////////////////////////////////////
	// Pose refinement
	//////////////////////////////////////////////////////////////////////////////

	if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask,
		tri_points2D, tri_points3D, &image.Qvec(),
		&image.Tvec(), &camera)) {
		std::cout << "refine failure" << std::endl;
		return false;
	}

	std::cout << "Locate Done" << std::endl;

}

std::string Lifetime::CreateUser(std::string& encoded_data) {
	std::vector<std::string> sv = util::split(encoded_data, ":");
	int scene_id = stoi(sv[1]);
	user_now_id++;
	users.push_back(User(user_now_id, scene_id, Eigen::Matrix4d::Identity()));
	std::string s = "id:" + std::to_string(user_now_id);
	std::cout << s << std::endl;
	return s;
}

std::string Lifetime::RestoreObjs(int& user_id) {
	std::string msg;
	std::cout << "The objs are:" << std::endl;
	for (auto obj : scenes[users[user_id - 1].scene_id_ - 1].objs_) {
		Eigen::Matrix3d obj_r = obj.r_;
		Eigen::Vector3d obj_t = obj.t_;
		float s = obj.scaleFactor;
		Eigen::Matrix3d Rwc = users[user_id - 1].T_.inverse().linear() * obj_r;
		Eigen::Vector3d C = users[user_id - 1].T_.inverse() * obj_t;
		// translate t from colmap to arcore
		C = C / users[user_id - 1].scale_;
		Eigen::Quaterniond q(Rwc);
		// Order as: w,x,y,z
		msg = msg + std::to_string(q.w()) + " " + std::to_string(q.x()) + " " + std::to_string(q.y()) + " "
			+ std::to_string(q.z()) + " " + std::to_string(C.x()) + " " + std::to_string(C.y()) + " " + std::to_string(C.z()) + " " + std::to_string(s);
		msg += ":";
		std::cout << "Rwc: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
			<< " C: " << C.x() << " " << C.y() << " " << C.z()
			<< " scaleFactor: " << s << std::endl;
	}
	return msg;
}

std::string Lifetime::RestoreAllObjsOfOther(int& restore_id, int& user_id) {
	std::string msg;
	std::cout << "The objs are:" << std::endl;
	for (auto obj : scenes[users[restore_id - 1].scene_id_ - 1].objs_) {
		Eigen::Matrix3d obj_r = obj.r_;
		Eigen::Vector3d obj_t = obj.t_;
		float s = obj.scaleFactor;
		Eigen::Matrix3d Rwc = users[user_id - 1].T_.inverse().linear() * obj_r;
		Eigen::Vector3d C = users[user_id - 1].T_.inverse() * obj_t;
		// translate t from colmap to arcore
		C = C / users[user_id - 1].scale_;
		Eigen::Quaterniond q(Rwc);
		// Order as: w,x,y,z
		msg = msg + std::to_string(q.w()) + " " + std::to_string(q.x()) + " " + std::to_string(q.y()) + " "
			+ std::to_string(q.z()) + " " + std::to_string(C.x()) + " " + std::to_string(C.y()) + " " + std::to_string(C.z()) + " " + std::to_string(s);
		msg += ":";
		std::cout << "Rwc: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
			<< " C: " << C.x() << " " << C.y() << " " << C.z()
			<< " scaleFactor: " << s << std::endl;
	}
	return msg;
}

bool Lifetime::CreateObj(std::string& encoded_data) {
	// 数据是Rwc C
	std::vector<std::string> sv = util::split(encoded_data, ":");
	if (sv[0].compare("g") != 0) {
		int user_id = std::stoi(sv[0]);
		Eigen::Quaterniond q(stod(sv[1]), stod(sv[2]), stod(sv[3]), stod(sv[4]));
		Eigen::Matrix3d Rwc = q.matrix();
		Eigen::Vector3d C;
		C << stod(sv[5]), stod(sv[6]), stod(sv[7]);
		float s = stof(sv[8]);
		std::cout << "Recieved an obj:" << std::endl;
		std::cout << "Rwc: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
			<< " C: " << C.x() << " " << C.y() << " " << C.z() << " ScaleFactor: " << s << std::endl;
		// unite the scale
		C *= users[user_id - 1].scale_;
		C = users[user_id - 1].T_ * C;
		Rwc = users[user_id - 1].T_.linear() * Rwc;
		Eigen::Quaterniond qa(Rwc);
		std::cout << "The obj is transformed as:" << std::endl;
		std::cout << "q: " << qa.w() << " " << qa.x() << " " << qa.y() << " " << qa.z()
			<< " C: " << C.x() << " " << C.y() << " " << C.z() << " ScaleFactor: " << s << std::endl;
		scenes[users[user_id - 1].scene_id_ - 1].objs_.push_back(VirtualObj(Rwc, C, s));

		return true;
	}
	else {
		// It's like: g:id:qw:qx:qy:qz:x:y:z:scale
		int user_id = std::stoi(sv[1]);
		Eigen::Quaterniond q(stod(sv[2]), stod(sv[3]), stod(sv[4]), stod(sv[5]));
		Eigen::Matrix3d Rwc = q.matrix();
		Eigen::Vector3d C;
		C << stod(sv[6]), stod(sv[7]), stod(sv[8]);
		float s = stof(sv[9]);
		std::cout << "Update an obj:" << std::endl;
		std::cout << "Rwc: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
			<< " C: " << C.x() << " " << C.y() << " " << C.z() << " ScaleFactor: " << s << std::endl;
		// unite the scale
		C *= users[user_id - 1].scale_;
		C = users[user_id - 1].T_ * C;
		Rwc = users[user_id - 1].T_.linear() * Rwc;
		Eigen::Quaterniond qa(Rwc);
		std::cout << "The obj is updated as:" << std::endl;
		std::cout << "q: " << qa.w() << " " << qa.x() << " " << qa.y() << " " << qa.z()
			<< " C: " << C.x() << " " << C.y() << " " << C.z() << " ScaleFactor: " << s << std::endl;
		scenes[users[user_id - 1].scene_id_ - 1].objs_.pop_back();
		scenes[users[user_id - 1].scene_id_ - 1].objs_.push_back(VirtualObj(Rwc, C, s));
		return true;
	}

}

std::string Lifetime::CreateObjWithOtherUser(int& restore_id, int& user_id) {
	std::string msg;
	std::cout << "The obj in user " << user_id << " is:" << std::endl;
	VirtualObj obj = scenes[users[restore_id - 1].scene_id_ - 1].objs_.back();
	Eigen::Matrix3d obj_r = obj.r_;
	Eigen::Vector3d obj_t = obj.t_;
	float s = obj.scaleFactor;
	Eigen::Matrix3d Rwc = users[user_id - 1].T_.inverse().linear() * obj_r;
	Eigen::Vector3d C = users[user_id - 1].T_.inverse() * obj_t;
	// translate t from colmap to arcore
	C = C / users[user_id - 1].scale_;
	Eigen::Quaterniond q(Rwc);
	// Order as: w,x,y,z
	msg = msg + std::to_string(q.w()) + " " + std::to_string(q.x()) + " " + std::to_string(q.y()) + " "
		+ std::to_string(q.z()) + " " + std::to_string(C.x()) + " " + std::to_string(C.y()) + " " + std::to_string(C.z()) + " " + std::to_string(s);
	msg += ":";
	std::cout << "Rwc: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
		<< " C: " << C.x() << " " << C.y() << " " << C.z()
		<< " scaleFactor: " << s << std::endl;
	return msg;
}

// Encode_data Format is like: 1:initial:A:user_id:img1:qw qx qy qz x1 y1 z1:f1:img2:qw qx qy qz x2 y2 z2:f2
// If single frame: 1:initial:A:user_id:img1:qw qx qy qz x1 y1 z1:f1
// @ brief: parse data when message include picture data
// @ param: singleShot: the camera is monocular or stereo
//			 encoded_data: data received from app
//			 locationNum: which scene is used to locate
//			 model: the locate model(locate use to initialize or fix)
//			 platform: the platform used to build the app
//			 _xyz: position and orientation of the camera
//			 f: focal length
//			 picture: picture data encoded by base64
void ParsePictureRequest(std::string& const encoded_data, int& locationNum, std::string& model,
	Platform platform, int& user_id, std::string& xyz_1, std::string& f_1, std::string& xyz_2,
	std::string f_2, std::vector<BYTE>& picture_1, std::vector<BYTE>& picture_2) {
	std::vector<std::string> sv = util::split(encoded_data, ":");
	locationNum = std::stoi(sv[0]);
	model = sv[1];
	std::string p = sv[2];
	if (p == "A") {
		platform = Platform::ANDROID;
	}
	else if (p == "U") {
		platform = Platform::UNITY;
	}
	else if (p == "I") {
		platform = Platform::IOS;
	}
	else {
		platform = Platform::UNSUPPORTED;
	}
	user_id = std::stoi(sv[3]);
	picture_1 = util::base64_decode(sv[4]);
	xyz_1 = sv[5]; // qt1
	f_1 = sv[6];
	picture_2 = util::base64_decode(sv[4]);
	xyz_2 = sv[7]; // qt2
	f_2 = sv[8];
}

// @ brief: when the scene is determined, use to compute the Translate matrix from ARCore to Colmap
bool Lifetime::ComputeT(std::string& encoded_data, const bool singleShot, const int scene_id) {
	// Format is like: A:user_id:img1:qw qx qy qz x1 y1 z1:f1:img2:qw qx qy qz x2 y2 z2:f2
	// If single frame: A:user_id:img1:qw qx qy qz x1 y1 z1:f1
	Eigen::Transform<double, 3, Eigen::Affine> T;
	Platform platform;
	std::string model, xyz_1, f_1, xyz_2, f_2;
	int user_id, locationNum;
	double scale;
	std::vector<BYTE> picture_1, picture_2;

	ParsePictureRequest(encoded_data, locationNum, model, platform, user_id, xyz_1, f_1,
		xyz_2, f_2, picture_1, picture_2);

	// monocular
	if (singleShot) {
		if (!SaveImageAndComputeTMono(
			picture_1, xyz_1, f_1, T, platform, scene_id)) {
			return false;
		}
	}
	// stereo
	else {
		if (!SaveImageAndComputeTStereo(
			picture_1, picture_2, xyz_1, xyz_2, f_1, T, scale, platform, scene_id)) {
			return false;
		}
	}

	users[user_id - 1].T_ = T;
	users[user_id - 1].scale_ = scale;
	return true;
}

// @ brief: choose scene that is most likely to include picture,
// and use to compute the Translate matrix from ARCore to Colmap
// Format is like: 1:initial:A:user_id:img1:qw qx qy qz x1 y1 z1:f1:img2:qw qx qy qz x2 y2 z2:f2
// If single frame: 1:initial:A:user_id:img1:qw qx qy qz x1 y1 z1:f1
int Lifetime::ComputeT(std::string& encoded_data, const bool singleShot, const std::vector<std::string>& const locations) {
	Eigen::Transform<double, 3, Eigen::Affine> T;
	Platform platform;
	std::string model, xyz_1, f_1, xyz_2, f_2;
	int user_id, locationNum;
	double scale;
	std::vector<BYTE> picture_1, picture_2;

	ParsePictureRequest(encoded_data, locationNum, model, platform, user_id, xyz_1, f_1,
		xyz_2, f_2, picture_1, picture_2);

	std::string sceneLocation;
	if (locationNum != 0) {
		sceneLocation = locations[locationNum - 1];
	}
	else {
		sceneLocation = "";
	}
	if (singleShot) {
		if (!FindMatchedSceneAndComputeTMono(user_id, picture_1, xyz_1, f_1, T, platform, sceneLocation, model))
			return -1;
	}
	else {
		if (!FindMatchedSceneAndComputeTStereo(user_id, picture_1, xyz_1, f_1, T, platform, sceneLocation, model))
			return -1;
	}
	users[user_id - 1].T_ = T;
	users[user_id - 1].scale_ = scale;
	std::string locResult = scenes[users[user_id - 1].scene_id_ - 1].location_;
	for (int i = 0; i < locations.size(); ++i) {
		if (locResult == locations[i]) return i;
	}
}

bool Lifetime::SaveImageAndComputeTMono(std::vector<BYTE>& str_decoded_byte, std::string& xyz,
	std::string f_a, Eigen::Transform<double, 3, Eigen::Affine>& T, Platform platform, const int scene_id) {
	Timer timer;
	timer.Start();

	f_a = util::Trim(f_a);
	double focal_length_a = std::stod(f_a);

	//trans data to freeimage bitmap 
	std::string time = util::getNowTime();
	cv::Mat img = cv::imdecode(str_decoded_byte, cv::IMREAD_COLOR);
	std::string imgName = time + "1.jpg";
	cv::imwrite(temp_image_path + "\\images\\" + imgName, img);

	std::vector<FIBITMAP*> fibitmaps;

	HBITMAP hbmp = util::ConvertCVMatToBMP(img);
	if (hbmp == nullptr) {
		std::cout << "nullptr error 1" << std::endl;
		return false;
	}
	// Grey scale
	FIBITMAP* dib1 = util::ConvertHBitmapToFiBitmap(hbmp);
	fibitmaps.emplace_back(dib1);

	std::vector<std::string> imageNames;
	imageNames.push_back(imgName);

	// 储存pose
	Eigen::Vector4d q1, q2;
	q1 = Eigen::Vector4d::Identity();
	q2 = q1;
	Eigen::Vector3d t1, t2;
	t1 = Eigen::Vector3d::Identity();
	t2 = t1;


	if (!Locate(scenes[scene_id - 1], imageNames, fibitmaps, focal_length_a, q1, q2, t1, t2))
		return false;

	Eigen::Quaterniond qarcore;
	Eigen::Vector3d tarcore;

	// parse xyz
	if (!util::ParseRAndT(platform, xyz, qarcore, tarcore)) {
		return false;
	}

	std::cout << qarcore.w() << " " << qarcore.x() << " " << qarcore.y() << " " << qarcore.z() << " "
		<< tarcore.x() << " " << tarcore.y() << " " << tarcore.z() << std::endl;
	// cal T
	T = util::CalFinalT(q1, q2, t1, t2, qarcore, tarcore);
	return true;
}


bool Lifetime::SaveImageAndComputeTStereo(std::vector<BYTE>& str_decoded_byte1, std::vector<BYTE>& str_decoded_byte2,
	std::string& xyz_a, std::string& xyz_b, std::string f_a, Eigen::Transform<double, 3, Eigen::Affine>& T,
	double& scale, Platform platform, const int scene_id) {
	Timer timer;
	timer.Start();

	// 两幅图A和B
	f_a = util::Trim(f_a);
	double focal_length_a = std::stod(f_a);

	Eigen::Quaterniond qarcore_a, qarcore_b;
	Eigen::Vector3d tarcore_a, tarcore_b;
	if (!util::ParseRAndT(platform, xyz_a, qarcore_a, tarcore_a) || !util::ParseRAndT(platform, xyz_b, qarcore_b, tarcore_b)) {
		return false;
	}


	std::string time = util::getNowTime();
	cv::Mat img1 = cv::imdecode(str_decoded_byte1, cv::IMREAD_COLOR);
	std::cout << img1.cols << " " << img1.rows << std::endl;
	cv::Mat img2 = cv::imdecode(str_decoded_byte2, cv::IMREAD_COLOR);
	std::cout << img2.cols << " " << img2.rows << std::endl;
	std::string img1Name = time + "1.jpg";
	std::string img2Name = time + "2.jpg";

	std::vector<FIBITMAP*> fibitmaps;

	HBITMAP hbmp1 = util::ConvertCVMatToBMP(img1);
	HBITMAP hbmp2 = util::ConvertCVMatToBMP(img2);
	if (hbmp1 == nullptr || hbmp2 == nullptr) {
		std::cout << "nullptr error 1" << std::endl;
		return 0;
	}
	// Grey scale
	FIBITMAP* dib1 = util::ConvertHBitmapToFiBitmap(hbmp1);
	FIBITMAP* dib2 = util::ConvertHBitmapToFiBitmap(hbmp2);

	fibitmaps.emplace_back(dib1);
	fibitmaps.emplace_back(dib2);

	std::vector<std::string> imageNames;
	imageNames.push_back(img1Name);
	imageNames.push_back(img2Name);

	// 储存pose
	Eigen::Vector4d q1, q2;
	q1 = Eigen::Vector4d::Identity();
	q2 = q1;
	Eigen::Vector3d t1, t2;
	t1 = Eigen::Vector3d::Identity();
	t2 = t1;


	if (!Locate(scenes[scene_id - 1], imageNames, fibitmaps, focal_length_a, q1, q2, t1, t2)) {
		return false;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////
	Eigen::Transform<double, 3, Eigen::Affine> T1, T2, T2T, TA, TB;
	Eigen::Quaterniond qcolmap_a(q1[0], q1[1], q1[2], q1[3]);
	Eigen::Vector3d tcolmap_a(t1[0], t1[1], t1[2]);
	Eigen::Quaterniond qcolmap_b(q2[0], q2[1], q2[2], q2[3]);
	Eigen::Vector3d tcolmap_b(t2[0], t2[1], t2[2]);

	// scale = |Cb_colmap-Ca_colmap|/|Cb_arcore-Ca_arcore|
	Eigen::Vector3d ccolmap_a = -qcolmap_a.matrix().transpose() * tcolmap_a;
	Eigen::Vector3d ccolmap_b = -qcolmap_b.matrix().transpose() * tcolmap_b;
	Eigen::Vector3d carcore_b = -qarcore_b.matrix().transpose() * tarcore_b;
	Eigen::Vector3d carcore_a = -qarcore_a.matrix().transpose() * tarcore_a;

	scale = (ccolmap_b - ccolmap_a).norm() / (carcore_b - carcore_a).norm();
	std::cout << "scale: " << scale << std::endl;
	bool test = false;
	if (test && (scale < 0.96 || scale>1.04)) {
		std::cout << "scale error too much" << std::endl;
		return false;
	}
	// 对arcore的尺度进行纠正 C_colmap = C_arcore * scale
	carcore_a *= scale;
	carcore_b *= scale;
	tarcore_a = -qarcore_a.matrix() * carcore_a;
	tarcore_b = -qarcore_b.matrix() * carcore_b;

	// the 1st image
	Eigen::Matrix4d T_arcore_world2Cam_a = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d T_colmap_world2Cam_a = Eigen::Matrix4d::Identity();
	T_arcore_world2Cam_a.block(0, 0, 3, 3) = qarcore_a.matrix();
	T_arcore_world2Cam_a.block(0, 3, 3, 1) = tarcore_a;
	T_colmap_world2Cam_a.block(0, 0, 3, 3) = qcolmap_a.matrix();
	T_colmap_world2Cam_a.block(0, 3, 3, 1) = tcolmap_a;
	T1 = T_arcore_world2Cam_a;
	T2 = T_colmap_world2Cam_a;
	T2T = T2.inverse();
	TA = T2T * T1;

	// the 2nd image
	Eigen::Matrix4d T_arcore_world2Cam_b = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d T_colmap_world2Cam_b = Eigen::Matrix4d::Identity();
	T_arcore_world2Cam_b.block(0, 0, 3, 3) = qarcore_b.matrix();
	T_arcore_world2Cam_b.block(0, 3, 3, 1) = tarcore_b;
	T_colmap_world2Cam_b.block(0, 0, 3, 3) = qcolmap_b.matrix();
	T_colmap_world2Cam_b.block(0, 3, 3, 1) = tcolmap_b;
	T1 = T_arcore_world2Cam_b;
	T2 = T_colmap_world2Cam_b;
	T2T = T2.inverse();
	TB = T2T * T1;

	// Validation
	// 根据TA求图b在ARCore世界坐标系下的相机中心然后与真实值对比
	Eigen::Vector3d c_b_arcore = TA.inverse() * ccolmap_b;
	c_b_arcore /= scale;
	carcore_b /= scale;
	std::cout << c_b_arcore << std::endl;
	std::cout << carcore_b << std::endl;
	double error_a = (c_b_arcore - carcore_b).norm();
	std::cout << error_a << std::endl;
	// 根据TB求图a在ARCore世界坐标系下的相机中心然后与真实值对比
	Eigen::Vector3d c_a_arcore = TB.inverse() * ccolmap_a;
	c_a_arcore /= scale;
	carcore_a /= scale;
	std::cout << c_a_arcore << std::endl;
	std::cout << carcore_a << std::endl;
	double error_b = sqrt((c_a_arcore - carcore_a).norm());
	std::cout << error_b << std::endl;

	// 比较误差 单位m
	/*if (error_a > 0.8 && error_b > 0.8) {
	std::cout << "Localization Error Too Large(> 0.8m)" << std::endl;
	return false;
	}*/
	if (error_a < error_b) {
		T = TA;
	}
	else {
		T = TB;
	}
	return true;
}

bool Lifetime::FindMatchedSceneAndComputeTMono(int user_id, std::vector<BYTE>& str_decoded_byte, std::string& xyz,
	std::string f_a, Eigen::Transform<double, 3, Eigen::Affine>& T, Platform platform,
	std::string sceneLocation, std::string model)
{
	Timer timer;
	timer.Start();
	User& now_user = users[user_id - 1];
	f_a = util::Trim(f_a);
	double focal_length_a = std::stod(f_a);

	//trans data to freeimage bitmap 
	cv::Mat img = cv::imdecode(str_decoded_byte, cv::IMREAD_COLOR);
	HBITMAP hbmp = util::ConvertCVMatToBMP(img);
	if (hbmp == nullptr) {
		std::cout << "nullptr error 1" << std::endl;
		return false;
	}
	// save image
	std::vector<std::string> imageNames;
	std::string time = util::getNowTime();
	std::string imgName = time + "1.jpg";
	cv::imwrite(temp_image_path + "\\images\\" + imgName, img);
	imageNames.push_back(imgName);

	// save pose
	Eigen::Vector4d q1, q2;
	q1 = Eigen::Vector4d::Identity();
	q2 = q1;
	Eigen::Vector3d t1, t2;
	t1 = Eigen::Vector3d::Identity();
	t2 = t1;

	// Extract features
	std::cout << "Start Extracting" << std::endl;
	std::vector<Database*> databases;
	for (auto& s : scenes) {
		databases.emplace_back(&s.database_);
	}
	OptionManager options;
	options.AddMapperOptions();
	options.AddExtractionOptions();
	options.AddVocabTreeMatchingOptions();


	MyMultiDatabaseSiftFeatureExtractor* myExtractor = GetMyFeaturesExtractor(imageNames,
		std::vector<FIBITMAP*>{util::ConvertHBitmapToFiBitmap(hbmp)}, options);
	myExtractor->ExtractFeatrues();
	std::cout << "Extract Done" << std::endl;


	// multi-scenes match
	bool locate_success = false;
	/*std::vector<bool> locate_success(scenes.size(), false);
	std::vector<int> match_nums(scenes.size(), 0);*/
	auto match_threads = now_user.match_thread_.get();
	auto match_scores = now_user.match_scores_.get();
	match_scores->resize(scenes.size(), 0);

	int query_down = 0;

	// if choose specific type scene
	if (sceneLocation != "") {
		for (auto& scene : scenes) {
			// when traverse other type scne, skip
			if (scene.location_ != sceneLocation) continue;
			if (scene.model_ != model) continue;
			std::vector<FIBITMAP*> fitmaps{ util::ConvertHBitmapToFiBitmap(hbmp) };
			match_threads->emplace_back(std::thread(&ThreadLocate, std::ref(now_user), std::ref(scene),
				imageNames, fitmaps, focal_length_a,
				std::ref(q1), std::ref(q2), std::ref(t1), std::ref(t2), std::ref(locate_success), std::ref(query_down)));
		}
	}
	else { // or search in all scenes
		for (auto& scene : scenes) {
			if (scene.model_ != model) continue;
			//if (test++ != 0) break;
			std::vector<FIBITMAP*> fitmaps{ util::ConvertHBitmapToFiBitmap(hbmp) };
			match_threads->emplace_back(std::thread(&ThreadLocate, std::ref(now_user), std::ref(scene),
				imageNames, fitmaps, focal_length_a,
				std::ref(q1), std::ref(q2), std::ref(t1), std::ref(t2), std::ref(locate_success), std::ref(query_down)));
		}
	}

	for (auto& th : *match_threads) {
		th.join();
	}

	// if locate failed return
	std::cout << "locate success" << locate_success << std::endl;
	if (!locate_success) {
		now_user.scene_id_ = -1;
		match_threads->clear();
		match_scores->clear();
		myExtractor->DeleteImageInfoInDatabase(-1);
		return false;
	}

	int max_num = 0;
	int max_id = 0;
	for (int i = 0; i < match_scores->size(); ++i) {
		if (max_num < (*match_scores)[i]) {
			max_num = (*match_scores)[i];
			max_id = i;
		}
	}
	now_user.scene_id_ = max_id + 1;
	myExtractor->DeleteImageInfoInDatabase(max_id);


	Scene* now_scene = &scenes[now_user.scene_id_ - 1];

	std::cout << "Locate Down" << std::endl;
	std::cout << "Locate result:\nQuaternion:\nX:" << q1.x() << ", Y: " << q1.y() << " , Z: " << q1.z() << ", W: " << q1.w() << "\n"
		<< "Transform:\nX : " << t1.x() << ", Y : " << t1.y() << ", Z : " << t1.z() << std::endl;

	// parse xyz
	Eigen::Quaterniond qarcore;
	Eigen::Vector3d tarcore;

	if (!util::ParseRAndT(platform, xyz, qarcore, tarcore)) {
		match_threads->clear();
		match_scores->clear();
		return false;
	}
	std::cout << "Arcore Parse result:\nQuaternion:\nX:" << qarcore.x() << ", Y: " << qarcore.y() << " , Z: "
		<< qarcore.z() << ", W: " << qarcore.w() << "\n"
		<< "Transform:\nX : " << tarcore.x() << ", Y : " << tarcore.y() << ", Z : " << tarcore.z() << std::endl;

	// cal T
	T = util::CalFinalT(q1, q2, t1, t2, qarcore, tarcore);

	// clear thread vector
	match_threads->clear();
	match_scores->clear();
	return true;
}

bool Lifetime::FindMatchedSceneAndComputeTStereo(int user_id, std::vector<BYTE>& str_decoded_byte, std::string& xyz,
	std::string f_a, Eigen::Transform<double, 3, Eigen::Affine>& T, Platform platform, std::string sceneLocation, std::string model) {
	// TODO: stereo method
}


// todo: test
bool Lifetime::Locate(Scene& s, std::vector<std::string>& imageNames, std::vector<FIBITMAP*>& fibitmaps,
	double focal_length_a, Eigen::Vector4d& q1, Eigen::Vector4d& q2,
	Eigen::Vector3d& t1, Eigen::Vector3d& t2) {
	Timer timer;
	timer.Start();
	OptionManager options;
	options.AddMapperOptions();
	options.AddExtractionOptions();
	options.AddVocabTreeMatchingOptions();

	//////////////////////////////////////////////
	// 特征点处理部分
	//////////////////////////////////////////////
	// Extract Features
	std::cout << "Start Extracting" << std::endl;
	MyMultiDatabaseSiftFeatureExtractor* myExtractor = GetMyFeaturesExtractor(imageNames,
		fibitmaps, options);
	myExtractor->ExtractFeatrues();
	std::cout << "Extract Done" << std::endl;
	timer.PrintSeconds();

	// Features Matching
	std::vector<image_t> imageIds;
	VocabTreeFeatureMatcher* myMatcher = GetVocabTreeFeatureMatcher(s, options, imageNames, imageIds);
	myMatcher->IndexAndMatch(imageIds, s.visual_index_, s.image_nums_, nullptr);
	std::cout << "Match Done" << std::endl;
	timer.PrintSeconds();


	//////////////////////////////////////////////
	// 定位部分
	//////////////////////////////////////////////

	Camera cam1 = s.database_.ReadCamera(s.database_.ReadImage(imageIds[0]).CameraId());
	cam1.SetFocalLengthX(focal_length_a);
	cam1.SetFocalLengthY(focal_length_a);
	s.database_.UpdateCamera(cam1);


	auto mapper_options = options.mapper->Mapper();
	mapper_options.abs_pose_max_error = 0.5; //1.0
	mapper_options.abs_pose_min_num_inliers = 10; // 5
	mapper_options.abs_pose_min_inlier_ratio = 0.3; // 0.3
	mapper_options.min_focal_length_ratio = 0.3;
	mapper_options.max_focal_length_ratio = 3;
	for (auto name : imageNames) {
		Image& image = s.database_.ReadImageWithName(name);
		Camera& camera = s.database_.ReadCamera(image.CameraId());
		bool pnpResult = SolvePnP(s, mapper_options, name, image, camera);
		if (!pnpResult) {
			return false;
		}
		std::cout << camera.FocalLengthX() << " " << focal_length_a << std::endl;

		if (name.compare(imageNames[0]) == 0) {
			q1 = image.Qvec();
			t1 = image.Tvec();
			std::cout << q1[0] << " " << q1[1] << " " << q1[2] << " " << q1[3] << " " << t1[0] << " " << t1[1] << " " << t1[2] << std::endl;
		}
		else if (imageNames.size() == 2 & name.compare(imageNames[1]) == 0) {
			q2 = image.Qvec();
			t2 = image.Tvec();
			std::cout << q2[0] << " " << q2[1] << " " << q2[2] << " " << q2[3] << " " << t2[0] << " " << t2[1] << " " << t2[2] << std::endl;
		}
		else {
			std::cout << "奇怪的一个错误啊！" << std::endl;
			return false;
		}
	}

	if (imageNames.size() == 1) {
		if (q1[0] == 1) {
			std::cout << "locate failed, please try again..." << std::endl;
			return false;
		}
	}
	else {
		if (q1[0] == 1 || q2[0] == 1) {
			std::cout << "locate failed, please try again..." << std::endl;
			return false;
		}
	}
	return true;
}

// todo: test
void Lifetime::ThreadLocate(User& const u, Scene& s, OptionManager& options, std::vector<std::string> imageNames, std::vector<FIBITMAP*> fibitmaps,
	const double focal_length_a, Eigen::Vector4d& q1, Eigen::Vector4d& q2,
	Eigen::Vector3d& t1, Eigen::Vector3d& t2, bool& match_suceess, int& query_down) {

	// user ptr
	auto match_score = u.match_scores_.get();
	auto match_cv = u.match_cv_.get();
	auto match_mutex = u.match_mutex_.get();

	std::vector<image_t> imageIds;
	VocabTreeFeatureMatcher* myMatcher = GetVocabTreeFeatureMatcher(s, options, imageNames, imageIds);
	myMatcher->IndexAndMatch(imageIds, s.visual_index_, s.image_nums_, nullptr);
	std::cout << "Match Done" << std::endl;

	//todo: complete vocabulary tree function( try to use one vb tree)
	/*myMatcher.QueryScore(query_options, imageIds, s.visual_index_, images_scores);
	std::cout << "=====Query down====" << std::endl;
	float average_score = 0;
	for (auto& image_score : images_scores[0]) {
		average_score += image_score.score;
	}
	average_score /= images_scores[0].size();
	(*match_score)[s.id_ - 1] = (average_score);
	query_mutex.lock();
	++query_down;
	query_mutex.unlock();
	std::unique_lock<std::mutex> lck(*match_mutex);
	while (!u.match_down_)
		match_cv->wait(lck);
	std::cout << "scene id: " << s.id_ << " user scene id: " << u.scene_id_ << std::endl;
	if (s.id_ != u.scene_id_) {
		return;
	}*/
	//print scores
	//std::cout << "Scene id " << s.id_ << ":" << std::endl;
	//for (int i = 0; i < images_scores.size(); ++i) {
	//	std::cout << "Image No." + i << ":" << std::endl;
	//	for (auto& image_score : images_scores[i]) {
	//		std::cout << "ImageId-" << image_score.image_id << ": " << image_score.score << std::endl;
	//	}
	//}

	Camera cam1 = s.database_.ReadCamera(s.database_.ReadImage(imageIds[0]).CameraId());
	cam1.SetFocalLengthX(focal_length_a);
	cam1.SetFocalLengthY(focal_length_a);
	s.database_.UpdateCamera(cam1);

	//////////////////////////////////////////////
	// 定位部分
	//////////////////////////////////////////////

	// Get match nums of image
	//int matchSum = 0;
	//for (auto name : imageNames) {
	//	Image& image = s.database_.ReadImageWithName(name);
	//	for (auto im : s.database_.ReadAllImages()) {
	//		if (im.ImageId() == image.ImageId()) continue;
	//		if (!s.reconstruction_.ExistsImage(im.ImageId())) continue;
	//		// 如果存在匹配
	//		if (s.database_.ExistsMatches(image.ImageId(), im.ImageId())) {
	//			// 得到这些匹配
	//			matchSum += s.database_.ReadMatches(image.ImageId(), im.ImageId()).size();
	//		}
	//	}
	//	std::cout << "Image " << name << " has match: " << matchSum << std::endl;
	//}

	//////////////////////////////////////////////
	// PNP
	//////////////////////////////////////////////

	auto mapper_options = options.mapper->Mapper();
	mapper_options.abs_pose_max_error = 0.5; //1.0
	mapper_options.abs_pose_min_num_inliers = 10; // 5
	mapper_options.abs_pose_min_inlier_ratio = 0.3; // 0.3
	mapper_options.min_focal_length_ratio = 0.3;
	mapper_options.max_focal_length_ratio = 3;

	for (auto name : imageNames) {
		Image& image = s.database_.ReadImageWithName(name);
		Camera& camera = s.database_.ReadCamera(image.CameraId());

		bool pnpResult = SolvePnP(s, mapper_options, name, image, camera);
		if (!pnpResult) {
			myMatcher->ClearMatch(image.ImageId());
			return;
		}
		std::cout << camera.FocalLengthX() << " " << focal_length_a << std::endl;

		if (name.compare(imageNames[0]) == 0) {
			q1 = image.Qvec();
			t1 = image.Tvec();
			std::cout << q1[0] << " " << q1[1] << " " << q1[2] << " " << q1[3] << " " << t1[0] << " " << t1[1] << " " << t1[2] << std::endl;
		}
		else if (imageNames.size() == 2 & name.compare(imageNames[1]) == 0) {
			q2 = image.Qvec();
			t2 = image.Tvec();
			std::cout << q2[0] << " " << q2[1] << " " << q2[2] << " " << q2[3] << " " << t2[0] << " " << t2[1] << " " << t2[2] << std::endl;
		}
		else {
			std::cout << "奇怪的一个错误啊！" << std::endl;
			myMatcher->ClearMatch(image.ImageId());
			return;
		}
	}

	if (imageNames.size() == 1) {
		if (q1[0] == 1) {
			std::cout << "locate failed, please try again..." << std::endl;
			return;
		}
	}
	else {
		if (q1[0] == 1 || q2[0] == 1) {
			std::cout << "locate failed, please try again..." << std::endl;
			return;
		}
	}
	match_suceess = true;
	return;
}





VLOC_API std::vector<std::string> split(std::string& Input, const char* Regex) {
	std::vector<std::string> Result;
	int pos = 0;
	int npos = 0;
	int regexlen = strlen(Regex);
	while ((npos = Input.find(Regex, pos)) != -1) {
		std::string tmp = Input.substr(pos, npos - pos);
		Result.push_back(tmp);
		pos = npos + regexlen;
	}
	Result.push_back(Input.substr(pos, Input.length() - pos));
	return Result;
}

