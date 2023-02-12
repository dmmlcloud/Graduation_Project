#include "mongoose.h"
#include "io.h"
#include "VLoc.h"

Lifetime lifetime;

const std::vector<std::string> locationsList = { "outdoor", "indoor", "inpassage", "outpassage", "platform", "outplatform", "escape", "inpassageFix" };

static void ev_handler(struct mg_connection* c, int ev, void* p) { // 处理接收到的编码

	std::string name = "string";
	char buffer[8000000];

	// UnityWebRequest
	if (ev == MG_EV_HTTP_REQUEST) {
		struct http_message* hm = (struct http_message*)p;

		mg_get_http_var(&hm->body, name.c_str(), buffer, sizeof(buffer));
		std::string encoded_data(buffer);
		int encoded_len = encoded_data.length();

		// test
		std::cout << "receive data:" << encoded_data << std::endl;
		std::cout << "receive data's length:" << encoded_len << std::endl;

		Scene* now_scene;

		if (encoded_len < 50) { // 收到的编码长度小于50，只能是创建用户或者恢复objs
			if (encoded_data[0] == CREATE_USER) { // 接收到连接创建用户并指定场景请求
												  // c:scene_id
				encoded_data += ":1";
				std::string user_id = lifetime.CreateUser(encoded_data);
				mg_send_head(c, 200, user_id.length(), "Content-Type: text/plain");
				mg_printf(c, "%s", user_id);
			}
			else if (encoded_data[0] == RESTORE_OBJ) { // 接收到用户传递所有objs请求
													   // r:user_id				   
				std::vector<std::string> sv = split(encoded_data, ":");
				int user_id = std::stoi(sv[1]);

				if (lifetime.scenes[lifetime.users[user_id - 1].scene_id_ - 1].objs_.empty()) {
					mg_send_head(c, 200, 7, "Content-Type: text/plain");
					mg_printf(c, "no objs");
					return;
				}
				std::string msg = lifetime.RestoreObjs(user_id);
				mg_send_head(c, 200, msg.length(), "Content-Type: text/plain");
				mg_printf(c, "%s", msg.c_str());
			}
			else { // 无效的编码
				mg_send_head(c, 200, 12, "Content-Type: text/plain");
				mg_printf(c, "Invalid data");
			}
		}
		else if (encoded_len < 300) { // 编码长度小于300，是收到一个obj的位姿并加入vector中或者更新最近一个obj的位姿
									  // user_id:qw:qx:qy:qz:tx:ty:tz:scale

			if (!lifetime.CreateObj(encoded_data)) {
				return;
			}
			mg_send_head(c, 200, 19, "Content-Type: text/plain");
			mg_printf(c, "Create Successfully");
		}
		else { // 编码长度大于300，是收到初始化的图片编码，计算变换T和尺度scale, 单帧的话把第二个参数设为true
			   // Format is like: 当前场景代码（Outdoor:1, Indoor:2, Inpassage:3, Outpassage:4, Platform:5, Outplatform:6, Escape:7）
			   // 模式代码(I/F):系统代码(A/U/I):user_id:img1:qw qx qy qz x1 y1 z1:f1:img2:qw qx qy qz x2 y2 z2:f2
			int now_scene = lifetime.FindMatchAndComputeT(encoded_data, true, locationsList);
			++now_scene;
			if (now_scene != 0) {
				std::string msg = "scene_id:" + std::to_string(now_scene);
				std::cout << msg;
				mg_send_head(c, 200, 10, "Content-Type: text/plain");
				mg_printf(c, "%s", msg);
				return;
			}
			else {
				mg_send_head(c, 200, 11, "Content-Type: text/plain");
				mg_printf(c, "Init Failed");
			}
		}
	}
	return;
}

void getModels(std::string path, std::vector<std::string>& res) {
	struct _finddata_t fileInfo;
	std::string p;
	intptr_t hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileInfo);
	if (hFile == -1)
	{
		std::cout << "cannot match the path" << std::endl;
		return;
	}
	do
	{
		if (fileInfo.attrib & _A_SUBDIR) {
			if (strcmp(fileInfo.name, "..") != 0 && strcmp(fileInfo.name, ".") != 0)
				res.emplace_back(fileInfo.name);
		}
	} while (!_findnext(hFile, &fileInfo));
	_findclose(hFile);
}

int main(int argc, char** argv) {
	lifetime.Init(argv);
	std::cout << "Initing..." << std::endl;
	std::vector<std::string> typeList;
	std::string models_path = "D:\\3DReconstruction\\Badaling\\models";
	getModels(models_path, typeList);
	VirtualObj vo1;
	Eigen::Quaterniond q(1, 0, 0, 0);
	vo1.r_ = q.matrix();
	vo1.scaleFactor = 1;
	vo1.t_ = Eigen::Vector3d(0, 0, 0);
	for (std::string type : typeList) {
		std::vector<std::string> sceneList;
		std::string type_path = models_path + "\\" + type;
		getModels(type_path, sceneList);
		for (std::string scene : sceneList) {
			std::vector<std::string> partList;
			std::string scene_path = type_path + "\\" + scene;
			getModels(scene_path, partList);
			for (std::string part : partList) {
				++lifetime.scene_now_id;
				std::string nowPath = scene_path + "\\" + part;
				Scene s(lifetime.scene_now_id, nowPath);
				s.location_ = scene;
				s.model_ = type;
				s.ReadReconstruction();
				std::cout << "Read " << scene << " " << part << " Reconstruction Done" << std::endl;
				s.ReadDatabase();
				std::cout << "Read " << scene << " " << part << " Database Done" << std::endl;
				s.ReadVI();
				std::cout << "Read " << scene << " " << part << " Visual Index Done" << std::endl;
				s.objs_.emplace_back(vo1);
				lifetime.scenes.emplace_back(s);
			}
		}
	}

	// place model at origin

	std::cout << "Init Done" << std::endl;

	struct mg_mgr mgr;
	struct mg_connection* c;

	mg_mgr_init(&mgr, NULL);
	c = mg_bind(&mgr, "8003", ev_handler);
	mg_set_protocol_http_websocket(c);

	while (1) {
		mg_mgr_poll(&mgr, 100);
	}
	mg_mgr_free(&mgr);

	return 0;
}