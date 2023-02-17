#include <tools/depth_estimation.h>
#include <Python.h>

namespace colmap{

DepthEstimation::DepthEstimation() {
    checkpointPath = "checkpoints/depthformer_swinl_w7_22k_nyu.pth";
    configPath = "configs/depthformer/depthformer_swinl_22k_w7_nyu.py";
    show = false;
    showDir = "results";
}

DepthEstimation::DepthEstimation(std::string configPath, std::string checkpointPath,
                                bool show, std::string showDir) : configPath(configPath), 
                                checkpointPath(checkpointPath), show(show), showDir(showDir) {

}

void DepthEstimation::EstimateDepth() {
    Py_Initialize();
    if(!Py_IsInitialized()) {
		std::cout << "python init fail" << std::endl;
		return;
	}
    PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('/home/lzy/my_project/Graduation_Project/depth_estimate')");

    PyObject* pModule = PyImport_ImportModule("read_depth_info");
	if( pModule == NULL ){
		std::cout <<"module not found" << std::endl;
		return;
	}

    PyObject* pFunc = PyObject_GetAttrString(pModule, "main");
	if( !pFunc || !PyCallable_Check(pFunc)){
		std::cout <<"not found function main" << std::endl;
		return;
	}

	PyObject* args = PyTuple_New(4);
    PyTuple_SetItem(args, 0, Py_BuildValue("s", configPath));
    PyTuple_SetItem(args, 1, Py_BuildValue("s", checkpointPath));
    PyTuple_SetItem(args, 2, Py_BuildValue("s", show));
    PyTuple_SetItem(args, 3, Py_BuildValue("s", showDir));
	
    
    PyObject* pRet = PyObject_CallObject(pFunc, args);
    
    int imageNum = PyList_Size(pRet);

    std::vector<std::vector<std::vector<double>>> result;

    result.resize(imageNum);

    for(int i = 0; i < imageNum; ++i) {
        PyObject* listItem = PyList_GetItem(pRet, i);
        int imageHeight = PyList_Size(listItem);
        result[i].resize(imageHeight);
        for(int heightIndex = 0; heightIndex < imageHeight; ++heightIndex) {
            PyObject* pixelItem = PyList_GetItem(listItem, imageHeight);
            int imageWidth = PyList_Size(pixelItem);
            result[i][heightIndex].resize(imageWidth); 
            for(int widthIndex = 0; widthIndex < imageWidth; ++widthIndex) {
                PyArg_Parse(PyList_GetItem(pixelItem, widthIndex), "d", &result[i][heightIndex][widthIndex]);
            }
            Py_DECREF(pixelItem);
        }
        Py_DECREF(listItem);
    }
    
    for(auto& image : result) {
        std::cout << "[ ";
        for(auto& height : image) {
            std::cout << "[ ";
            for(auto& width : height) {
                std::cout << width << ", ";
            }
            std::cout << "]," << std::endl;
        }
        std::cout << "]," << std:: endl;
   }
    
    Py_DECREF(args);
    Py_DECREF(pRet);
    Py_Finalize();


}

void DepthEstimation::GetDepthInfo(image_t image_id, double x, double y) {

}

}