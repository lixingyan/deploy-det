#include "TrtModel.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include "cuda_runtime.h"
#include <sys/stat.h>

inline bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}


//!初始化推理引擎，如果没有推理引擎，则从onnx模型构建推理引擎
TrtModel::TrtModel(std::string onnxfilepath, bool fp16, int maxbatch)
    :onnx_file_path{onnxfilepath}, FP16(fp16), maxBatch(maxbatch)
{
    auto idx = onnx_file_path.find(".onnx");
    auto basename = onnx_file_path.substr(0, idx);
    m_enginePath = basename + ".engine";

    if (file_exists(m_enginePath)){
        std::cout << "start building model from engine file: " << m_enginePath;
        this->Runtime();
    }else{
        std::cout << "start building model from onnx file: " << onnx_file_path;
        this->genEngine();
        this->Runtime();
    }
    this->trtIOMemory();
}


bool TrtModel::constructNetwork(
    std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network,
    std::unique_ptr<nvinfer1::IBuilderConfig>& config,
    std::unique_ptr<nvonnxparser::IParser>& parser)
{
    // 读取onnx模型文件开始构建模型
    auto parsed = parser->parseFromFile(onnx_file_path.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if(!parsed){
        std::cout<<" (T_T)~~~ ,Failed to parse onnx file."<<std::endl;
        return false;
    }

    auto input = network->getInput(0);
    auto input_dims = input->getDimensions();
    auto profile = builder->createOptimizationProfile(); 

    // 配置最小、最优、最大范围
    input_dims.d[0] = 1;                                                         
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = maxBatch;
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    // 判断是否使用半精度优化模型
    if(FP16)  config->setFlag(nvinfer1::BuilderFlag::kFP16);


    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);


    // 设置默认设备类型为 DLA
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);

    // 获取 DLA 核心支持情况
    int numDLACores = builder->getNbDLACores();
    if (numDLACores > 0) {
        std::cout << "DLA is available. Number of DLA cores: " << numDLACores << std::endl;

        // 设置 DLA 核心
        int coreToUse = 0; // 选择第一个 DLA 核心（可以根据实际需求修改）
        config->setDLACore(coreToUse);
        std::cout << "Using DLA core: " << coreToUse << std::endl;
    } else {
        std::cerr << "DLA not available on this platform, falling back to GPU." << std::endl;
        
        // 如果 DLA 不可用，则设置 GPU 回退
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        config->setDefaultDeviceType(nvinfer1::DeviceType::kGPU);
    }
    return true;
}


bool TrtModel::genEngine(){

    // 打印模型编译过程的日志
    sample::gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);

    // 创建builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if(!builder){
        std::cout<<" (T_T)~~~, Failed to create builder."<<std::endl;
        return false;
    }

    // 声明显性batch，创建network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if(!network){
        std::cout<<" (T_T)~~~, Failed to create network."<<std::endl;
        return false;
    }

    // 创建 config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config){
        std::cout<<" (T_T)~~~, Failed to create config."<<std::endl;
        return false;
    }

    // 创建parser 从onnx自动构建模型，否则需要自己构建每个算子
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if(!parser){
        std::cout<<" (T_T)~~~, Failed to create parser."<<std::endl;
        return false;
    }

    // 为网络设置config, 以及parse
    auto constructed = this->constructNetwork(builder, network, config, parser);
    if (!constructed)
    { 
        std::cout<<" (T_T)~~~,  Failed to Create an optimization profile and calibration configuration. (•_•)~ "<<std::endl;
        return false;
    }

    builder->setMaxBatchSize(1);
    // config->setMaxWorkspaceSize(1<<30);     /*在比较新的版本中，这个接口已经被弃用*/
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);      /*在新的版本中被使用*/

    auto profileStream = samplesCommon::makeCudaStream();
    if(!profileStream){
        std::cout<<" (T_T)~~~, Failed to makeCudaStream."<<std::endl;
        return false;
    }
    config->setProfileStream(*profileStream);

    // 创建序列化引擎文件
    auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if(!plan){
        std::cout<<" (T_T)~~~, Failed to SerializedNetwork."<<std::endl;
        return false;
    }

    //! 检查输入部分是否符合要求
    if(network->getNbInputs() == 1){
        auto mInputDims = network->getInput(0)->getDimensions();
        std::cout<<" ✨~ model input dims: "<<mInputDims.nbDims <<std::endl;
        for(size_t ii=0; ii<mInputDims.nbDims; ++ii){
            std::cout<<" ✨^_^ model input dim"<<ii<<": "<<mInputDims.d[ii] <<std::endl;
        }
    } else {
        std::cout<<" (T_T)~~~, please check model input shape "<<std::endl;
        return false;
    }

    //! 检查输出部分是否符合要求
    if(network->getNbOutputs() == 1){
        for(size_t i=0; i<network->getNbOutputs(); ++i){
            auto mOutputDims = network->getOutput(i)->getDimensions();
            std::cout<<" ✨~ model output dims: "<<mOutputDims.nbDims <<std::endl;
            for(size_t jj=0; jj<mOutputDims.nbDims; ++jj){
                std::cout<<" ✨^_^ model output dim"<<jj<<": "<<mOutputDims.d[jj] <<std::endl;
            }
        }
        
    } else {
        std::cout<<" (T_T)~~~, please check model output shape "<<std::endl;
        return false;
    }

    // 序列化保存推理引擎文件文件
    std::ofstream engine_file(m_enginePath, std::ios::binary);
    if(!engine_file.good()){
        std::cout<<" (T_T)~~~, Failed to open engine file"<<std::endl;
        return false;
    }
    engine_file.write((char *)plan->data(), plan->size());
    engine_file.close();

    std::cout << " ~~Congratulations! 🎉🎉🎉~  Engine build success!!! ✨✨✨~~ " << std::endl;

}


std::vector<unsigned char>TrtModel::load_engine_file()
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(m_enginePath, std::ios::binary);
    if(!engine_file.is_open()){
        std::cout<<" (T_T)~~~, Unable to load engine file O_O."<<std::endl;
        return engine_data;
    }
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
    return engine_data;
}


bool TrtModel::trtIOMemory() {

    m_inputDims   = m_context->getBindingDimensions(0);
    m_outputDims = m_context->getBindingDimensions(1);

    std::cout<<"after optimizer input shape: "<<m_context->getBindingDimensions(0)<<std::endl;
    std::cout<<"after optimizer output shape: "<<m_context->getBindingDimensions(1)<<std::endl;
    this->kInputH=m_inputDims.d[2];
    this->kInputW=m_inputDims.d[3];
    this->ImageC=m_inputDims.d[1];

    CUDA_CHECK(cudaStreamCreate(&m_stream));

    m_inputSize = m_inputDims.d[0] * m_inputDims.d[1] * m_inputDims.d[2] * m_inputDims.d[3] * sizeof(float);
    m_imgArea = m_inputDims.d[2] * m_inputDims.d[3];
    m_outputSize = m_outputDims.d[0] * m_outputDims.d[1] * m_outputDims.d[2] * sizeof(float);


    // 改进这里的内存分配错误处理
    if (cudaMallocHost(&m_inputMemory[0], m_inputSize)!= cudaSuccess) {
        std::cerr << "Failed to allocate host memory for input. Error code: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        // 可以选择抛出异常或返回错误码，这里简单示例为返回false
        return false;
    }
    if (cudaMallocHost(&m_outputMemory[0], m_outputSize)!= cudaSuccess) {
        std::cerr << "Failed to allocate host memory for output. Error code: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        // 释放之前已分配的内存（如果需要）
        cudaFreeHost(m_inputMemory[0]);
        // 返回错误码或抛出异常
        return false;
    }
    if (cudaMalloc(&m_inputMemory[1], m_inputSize)!= cudaSuccess) {
        std::cerr << "Failed to allocate device memory for input. Error code: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        // 释放之前已分配的内存（如果需要）
        cudaFreeHost(m_inputMemory[0]);
        cudaFreeHost(m_outputMemory[0]);
        // 返回错误码或抛出异常
        return false;
    }
    if (cudaMalloc(&m_outputMemory[1], m_outputSize)!= cudaSuccess) {
        std::cerr << "Failed to allocate device memory for output. Error code: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        // 释放之前已分配的内存（如果需要）
        cudaFreeHost(m_inputMemory[0]);
        cudaFreeHost(m_outputMemory[0]);
        cudaFree(m_inputMemory[1]);
        // 返回错误码或抛出异常
        return false;
    }

    // 创建m_bindings，之后再寻址就直接从这里找
    m_bindings[0] = m_inputMemory[1];
    m_bindings[1] = m_outputMemory[1];

    return true; // 成功时返回true
}


bool TrtModel::Runtime(){

    // 初始化trt插件
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    
    // 加载序列化的推理引擎
    auto plan = this->load_engine_file();

    // / 打印模型推理过程的日志
    sample::setReportableSeverity(sample::Severity::kINFO);

    // 创建推理引擎
    m_runtime.reset(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if(!m_runtime){
        std::cout<<" (T_T)~~~, Failed to create runtime."<<std::endl;
        return false;
    }

    // 反序列化推理引擎
    m_engine.reset(m_runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if(!m_engine){
        std::cout<<" (T_T)~~~, Failed to deserialize."<<std::endl;
        return false;
    }

    // 获取优化后的模型的输入维度和输出维度
    // int nbBindings = m_engine->getNbBindings(); // trt8.5 以前版本
    int nbBindings = m_engine->getNbIOTensors();  // trt8.5 以后版本
    for (int i = 0; i < nbBindings; i++)
    {
        auto dims = m_engine->getBindingDimensions(i);

        auto size = dims.d[0]*dims.d[1]*dims.d[2]*dims.d[3]*sizeof(float);

        auto name = m_engine->getBindingName(i);
        auto bingdingType = m_engine->getBindingDataType(i);

        std::cout << "Binding " << i << ": " << name << ", size: " << size << ", dims: " << dims << ", type: " << int(bingdingType) << std::endl;
    }


    // 推理执行上下文
    m_context.reset(m_engine->createExecutionContext());
    if(!m_context){
        std::cout<<" (T_T)~~~, Failed to create ExecutionContext."<<std::endl;
        return false;
    }

    auto input_dims = m_context->getBindingDimensions(0);
    input_dims.d[0] = maxBatch;

    // 设置当前推理时，input大小
    m_context->setBindingDimensions(0, input_dims);

    std::cout << " ~~Congratulations! 🎉🎉🎉~  runtime deserialize success!!! ✨✨✨~~ " << std::endl;

}


cv::Mat TrtModel::letterbox(cv::Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->kInputH;
	*neww = this->kInputW;
	cv::Mat dstimg;
	if (srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->kInputH; 
			*neww = int(this->kInputW / hw_scale);
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((this->kInputW - *neww) * 0.5);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->kInputW - *neww - *left, cv::BORDER_CONSTANT, 0);
		}
		else {
			*newh = (int)this->kInputH * hw_scale;
			*neww = this->kInputW;
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*top = (int)(this->kInputH - *newh) * 0.5;
			cv::copyMakeBorder(dstimg, dstimg, *top, this->kInputH - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 0);
		}
	}
	else {
		cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}

void TrtModel::base2process(cv::Mat &warp_dst)
{

    /*Preprocess -- host端进行normalization和BGR2RGB, NHWC->NCHW*/
    if (warp_dst.data == nullptr) std::cerr<<("ERROR: Image file not founded! Program terminated"); 
    if (warp_dst.channels()==3){
        int index {0};
        int offset_ch0 = m_imgArea * 0;
        int offset_ch1 = m_imgArea * 1;
        int offset_ch2 = m_imgArea * 2;
        for (int i = 0; i < m_inputDims.d[2]; i++) {
            for (int j = 0; j < m_inputDims.d[3]; j++) {
                index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
                m_inputMemory[0][offset_ch2++] = warp_dst.data[index + 0] / 255.0f;
                m_inputMemory[0][offset_ch1++] = warp_dst.data[index + 1] / 255.0f;
                m_inputMemory[0][offset_ch0++] = warp_dst.data[index + 2] / 255.0f;
            }
    }
    /*Preprocess -- 将host的数据移动到device上*/
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyHostToDevice, m_stream));
    }else if (warp_dst.channels()==1){
        int index {0};
        int offset_ch = m_imgArea*0 ;

        for (int i = 0; i < m_inputDims.d[2]; i++) {
            for (int j = 0; j < m_inputDims.d[3]; j++) {
                index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
                m_inputMemory[0][offset_ch++] = warp_dst.data[index + 0] / 255.0f;
            }
    }
    /*Preprocess -- 将host的数据移动到device上*/
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyHostToDevice, m_stream));

    }

}


void TrtModel::nms(std::vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // 降序排列

	std::vector<bool> remove_flags(input_boxes.size(),false);

	auto iou = [](const BoxInfo& box1,const BoxInfo& box2)
	{
		float xx1 = std::max(box1.x1, box2.x1);
		float yy1 = std::max(box1.y1, box2.y1);
		float xx2 = std::min(box1.x2, box2.x2);
		float yy2 = std::min(box1.y2, box2.y2);
		// 交集
		float w = std::max(0.0f, xx2 - xx1 + 1);
		float h = std::max(0.0f, yy2 - yy1 + 1);
		float inter_area = w * h;
		// 并集
		float union_area = std::max(0.0f,box1.x2-box1.x1) * std::max(0.0f,box1.y2-box1.y1)
						   + std::max(0.0f,box2.x2-box2.x1) * std::max(0.0f,box2.y2-box2.y1) - inter_area;
		return inter_area / union_area;
	};
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if(remove_flags[i]) continue;
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if(remove_flags[j]) continue;
			if(input_boxes[i].label == input_boxes[j].label && iou(input_boxes[i],input_boxes[j])>=this->kNmsThresh)
			{
				remove_flags[j] = true;
			}
		}
	}
	int idx_t = 0;
    // remove_if()函数 remove_if(beg, end, op) //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &remove_flags](const BoxInfo& f) { return remove_flags[idx_t++]; }), input_boxes.end());
}

std::vector<BoxInfo> TrtModel::doInference(cv::Mat& frame) {

    int newh = 0, neww = 0, padh = 0, padw = 0;

    
    std::vector<BoxInfo> generate_boxes {};

    auto warp_dst = this->letterbox(frame, &newh, &neww, &padh, &padw);     // letterbox

    this->base2process(warp_dst);

    bool status = this->m_context->enqueueV2((void**)m_bindings, m_stream, nullptr);
    if (!status) std::cerr << "Failed to execute inference." << std::endl;
    
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemory[0], m_outputMemory[1], m_outputSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    float* pdata = (float*)m_outputMemory[0];

    float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
    for(int i = 0; i < m_outputDims.d[1]; ++i){ // 遍历所有的num_pre_boxes
        int index = i * m_outputDims.d[2];      // prob[b*num_pred_boxes*(classes+5)]  
        float obj_conf = pdata[index + 4];      // 置信度分数
        if (obj_conf > this->kConfThresh)       // 大于阈值
        {
            float* max_class_pos = std::max_element(pdata + index + 5, pdata + index + m_outputDims.d[2]);
            (*max_class_pos) *= obj_conf;       // 最大的类别分数*置信度
            if ((*max_class_pos) > this->kConfThresh)  // 再次筛选
            { 
                //const int class_idx = classIdPoint.x;
                float cx = pdata[index];   // x
                float cy = pdata[index+1]; // y
                float w = pdata[index+2];  // w
                float h = pdata[index+3];  // h
                float xmin = (cx - padw - 0.5 * w)*ratiow ;
                float ymin = (cy - padh - 0.5 * h)*ratioh ;
                float xmax = (cx - padw + 0.5 * w)*ratiow ;
                float ymax = (cy - padh + 0.5 * h)*ratioh ;
                generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, (*max_class_pos), max_class_pos-(pdata + index + 5) });
            }
        }
    }
    this->nms(generate_boxes);
    return generate_boxes;

}

// 在图像上绘制检测结果
bool TrtModel::draw_bbox(cv::Mat& frame, std::vector<BoxInfo> result){
    for (size_t i = 0; i < result.size(); ++i){
        int xmin = int(result[i].x1);
		int ymin = int(result[i].y1);
        cv::Scalar color = cv::Scalar(this->COLORS_HEX[result[i].label+6][0], this->COLORS_HEX[result[i].label+6][1], this->COLORS_HEX[result[i].label+6][2]);
        std::string label = cv::format("%.2f", result[i].score);
        label = this->CLASS_NAMES[result[i].label] + ":" + label;
        cv::rectangle(frame, cv::Point(xmin, ymin), cv::Point(int(result[i].x2), int(result[i].y2)), color, 1);
        cv::putText(frame, label, cv::Point(xmin,ymin-10), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,255), 1);

	}
    result.clear();
}

