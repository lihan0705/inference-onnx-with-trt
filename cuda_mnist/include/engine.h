#include "NvInfer.h"
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>


using namespace nvinfer1;


// Options for the network
struct Options {
    // Use 16 bit floating point type for inference
    bool FP16 = false;
    // Batch sizes to optimize for.
    std::vector<int32_t> optBatchSizes;
    // Maximum allowable batch size
    int32_t maxBatchSize = 16;
    // Max allowable GPU memory to be used for model conversion, in bytes.
    // Applications should allow the engine builder as much workspace as they can afford;
    // at runtime, the SDK allocates no more than this and typically less.
    size_t maxWorkspaceSize = 4000000000;
    // GPU device index
    int deviceIndex = 0;
};

// Class to extend TensorRT logger
class Logger : public ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

class Engine{
    public:
        Engine(const Options& option);
        ~Engine();

        bool build(std::string onnx_path);
        bool load_network();
        bool run_inference(const std::vector<cv::Mat>& input_img_list, std::vector<std::vector<float>>& feat_vec);

    private:
     // Converts the engine options into a string
    std::string serializeEngineOptions(const Options& options);

    void getGPUUUIDs(std::vector<std::string>& gpuUUIDs);

    bool doesFileExist(const std::string& filepath);

    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options& m_options;
    Logger m_logger;       
};
