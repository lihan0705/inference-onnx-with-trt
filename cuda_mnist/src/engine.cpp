#include <iostream>
#include "../include/engine.h"
#include "NvOnnxParser.h"


void Logger::log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING){
        std::cout << msg << std::endl;       
        }
}


bool Engine::doesFileExist(const std::string &filepath){
    std::ifstream f(filepath.c_str());
    return f.good()
}
/*
class CMyClass {
    CMyClass(int x, int y);
    int m_x;
    int m_y;
};

CMyClass::CMyClass(int x, int y) : m_y(y), m_x(m_y)
{
};
*/
Engine::Engine(const Options &options):m_options(options){}

Engine::build(std::string onnxModelPath){
        m_engineName = serializeEngineOptions(m_options);

        if (doesFileExist(m_engineName)) {
            std::cout << "Engine found, not regenerating..." << std::endl;
            return true;
        }

        std::cout << "Engine not found, generating..." << std::endl;
        // build instance builder
        // auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
        nvinfer1::IBuilder* builder = createInferBuilder(logger);

        if (!builder) {
            return false;
        }
        
        //creat a network defintion
        uint32_t flag = 1U <<static_cast<uint32_t>
        (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 

        // Create a parser for reading the onnx file.
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);
        nvonnxparser::IParser*  parser = createParser(*network, logger);
        if (!parser) {
            return false;
        }

        // We are going to first read the onnx file into memory, then pass that buffer to the parser.
        // Had our onnx model file been encrypted, this approach would allow us to first decrypt the buffer.
        if (!parser->parseFromFile(onnxModelPath.data(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        {
            std::cerr << ": failed to parse onnx model file, please check the onnx version and trt support op!"
                    << std::endl;
            exit(-1);
        }

        // build config 
        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
        if (!config) {
            return false;
        }

        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);

        nvinfer1::IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);


        // Save the input height, width, and channels.
        // Require this info for inference.
        const auto input = network->getInput(0);
        const auto output = network->getOutput(0);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t inputC = inputDims.d[1];
        int32_t inputH = inputDims.d[2];
        int32_t inputW = inputDims.d[3];

        IOptimizationProfile* defaultProfile = builder->createOptimizationProfile();
        defaultProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
        defaultProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, Dims4(1, inputC, inputH, inputW));
        defaultProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
        config->addOptimizationProfile(defaultProfile);
        config->setMaxWorkspaceSize(m.option.maxWorkspaceSize);

        if (m_options.FP16){
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        // Write the engine to disk
        std::ofstream outfile(m_engineName, std::ofstream::binary);
        outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

        std::cout << "Success, saved engine to " << m_engineName << std::endl;
        return true;
}