#pragma once

#include <mitsuba/core/properties.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/vaehelper.h>

NAMESPACE_BEGIN(mitsuba)
MI_VARIANT VaeHelper<Float, Spectrum>::VaeHelper() { 
    //m_kernelEpsScale = kernelEpsScale;
    m_kernelEpsScale = 1.0f;
}

MI_VARIANT VaeHelper<Float, Spectrum>::~VaeHelper() {}

MI_VARIANT void VaeHelper<Float, Spectrum>::sampleGaussianVector(float *data, Sampler *sampler, int nVars) {
    bool odd = nVars % 2;
    int idx  = 0;
    for (int i = 0; i < nVars / 2; ++i) {
        Point2f uv = warp::square_to_std_normal(sampler->next_2d());
        data[idx]  = dr::slice(uv[0], 0);
        ++idx;
        data[idx] = dr::slice(uv[1], 0);
        ++idx;
    }
    if (odd)
        data[idx] = dr::slice(warp::square_to_std_normal(sampler->next_2d())[0], 0);
}

MI_VARIANT
bool VaeHelper<Float, Spectrum>::prepare(const Scene *scene, const std::vector<ref<Shape>> &shapes, const Spectrum &sigmaT, const Spectrum &albedo, float g, float eta, const std::string &modelName, const std::string &absModelName,
                                         const std::string &angularModelName, const std::string &outputDir, int batchSize, const PolyFitConfig &pfConfig) {
    m_batchSize = batchSize;

    // Build acceleration data structure to compute polynomial fits efficiently
    m_averageMedium = MediumParameters<Spectrum>(albedo, g, eta, sigmaT);

    if (modelName == "None") {
        m_polyOrder = pfConfig.order;
        precomputePolynomials(shapes, m_averageMedium, pfConfig);
        return true; // Dont load any ML models
    }
    std::string modelPath         = outputDir + "models/" + modelName + "/";
    std::string absModelPath      = outputDir + "models_abs/" + absModelName + "/";
    std::string angularModelPath  = outputDir + "models_angular/" + angularModelName + "/";
    std::string configFile        = modelPath + "training-metadata.json";
    std::string angularConfigFile = angularModelPath + "training-metadata.json";

    m_config    = VaeConfig(configFile, angularModelName != "None" ? angularConfigFile : "", outputDir);
    m_polyOrder = m_config.polyOrder;
    precomputePolynomials(shapes, m_averageMedium, pfConfig);
    // VaeHelper<Float, Spectrum>::prepare(scene, shapes, sigmaT, albedo, g, eta, modelName, absModelName, angularModelName, outputDir, batchSize, pfConfig);

    std::string graph_path         = modelPath + "frozen_model.pb";
    std::string abs_graph_path     = absModelPath + "frozen_model.pb";
    std::string angular_graph_path = angularModelPath + "frozen_model.pb";

    std::string absVariableDir     = absModelPath + "/variables/";
    std::string scatterVariableDir = modelPath + "/variables/";
    std::cout << "Loading model " << modelName << std::endl;
    std::cout << "absVariableDir: " << absVariableDir << std::endl;
    std::cout << "scatterVariableDir: " << scatterVariableDir << std::endl;

    std::cout << "m_config.configName: " << m_config.configName << std::endl;

    if (m_config.configName == "FinalSharedLs/AbsSharedSimComplex") {
        std::cout << "Using VAE Shared Similarity Theory Based\n";
        /*scatterModel = std::unique_ptr<ScatterModelBase>(
            new ScatterModelSimShared<3, 4>(scatterVariableDir, absVariableDir, m_config.stats, "mlsPolyLS3", "LS",
                                            true));*/
    } else if (m_config.configName == "VaeFeaturePreSharedSim2/AbsSharedSimComplex") {
        std::cout << "Using VAE Shared Similarity Theory Based\n";
        /*scatterModel = std::unique_ptr<ScatterModelBase>(
            new ScatterModelSimShared<3, 4>(scatterVariableDir, absVariableDir, m_config.stats, "mlsPolyLS3", "LS",
                                            false));*/
    } else if (m_config.configName == "VaeEfficient" || m_config.configName == "VaeEfficientHardConstraint" || m_config.configName == "VaeEfficientSim") {
        std::cout << "Using VAE Efficient\n";
        /*scatterModel = std::unique_ptr<ScatterModelBase>(
            new ScatterModelEfficient<3, 4>(scatterVariableDir, absVariableDir, m_config.stats, "mlsPolyLS3"));*/
    } else {
        Log(Info, "COFIG NOT RECOGNIZED: Using VAE Shared Similarity Theory Based (EPSILON SCALE)");
        Log(Info, "Prediction space is %f", m_config.predictionSpace);
        scatterModel = std::unique_ptr<ScatterModelBase<Float, Spectrum>>(new ScatterModelSimShared<3, 4, 64, 64, Float, Spectrum, Sampler>(scatterVariableDir, absVariableDir, m_config.stats, "mlsPolyLS3", m_config.predictionSpace, true));

        // std::cout << "Using VAE Baseline\n";
        // scatterModel = std::unique_ptr<ScatterModelBase>(new ScatterModel<3, 8>(scatterVariableDir, absVariableDir, m_config.stats, "mlsPolyLS3"));
    }

    m_effectiveAlbedo = Volpath3D<Float, Spectrum>::effectiveAlbedo(albedo);
    Log(Info, "Done preprocessing");
    return true;
}

MI_IMPLEMENT_CLASS_VARIANT(VaeHelper, Object, "VaeHelper")
MI_INSTANTIATE_CLASS(VaeHelper)
NAMESPACE_END(mitsuba)