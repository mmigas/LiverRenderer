#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mitsuba/render/sss_particle_tracer.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "json.hpp"

using json = nlohmann::json;

NAMESPACE_BEGIN(mitsuba)
template <typename Float, typename Spectrum> class /*MI_EXPORT_LIB*/ NetworkHelpers {
public:
    static void onb(const Eigen::Vector3f &n, Eigen::Vector3f &b1, Eigen::Vector3f &b2) {
        float sign    = copysignf(1.0f, n[2]);
        const float a = -1.0f / (sign + n[2]);
        const float b = n[0] * n[1] * a;
        b1            = Eigen::Vector3f(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
        b2            = Eigen::Vector3f(b, sign + n[1] * n[1] * a, -n[1]);
    }

    // static constexpr int nChooseK(int n, int k) {
    //     float result = 1.0f;
    //     for (int i = 1; i <= k; ++i) {
    //         result *= (float) (n - (k - i)) / ((float) i);
    //     }
    //     return std::round(result);
    // }

    static inline constexpr int nChooseK(int n, int k) { return (k == 0 || n == k) ? 1 : nChooseK(n - 1, k - 1) + nChooseK(n - 1, k); }

    static inline constexpr int nPolyCoeffs(int polyOrder) { return nChooseK(3 + polyOrder, polyOrder); }

    static inline constexpr int nInFeatures(int polyOrder) { return nPolyCoeffs(polyOrder) + 3; }

    static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    template <int Size> static drjit::Array<Float, Size> loadVectorDynamic(const std::string &filename) {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open())
            std::cout << "FILE NOT FOUND: " << filename << std::endl;
        std::cout << "Loading " << filename << std::endl;
        int32_t nDims;
        f.read(reinterpret_cast<char *>(&nDims), sizeof(nDims));
        int32_t size;
        f.read(reinterpret_cast<char *>(&size), sizeof(size));
        drjit::Array<Float, Size> ret(size);
        for (int i = 0; i < size; ++i) {
            f.read(reinterpret_cast<char *>(&ret[i]), sizeof(ret[i]));
        }
        return ret;
    }

    template <int Rows, int Cols> static drjit::Matrix<Float, Rows> loadMatrixDynamic(const std::string &filename) {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open())
            std::cout << "FILE NOT FOUND " << filename << std::endl;
        std::cout << "Loading " << filename << std::endl;

        int32_t nDims;
        f.read(reinterpret_cast<char *>(&nDims), sizeof(nDims));

        int32_t rows, cols;
        f.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        f.read(reinterpret_cast<char *>(&cols), sizeof(cols));
        Eigen::MatrixXf ret(rows, cols);
        drjit::Matrix<Float, Rows> matrix(rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < rows; ++j) {
                f.read(reinterpret_cast<char *>(&matrix[i][j]), sizeof(matrix[i][j]));
            }
        }
        return ret;
    }

    static Eigen::Vector3f localToWorld(const Eigen::Vector3f &inPos, const Eigen::Vector3f &inNormal, const Eigen::Vector3f &outPosLocal, bool predictInTangentSpace) {
        if (predictInTangentSpace) {
            Eigen::Vector3f tangent1, tangent2;
            onb(inNormal, tangent1, tangent2);
            return inPos + outPosLocal[0] * tangent1 + outPosLocal[1] * tangent2 + outPosLocal[2] * inNormal;
        } else {
            return inPos + outPosLocal;
        }
    }

    template <size_t PolyOrder = 3, bool useSimilarityTheory = false>
    static Eigen::Matrix<Float, nInFeatures(PolyOrder), 1> preprocessFeatures(const Spectrum &albedo, float g, float ior, const Spectrum &sigmaT, const dr::Array<Float, nPolyCoeffs(PolyOrder)> &shapeFeatures, float albedoMean, float albedoStdInv, float gMean, float gStdInv, const dr::Array<Float, nPolyCoeffs(PolyOrder)> &shapeFeatMean,
                                                                              const dr::Array<Float, nPolyCoeffs(PolyOrder)> &shapeFeatStdInv) {
        float effectiveAlbedo;
        if (useSimilarityTheory) {
            Spectrum sigmaS  = albedo * sigmaT;
            Spectrum sigmaA  = sigmaT - sigmaS;
            Spectrum albedoP = (1 - g) * sigmaS / ((1 - g) * sigmaS + sigmaA);
            effectiveAlbedo  = dr::slice(dr::mean(Volpath3D<Float, Spectrum>::effectiveAlbedo(albedoP)), 0);
        } else {
            effectiveAlbedo = dr::slice(dr::mean(Volpath3D<Float, Spectrum>::effectiveAlbedo(albedo)), 0);
        }
        float albedoNorm = (effectiveAlbedo - albedoMean) * albedoStdInv;
        float gNorm      = (g - gMean) * gStdInv;
        float iorNorm    = 2.0f * (ior - 1.25f);
        // Eigen::Matrix<float, nPolyCoeffs(PolyOrder), 1> shapeFeaturesNorm = (shapeFeatures - shapeFeatMean).cwiseProduct(shapeFeatStdInv);
        Eigen::Matrix<float, nInFeatures(PolyOrder), 1> features;
        // features.segment(0, nPolyCoeffs(PolyOrder)) = shapeFeaturesNorm;
        features[nPolyCoeffs(PolyOrder)]     = albedoNorm;
        features[nPolyCoeffs(PolyOrder) + 1] = gNorm;
        features[nPolyCoeffs(PolyOrder) + 2] = iorNorm;
        return features;
        return Eigen::Matrix<float, nInFeatures(PolyOrder), 1>();
    }
};

template <size_t PolyOrder /*= 3*/, size_t LayerWidth /*= 64*/, typename Float, typename Spectrum> class MI_EXPORT_LIB AbsorptionModel {
public:
    typedef drjit::Matrix<Float, 20> ShapeVector;
    /*// typedef Eigen::Matrix<Float, NetworkHelpers<Float, Spectrum>::nPolyCoeffs(PolyOrder), 1> ShapeVector;

    AbsorptionModel() {}

    // AbsorptionModel(const std::string &variablePath, const VaeConfig &config) {
    AbsorptionModel(const std::string &variablePath, const json &stats, const std::string &shapeFeaturesName) {
        absorption_mlp_fcn_0_biases  = NetworkHelpers<Float, Spectrum>::loadVectorDynamic(variablePath + "/absorption_mlp_fcn_0_biases.bin");
        absorption_mlp_fcn_1_biases  = NetworkHelpers<Float, Spectrum>::loadVectorDynamic(variablePath + "/absorption_mlp_fcn_1_biases.bin");
        absorption_mlp_fcn_2_biases  = NetworkHelpers<Float, Spectrum>::loadVectorDynamic(variablePath + "/absorption_mlp_fcn_2_biases.bin");
        absorption_dense_bias        = NetworkHelpers<Float, Spectrum>::loadVectorDynamic(variablePath + "/absorption_dense_bias.bin");
        absorption_mlp_fcn_0_weights = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic(variablePath + "/absorption_mlp_fcn_0_weights.bin");
        absorption_mlp_fcn_1_weights = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic(variablePath + "/absorption_mlp_fcn_1_weights.bin");
        absorption_mlp_fcn_2_weights = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic(variablePath + "/absorption_mlp_fcn_2_weights.bin");
        absorption_dense_kernel      = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic(variablePath + "/absorption_dense_kernel.bin");

        m_gMean            = stats["g_mean"][0];
        m_gStdInv          = stats["g_stdinv"][0];
        m_albedoMean       = stats["effAlbedo_mean"][0];
        m_albedoStdInv     = stats["effAlbedo_stdinv"][0];
        std::string degStr = std::to_string(PolyOrder);
        for (int i = 0; i < NetworkHelpers<Float, Spectrum>::nPolyCoeffs(PolyOrder); ++i) {
            m_shapeFeatMean[i]   = stats[shapeFeaturesName + "_mean"][i];
            m_shapeFeatStdInv[i] = stats[shapeFeaturesName + "_stdinv"][i];
        }
    }

    float run(Spectrum albedo, float g, float ior, const Spectrum &sigmaT, const ShapeVector &polyCoeffs) const {
        Eigen::Matrix<float, NetworkHelpers<Float, Spectrum>::nInFeatures(PolyOrder), 1> input = NetworkHelpers<Float, Spectrum>::preprocessFeatures<PolyOrder>(
            albedo, g, ior, sigmaT, polyCoeffs, m_albedoMean, m_albedoStdInv, m_gMean, m_gStdInv, m_shapeFeatMean, m_shapeFeatStdInv);
        Eigen::Matrix<float, LayerWidth, 1> x = (absorption_mlp_fcn_0_weights * input + absorption_mlp_fcn_0_biases).cwiseMax(0.0f);
        x                                     = (absorption_mlp_fcn_1_weights * x + absorption_mlp_fcn_1_biases).cwiseMax(0.0f);
        x                                     = (absorption_mlp_fcn_2_weights * x + absorption_mlp_fcn_2_biases).cwiseMax(0.0f);
        Eigen::Matrix<float, 1, 1> x2         = absorption_dense_kernel * x + absorption_dense_bias;
        return NetworkHelpers<Float, Spectrum>::sigmoid(x2[0]);
    }

    Eigen::Matrix<float, LayerWidth, 1> absorption_mlp_fcn_0_biases;
    Eigen::Matrix<float, LayerWidth, 1> absorption_mlp_fcn_1_biases;
    Eigen::Matrix<float, LayerWidth, 1> absorption_mlp_fcn_2_biases;
    Eigen::VectorXf absorption_dense_bias;

    Eigen::Matrix<float, LayerWidth, NetworkHelpers<Float, Spectrum>::nInFeatures(PolyOrder)> absorption_mlp_fcn_0_weights;
    Eigen::Matrix<float, LayerWidth, LayerWidth> absorption_mlp_fcn_1_weights;
    Eigen::Matrix<float, LayerWidth, LayerWidth> absorption_mlp_fcn_2_weights;
    Eigen::Matrix<float, 1, LayerWidth> absorption_dense_kernel;

    ShapeVector m_shapeFeatMean, m_shapeFeatStdInv;
    float m_albedoMean, m_albedoStdInv, m_gMean, m_gStdInv;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW*/
};

template <typename Float, typename Spectrum> class MI_EXPORT_LIB ScatterModelBase : public Object {
public:
    MI_IMPORT_TYPES(Sampler)
    virtual std::pair<Vector<Float, 3>, Float> run(const Vector<Float, 3> &inPos, const Vector<Float, 3> &inDir, const Spectrum &albedo, float g, float ior, const Spectrum &sigmaT, Float polyScaleFactor, const dr::Matrix<Float, nPolyCoeffs(3)> &polyCoeffs, Sampler *sampler, drjit::Matrix<Float, 3> toAsTransform) const = 0;

    virtual ~ScatterModelBase() {}
    /*MI_DECLARE_CLASS()*/
};

MI_EXTERN_CLASS(ScatterModelBase)

template <size_t PolyOrder /*= 3*/, size_t NLatent /*= 4*/, size_t LayerWidth /*= 64*/, size_t PreLayerWidth /*= 64*/, typename Float, typename Spectrum, typename Sampler>
class MI_EXPORT_LIB ScatterModelSimShared : public ScatterModelBase<Float, Spectrum> {
public:
    ScatterModelSimShared() {}

    ScatterModelSimShared(const std::string &variablePath, const std::string &absVariablePath, const json &stats, const std::string &shapeFeaturesName, const std::string &predictionSpace = "LS", bool useEpsilonSpace = false) {
        m_useEpsilonSpace                          = useEpsilonSpace;
        absorption_dense_bias                      = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<1>(variablePath + "/absorption_dense_bias.bin");
        absorption_mlp_fcn_0_biases                = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<32>(variablePath + "/absorption_mlp_fcn_0_biases.bin");
        scatter_decoder_fcn_fcn_0_biases           = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<LayerWidth>(variablePath + "/scatter_decoder_fcn_fcn_0_biases.bin");
        scatter_decoder_fcn_fcn_1_biases           = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<LayerWidth>(variablePath + "/scatter_decoder_fcn_fcn_1_biases.bin");
        scatter_decoder_fcn_fcn_2_biases           = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<LayerWidth>(variablePath + "/scatter_decoder_fcn_fcn_2_biases.bin");
        shared_preproc_mlp_2_shapemlp_fcn_0_biases = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<PreLayerWidth>(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_biases.bin");
        shared_preproc_mlp_2_shapemlp_fcn_1_biases = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<PreLayerWidth>(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_biases.bin");
        shared_preproc_mlp_2_shapemlp_fcn_2_biases = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<PreLayerWidth>(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_biases.bin");
        scatter_dense_2_bias                       = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<3>(variablePath + "/scatter_dense_2_bias.bin");
        // scatter_decoder_fcn_fcn_0_weights           = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic        (variablePath + "/scatter_decoder_fcn_fcn_0_weights.bin");
        // scatter_decoder_fcn_fcn_1_weights           = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic        (variablePath + "/scatter_decoder_fcn_fcn_1_weights.bin");
        // scatter_decoder_fcn_fcn_2_weights           = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic        (variablePath + "/scatter_decoder_fcn_fcn_2_weights.bin");
        // shared_preproc_mlp_2_shapemlp_fcn_0_weights = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic        (variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_weights.bin");
        // shared_preproc_mlp_2_shapemlp_fcn_1_weights = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic        (variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_weights.bin");
        // shared_preproc_mlp_2_shapemlp_fcn_2_weights = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic        (variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_weights.bin");
        // scatter_dense_2_kernel                      = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic        (variablePath + "/scatter_dense_2_kernel.bin");

        // absorption_dense_kernel      = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic(variablePath + "/absorption_dense_kernel.bin");
        // absorption_mlp_fcn_0_weights = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic(variablePath + "/absorption_mlp_fcn_0_weights.bin");

        m_gMean            = stats["g_mean"][0];
        m_gStdInv          = stats["g_stdinv"][0];
        m_albedoMean       = stats["effAlbedo_mean"][0];
        m_albedoStdInv     = stats["effAlbedo_stdinv"][0];
        std::string degStr = std::to_string(PolyOrder);
        // for (int i = 0; i < NetworkHelpers<Float, Spectrum>::nPolyCoeffs(PolyOrder); ++i) {
        //     m_shapeFeatMean[i]   = stats[shapeFeaturesName + "_mean"][i];
        //     m_shapeFeatStdInv[i] = stats[shapeFeaturesName + "_stdinv"][i];
        // }
        // if (predictionSpace != "AS") {
        //     for (int i = 0; i < 3; ++i) {
        //         m_outPosMean[i] = stats["outPosRel" + predictionSpace + "_mean"][i];
        //         m_outPosStd[i]  = 1.0f / float(stats["outPosRel" + predictionSpace + "_stdinv"][i]);
        //     }
        // }

        m_useAsSpace = predictionSpace == "AS";

        std::cout << "predictionSpace: " << predictionSpace << std::endl;
        std::cout << "m_useAsSpace: " << m_useAsSpace << std::endl;

        // m_gMean            = 0.0f;
        // m_gStdInv          = 1.0f;
        // m_albedoMean       = 0.0f;
        // m_albedoStdInv     = 1.0f;
        // std::string degStr = std::to_string(PolyOrder);
        // for (int i = 0; i < NetworkHelpers::nPolyCoeffs(PolyOrder); ++i) {
        //     m_shapeFeatMean[i]   = 0.0f;
        //     m_shapeFeatStdInv[i] = 1.0f;
        // }
        // for (int i = 0; i < 3; ++i) {
        //     m_outPosMean[i] = 0.0f;
        //     m_outPosStd[i]  = 1.0f;
        // }
    }

    std::pair<Vector<Float, 3>, Float> run(const Vector<Float, 3> &inPos, const Vector<Float, 3> &inDir, const Spectrum &albedo, float g, float ior, const Spectrum &sigmaT, Float fitScaleFactor, const dr::Matrix<Float, nPolyCoeffs(3)> &polyCoeffs, Sampler *sampler, dr::Matrix<Float, 3>) const override {
        Spectrum sigmaTp = Volpath3D<Float, Spectrum>::getSigmaTp(albedo, g, sigmaT);
        // auto x = NetworkHelpers<Float, Spectrum>::preprocessFeatures<PolyOrder, true>(albedo, g, ior, sigmaT, polyCoeffs, m_albedoMean, m_albedoStdInv,
        // m_gMean,m_gStdInv, m_shapeFeatMean, m_shapeFeatStdInv);

        // Apply the preprocessing network
        /*Eigen::Matrix<float, PreLayerWidth, 1> features =
                (shared_preproc_mlp_2_shapemlp_fcn_0_weights * x + shared_preproc_mlp_2_shapemlp_fcn_0_biases).
                cwiseMax(0.0f);
        features = (shared_preproc_mlp_2_shapemlp_fcn_1_weights * features +
                    shared_preproc_mlp_2_shapemlp_fcn_1_biases).cwiseMax(0.0f);
        features = (shared_preproc_mlp_2_shapemlp_fcn_2_weights * features +
                    shared_preproc_mlp_2_shapemlp_fcn_2_biases).cwiseMax(0.0f);

        // Compute absorption
        Eigen::Matrix<float, 32, 1> absTmp = (absorption_mlp_fcn_0_weights * features + absorption_mlp_fcn_0_biases)
                .cwiseMax(0.0f);
        Eigen::Matrix<float, 1, 1> a = absorption_dense_kernel * absTmp + absorption_dense_bias;
        float absorption = NetworkHelpers<Float, Spectrum>::sigmoid(a[0]);

        if (dr::any_or<true>(sampler->next_1d() > absorption)) {
            absorption = 0.0f; // nothing gets absorbed instead
        } else {
            return std::make_pair(inPos, 1.0f); // all is absorbed
        }*/
        // Concatenate features with random numbers
        /*Eigen::Matrix<float, NLatent, 1> latent(NLatent);
        VaeHelper<Float, Spectrum>::sampleGaussianVector(latent.data(), sampler, NLatent);

        Eigen::Matrix<float, PreLayerWidth + NLatent, 1> featLatent;
        featLatent << latent, features;

        Eigen::Matrix<float, 64, 1> y = (scatter_decoder_fcn_fcn_0_weights * featLatent +
                                         scatter_decoder_fcn_fcn_0_biases).cwiseMax(0.0f);
        y = (scatter_decoder_fcn_fcn_1_weights * y + scatter_decoder_fcn_fcn_1_biases).cwiseMax(0.0f);
        y = (scatter_decoder_fcn_fcn_2_weights * y + scatter_decoder_fcn_fcn_2_biases).cwiseMax(0.0f);*/
        // Eigen::Vector3f outPos = scatter_dense_2_kernel /** y */+ scatter_dense_2_bias;

        /*if (m_useEpsilonSpace) {
            if (m_useAsSpace) {
                /*Vector<Float, 3> tmp = toAsTransform.preMult(Vector<Float, 3>(outPos[0], outPos[1], outPos[2])) /
                fitScaleFactor;
                outPos = Eigen::Vector3f(tmp.x, tmp.y, tmp.z) + inPos;
            } else {
                outPos = NetworkHelpers<Float, Spectrum>::localToWorld(inPos, -inDir, outPos, true);
                outPos = inPos + (outPos - inPos) / fitScaleFactor;
            }
        } else {
            outPos = outPos.cwiseProduct(m_outPosStd) + m_outPosMean;
            outPos = NetworkHelpers<Float, Spectrum>::localToWorld(inPos, -inDir, outPos, true);

        }*/
        return std::make_pair(Vector<Float, 3>(0.0f, 0.0f, 0.0f), 0.0f);
    }

    bool m_useEpsilonSpace{}, m_useAsSpace{};

    // Eigen::Matrix<float, 32, PreLayerWidth> absorption_mlp_fcn_0_weights;
    dr::Array<dr::Array<Float, PreLayerWidth>, 32> absorption_mlp_fcn_0_weights;
    // Eigen::Matrix<float, 32, 1> absorption_mlp_fcn_0_biases;
    dr::Array<Float, 32> absorption_mlp_fcn_0_biases;
    // Eigen::Matrix<float, 1, 32> absorption_dense_kernel;
    dr::Array<Float, 32> absorption_dense_kernel;
    // Eigen::Matrix<float, 1, 1> absorption_dense_bias;
    dr::Array<Float, 1> absorption_dense_bias;

    // Eigen::Matrix<float, LayerWidth, 1> scatter_decoder_fcn_fcn_0_biases;
    dr::Array<Float, LayerWidth> scatter_decoder_fcn_fcn_0_biases;
    // Eigen::Matrix<float, LayerWidth, 1> scatter_decoder_fcn_fcn_1_biases;
    dr::Array<Float, LayerWidth> scatter_decoder_fcn_fcn_1_biases;
    // Eigen::Matrix<float, LayerWidth, 1> scatter_decoder_fcn_fcn_2_biases;
    dr::Array<Float, LayerWidth> scatter_decoder_fcn_fcn_2_biases;
    // Eigen::Matrix<float, PreLayerWidth, 1> shared_preproc_mlp_2_shapemlp_fcn_0_biases;
    dr::Array<Float, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_0_biases;
    // Eigen::Matrix<float, PreLayerWidth, 1> shared_preproc_mlp_2_shapemlp_fcn_1_biases;
    dr::Array<Float, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_1_biases;
    // Eigen::Matrix<float, PreLayerWidth, 1> shared_preproc_mlp_2_shapemlp_fcn_2_biases;
    dr::Array<Float, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_2_biases;
    // Eigen::Matrix<float, 3, 1> scatter_dense_2_bias;
    dr::Array<Float, 3> scatter_dense_2_bias;

    // Eigen::Matrix<float, LayerWidth, PreLayerWidth + NLatent> scatter_decoder_fcn_fcn_0_weights;
    dr::Array<dr::Array<Float, PreLayerWidth + NLatent>, LayerWidth> scatter_decoder_fcn_fcn_0_weights;
    // Eigen::Matrix<float, LayerWidth, LayerWidth> scatter_decoder_fcn_fcn_1_weights;
    dr::Matrix<Float, LayerWidth> scatter_decoder_fcn_fcn_1_weights;
    // Eigen::Matrix<float, LayerWidth, LayerWidth> scatter_decoder_fcn_fcn_2_weights;
    dr::Matrix<Float, LayerWidth> scatter_decoder_fcn_fcn_2_weights;
    // Eigen::Matrix<float, 3, LayerWidth> scatter_dense_2_kernel;
    dr::Array<dr::Array<Float, LayerWidth>, 3> scatter_dense_2_kernel;

    // Eigen::Matrix<float, PreLayerWidth, NetworkHelpers<Float, Spectrum>::nInFeatures(PolyOrder)>     shared_preproc_mlp_2_shapemlp_fcn_0_weights;
    // dr::Array<dr::Array<Float, NetworkHelpers<Float, Spectrum>::nInFeatures(PolyOrder)>, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_0_weights;
    // Eigen::Matrix<float, PreLayerWidth, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_1_weights;
    dr::Matrix<Float, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_1_weights;
    // Eigen::Matrix<float, PreLayerWidth, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_2_weights;
    dr::Matrix<Float, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_2_weights;
    // Eigen::Matrix<float, NetworkHelpers<Float, Spectrum>::nPolyCoeffs(PolyOrder), 1> m_shapeFeatMean,     m_shapeFeatStdInv;
    // dr::Array<Float, NetworkHelpers<Float, Spectrum>::nPolyCoeffs(PolyOrder)> m_shapeFeatMean, m_shapeFeatStdInv;
    // Eigen::Vector3f m_outPosMean, m_outPosStd;
    Vector<Float, 3> m_outPosMean, m_outPosStd;
    float m_albedoMean{}, m_albedoStdInv{}, m_gMean{}, m_gStdInv{};
    /*MI_DECLARE_CLASS()*/
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

NAMESPACE_END(mitsuba)