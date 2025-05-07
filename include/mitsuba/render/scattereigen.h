#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
#include <mitsuba/render/sss_particle_tracer.h>

#include <Eigen/Core>
#include <Eigen/Dense>

using json = nlohmann::json;

NAMESPACE_BEGIN(mitsuba)
template <typename Float, typename Spectrum> class /*MI_EXPORT_LIB*/ NetworkHelpers {
public:
    static void onb(const dr::Array<Float, 3> &n, dr::Array<Float, 3> &b1, dr::Array<Float, 3> &b2) {
        Float sign    = dr::copysign(Float(1.0f), n[2]);
        const Float a = -1.0f / (sign + n[2]);
        const Float b = n[0] * n[1] * a;
        b1            = dr::Array<Float, 3>(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
        b2            = dr::Array<Float, 3>(b, sign + n[1] * n[1] * a, -n[1]);
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

        // Create the return array
        drjit::Array<Float, Size> ret;
        // Use a buffer of primitive float type
        std::vector<float> buffer(size);
        f.read(reinterpret_cast<char *>(buffer.data()), size * sizeof(float));

        // Explicitly convert each float to the Float type
        for (int i = 0; i < Size; ++i) {
            ret[i] = Float(buffer[i]);
            Log(Info, "ret[%d]: %f", i, ret[i]);
        }

        return ret;
    }

    template <int Size> static drjit::Matrix<Float, Size> loadMatrixDynamic(const std::string &filename) {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open())
            std::cout << "FILE NOT FOUND " << filename << std::endl;
        std::cout << "Loading " << filename << std::endl;

        int32_t nDims;
        f.read(reinterpret_cast<char *>(&nDims), sizeof(nDims));

        int32_t rows, cols;
        f.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        f.read(reinterpret_cast<char *>(&cols), sizeof(cols));
        drjit::Matrix<Float, Size> matrix(rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < rows; ++j) {
                float value;
                f.read(reinterpret_cast<char *>(&value), sizeof(value));
                matrix(i, j) = Float(value);
            }
        }

        return matrix;
    }
    template <int Col> static drjit::Array<Float, Col> loadColumn(const std::string &filename) {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open())
            std::cout << "FILE NOT FOUND: " << filename << std::endl;
        std::cout << "Loading " << filename << std::endl;
        int32_t nDims;
        f.read(reinterpret_cast<char *>(&nDims), sizeof(nDims));
        int32_t rows, cols;

        f.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        f.read(reinterpret_cast<char *>(&cols), sizeof(cols));

        // Create the return array
        drjit::Array<Float, Col> ret;
        // Use a buffer of primitive float type
        std::vector<float> buffer(cols);
        f.read(reinterpret_cast<char *>(buffer.data()), cols * sizeof(float));

        // Explicitly convert each float to the Float type
        for (int i = 0; i < Col; ++i) {
            ret[i] = Float(buffer[i]);
        }

        return ret;
    }

    template <int Row, int Col> static drjit::Array<drjit::Array<Float, Col>, Row> loadArrayArray(const std::string &filename) {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open())
            std::cout << "FILE NOT FOUND " << filename << std::endl;
        std::cout << "Loading " << filename << std::endl;

        int32_t nDims;
        f.read(reinterpret_cast<char *>(&nDims), sizeof(nDims));

        int32_t rows, cols;
        f.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        f.read(reinterpret_cast<char *>(&cols), sizeof(cols));
        drjit::Array<drjit::Array<Float, Col>, Row> matrix;
        for (int i = 0; i < Row; ++i) {
            for (int j = 0; j < Col; ++j) {
                float value;
                f.read(reinterpret_cast<char *>(&value), sizeof(value));
                matrix[i][j] = Float(value);
            }
        }
        return matrix;
    }
    static dr::Array<Float, 3> localToWorld(const dr::Array<Float, 3> &inPos, const dr::Array<Float, 3> &inNormal, const dr::Array<Float, 3> &outPosLocal, bool predictInTangentSpace) {
        if (predictInTangentSpace) {
            dr::Array<Float, 3> tangent1, tangent2;
            onb(inNormal, tangent1, tangent2);
            return inPos + outPosLocal[0] * tangent1 + outPosLocal[1] * tangent2 + outPosLocal[2] * inNormal;
        } else {
            return inPos + outPosLocal;
        }
    }

    template <size_t PolyOrder = 3, bool useSimilarityTheory = false>
    static dr::Array<Float, nInFeatures(PolyOrder)> preprocessFeatures(const Spectrum &albedo, float g, float ior, const Spectrum &sigmaT, const dr::Array<Float, nPolyCoeffs(3)> &shapeFeatures, float albedoMean, float albedoStdInv, float gMean, float gStdInv,
                                                                       const dr::Array<Float, nPolyCoeffs(PolyOrder)> &shapeFeatMean, const dr::Array<Float, nPolyCoeffs(PolyOrder)> &shapeFeatStdInv) {
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

        dr::Array<Float, nInFeatures(PolyOrder)> features;

        // Copy shape features with normalization
        for (int i = 0; i < nPolyCoeffs(PolyOrder); i++) {
            features[i] = (shapeFeatures[i] - shapeFeatMean[i]) * shapeFeatStdInv[i];
        }

        // Add remaining features
        features[nPolyCoeffs(PolyOrder)]     = Float(albedoNorm);
        features[nPolyCoeffs(PolyOrder) + 1] = Float(gNorm);
        features[nPolyCoeffs(PolyOrder) + 2] = Float(iorNorm);

        return features;
    }
};

template <size_t PolyOrder /*= 3*/, size_t LayerWidth /*= 64*/, typename Float, typename Spectrum> class MI_EXPORT_LIB AbsorptionModel {
public:
    typedef drjit::Array<Float, 20> ShapeVector;
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
    virtual std::pair<Vector<Float, 3>, Float> run(const Vector<Float, 3> &inPos, const Vector<Float, 3> &inDir, const Spectrum &albedo, float g, float ior, const Spectrum &sigmaT, Float polyScaleFactor, const dr::Array<Float, nPolyCoeffs(3)> &polyCoeffs,
                                                   Sampler *sampler, drjit::Matrix<Float, 3> toAsTransform) const = 0;

    virtual ~ScatterModelBase() {}
    /*MI_DECLARE_CLASS()*/
};

MI_EXTERN_CLASS(ScatterModelBase)

template <size_t PolyOrder /*= 3*/, size_t NLatent /*= 4*/, size_t LayerWidth /*= 64*/, size_t PreLayerWidth /*= 64*/, typename Float, typename Spectrum, typename Sampler> class MI_EXPORT_LIB ScatterModelSimShared : public ScatterModelBase<Float, Spectrum> {
public:
    MI_IMPORT_TYPES()
    ScatterModelSimShared() {}

    ScatterModelSimShared(const std::string &variablePath, const std::string &absVariablePath, const json &stats, const std::string &shapeFeaturesName, const std::string &predictionSpace = "LS", bool useEpsilonSpace = false) {
        m_useEpsilonSpace                           = useEpsilonSpace;
        absorption_dense_bias                       = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<1>(variablePath + "/absorption_dense_bias.bin");
        absorption_mlp_fcn_0_biases                 = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<32>(variablePath + "/absorption_mlp_fcn_0_biases.bin");
        scatter_decoder_fcn_fcn_0_biases            = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<LayerWidth>(variablePath + "/scatter_decoder_fcn_fcn_0_biases.bin");
        scatter_decoder_fcn_fcn_1_biases            = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<LayerWidth>(variablePath + "/scatter_decoder_fcn_fcn_1_biases.bin");
        scatter_decoder_fcn_fcn_2_biases            = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<LayerWidth>(variablePath + "/scatter_decoder_fcn_fcn_2_biases.bin");
        shared_preproc_mlp_2_shapemlp_fcn_0_biases  = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<PreLayerWidth>(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_biases.bin");
        shared_preproc_mlp_2_shapemlp_fcn_1_biases  = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<PreLayerWidth>(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_biases.bin");
        shared_preproc_mlp_2_shapemlp_fcn_2_biases  = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<PreLayerWidth>(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_biases.bin");
        scatter_dense_2_bias                        = NetworkHelpers<Float, Spectrum>::loadVectorDynamic<3>(variablePath + "/scatter_dense_2_bias.bin");
        scatter_decoder_fcn_fcn_0_weights           = NetworkHelpers<Float, Spectrum>::loadArrayArray<LayerWidth, PreLayerWidth + NLatent>(variablePath + "/scatter_decoder_fcn_fcn_0_weights.bin");
        scatter_decoder_fcn_fcn_1_weights           = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic<LayerWidth>(variablePath + "/scatter_decoder_fcn_fcn_1_weights.bin");
        scatter_decoder_fcn_fcn_2_weights           = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic<LayerWidth>(variablePath + "/scatter_decoder_fcn_fcn_2_weights.bin");
        shared_preproc_mlp_2_shapemlp_fcn_0_weights = NetworkHelpers<Float, Spectrum>::loadArrayArray<PreLayerWidth, NetworkHelpers<Float, Spectrum>::nInFeatures(PolyOrder)>(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_0_weights.bin");
        shared_preproc_mlp_2_shapemlp_fcn_1_weights = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic<PreLayerWidth>(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_1_weights.bin");
        shared_preproc_mlp_2_shapemlp_fcn_2_weights = NetworkHelpers<Float, Spectrum>::loadMatrixDynamic<PreLayerWidth>(variablePath + "/shared_preproc_mlp_2_shapemlp_fcn_2_weights.bin");
        scatter_dense_2_kernel                      = NetworkHelpers<Float, Spectrum>::loadArrayArray<3, LayerWidth>(variablePath + "/scatter_dense_2_kernel.bin");

        absorption_dense_kernel      = NetworkHelpers<Float, Spectrum>::loadColumn<32>(variablePath + "/absorption_dense_kernel.bin");
        absorption_mlp_fcn_0_weights = NetworkHelpers<Float, Spectrum>::loadArrayArray<32, PreLayerWidth>(variablePath + "/absorption_mlp_fcn_0_weights.bin");

        m_gMean            = stats["g_mean"][0];
        m_gStdInv          = stats["g_stdinv"][0];
        m_albedoMean       = stats["effAlbedo_mean"][0];
        m_albedoStdInv     = stats["effAlbedo_stdinv"][0];
        std::string degStr = std::to_string(PolyOrder);
        for (int i = 0; i < NetworkHelpers<Float, Spectrum>::nPolyCoeffs(PolyOrder); ++i) {
            float mean           = stats["mlsPoly3_mean"][i];
            float stdinv         = stats["mlsPoly3_stdinv"][i];
            m_shapeFeatMean[i]   = mean;
            m_shapeFeatStdInv[i] = stdinv;
        }
        if (predictionSpace != "AS") {
            for (int i = 0; i < 3; ++i) {
                // m_outPosMean[i] = stats["outPosRel" + predictionSpace + "_mean"][i];
                // m_outPosStd[i]  = 1.0f / float(stats["outPosRel" + predictionSpace + "_stdinv"][i]);
            }
        }

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

    std::pair<Vector<Float, 3>, Float> run(const Vector3f &inPos, const Vector3f &inDir, const Spectrum &albedo, float g, float ior, const Spectrum &sigmaT, Float fitScaleFactor, const dr::Array<Float, nPolyCoeffs(3)> &polyCoeffs, Sampler *sampler,
                                           Matrix3f) const override {
        Spectrum sigmaTp = Volpath3D<Float, Spectrum>::getSigmaTp(albedo, g, sigmaT);

        auto x = NetworkHelpers<Float, Spectrum>::preprocessFeatures<PolyOrder, true>(albedo, g, ior, sigmaT, polyCoeffs, m_albedoMean, m_albedoStdInv, m_gMean, m_gStdInv, m_shapeFeatMean, m_shapeFeatStdInv);

        Eigen::Matrix<float, NetworkHelpers<Float, Spectrum>::nInFeatures(PolyOrder), 1> xEigen;
        // Fill Eigen matrix
        for (int i = 0; i < NetworkHelpers<Float, Spectrum>::nInFeatures(PolyOrder); ++i) {
            xEigen(i) = (float) dr::slice(x[i], 0);
        }

        // Apply the preprocessing network
        Eigen::Matrix<float, 32, PreLayerWidth> absorption_mlp_fcn_0_weights_eigen;
        Eigen::Matrix<float, 32, 1> absorption_mlp_fcn_0_biases_eigen;

        Eigen::Matrix<float, PreLayerWidth, NetworkHelpers<Float, Spectrum>::nInFeatures(PolyOrder)> shared_preproc_mlp_2_shapemlp_fcn_0_weights_eigen;
        Eigen::Matrix<float, PreLayerWidth, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_1_weights_eigen;
        Eigen::Matrix<float, PreLayerWidth, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_2_weights_eigen;
        Eigen::Matrix<float, PreLayerWidth, 1> shared_preproc_mlp_2_shapemlp_fcn_0_biases_eigen;
        Eigen::Matrix<float, PreLayerWidth, 1> shared_preproc_mlp_2_shapemlp_fcn_1_biases_eigen;
        Eigen::Matrix<float, PreLayerWidth, 1> shared_preproc_mlp_2_shapemlp_fcn_2_biases_eigen;

        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < PreLayerWidth; ++j) {
                absorption_mlp_fcn_0_weights_eigen(i, j) = (float) dr::slice(absorption_mlp_fcn_0_weights[i][j], 0);
            }
            absorption_mlp_fcn_0_biases_eigen(i) = (float) dr::slice(absorption_mlp_fcn_0_biases[i], 0);
        }

        // Fill Eigen matrices
        for (int i = 0; i < PreLayerWidth; ++i) {
            for (int j = 0; j < NetworkHelpers<Float, Spectrum>::nInFeatures(PolyOrder); ++j) {
                shared_preproc_mlp_2_shapemlp_fcn_0_weights_eigen(i, j) = (float) drjit::slice(shared_preproc_mlp_2_shapemlp_fcn_0_weights[i][j], 0);
            }
            for (int j = 0; j < PreLayerWidth; ++j) {
                shared_preproc_mlp_2_shapemlp_fcn_1_weights_eigen(i, j) = (float) drjit::slice(shared_preproc_mlp_2_shapemlp_fcn_1_weights[i][j], 0);
                shared_preproc_mlp_2_shapemlp_fcn_2_weights_eigen(i, j) = (float) drjit::slice(shared_preproc_mlp_2_shapemlp_fcn_2_weights[i][j], 0);
            }

            shared_preproc_mlp_2_shapemlp_fcn_0_biases_eigen(i) = (float) drjit::slice(shared_preproc_mlp_2_shapemlp_fcn_0_biases[i], 0);
            shared_preproc_mlp_2_shapemlp_fcn_1_biases_eigen(i) = (float) drjit::slice(shared_preproc_mlp_2_shapemlp_fcn_1_biases[i], 0);
            shared_preproc_mlp_2_shapemlp_fcn_2_biases_eigen(i) = (float) drjit::slice(shared_preproc_mlp_2_shapemlp_fcn_2_biases[i], 0);
        }

        Eigen::Matrix<float, PreLayerWidth, 1> featuresEigen = (shared_preproc_mlp_2_shapemlp_fcn_0_weights_eigen * xEigen + shared_preproc_mlp_2_shapemlp_fcn_0_biases_eigen).cwiseMax(0.0f);

        dr::Array<Float, PreLayerWidth> features;
        for (size_t i = 0; i < PreLayerWidth; ++i) {
            // Compute dot product for this row
            Float sum = 0;
            for (size_t j = 0; j < NetworkHelpers<Float, Spectrum>::nInFeatures(PolyOrder); ++j) {
                sum += shared_preproc_mlp_2_shapemlp_fcn_0_weights[i][j] * x[j];
            }
            // Add bias and apply ReLU
            features[i] = dr::maximum(sum + shared_preproc_mlp_2_shapemlp_fcn_0_biases[i], Float(0.0f));
        }

        features = dr::maximum(shared_preproc_mlp_2_shapemlp_fcn_1_weights * features + shared_preproc_mlp_2_shapemlp_fcn_1_biases, 0.0f);
        features = dr::maximum(shared_preproc_mlp_2_shapemlp_fcn_2_weights * features + shared_preproc_mlp_2_shapemlp_fcn_2_biases, 0.0f);

        featuresEigen = (shared_preproc_mlp_2_shapemlp_fcn_1_weights_eigen * featuresEigen + shared_preproc_mlp_2_shapemlp_fcn_1_biases_eigen).cwiseMax(0.0f);
        featuresEigen = (shared_preproc_mlp_2_shapemlp_fcn_2_weights_eigen * featuresEigen + shared_preproc_mlp_2_shapemlp_fcn_2_biases_eigen).cwiseMax(0.0f);


        // Compute absorption
        Eigen::Matrix<float, 32, 1> absTmp_eigen = (absorption_mlp_fcn_0_weights_eigen * featuresEigen + absorption_mlp_fcn_0_biases_eigen).cwiseMax(0.0f);
        dr::Array<Float, 32> absTmp;
        for (size_t i = 0; i < 32; ++i) {
            // Compute dot product for this row
            Float sum = 0;
            for (size_t j = 0; j < PreLayerWidth; ++j) {
                sum += absorption_mlp_fcn_0_weights[i][j] * features[j];
            }
            // Add bias and apply ReLU
            absTmp[i] = dr::maximum(sum + absorption_mlp_fcn_0_biases[i], Float(0.0f));
        }
        // print Eigen matrix and jit array
        for (size_t i = 0; i < 32; ++i) {
            float eigenValue = absTmp_eigen[i];
            float jitValue   = (float) dr::slice(absTmp[i], 0);
            Log(Info, "Eigen: %f, Jit: %f", eigenValue, jitValue);
            if (dr::any_or<true>(drjit::round(eigenValue * 100000.0f) != drjit::round(jitValue * 100000.0f))) {
                Log(Info, "Different at index %d", i);
            }
        }
        // Eigen::Array<float, 1, 1> a       = absorption_dense_kernel * absTmp + absorption_dense_bias;
        dr::Array<Float, 1> a = dr::dot(absorption_dense_kernel, absTmp) + absorption_dense_bias;
        float absorption      = NetworkHelpers<Float, Spectrum>::sigmoid((float) dr::slice(a[0], 0));
        if (dr::any_or<true>(sampler->next_1d() > absorption)) {
            absorption = 0.0f; // nothing gets absorbed instead
        } else {
            return std::make_pair(inPos, 1.0f); // all is absorbed
        }
        // Concatenate features with random numbers
        dr::Array<float, NLatent> latent(NLatent);
        VaeHelper<Float, Spectrum>::sampleGaussianVector(latent.data(), sampler, NLatent);

        // Eigen::Matrix<float, PreLayerWidth + NLatent, 1> featLatent;
        dr::Array<Float, PreLayerWidth + NLatent> featLatent;
        for (size_t i = 0; i < NLatent; ++i) {
            featLatent[i] = Float(latent[i]);
        }

        // Copy features values
        for (size_t i = 0; i < PreLayerWidth; ++i) {
            featLatent[NLatent + i] = features[i];
        }
        // Eigen::Matrix<float, 64, 1> y = (scatter_decoder_fcn_fcn_0_weights * featLatent + scatter_decoder_fcn_fcn_0_biases).cwiseMax(0.0f);
        // y                             = (scatter_decoder_fcn_fcn_1_weights * y + scatter_decoder_fcn_fcn_1_biases).cwiseMax(0.0f);
        // y                             = (scatter_decoder_fcn_fcn_2_weights * y + scatter_decoder_fcn_fcn_2_biases).cwiseMax(0.0f);
        // Eigen::Vector3f outPos = scatter_dense_2_kernel * y + scatter_dense_2_bias;

        dr::Array<Float, 64> y;
        for (size_t i = 0; i < 64; ++i) {
            // Compute dot product for this row
            Float sum = 0;
            for (size_t j = 0; j < PreLayerWidth + NLatent; ++j) {
                sum += scatter_decoder_fcn_fcn_0_weights[i][j] * featLatent[j];
            }
            // Add bias and apply ReLU
            y[i] = dr::maximum(sum + scatter_decoder_fcn_fcn_0_biases[i], Float(0.0f));
        }

        // Dot between matrix and vector
        dr::Array<Float, 64> result1;
        for (size_t i = 0; i < 64; ++i) {
            Float sum = 0;
            for (size_t j = 0; j < 64; ++j) {
                sum += scatter_decoder_fcn_fcn_1_weights[i][j] * y[j];
            }
            result1[i] = sum + scatter_decoder_fcn_fcn_1_biases[i];
        }
        y = dr::maximum(result1, 0.0f);

        dr::Array<Float, 64> result2;
        for (size_t i = 0; i < 64; ++i) {
            Float sum = 0;
            for (size_t j = 0; j < 64; ++j) {
                sum += scatter_decoder_fcn_fcn_2_weights[i][j] * y[j];
            }
            result2[i] = sum + scatter_decoder_fcn_fcn_2_biases[i];
        }
        y = dr::maximum(result2, 0.0f);
        Vector3f outPos;
        for (size_t i = 0; i < 3; ++i) {
            Float sum = 0;
            for (size_t j = 0; j < 64; ++j) {
                sum += scatter_dense_2_kernel[i][j] * y[j];
            }
            outPos[i] = sum + scatter_dense_2_bias[i];
        }

        if (m_useEpsilonSpace) {
            if (m_useAsSpace) {
                // Vector<Float, 3> tmp = toAsTransform.preMult(Vector<Float, 3>(outPos[0], outPos[1], outPos[2])) /fitScaleFactor;
                // outPos = Eigen::Vector3f(tmp.x, tmp.y, tmp.z) + inPos;
            } else {
                outPos = NetworkHelpers<Float, Spectrum>::localToWorld(inPos, -inDir, outPos, true);
                outPos = inPos + (outPos - inPos) / fitScaleFactor;
            }
        } else {
            // outPos = outPos.cwiseProduct(m_outPosStd) + m_outPosMean;
            // outPos = NetworkHelpers<Float, Spectrum>::localToWorld(inPos, -inDir, outPos, true);
        }
        return std::make_pair(outPos, Float(absorption));
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
    dr::Array<dr::Array<Float, NetworkHelpers<Float, Spectrum>::nInFeatures(PolyOrder)>, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_0_weights;
    // Eigen::Matrix<float, PreLayerWidth, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_1_weights;
    dr::Matrix<Float, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_1_weights;
    // Eigen::Matrix<float, PreLayerWidth, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_2_weights;
    dr::Matrix<Float, PreLayerWidth> shared_preproc_mlp_2_shapemlp_fcn_2_weights;
    // Eigen::Matrix<float, NetworkHelpers<Float, Spectrum>::nPolyCoeffs(PolyOrder), 1> m_shapeFeatMean,     m_shapeFeatStdInv;
    dr::Array<Float, NetworkHelpers<Float, Spectrum>::nPolyCoeffs(PolyOrder)> m_shapeFeatMean, m_shapeFeatStdInv;
    // Eigen::Vector3f m_outPosMean, m_outPosStd;
    Vector<Float, 3> m_outPosMean, m_outPosStd;
    float m_albedoMean{}, m_albedoStdInv{}, m_gMean{}, m_gStdInv{};
    /*MI_DECLARE_CLASS()*/
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

NAMESPACE_END(mitsuba)