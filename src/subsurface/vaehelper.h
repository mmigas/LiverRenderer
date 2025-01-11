#pragma once
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sss_particle_tracer.h>
#include <mitsuba/render/polynomials.h>
#include "vaeconfig.h"

NAMESPACE_BEGIN(mitsuba)
    /**
     * Helper Class to sample using the VAE
     */
    template<typename Float, typename Spectrum>
    class VaeHelper : public Object {
    public:
        MI_IMPORT_TYPES(Scene, Sampler, Shape)

        static void sampleGaussianVector(float* data, Sampler* sampler, int nVars) {
            bool odd = nVars % 2;
            int idx = 0;
            for (int i = 0; i < nVars / 2; ++i) {
                Point2f uv = warp::square_to_std_normal(sampler->next2D());
                data[idx] = uv.x;
                ++idx;
                data[idx] = uv.y;
                ++idx;
            }
            if (odd)
                data[idx] = warp::square_to_std_normal(sampler->next2D()).x;
        }

        /*static void sampleUniformVector(float* data, Sampler* sampler, int nVars) {
            for (int i = 0; i < nVars; ++i) {
                data[i] = sampler->next1D();
            }
        }*/

        VaeHelper() {
        }

        virtual bool prepare(const Scene* scene, const std::vector<ref<Shape>>& shapes, const Spectrum& sigmaT,
                             const Spectrum& albedo, float g, float eta, const std::string& modelName,
                             const std::string& absModelName, const std::string& angularModelName,
                             const std::string& outputDir, int batchSize, const PolyFitConfig& pfConfig) {
            m_batchSize = batchSize;

            // Build acceleration data structure to compute polynomial fits efficiently
            m_averageMedium = MediumParameters<Float, Spectrum>(albedo, g, eta, sigmaT);

            if (modelName == "None") {
                m_polyOrder = pfConfig.order;
                //precomputePolynomials(shapes, medium, pfConfig);
                return true; // Dont load any ML models
            }
            std::string modelPath = outputDir + "models/" + modelName + "/";
            std::string absModelPath = outputDir + "models_abs/" + absModelName + "/";
            std::string angularModelPath = outputDir + "models_angular/" + angularModelName + "/";
            std::string configFile = modelPath + "training-metadata.json";
            std::string angularConfigFile = angularModelPath + "training-metadata.json";

            m_config = VaeConfig(configFile, angularModelName != "None" ? angularConfigFile : "", outputDir);
            m_polyOrder = m_config.polyOrder;
            //precomputePolynomials(shapes, medium, pfConfig);
            return true;
        }

        virtual ScatterSamplingRecord<Float, Spectrum> sample(const Scene* scene, const Point3f& p, const Vector3f& d,
                                                              const Vector3f& polyNormal, const Spectrum& sigmaT,
                                                              const Spectrum& albedo, float g, float eta,
                                                              Sampler* sampler, const SurfaceInteraction3f* its,
                                                              bool projectSamples, int channel = 0) const {
            std::cout << "NOT IMPLEMENTED\n";
            return ScatterSamplingRecord<Float, Spectrum>();
        };

        /*
        virtual void sampleBatched(const Scene* scene, const Point3f& p, const Vector3f& d, const Spectrum& sigmaT,
                                   const Spectrum& albedo, float g, float eta, Sampler* sampler, const Interaction3f* its,
                                   int nSamples, bool projectSamples, std::vector<ScatterSamplingRecord<Float, Spectrum>>& sRec) const {
            std::cout << "NOT IMPLEMENTED\n";
        };

        virtual void sampleRGB(const Scene* scene, const Point3f& p, const Vector3f& d, const Spectrum& sigmaT,
                               const Spectrum& albedo, float g, float eta, Sampler* sampler, const Interaction3f* its,
                               bool projectSamples, ScatterSamplingRecord<Float, Spectrum>* sRec) const {
            std::cout << "NOT IMPLEMENTED\n";
        };
        */

        const VaeConfig& getConfig() const {
            return m_config;
        };

        /*void precomputePolynomialsImpl(const std::vector<Shape*>& shapes,
                                       const MediumParameters<Float, Spectrum>& medium,
                                       int channel, const PolyFitConfig& pfConfig);

        void precomputePolynomials(const std::vector<Shape*>& shapes, const MediumParameters<Float, Spectrum>& medium,
                                   const PolyFitConfig& pfConfig);

        static size_t numPolynomialCoefficients(size_t deg) {
            return (deg + 1) * (deg + 2) * (deg + 3) / 6;
        }

        std::vector<float> getPolyCoeffs(const Point3f& p, const Vector3f& d, Float sigmaT_scalar, Float g,
                                         const Spectrum& albedo, const Interaction3f* its, bool useLightSpace,
                                         std::vector<float>& shapeCoeffs, std::vector<float>& tmpCoeffs,
                                         bool useLegendre = false, int channel = 0) const;*/

        template<size_t PolyOrder = 3>
        Eigen::Matrix<Float, PolyUtils<Float, Spectrum>::nPolyCoeffs(PolyOrder), 1> getPolyCoeffsEigen(
            const Point3f& p, const Vector3f& d,
            const Vector3f& polyNormal,
            const SurfaceInteraction3f* its, bool useLightSpace,
            bool useLegendre = false, int channel = 0) const {
            if (its) {
                // const Eigen::VectorXf &c = its->polyCoeffs;
                const float* coeffs = its->polyCoeffs[channel];
                if (useLightSpace) {
                    Vector3f s, t;
                    Vector3f n = -d;
                    Volpath3D<Float, Spectrum>::onbDuff(n, s, t);
                    Eigen::Matrix<Float, PolyUtils<Float, Spectrum>::nPolyCoeffs(PolyOrder), 1> shapeCoeffs =
                            PolyUtils<Float, Spectrum>::rotatePolynomialEigen < PolyOrder > (coeffs, s, t, n);
                    if (useLegendre)
                        PolyUtils<Float, Spectrum>::legendreTransform(shapeCoeffs);
                    return shapeCoeffs;
                } else {
                    Eigen::Matrix<Float, PolyUtils<Float, Spectrum>::nPolyCoeffs(PolyOrder), 1> shapeCoeffs;
                    for (int i = 0; i < its->nPolyCoeffs; ++i)
                        shapeCoeffs[i] = coeffs[i];
                    if (useLegendre)
                        PolyUtils<Float, Spectrum>::legendreTransform(shapeCoeffs);
                    return shapeCoeffs;
                }
            }
            return Eigen::Matrix<Float, PolyUtils<Float, Spectrum>::nPolyCoeffs(PolyOrder), 1>::Zero();
        }

        template<size_t PolyOrder>
        std::pair<Eigen::Matrix<float, PolyUtils<Float, Spectrum>::nPolyCoeffs(PolyOrder), 1>, drjit::Matrix<Float, 3>>
        getPolyCoeffsAs(const Point3f& p, const Vector3f& d,
                        const Vector3f& polyNormal,
                        const SurfaceInteraction3f* its, int channel = 0) const {
            assert(its);
            const float* coeffs = its->polyCoeffs[channel];
            drjit::Matrix<Float, 3> transf = Volpath3D<Float, Spectrum>::azimuthSpaceTransformNew(-d, polyNormal);
            Eigen::Matrix<float, PolyUtils<Float, Spectrum>::nPolyCoeffs(PolyOrder), 1> shapeCoeffs =
                    PolyUtils<Float, Spectrum>::rotatePolynomialEigen < PolyOrder > (
                        coeffs, transf.entry(0), transf.entry(1), transf.entry(2));
            return std::make_pair(shapeCoeffs, transf);
        }

    protected:
        VaeConfig m_config;

        //std::vector<std::vector<ConstraintKdTree>> m_trees;
        size_t m_batchSize;
        int m_polyOrder;
        MediumParameters<Float, Spectrum> m_averageMedium;
    };

NAMESPACE_END(mitsuba)
