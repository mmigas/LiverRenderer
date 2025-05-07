#pragma once

#include <chrono>
#include <drjit/call.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/polynomials.h>
#include <mitsuba/render/scattereigen.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sss_particle_tracer.h>
#include <mitsuba/render/vaeconfig.h>

NAMESPACE_BEGIN(mitsuba)
/**
 * Helper Class to sample using the VAE
 */
template <typename Float, typename Spectrum> class VaeHelper : public Object {
public:
    MI_IMPORT_TYPES(Scene, Sampler, Shape, Mesh)
    mutable double totalMsGetPolyCoeffs      = 0.0;
    mutable double totalRunScatterNetwork    = 0.0;
    mutable double totalRunAbsorptionNetwork = 0.0;
    mutable double totalProjectSamples       = 0.0;
    mutable double totalSetupTime            = 0.0;
    mutable double numScatterEvaluations     = 0.0;
    mutable double totalSampleTime           = 0.0;
    VaeHelper();

    ~VaeHelper();
    static void sampleGaussianVector(float *data, Sampler *sampler, int nVars);

    /*static void sampleUniformVector(float* data, Sampler* sampler, int nVars) {
        for (int i = 0; i < nVars; ++i) {
            data[i] = sampler->next1D();
        }
    }*/

    virtual bool prepare(const Scene *scene, const std::vector<ref<Shape>> &shapes, const Spectrum &sigmaT, const Spectrum &albedo, float g, float eta, const std::string &modelName, const std::string &absModelName, const std::string &angularModelName,
                         const std::string &outputDir, int batchSize, const PolyFitConfig &pfConfig);

    virtual ScatterSamplingRecord<Float, Spectrum> sample(const Scene *scene, const Point3f &p, const Vector3f &d, const Vector3f &polyNormal, const Spectrum &sigmaT, const Spectrum &albedo, float g, float eta, Sampler *sampler,
                                                          const SurfaceInteraction3f *its, bool projectSamples, int channel = 0) const {
        // {
        //     const Eigen::Vector3f a(0.f, 0.f, 0.f);
        //     const Eigen::Vector3f b(1.f, 0.f, 0.f);
        //     Eigen::VectorXf latent(m_config.nLatent);
        //     for (int i = 0; i < m_config.nLatent; ++i)
        //         latent[i] = 1.0f;
        //     AbsorptionModel<3>::ShapeVector shapeCoeff1;
        //     for (int i = 0; i < shapeCoeff1.size(); ++i)
        //         shapeCoeff1[i] = 1.0f;
        //     Eigen::Vector3f tmp;
        //     float tmp2;
        //     std::tie(tmp, tmp2) = scatterModel->run(a, b, Spectrum(0.5f), 0.5f, 1.3f, Spectrum(10.0f), shapeCoeff1, latent);
        //     std::cout << "tmp2: " << tmp2 << std::endl;
        //     std::cout << "tmp[0]: " << tmp[0] << std::endl;
        //     std::cout << "tmp[1]: " << tmp[1] << std::endl;
        //     std::cout << "tmp[2]: " << tmp[2] << std::endl;
        // }

        auto sampleStart = std::chrono::steady_clock::now();
#ifdef GETTIMINGS
        auto setupStart = std::chrono::steady_clock::now();
#endif
        typename AbsorptionModel<3, 64, Float, Spectrum>::ShapeVector shapeCoeffEigen;
        typename AbsorptionModel<3, 64, Float, Spectrum>::ShapeVector shapeCoeffEigenWs;
        // Matrix3x3 asTransform;
        drjit::Matrix<Float, 3> asTransform;
        if (m_config.predictionSpace == "AS") {
            const float *coeffs = its->polyCoeffs[channel];
            for (int i = 0; i < shapeCoeffEigenWs.size(); ++i)
                shapeCoeffEigenWs[i] = coeffs[i];
            // std::tie(shapeCoeffEigen, asTransform) = getPolyCoeffsAs < 3 > (p, d, polyNormal, its, channel);
        } else {
            shapeCoeffEigen = VaeHelper<Float, Spectrum>::getPolyCoeffsEigen<3>(p, d, polyNormal, its, m_config.predictionSpace == "LS", m_config.useLegendre, channel);
        }
#ifdef GETTIMINGS
        auto setupEnd  = std::chrono::steady_clock::now();
        auto setupDiff = setupEnd - setupStart;
        totalMsGetPolyCoeffs += std::chrono::duration<double, std::milli>(setupDiff).count();
#endif

        Spectrum albedoChannel(albedo[channel]);
        Spectrum sigmaTChannel(sigmaT[channel]);
        MediumParameters<Spectrum> medium(albedoChannel, g, eta, sigmaTChannel);

        Float kernelEps      = PolyUtils<Float, Spectrum>::getKernelEps(m_averageMedium, channel, m_kernelEpsScale);
        Float fitScaleFactor = PolyUtils<Float, Spectrum>::getFitScaleFactor(kernelEps);

#ifdef GETTIMINGS
        auto scatteringStart = std::chrono::steady_clock::now();
#endif

        Float absorption;
        Vector3f outPos;
        std::tie(outPos, absorption) = scatterModel->run(p, d, medium.albedo, medium.g, medium.eta, medium.sigmaT, fitScaleFactor, shapeCoeffEigen, sampler, asTransform);
#ifdef GETTIMINGS
        auto scatteringEnd  = std::chrono::steady_clock::now();
        auto scatteringDiff = scatteringEnd - scatteringStart;
        totalRunScatterNetwork += std::chrono::duration<double, std::milli>(scatteringDiff).count();
#endif
        ScatterSamplingRecord<Float, Spectrum> sRec;
        sRec.throughput = Spectrum(1.0f - absorption);
        sRec.outDir     = Vector3f(1.0f);
        Point3f sampledP(outPos[0], outPos[1], outPos[2]);
        sRec.p       = sampledP; // Rescale sampled points using sigmaT
        sRec.isValid = dr::any_or<true>(absorption < 1.0f);
        numScatterEvaluations++;
#ifdef GETTIMINGS
        auto projectionStart = std::chrono::steady_clock::now();
#endif

        if (m_config.predictionSpace == "AS") {
            // PolyUtils<Float, Spectrum>::projectPointsToSurface<nPolyCoeffs(3)>(scene, p, -d, sRec, shapeCoeffEigenWs, m_config.polyOrder, false,
            // fitScaleFactor, kernelEps);
        } else {
            PolyUtils<Float, Spectrum>::projectPointsToSurface(scene, p, -d, sRec, shapeCoeffEigen, m_config.polyOrder, m_config.predictionSpace == "LS", fitScaleFactor, kernelEps);
        }

#ifdef GETTIMINGS
        auto projectionEnd  = std::chrono::steady_clock::now();
        auto projectionDiff = projectionEnd - projectionStart;
        totalProjectSamples += std::chrono::duration<double, std::milli>(projectionDiff).count();
#endif
        auto sampleEnd  = std::chrono::steady_clock::now();
        auto sampleDiff = sampleEnd - sampleStart;
        totalSampleTime += std::chrono::duration<double, std::milli>(sampleDiff).count();

        return sRec;
    }

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

    const VaeConfig &getConfig() const { return m_config; };

    void precomputePolynomialsImpl(const std::vector<ref<Shape>> &shapes, const MediumParameters<Spectrum> &medium, int channel, const PolyFitConfig &pfConfig) {

        Float kernelEps = PolyUtils<Float, Spectrum>::getKernelEps(medium, channel, pfConfig.kernelEpsScale);

        ref<Sampler> sampler = static_cast<Sampler *>(PluginManager::instance()->create_object<Sampler>(Properties("independent")));
        sampler->seed(0);
        for (size_t shapeIdx = 0; shapeIdx < shapes.size(); ++shapeIdx) {
            if (!shapes[shapeIdx]->has_subsurface()) {
                continue;
            }
            int nSamples = dr::slice(dr::maximum(shapes[shapeIdx]->surface_area() * 2.0f / kernelEps, 1024), 0);
            // TODO: Remove slice to run in cuda
            std::vector<Point3f> sampled_p;
            std::vector<Vector3f> sampled_n;
            for (int i = 0; i < nSamples; ++i) {
                PositionSample3f pRec;
                shapes[shapeIdx]->sample_position(pRec.time, sampler->next_2d()); // TODO: Handle multiple shapes
                sampled_p.push_back(pRec.p);
                sampled_n.push_back(pRec.n);
            }
            // m_trees[channel].push_back(ConstraintKdTree());
            // m_trees[channel].back().build(sampled_p, sampled_n);
            auto nPoints = sampled_p.size();

            for (size_t i = 0; i < nPoints; ++i) {
                ConstraintKdTree<Float, Spectrum>::ExtraData d;
                d.p           = sampled_p[i];
                d.n           = sampled_n[i];
                d.sampleCount = 1;
                d.avgN        = Vector3f(-10, -10, -10);
                d.avgP        = Point3f(-100, -100, -100);
                // m_trees[channel][i] = d;
            }
            Mesh *mesh = (Mesh *) shapes[shapeIdx].get(); // From
            // TriMesh to Mesh
            // Log(Info, "Precomputing Coeffs on Mesh... (%d vertices, %d constraints)", mesh->vertex_count(), sampled_p.size());

            if (!mesh->hasPolyCoeffs())
                mesh->createPolyCoeffsArray();
            mesh->setHasRgb(medium.isRgb());
            PolyStorage *polyCoeffs = mesh->getPolyCoeffs();

            if (!polyCoeffs)
                // Log(Error, "poly coeffs null");

                auto polyFittingStart = std::chrono::steady_clock::now();

#pragma parallel for
            for (int i = 0; i < mesh->vertex_count(); ++i) {
                PolyFitRecord<Float, Spectrum> pfRec;
                pfRec.p                    = mesh->vertex_positions_buffer()[i];
                pfRec.d                    = mesh->vertex_normals_buffer()[i];
                pfRec.n                    = mesh->vertex_normals_buffer()[i];
                pfRec.kernelEps            = kernelEps;
                pfRec.config               = pfConfig;
                pfRec.config.useLightspace = false;
                Polynomial<Float> result;
                std::vector<Point3f> pts;
                std::vector<Vector3f> dirs;
                // std::tie(result, pts, dirs) = PolyUtils<Float, Spectrum>::fitPolynomial(pfRec, &m_trees[channel].back());
                for (int j = 0; j < result.coeffs.size(); ++j) {
                    polyCoeffs[i].coeffs[channel][j] = result.coeffs[j];
                    // polyCoeffs[i].kernelEps[channel] = kernelEps;
                    polyCoeffs[i].nPolyCoeffs = result.coeffs.size();
                }
            }
            auto polyFittingEnd = std::chrono::steady_clock::now();
            // auto polyFittingDuration = polyFittingEnd - polyFittingStart;
            // double totalMs           = std::chrono::duration<double, std::milli>(polyFittingDuration).count();
            // Log(Info, "Done precomputing coeffs. Took %f ms", totalMs);
        }
    }

    void precomputePolynomials(const std::vector<ref<Shape>> &shapes, const MediumParameters<Spectrum> &medium, const PolyFitConfig &pfConfig) {
        // m_trees.emplace_back();
        if (medium.isRgb()) {
            // m_trees.emplace_back();
            // m_trees.emplace_back();

            precomputePolynomialsImpl(shapes, medium, 0, pfConfig);
            precomputePolynomialsImpl(shapes, medium, 1, pfConfig);
            precomputePolynomialsImpl(shapes, medium, 2, pfConfig);
        } else {
            precomputePolynomialsImpl(shapes, medium, 0, pfConfig);
        }
        Log(Info, "Done precomputing polynomials");
    }

    static size_t numPolynomialCoefficients(size_t deg) { return (deg + 1) * (deg + 2) * (deg + 3) / 6; }

    std::vector<float> getPolyCoeffs(const Point3f &p, const Vector3f &d, Float sigmaT_scalar, Float g, const Spectrum &albedo, const Interaction3f *its, bool useLightSpace, std::vector<float> &shapeCoeffs, std::vector<float> &tmpCoeffs,
                                     bool useLegendre = false, int channel = 0) const;

    template <size_t PolyOrder = 3>
    dr::Array<Float, nPolyCoeffs(PolyOrder)> getPolyCoeffsEigen(const Point3f &p, const Vector3f &d, const Vector3f &polyNormal, const SurfaceInteraction3f *its, bool useLightSpace, bool useLegendre = false, int channel = 0) const {
        if (its) {
            // const Eigen::VectorXf &c = its->polyCoeffs;
            const float *coeffs = its->polyCoeffs[channel];
            if (useLightSpace) {
                Vector3f s, t;
                Vector3f n = -d;
                Volpath3D<Float, Spectrum>::onbDuff(n, s, t);
                dr::Array<Float, nPolyCoeffs(PolyOrder)> shapeCoeffs = PolyUtils<Float, Spectrum>::rotatePolynomialEigen<PolyOrder>(coeffs, s, t, n);
                /*if (useLegendre)
                    PolyUtils<Float, Spectrum>::legendreTransform(shapeCoeffs);*/
                // return shapeCoeffs;
            } else {
                dr::Array<Float, nPolyCoeffs(PolyOrder)> shapeCoeffs;
                for (int i = 0; i < its->nPolyCoeffs; ++i)
                    shapeCoeffs[i] = coeffs[i];
                /*if (useLegendre)
                    PolyUtils<Float, Spectrum>::legendreTransform(shapeCoeffs);*/
                // return shapeCoeffs;
            }
        }
        return dr::zeros<Float>(nPolyCoeffs(PolyOrder));
    }

    template <size_t PolyOrder> std::pair<dr::Array<Float, nPolyCoeffs(PolyOrder)>, drjit::Matrix<Float, 3>> getPolyCoeffsAs(const Point3f &p, const Vector3f &d, const Vector3f &polyNormal, const SurfaceInteraction3f *its, int channel = 0) const {
        assert(its);
        const float *coeffs                                  = its->polyCoeffs[channel];
        drjit::Matrix<Float, 3> transf                       = Volpath3D<Float, Spectrum>::azimuthSpaceTransformNew(-d, polyNormal);
        dr::Array<Float, nPolyCoeffs(PolyOrder)> shapeCoeffs = PolyUtils<Float, Spectrum>::rotatePolynomialEigen<PolyOrder>(coeffs, transf.entry(0), transf.entry(1), transf.entry(2));
        return std::make_pair(shapeCoeffs, transf);
    }
    MI_DECLARE_CLASS()
protected:
    // AbsorptionModel<3, 64, Float, Spectrum> absModel;
    std::unique_ptr<ScatterModelBase<Float, Spectrum>> scatterModel;
    std::string m_id;
    VaeConfig m_config;
    Spectrum m_effectiveAlbedo;
    Float m_kernelEpsScale;

    // std::vector<std::vector<ConstraintKdTree>> m_trees;
    size_t m_batchSize{};
    int m_polyOrder{};
    MediumParameters<Spectrum> m_averageMedium;
};
MI_EXTERN_CLASS(VaeHelper)

NAMESPACE_END(mitsuba)
