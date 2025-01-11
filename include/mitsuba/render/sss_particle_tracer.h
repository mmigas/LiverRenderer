#pragma once

#include "fwd.h"
#include <mitsuba/core/ray.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/polynomials_structs.h>

NAMESPACE_BEGIN(mitsuba)
    class ConstraintKdTree;

    template<typename Float, typename Spectrum>
    struct ScatterSamplingRecord { // Record used at runtime by samplers abstracting SS scattering
        Point<Float, 3> p;
        Vector<Float, 3> n;
        Vector<Float, 3> outDir;
        bool isValid = false;
        Spectrum throughput;
        int sampledColorChannel = -1;
    };

    template<typename Float, typename Spectrum>
    class ScatterSamplingRecordArray : public Object {
    public:
        ScatterSamplingRecordArray(size_t size) : data(size) {
        }

        std::vector<ScatterSamplingRecord<Float, Spectrum>> data;
    };


    template<typename Float, typename Spectrum>
    std::tuple<Ray<Point<Float, 3>, Spectrum>, Vector<Float, 3>> sampleShape(
        const Shape<Float, Spectrum>* shape, const MediumParameters<Float, Spectrum>& medium,
        Sampler<Float, Spectrum>* sampler) {
        for (int i = 0; i < 1000; ++i) { // After 1000 trials we should have sampled something inside the object...
            Float time = 0.0f;
            PositionSample<Float, Spectrum> pRec = shape->sample_Position(time, sampler->next2D());

            Point<Float, 3> o = pRec.p;
            Vector<Float, 3> normal = pRec.n;

            Vector<Float, 3> localDir = warp::square_to_cosine_hemisphere(sampler->next2D());
            // Evaluate fresnel term to decide whether to use direction or not
            Float cosThetaT;
            Float F = fresnelDielectricExt(Frame<Float>::cosTheta(localDir), cosThetaT, medium.eta);
            if (sampler->next1D() <= F)
                continue;
            else
                localDir = refract(localDir, cosThetaT, medium.eta);
            // Rotate to world coordinates
            Vector<Float, 3> sampleD = Frame(normal).toWorld(localDir);

            return std::make_tuple(Ray(o + drjit::Epsilon * sampleD, sampleD, 0.0f), normal);
        }
        return std::make_tuple(Ray<Point<Float, 3>, Spectrum>(), Vector<Float, 3>());
    }

    template<typename Float, typename Spectrum>
    std::tuple<Ray<Point<Float, 3>, Spectrum>, Vector<Float, 3>> sampleShapeFixedInDir(
        const Shape<Float, Spectrum>* shape, Sampler<Float, Spectrum>* sampler, const Vector<Float, 3> inDirLocal) {
        Float time = 0.0f;
        PositionSample<Float, Spectrum> pRec = shape->sample_Position(time, sampler->next2D());

        Point<Float, 3> o = pRec.p;
        Vector<Float, 3> normal = pRec.n;

        // Rotate to world coordinates
        Vector<Float, 3> sampleD = Frame(normal).toWorld(inDirLocal);
        return std::make_tuple(Ray(o + drjit::Epsilon * sampleD, sampleD, 0.0f), normal);
    }


    template<typename Float>
    inline Vector<Float, 3> sampleHG(const Vector<Float, 3>& d, const Point<Float, 2>& sample, float g) {
        Float cosTheta;
        if (std::abs(g) < drjit::Epsilon) {
            cosTheta = 1 - 2 * sample.x;
        } else {
            Float sqrTerm = (1 - g * g) / (1 - g + 2 * g * sample.x);
            cosTheta = (1 + g * g - sqrTerm * sqrTerm) / (2 * g);
        }
        Float sinTheta = drjit::safe_sqrt(1.0f - cosTheta * cosTheta);
        Float sinPhi, cosPhi;
        drjit::sincos(2 * drjit::Pi<Float> * sample.y, &sinPhi, &cosPhi);
        drjit::sincos(acos(cosTheta), &sinTheta, &cosTheta);
        return Frame(d).toWorld(Vector(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));
    }


    template<typename Float, typename Spectrum>
    class Volpath3D : public Object {
    public:
        MI_IMPORT_TYPES(Scene, Sampler, Texture)
        MI_IMPORT_OBJECT_TYPES()

        static void onbDuff(const Vector3f& n, Vector3f& b1, Vector3f& b2) {
            Float one = 1.0f;
            Float sign = dr::copysign(one, n[2]);
            const Float a = -1.0f / (sign + n[2]);
            const Float b = n[0] * n[1] * a;
            b1 = Vector3f(1.0f + sign * n[0] * n[1] * a, sign * b, -sign * n[1]);
            b2 = Vector3f(b, sign + n[1] * n[1] * a, -n[1]);
        }

        static drjit::Matrix<Float, 3> azimuthSpaceTransform(const Vector3f& refDir, const Vector3f& normal) {
            // Transform: World Space to Tangent Space
            Vector3f s, t;
            onbDuff(normal, s, t);
            drjit::Matrix<Float, 3> tsTransform(s.x, s.y, s.z, t.x, t.y, t.z, normal.x, normal.y, normal.z);

            // Transform: Tangent Space => (0, y, z) Space
            Vector3f ts_refDir = tsTransform * refDir;
            float phi = atan2(ts_refDir.x, ts_refDir.y);
            drjit::Matrix<Float, 3> rotMat({std::cos(phi), std::sin(phi), 0.f}, {-std::sin(phi), std::cos(phi), 0.f},
                                           {0.f, 0.f, 1.f});
            Vector3f newRefDir = rotMat * ts_refDir;
            // Transform: (0, y, z) => Light Space
            onbDuff(newRefDir, s, t);
            drjit::Matrix<Float, 3> lsTransform(s.x, s.y, s.z, t.x, t.y, t.z, newRefDir.x, newRefDir.y, newRefDir.z);
            return lsTransform * rotMat * tsTransform;
        }


        static drjit::Matrix<Float, 3> azimuthSpaceTransformNew(const Vector3f& light_dir, const Vector3f& normal) {
            Vector3f t1 = normalize(cross(normal, light_dir));
            Vector3f t2 = normalize(cross(light_dir, t1));
            if (drjit::any_or<true>(drjit::abs(drjit::dot(normal, light_dir)) > 0.99999f)) {
                Vector3f s, t;
                onbDuff(light_dir, s, t);
                drjit::Matrix<Float, 3> lsTransform(s.x(), s.y(), s.z(), t.x(), t.y(), t.z(), light_dir.x(),
                                                    light_dir.y(),
                                                    light_dir.z());
                return lsTransform;
            }
            drjit::Matrix<Float, 3> lsTransform(t1.x(), t1.y(), t1.z(),
                                                t2.x(), t2.y(), t2.z(),
                                                light_dir.x(), light_dir.y(), light_dir.z());
            return lsTransform;
        }

        static constexpr int nChooseK(int n, int k) {
            return (k == 0 || n == k) ? 1 : nChooseK(n - 1, k - 1) + nChooseK(n - 1, k);
        }

        static constexpr int nPolyCoeffs(int polyOrder) {
            return nChooseK(3 + polyOrder, polyOrder);
        }

        struct PathSampleResult {
            enum EStatus { EValid, EAbsorbed, EInvalid };

            Point3f pOut;
            Vector3f dOut, outNormal;
            Spectrum throughput;
            int bounces;
            EStatus status;
        };

        struct TrainingSample {
            Vector3f dIn, dOut, inNormal, outNormal;
            Point3f pIn, pOut;
            Spectrum throughput, albedo, sigmaT;
            size_t bounces;
            Float absorptionProb;
            Float absorptionProbVar;
            Float g;
            Float ior;
            std::vector<float> shapeCoeffs;
            std::vector<float> shCoefficients;
        };

        struct SamplingConfig {
            bool ignoreZeroScatter = true;
            bool disableRR = false;
            bool importanceSamplePolynomials = false;
            int maxBounces = 10000;
            float polynomialStepSize = 0.1f;
            PolyFitConfig polyCfg;
        };

        /*static PathSampleResult samplePath(const Scene* scene, const Polynomial<Float>* polynomial, Sampler* sampler,
                                           Ray3f ray,
                                           const MediumParameters<Float, Spectrum>& medium,
                                           const SamplingConfig& samplingConfig) {
            PathSampleResult r;
            r.throughput = Spectrum(1.0f);
            r.status = PathSampleResult::EStatus::EInvalid;

            Float sigmaT = medium.sigmaT.average();
            for (size_t bounces = 0; bounces < samplingConfig.maxBounces; ++bounces) {
                if (samplingConfig.ignoreZeroScatter && bounces == 0) {
                    // Trace the first ray segment such that there is no
                    // intersection
                    Interaction3f its;
                    if (scene)
                        scene->rayIntersect(ray, its);
                    else
                        its = intersectPolynomial(ray, *polynomial, samplingConfig.polynomialStepSize, false, bounces);

                    if ((scene && !its.isValid()) ||
                        (its.isValid() && dot(its.shFrame.n, ray.d) <= 0)) { // discard illegal paths
                        // scene && !its.isValid(): For polynomials its common to create "infinite objects where there wont be
                        // any intersection"
                        r.bounces = bounces;
                        return r;
                    }
                    Float t;
                    if (!its.isValid())
                        t = -log(1 - sampler->next1D()) / sigmaT;
                    else
                        t = -log(1 - sampler->next1D() * (1 - std::exp(-sigmaT * its.t))) / sigmaT;

                    r.throughput *= medium.albedo;
                    ray = Ray(ray.o + t * ray.d, sampleHG(ray.d, sampler->next2D(), medium.g), 0.0f,
                              std::numeric_limits<float>::infinity(), 0.0f);
                } else {
                    Float t = -log(1 - sampler->next1D()) / sigmaT;
                    ray.maxt = t;
                    Interaction3f its;
                    if (scene)
                        scene->rayIntersect(ray, its);
                    else
                        its = intersectPolynomial(ray, *polynomial, samplingConfig.polynomialStepSize, false, bounces);

                    if (its.isValid() && dot(its.shFrame.n, ray.d) <= 0) { // break if somehow hit object from outside
                        return r;
                    }
                    if (its.isValid()) { // If we hit the object, potentially exit
                        // Check if light gets reflected internally
                        Float cosThetaT;
                        Float F = fresnelDielectricExt(Frame::cosTheta(its.wi), cosThetaT, medium.eta);
                        if (sampler->next1D() > F) { // refract
                            // Technically, radiance should be scaled here by 1/eta**2
                            // => Since we dont scale when entering medium, we dont need it
                            r.dOut = its.shFrame.toWorld(refract(its.wi, cosThetaT, medium.eta)); // s.dOut = ray.d;
                            r.pOut = its.p;
                            r.outNormal = its.shFrame.n;
                            r.bounces = bounces;
                            r.status = PathSampleResult::EStatus::EValid;
                            return r;
                        } else {
                            ray = Ray(its.p, its.shFrame.toWorld(reflect(its.wi)), 0.0f);
                        }
                    } else {
                        r.throughput *= medium.albedo;
                        // At every medium interaction, multiply by the m.albedo of the current sample
                        ray = Ray(ray.o + t * ray.d, sampleHG(ray.d, sampler->next2D(), medium.g), 0.0f,
                                  std::numeric_limits<float>::infinity(), 0.0f);
                    }
                }
                // if the throughput is too small, perform russion roulette
                // Float rrProb = 1 - m.albedo.average();
                Float rrProb = 1.0f - r.throughput.max();
                if (samplingConfig.disableRR)
                    rrProb = 0.0f;
                if (sampler->next1D() < rrProb) {
                    r.status = PathSampleResult::EStatus::EAbsorbed;
                    r.bounces = bounces;
                    return r;
                } else {
                    r.throughput *= 1 / (1 - rrProb);
                }
            }
            r.status = PathSampleResult::EStatus::EAbsorbed; // count paths exceeding max bounce as absorbed
            return r;
        }*/

        /*static std::vector<TrainingSample>
        samplePathsBatch(const Scene* scene, const Shape* shape, const MediumParameters<Float, Spectrum>& medium,
                         const SamplingConfig& samplingConfig, size_t batchSize, size_t nAbsSamples,
                         const Point3f* inPos,
                         const Vector3f* inDir, Sampler* sampler, const Polynomial<Float>* polynomial = nullptr,
                         const ConstraintKdTree* kdtree = nullptr, int polyOrder = 3) {
            size_t maxBounces = 10000;
            Polynomial<Float> fitPolynomial;
            fitPolynomial.order = polyOrder;
            const Polynomial<Float>* tracedPoly = polynomial ? polynomial : &fitPolynomial;

            Ray3f sampledRay;
            Vector3f normal(1, 0, 0);
            std::vector<float> shCoefficients;
            if (inPos && inDir) {
                sampledRay = Ray(*inPos, *inDir, 0.0f);
            } else {
                generateStartingConfiguration(inDir, shape, sampler, medium, kdtree,
                                              samplingConfig.importanceSamplePolynomials,
                                              sampledRay, normal, fitPolynomial, samplingConfig);
            }
            size_t nAbsorbed = 0;
            size_t nEscaped = 0;
            size_t nIter = 0;
            size_t nValidSamples = 0;
            // Sample until we have enough samples to fill the current batch
            std::vector<TrainingSample> batchTrainSamples;
            while (batchTrainSamples.size() < batchSize || nValidSamples < nAbsSamples) {
                // Resample the inpos/indir if many samples are not valid
                if ((nEscaped > 2 * nAbsSamples) || (nAbsorbed > 100 * std::max(batchSize, nAbsSamples))) {
                    nIter = 0;
                    nAbsorbed = 0;
                    nEscaped = 0;
                    nValidSamples = 0;
                    batchTrainSamples.clear();
                    if (inPos && inDir) {
                        break; // If inpos and direction are invalid, we can just break and not return any samples
                    } else {
                        generateStartingConfiguration(inDir, shape, sampler, medium, kdtree,
                                                      samplingConfig.importanceSamplePolynomials, sampledRay, normal,
                                                      fitPolynomial, samplingConfig);
                    }
                }
                // Regenerate random ray direction (for now we average this out!)
                PathSampleResult r = Volpath3D::samplePath(scene, tracedPoly, sampler, sampledRay, medium,
                                                           samplingConfig);
                switch (r.status) {
                    case PathSampleResult::EStatus::EInvalid:
                        nEscaped++;
                        break;
                    case PathSampleResult::EStatus::EAbsorbed:
                        nValidSamples++;
                        if (nIter < nAbsSamples) // Only compute stats for the nAbsSamples samples
                            nAbsorbed++;
                        break;
                    case PathSampleResult::EStatus::EValid:
                        nValidSamples++;
                        TrainingSample s;
                        s.pIn = sampledRay.o;
                        s.dIn = sampledRay.d;
                        s.dOut = r.dOut;
                        s.pOut = r.pOut;
                        s.inNormal = normal;
                        s.outNormal = r.outNormal;
                        s.throughput = r.throughput;
                        s.albedo = medium.albedo;
                        s.sigmaT = medium.sigmaT;
                        s.g = medium.g;
                        s.ior = medium.eta;
                        s.bounces = r.bounces;

                        if (tracedPoly) {
                            s.shapeCoeffs = tracedPoly->coeffs;
                        }

                        if (batchTrainSamples.size() < batchSize)
                            batchTrainSamples.push_back(s);
                        break;
                }
                nIter++;
            }

            // For all the samples in the current batch, multiply their contribution by the probability of a sample being
            // absorbed
            Float absorptionProb = (Float)nAbsorbed / (Float)nAbsSamples;
            for (size_t k = 0; k < batchTrainSamples.size(); ++k) {
                batchTrainSamples[k].absorptionProb = absorptionProb;
                batchTrainSamples[k].absorptionProbVar =
                        absorptionProb * (1 - absorptionProb) / (Float)(nAbsSamples - 1);
            }
            return batchTrainSamples;
        }*/

        /*static std::vector<TrainingSample>
        samplePaths(const Scene* scene, const Shape* shape, const std::vector<MediumParameters<Float, Spectrum>>& medium,
                    const SamplingConfig& samplingConfig, size_t nSamples, size_t batchSize, size_t nAbsSamples,
                    const Point3f* inPos, const Vector3f* inDir, Sampler* sampler, const Polynomial<Float>* polynomial = nullptr,
                    const ConstraintKdTree* kdtree = nullptr) {
            std::vector<TrainingSample> trainingSamples;
            for (size_t i = 0; i < nSamples / batchSize; ++i) {
                // Get the parameters for the current batch
                size_t paramIdx = std::min(i, medium.size() - 1);
                auto batchTrainSamples =
                        samplePathsBatch(scene, shape, medium[paramIdx], samplingConfig, batchSize, nAbsSamples, inPos, inDir,
                                         sampler, polynomial, kdtree, samplingConfig.polyCfg.order);
                for (auto& b: batchTrainSamples)
                    trainingSamples.push_back(b);
            }
            return trainingSamples;
        }*/

        static bool acceptPolynomial(const Polynomial<Float>& polynomial, Sampler* sampler) {
            float a = 0.0f;
            for (int i = 4; i < polynomial.coeffs.size(); ++i) {
                float c = polynomial.coeffs[i];
                a += c * c;
            }
            a = std::log(a + 1e-4f);
            float sigmoid = 1.0f / (1.0f + std::exp(-a));
            sigmoid *= sigmoid;
            sigmoid *= sigmoid;
            return sampler->next1D() <= sigmoid;
        }

        static Float effectiveAlbedo(const Float& albedo) {
            return -drjit::log(1.0f - albedo * (1.0f - drjit::exp(-8.0f))) / 8.0f;
        }

        static Spectrum getSigmaTp(const Spectrum& albedo, float g, const Spectrum& sigmaT) {
            Spectrum sigmaS = albedo * sigmaT;
            Spectrum sigmaA = sigmaT - sigmaS;
            return (1 - g) * sigmaS + sigmaA;
        }

        static Spectrum effectiveAlbedo(const Spectrum& albedo) {
            float r = effectiveAlbedo(albedo[0]);
            float g = effectiveAlbedo(albedo[1]);
            float b = effectiveAlbedo(albedo[2]);
            Spectrum ret;
            ret.fromLinearRGB(r, g, b);
            return ret;
        }

        static Interaction<Float, Spectrum> intersectPolynomial(const Ray<Point3f, Spectrum>& ray,
                                                                const Polynomial<Float>& polynomial, Float stepSize,
                                                                bool printDebug = false, int nBounce = 0) {
            const int* pX = derivPermutation(polynomial.order, 0);
            const int* pY = derivPermutation(polynomial.order, 1);
            const int* pZ = derivPermutation(polynomial.order, 2);

            // Naive ray marching (for testing)
            float initVal;
            for (int i = 0; i < 50 / stepSize; ++i) {
                Float t = stepSize * i + ray.mint;
                bool maxTReached = false;
                if (t > ray.maxt) {
                    // If we stepped too far, check if we maybe intersect on remainder of segment
                    t = ray.maxt;
                    maxTReached = true;
                }

                float polyValue = evalPoly(polynomial.refPos, ray.o + ray.d * t, polynomial.order,
                                           polynomial.scaleFactor, polynomial.useLocalDir, polynomial.refDir,
                                           polynomial.coeffs);
                if (i == 0) {
                    initVal = polyValue;
                    // if (initVal == 0.0f) {
                    //     std::cout << "Self intersection (increase ray epsilon?), bounce: " << nBounce << std::endl;
                    // }
                } else {
                    if (polyValue * initVal <= 0) {
                        // We observed a sign change between two variables
                        // Assuming we are in a regime where the derivative makes sense, perform newton-bisection iterations to find the actual intersection location


                        // [a, b] = interval of potential solution
                        float a = stepSize * (i - 1) + ray.mint;
                        float b = stepSize * i + ray.mint;
                        if (printDebug) {
                            std::cout << "i - 1: " << i - 1 << std::endl;
                            std::cout << "i: " << i << std::endl;
                            std::cout << "a: " << (ray.o + ray.d * a).toString() << std::endl;
                            std::cout << "b: " << (ray.o + ray.d * b).toString() << std::endl;
                        }
                        t = 0.5 * (a + b);
                        Vector3f deriv;
                        for (int j = 0; j < 5; ++j) {
                            if (!(t >= a && t <= b)) // if t is out of bounds, go back to bisection method
                                t = 0.5f * (a + b);

                            std::tie(polyValue, deriv) = evalPolyGrad(polynomial.refPos, ray.o + ray.d * t,
                                                                      polynomial.order, pX, pY, pZ,
                                                                      polynomial.scaleFactor, polynomial.useLocalDir,
                                                                      polynomial.refDir, polynomial.coeffs);
                            if ((initVal < 0 && polyValue < 0) || (initVal > 0 && polyValue > 0))
                                a = t;
                            else
                                b = t;

                            t = t - polyValue / (dot(deriv, ray.d) * polynomial.scaleFactor); // update t with newton
                            if (std::abs(polyValue) < 1e-5)
                                break;
                        }
                        if (t > ray.maxt) {
                            break;
                        }
                        // deriv = Vector(1,0,0);
                        Interaction3f its;
                        its.p = ray.o + ray.d * t;
                        its.shFrame = Frame(normalize(deriv));
                        its.geoFrame = its.shFrame;
                        its.t = t;
                        its.wi = its.shFrame.toLocal(-ray.d);
                        return its;
                    }
                }
                if (maxTReached)
                    break;
            }
            Interaction3f its;
            its.t = std::numeric_limits<Float>::infinity();
            return its;
        }


        MI_DECLARE_CLASS()
    };

    /*template<typename Float, typename Spectrum>
    void generateStartingConfiguration(const Vector<Float, 3>* inDir, const Shape<Float, Spectrum>* shape, Sampler<Float, Spectrum>* sampler,
                                       const MediumParameters<Float, Spectrum>& medium, const ConstraintKdTree* kdtree,
                                       bool importanceSamplePolynomials, Ray<Point<Float, 3>, Spectrum>& sampledRay, Vector<Float, 3>& normal,
                                       PolyUtils::Polynomial& poly, const Volpath3D<Float, Spectrum>::SamplingConfig& samplingConfig) {
        assert(shape);
        if (kdtree) {
            for (int i = 0; i < 10; ++i) {
                if (inDir) {
                    std::tie(sampledRay, normal) = sampleShapeFixedInDir(shape, sampler, *inDir);
                } else {
                    std::tie(sampledRay, normal) = sampleShape(shape, medium, sampler);
                }
                PolyUtils::PolyFitRecord pfRec;
                pfRec.p = sampledRay.o;
                pfRec.d = -sampledRay.d;
                pfRec.n = normal;
                pfRec.kernelEps = PolyUtils::getKernelEps(medium);
                pfRec.config = samplingConfig.polyCfg;
                std::vector<Point<Float, 3>> pos;
                std::vector<Vector<Float, 3>> dirs;
                std::tie(poly, pos, dirs) = PolyUtils::fitPolynomial(pfRec, kdtree);
                if (!importanceSamplePolynomials || Volpath3D<Float, Spectrum>::acceptPolynomial(poly, sampler)) {
                    if (samplingConfig.polyCfg.hardSurfaceConstraint) {
                        Vector polyNormal = PolyUtils::adjustRayForPolynomialTracing(sampledRay, poly, normal);
                        normal = polyNormal; // Return the polynomial normal
                        return;
                    } else {
                        std::cout << "CURRENTLY ONLY POLYS WITH HARD SURFACE CONSTRAINT = TRUE SUPPORTED (in generateStartingConfiguration)\n";
                        exit(1);
                    }
                }
            }
            std::cout << "Failed to sample location on the surface\n";
        } else {
            if (inDir) {
                std::tie(sampledRay, normal) = sampleShapeFixedInDir(shape, sampler, *inDir);
            } else {
                std::tie(sampledRay, normal) = sampleShape(shape, medium, sampler);
            }
            return;
        }
    }*/

NAMESPACE_END(mitsuba)
