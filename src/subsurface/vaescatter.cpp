/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#include <fstream>
#include <utility>
#include <vector>
#include <chrono>


#include "vaehelper.h"
#include "vaehelpereigen.h"
//#include "vaehelperpt.h"

#ifdef USETF
#include "vaehelpertf.h"
#endif


//#include <mitsuba/core/statistics.h>
#include <mitsuba/core/struct.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/subsurface.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/polynomials_structs.h>
//#include "vaeconfig.h"

#include <fstream>

#include "../../cmake-build-release/ext/zlib/zconf.h"
#include "mitsuba/render/bsdf.h"
#include "mitsuba/render/polynomials.h"


NAMESPACE_BEGIN(mitsuba)
    template<typename Spectrum>
    void logError(int line, const Spectrum& value) {
        if (!(std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2])))
            std::cout << "invalid sample line " << line << " tag " << value[0] << " " << value[1] << " " << value[2] <<
                    std::endl;
    }


    /*
    template <typename Float>
    void logError (int line, const Float& value) {
        if (!std::isfinite(value))
            std::cout << "invalid float " << line << std::endl;
    }
    */

    //#define CHECK_VALID( value ) logError(__LINE__, value)


    //static StatsCounter avgPathLength("PTracer", "Average path length", EAverage);

    template<typename Float, typename Spectrum>
    class VaeScatter : public Subsurface<Float, Spectrum> {
        MI_IMPORT_BASE(Subsurface)
        MI_IMPORT_TYPES(Texture, Scene, Sampler, Sensor, BSDF, EmitterPtr)

    public:
        VaeScatter(const Properties& props): Base(props) {
            m_scatter_model = props.get<std::string>("vaescatter", "");
            m_use_ptracer = props.get<bool>("bruteforce", false);
            m_use_ptracer_direction = props.get<bool>("useptracerdirection", false);
            m_use_polynomials = props.get<bool>("usepolynomials", false);
            m_use_difftrans = props.get<bool>("difftrans", false);
            m_use_mis = props.get<bool>("usemis", false);
            m_disable_absorption = props.get<bool>("disableabsorption", false);
            m_disable_projection = props.get<bool>("disableprojection", false);
            m_visualize_invalid_samples = props.get<bool>("showinvalidsamples", false);
            m_visualize_absorption = props.get<bool>("visualizeabsorption", false);
            // m_ignoreavgconstraints = props.getBoolean("ignoreavgconstraints", false);
            // m_low_kdtree_threshold = props.getBoolean("lowkdtreethreshold", false);

            Spectrum sigmaS, sigmaA;
            Spectrum g;
            //lookupMaterial(props, sigmaS, sigmaA, g, &m_eta);

            if (props.has_property("forceG")) {
                g = Spectrum(props.get<float>("forceG"));
                sigmaS = sigmaS / (Spectrum(1) - g);
            }

            Spectrum sigmaT = sigmaS + sigmaA;
            Spectrum albedo = sigmaS / sigmaT;

            //m_albedo = props.get<Spectrum>("albedo", albedo);
            //m_albedoTexture = props.get<Texture>(m_albedo);

            //m_sigmaT = props.getSpectrum("sigmaT", sigmaT); //
            m_g = props.get<float>("g", /*g.average()*/ 1.0f);


            m_medium.albedo = m_albedo;
            m_medium.sigmaT = m_sigmaT;
            m_medium.g = m_g;
            m_medium.eta = m_eta;

            m_use_rgb = m_medium.isRgb();

            m_modelName = props.get<std::string>("modelname", "0029_mlpshapefeaturesdeg3");
            m_absModelName = props.get<std::string>("absmodelname", "None");
            m_angularModelName = props.get<std::string>("angularmodelname", "None");
            m_outputDir = props.get<std::string>("outputdir", "/hdd/code/mitsuba-ml/pysrc/outputs/vae3d/");
            m_sssSamples = props.get<int>("sampleCount", 1);
            m_polyOrder = props.get<int>("polyOrder", 3);

            m_polyGlobalConstraintWeight = props.get<float>("polyGlobalConstraintWeight", -1.0f);
            m_polyRegularization = props.get<float>("polyRegularization", -1.0f);
            m_kernelEpsScale = props.get<float>("kernelEpsScale", 1.0f);

            PolyFitConfig pfConfig;
            m_polyGlobalConstraintWeight = m_polyGlobalConstraintWeight < 0.0f
                                               ? pfConfig.globalConstraintWeight
                                               : m_polyGlobalConstraintWeight;
            m_polyRegularization = m_polyRegularization < 0.0f ? pfConfig.regularization : m_polyRegularization;


            float roughness = props.get<float>("roughness", 0.f);
            Properties props2(roughness == 0 ? "dielectric" : "roughdielectric");
            props2.set_float("intIOR", m_eta);
            props2.set_float("extIOR", 1.f);
            if (roughness > 0)
                props2.set_float("alpha", roughness);
            m_bsdf = PluginManager::instance()->create_object<BSDF>(props2);


            // m_vaehelper = new VaeHelperTf();

            if (m_use_ptracer)
                //m_vaehelper = new VaeHelperPtracer(m_use_polynomials, m_use_difftrans, m_disable_projection, m_kernelEpsScale);
                Log(Error, "VaeHelperPtracer not implemented");
            else
                m_vaehelper = new VaeHelperEigen<Float, Spectrum>(m_kernelEpsScale);
        }

        virtual ~VaeScatter() override {
            Log(Info, "Done rendering VaeScatter, printing stats. Only accurate for SINGLE THREADED execution!");
            std::cout << "numScatterEvaluations: " << numScatterEvaluations << std::endl;
            std::cout << "totalScatterTime: " << totalScatterTime << std::endl;
            std::cout << "totalScatterTime / numScatterEvaluations: " << totalScatterTime / numScatterEvaluations <<
                    std::endl;
        }


        inline Float miWeight(Float pdfA, Float pdfB) const {
            pdfA *= pdfA;
            pdfB *= pdfB;
            return pdfA / (pdfA + pdfB);
        }

        inline Vector3f refract(const Vector3f& wi, Float cosThetaT, Float eta) const {
            Float scale = drjit::if_stmt(std::make_tuple(cosThetaT), cosThetaT < 0,
                                         [&](Float eta) {
                                             return -(1.0f / eta);
                                         },
                                         [&](Float eta) {
                                             return -eta;
                                         });
            return Vector(scale * wi.x(), scale * wi.y(), cosThetaT);
        }

        Float FresnelMoment1(Float eta) const {
            Float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta,
                    eta5 = eta4 * eta;
            Float return_float = drjit::if_stmt(std::make_tuple(eta), eta < 1,
                                                [&](const auto&) {
                                                    return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945f *
                                                           eta3 +
                                                           2.49277f * eta4 - 0.68441f * eta5;
                                                },
                                                [&](const auto&) {
                                                    return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f *
                                                           eta3 -
                                                           1.27198f * eta4 + 0.12746f * eta5;
                                                }
            );
            return return_float;
        }

        // Factor to correct throughput after refraction on outgoing direction
        Float Sw(const Vector3f& w) const { //TODO: When does this make a difference
            Float c = 1 - 2 * FresnelMoment1(1 / m_eta);
            Float cosThetaT;
            // return (1 - fresnelDielectricExt(Frame::cosTheta(w), cosThetaT, m_eta)) / (c * M_PI);
            Float eta = m_eta;
            return (1 - fresnel_ext(Frame<Float>::cos_theta(w), cosThetaT, eta)) / (c);
        }

        std::pair<Ray3f, Interaction3f> escapeObject(const Ray3f& ray, const Scene* scene) const {
            Ray3f escapeRay(ray.o, ray.d, ray.time);
            Interaction3f rayIts;
            /*
            // Continue the ray until we eventually either hit an object or dont hit anything
            Float eps = math::ShadowEpsilon<Float> * (1.0f + std::max(std::abs(ray.o.x),
                                                               std::max(std::abs(ray.o.y), std::abs(ray.o.z))));
            for (int i = 0; i < m_maxSelfIntersections; ++i) {
                if (scene->rayIntersect(escapeRay, rayIts) && rayIts.has_subsurface() && Frame<Float>::cos_theta(
                        rayIts.wi) < 0) {
                    escapeRay = Ray(rayIts.p, ray.d, eps, 100000.0f, ray.time);
                } else {
                    break;
                }
            }*/
            return std::make_pair(escapeRay, rayIts);
        }

        bool isShadowedIgnoringSssObject(Ray3f ray, const Scene* scene, Float emitterDist) const {
            /*Float eps = math::ShadowEpsilon<Float> * (1.0f + std::max(std::abs(ray.o.x),
                                                               std::max(std::abs(ray.o.y), std::abs(ray.o.z))));
            Interaction3f shadowIts;
            for (int i = 0; i < m_maxSelfIntersections; ++i) {
                if (scene->rayIntersect(ray, shadowIts)) {
                    if (shadowIts.hasSubsurface() && Frame<Float>::cos_theta(shadowIts.wi) < 0) {
                        ray = Ray(shadowIts.p, ray.d, eps, (1 - eps) * (emitterDist - shadowIts.t), shadowIts.time);
                    } else {
                        return true;
                    }
                } else {
                    return false;
                }
            }*/
            return true;
        }


        static inline Vector3f reflect(const Vector3f& wi) {
            return Vector(-wi.x(), -wi.y(), wi.z());
        }


        Spectrum LoImpl(const Scene* scene, Sampler* sampler, const SurfaceInteraction3f& its, const Vector3f& d,
                        UInt32 depth, bool recursiveCall) const {
            its.predAbsorption = Spectrum(0.0f);

            // If we use the multichannel integrator to render, we need to use its child integrator here
            MonteCarloIntegrator<Float, Spectrum>* integrator = (MonteCarloIntegrator<Float, Spectrum>*)scene->
                    integrator();
            /*if (!sRecSingleTls.get()) {
                sRecSingleTls.set(new ScatterSamplingRecordArray(1));
            }
            if (!sRecBatchedTls.get()) {
                sRecBatchedTls.set(new ScatterSamplingRecordArray(m_sssSamples));
            }
            if (!sRecRgbTls.get()) {
                sRecRgbTls.set(new ScatterSamplingRecordArray(3));
            }*/

            // {
            //     if (!recursiveCall) {
            //         BSDFSamplingRecord bRec(its, sampler, ERadiance);
            //         bRec.typeMask = BSDF::ETransmission;
            //         Float bsdfPdf;
            //         Spectrum bsdfWeight = m_bsdf->sample(bRec, bsdfPdf, sampler->next2D());
            //         auto &sRecSingle2 = sRecSingleTls.get()->data;
            //         auto &sRecBatched2 = sRecBatchedTls.get()->data;
            //         auto &sRecRgb2 = sRecRgbTls.get()->data;
            //         auto &sRec2 = (depth == 1 && m_use_rgb) ? sRecRgb2 : sRecSingle2;
            //         int nSamples2 = sRec2.size();
            //         Vector refractedD = -its.toWorld(bRec.wo);

            //         sampleOutgoingPosition(scene, its, refractedD, sampler, sRec2, nSamples2);
            //         for (int i = 0; i < nSamples2; ++i) {
            //             its.predAbsorption += sRec2[i].throughput / nSamples2 / 3.0;
            //         }
            //     }
            // }
            //BSDFSamplingRecord bRec(its, sampler, ERadiance);
            Float bsdfPdf;
            BSDFContext ctx;
            Mask active = true;
            auto [bRec, bsdfWeight] = m_bsdf->sample(ctx, its, sampler->next_1d(active), sampler->next_2d(active),
                                                     active);
            /*if ((!recursiveCall && ((bRec.sampledType & BSDF::EReflection) != 0)) ||
                (recursiveCall && ((bRec.sampledType & BSDF::ETransmission) != 0))) {
                RadianceQueryRecord query(scene, sampler);
                query.type = RadianceQueryRecord::ERadiance;
                query.depth = depth + 1;
                query.its.sampledColorChannel = its.sampledColorChannel;
                return bsdfWeight * integrator->Li(RayDifferential(its.p, its.toWorld(bRec.wo), its.time), query);
            }*/

            Vector3f refractedD = -its.to_world(bRec.wo);
            // Trace a ray to determine depth through object, then decide whether we should use 0-scattering or multiple scattering
            Ray3f zeroScatterRay(its.p, -refractedD, its.time);
            SurfaceInteraction3f zeroScatterIts = scene->ray_intersect(zeroScatterRay);
            if (drjit::any_or<true>(!zeroScatterIts.is_valid())) {
                return Spectrum(0.0f);
            }
            Float average;
            if constexpr (is_rgb_v<Spectrum>) {
                average = (-m_sigmaT.r() + -m_sigmaT.g() + -m_sigmaT.b()) / 3;
            }
            if (drjit::any_or<true>(sampler->next_1d() > 1 - drjit::exp(average * zeroScatterIts.t))) {
                // Ray passes through object without scattering
                /*RadianceQueryRecord query(scene, sampler);
                // query.newQuery(RadianceQueryRecord::ERadiance | RadianceQueryRecord::EIntersection, its.shape->getExteriorMedium());
                query.newQuery(RadianceQueryRecord::ERadiance, its.shape->getExteriorMedium());
                query.depth = depth + 1;*/

                if (drjit::any_or<true>(depth > 10))
                    return Spectrum(0.0f);

                if (drjit::any_or<true>(zeroScatterIts.has_subsurface())) {
                    //return bsdfWeight * LoImpl(scene, sampler, zeroScatterIts, refractedD, depth + 1, true);
                } else
                    return Spectrum(0.0f);
            }


            ScatterSamplingRecord<Float, Spectrum> sRecRgb[3];
            ScatterSamplingRecord<Float, Spectrum> sRecSingle[1];

            int nSamples = drjit::if_stmt(std::make_tuple(depth, m_use_rgb),
                                          depth == 1 && m_use_rgb,
                                          [&](const auto&, auto) {
                                              return 3;
                                          }, // true
                                          [&](const auto&, auto) {
                                              return 1;
                                          } // false
            );

            ScatterSamplingRecord<Float, Spectrum>* sRec = nSamples == 3 ? sRecRgb : sRecSingle;
            sampleOutgoingPosition(scene, its, refractedD, sampler, sRec, nSamples);
            Spectrum result(0.0f);
            Spectrum resultNoAbsorption(0.0f);
            int nMissed = 0;
            for (int i = 0; i < nSamples; ++i) {
                // its.predAbsorption += sRec[i].throughput;

                if (!sRec[i].isValid) {
                    nMissed++;
                    if (m_visualize_invalid_samples) {
                        Spectrum tmp = Color3f(100.0f, 0.0f, 0.0f);
                        result += tmp;
                    }
                    continue;
                }
                Spectrum throughput = bsdfWeight * m_eta * m_eta;
                // This eta multiplication accounts for outgoing location
                if (!m_disable_absorption)
                    throughput *= sRec[i].throughput;

                /*if (m_use_ptracer_direction) {
                    refractedD = sRec[i].outDir;
                    RayDifferential indirectRay(sRec[i].p, refractedD, 0.0f);
                    Interaction3f indirectRayIts;
                    indirectRayIts.sampledColorChannel = sRec[i].sampledColorChannel;
                    scene->rayIntersect(indirectRay, indirectRayIts);
                    RadianceQueryRecord query(scene, sampler);
                    query.newQuery(
                        RadianceQueryRecord::ERadiance | RadianceQueryRecord::EIntersection,
                        its.shape->getExteriorMedium()); //exiting the current shape
                    query.depth = depth + 1;
                    query.its = indirectRayIts;
                    result += throughput * integrator->sample(indirectRay, query);
                    continue;
                }*/

                if (m_visualize_absorption) {
                    result += throughput;
                    continue;
                }

                const Vector3f& normal = sRec[i].n;
                const Point3f& outPosition = sRec[i].p; { // Sample 'bsdf' for outgoing ray
                    Frame3f frame(normal);
                    Vector3f bsdfLocalDir = warp::square_to_cosine_hemisphere(sampler->next_2d());
                    Float bsdfPdf = warp::square_to_cosine_hemisphere_pdf(bsdfLocalDir);
                    Vector3f bsdfRayDir = frame.to_world(bsdfLocalDir);
                    SurfaceInteraction3f bsdfRayIts;
                    Ray3f bsdfSampleRay;

                    if (m_disable_projection) {
                        // Escape the ray: Find new ray until the ray doesnt intersect the SSS object form inside anymore
                        //std::tie(bsdfSampleRay, bsdfRayIts) = escapeObject(bsdfSampleRay, scene);
                    }

                    bsdfRayIts = scene->ray_intersect(bsdfSampleRay);
                    bsdfRayIts.sampledColorChannel = sRec[i].sampledColorChannel;

                    /*
                    RadianceQueryRecord query(scene, sampler);
                    query.newQuery(
                        RadianceQueryRecord::ERadiance | RadianceQueryRecord::EIntersection,
                        its.shape->getExteriorMedium()); //exiting the current shape
                    query.depth = depth + 1;
                    query.its = bsdfRayIts;
                    query.extra |= RadianceQueryRecord::EIgnoreFirstBounceEmitted;
                    */

                    // Evaluate illumination PDF in ray direction
                    /*Spectrum emitted(0.0f);
                    Float lumPdf = 0.0f;
                    if (drjit::any_or<true>(bsdfRayIts.is_valid())) { // If emitter was hit, we can apply MIS
                        if (bsdfRayIts.shape->is_emitter()) {
                            DirectionSample3f dRec;
                            //dRec.ref = outPosition;
                            //dRec.refN = normal;
                            emitted = bsdfRayIts.Le(-bsdfSampleRay.d);
                            //dRec.setQuery(bsdfSampleRay, bsdfRayIts);
                            lumPdf = scene->pdf_emitter_direction(bsdfRayIts, dRec);
                        }
                    } else {
                        const Emitter<Float, Spectrum>* env = scene->environment();
                        if (env) {
                            DirectionSample3f dRec;
                            dRec.refN = Vector3f(0.0f);
                            emitted = env->eval_environment(bsdfSampleRay);
                            if (env->fillDirectSamplingRecord(dRec, bsdfSampleRay))
                                lumPdf = scene->pdfEmitterDirect(dRec);
                        }
                    }*/

                    Mask active = true;
                    auto [spec, mask] = integrator->sample(scene, sampler, bsdfSampleRay, nullptr, nullptr, active);
                    Spectrum indirect = throughput * spec;
                    if (m_use_mis) {
                        /*Spectrum l = emitted * miWeight(bsdfPdf, lumPdf) * Sw(bsdfLocalDir);
                        result += indirect + throughput * l;
                        resultNoAbsorption += indirect + l;*/
                    } else {
                        Spectrum t = indirect * Sw(bsdfLocalDir);
                        //result += t;
                        //resultNoAbsorption += t;
                    }
                } { // Peform next event estimation and MIS with the path traced result
                    auto [emitterRec , emitterSampleValue] = scene->sample_emitter_direction(its,
                        sampler->next_2d(),
                        !m_disable_projection); // dont test visibility if we dont project
                    if (m_disable_projection) {
                        /*Ray3f shadowRay(emitterRec.ref, emitterRec.d, drjit::Epsilon, emitterRec.dist * (1 - math::ShadowEpsilon), emitterRec.time);
                        if (isShadowedIgnoringSssObject(shadowRay, scene, emitterRec.dist)) {
                            emitterSampleValue = Spectrum(0.0f);
                        }*/
                    }

                    //if (!emitterSampleValue.isZero()) {
                    EmitterPtr emitter = static_cast<EmitterPtr>(emitterRec.emitter);
                    const Float bsdfVal = drjit::InvPi<Float> * drjit::maximum(drjit::dot(emitterRec.d, normal), 0.0f);

                    Frame3f local(normal);
                    if (drjit::any_or<true>(bsdfVal > 0)) {
                        Float bsdfPdf;
                        //bool hasFlag = (bool)has_flag(emitter->flags(), EmitterFlags::Surface);
                        auto mask = emitter->flags() & (uint32_t)EmitterFlags::Surface;
                        if (drjit::any_or<true>(mask)) {
                            bsdfPdf = bsdfVal;
                        } else {
                            bsdfPdf = 0;
                        }
                        if (m_use_mis) {
                            /*Spectrum emitted = emitterSampleValue * bsdfVal * miWeight(emitterRec.pdf, bsdfPdf) * Sw(local.toLocal(emitterRec.d));
                            result += throughput * emitted;
                            resultNoAbsorption += emitted;*/
                        } else {
                            Spectrum emitted = emitterSampleValue * bsdfVal * Sw(local.to_local(emitterRec.d));
                            result += throughput * emitted;
                            resultNoAbsorption += emitted;
                        }
                    }
                    //}
                }
            }
            if (m_use_ptracer) {
                //its.predAbsorption = Spectrum(1.0 - (Float)nMissed / (Float)nSamples);
                //its.missedProjection = 0.0f;
            } else {
                its.missedProjection = (float)nMissed / (float)nSamples;
                // its.predAbsorption /= nSamples;
            }
            its.filled = true;
            if (!m_use_ptracer)
                its.noAbsorption = resultNoAbsorption / nSamples;

            if constexpr (is_rgb_v<Spectrum>) {
                if (drjit::any_nested(drjit::isnan(result[0]) || drjit::isnan(result[1]) || drjit::isnan(result[2]))) {
                    Log(Warn, "VaeScatter encountered NaN value!");
                    return Spectrum(0.0f);
                }
            }
            return result / nSamples;
        }


        Spectrum sample(const Scene* scene, Sampler* sampler, const SurfaceInteraction3f& si, const Vector3f& d,
                        UInt32 depth) const override {
            if (drjit::any_or<true>(drjit::dot(si.sh_frame.n, d) < 0.0f))
                // Discard if we somehow hit the object from inside
                return Spectrum(0.0f);
            return LoImpl(scene, sampler, si, d, depth, false);
        }


        void configure() {
        }

        void sampleOutgoingPosition(const Scene* scene, const SurfaceInteraction3f& its, const Vector3f& d,
                                    Sampler* sampler,
                                    ScatterSamplingRecord<Float, Spectrum>* sRec, int nSamples) const {
            //UnpolarizedSpectrum albedo = m_albedoTexture->eval(its);
            if (nSamples == 3) {
                Vector3f inDir = -d;
                if (m_use_ptracer) {
                    /*for (int i = 0; i < 3; ++i) {
                        sRec[i] = m_vaehelper->sample(scene, its.p, d, Vector3f(0.0f, 1.0f, 0.0f), m_sigmaT, m_albedo,
                                                      m_g, m_eta, sampler, &its, true, i);
                        Spectrum tmp = sRec[i].throughput;
                        sRec[i].throughput = Spectrum(0.0f);
                        sRec[i].throughput[i] = tmp[i] * 3.0f;
                    }
                    return;*/
                }
                for (int i = 0; i < 3; ++i) {
                    Vector3f inDir2 = inDir;
                    Float polyScale = PolyUtils<Float, Spectrum>::getFitScaleFactor(m_medium, i);
                    Vector3f polyNormal = PolyUtils<Float, Spectrum>::adjustRayDirForPolynomialTracing(
                        inDir2, its, 3, polyScale, i);
                    sRec[i] = m_vaehelper->sample(scene, its.p, inDir2, polyNormal, m_sigmaT, m_albedo, m_g, m_eta,
                                                  sampler, &its, true, i);
                    Spectrum tmp = sRec[i].throughput;
                    sRec[i].throughput = Spectrum(0.0f);
                    sRec[i].throughput[i] = tmp[i] * 3.0f;
                }
            } /*else {
                assert(nSamples == 1);
                if (m_use_rgb) {
                    // Sample random color channel to sample

                    bool randomSampleChannel = its.sampledColorChannel < 0;
                    int channel = randomSampleChannel ? int(3 * sampler->next1D()) : its.sampledColorChannel;

                    Vector3f inDir = -d;
                    if (m_use_ptracer) {
                        sRec[0] = m_vaehelper->sample(scene, its.p, d, Vector3f(0.0f, 1.0f, 0.0f),
                                                      m_sigmaT, albedo, m_g, m_eta, sampler, &its, true, channel);
                    } else {
                        Vector3f polyNormal = PolyUtils<Float, Spectrum>::adjustRayDirForPolynomialTracing(inDir, its, 3, PolyUtils<Float, Spectrum>::getFitScaleFactor(m_medium, channel), channel);
                        sRec[0] = m_vaehelper->sample(scene, its.p, inDir, polyNormal, m_sigmaT, albedo, m_g, m_eta, sampler, &its, true, channel);
                    }

                    Spectrum tmp = sRec[0].throughput * (randomSampleChannel ? 3.0f : 1.0f);
                    sRec[0].throughput = Spectrum(0.0f);
                    sRec[0].throughput[channel] = tmp[channel];
                    sRec[0].sampledColorChannel = channel;
                    return;
                } else {
                    Vector3f inDir = -d;
                    if (m_use_ptracer) {
                        sRec[0] = m_vaehelper->sample(scene, its.p, d, Vector3f(0.0f, 1.0f, 0.0f), m_sigmaT, albedo, m_g, m_eta, sampler, &its, true);
                        return;
                    }
                    Vector3f polyNormal = PolyUtils<Float, Spectrum>::adjustRayDirForPolynomialTracing(inDir, its, 3, PolyUtils<Float, Spectrum>::getFitScaleFactor(m_medium));
                    sRec[0] = m_vaehelper->sample(scene, its.p, inDir, polyNormal, m_sigmaT, albedo, m_g, m_eta, sampler, &its, true);
                }
            }*/
        }

        void preprocess(const ref<Scene> scene) override {
            Log(Info, "Preprocessing SSS");
            ref<Sampler> sampler = PluginManager::instance()->create_object<Sampler>(Properties("independent"));
            Log(Info, "n shapes %d", scene.get()->shapes().size());

            auto preprocStart = std::chrono::steady_clock::now();

            PolyFitConfig pfConfig;
            pfConfig.regularization = m_polyRegularization;
            pfConfig.globalConstraintWeight = m_polyGlobalConstraintWeight;
            pfConfig.order = m_polyOrder;
            pfConfig.kernelEpsScale = m_kernelEpsScale;
            m_vaehelper->prepare(scene.get(), scene->shapes(), m_sigmaT, m_albedo, m_g, m_eta, m_modelName,
                                 m_absModelName, m_angularModelName, m_outputDir, m_sssSamples, pfConfig);
            if (!m_use_ptracer) // if ML model is used, the config is well defined
                m_polyOrder = m_vaehelper->getConfig().polyOrder;


            auto preprocEnd = std::chrono::steady_clock::now();
            auto preprocDiff = preprocEnd - preprocStart;
            double totalSecondsPreproc = std::chrono::duration<double, std::milli>(preprocDiff).count() / 1000.0;
            Log(Info, "Preprocessing time: %fs", totalSecondsPreproc);
        }

    protected:
        MI_DECLARE_CLASS()

    private:
        float m_eta;
        ref<Texture> m_albedoTexture;
        ref<BSDF> m_bsdf;
        // ref<Texture> m_sigmaT;
        Spectrum m_albedo, m_sigmaT;
        float m_g;
        float m_polyGlobalConstraintWeight, m_polyRegularization, m_kernelEpsScale;
        std::string m_scatter_model, m_modelName, m_absModelName, m_angularModelName, m_outputDir;
        int m_sssSamples, m_polyOrder;
        int m_maxSelfIntersections = 10;
        MediumParameters<Float, Spectrum> m_medium;
        ref<VaeHelper<Float, Spectrum>> m_vaehelper;
        bool m_use_ptracer, m_use_difftrans, m_use_mis, m_use_polynomials, m_disable_projection,
                m_disable_absorption, m_visualize_invalid_samples, m_visualize_absorption,
                m_use_ptracer_direction, m_use_rgb;

        mutable double totalScatterTime = 0.0;
        mutable double numScatterEvaluations = 0.0;

        /*mutable ThreadLocal<ScatterSamplingRecordArray> sRecSingleTls;
        mutable ThreadLocal<ScatterSamplingRecordArray> sRecBatchedTls;
        mutable ThreadLocal<ScatterSamplingRecordArray> sRecRgbTls;*/
    };


    MI_IMPLEMENT_CLASS_VARIANT(VaeScatter, Subsurface)
    MI_EXPORT_PLUGIN(VaeScatter, "VaeScatter Subsurface Scattering")
NAMESPACE_END(mitsuba)
