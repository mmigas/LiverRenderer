#pragma once

#include <mitsuba/core/platform.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/integrator.h>

#include "nanothread/nanothread.h"


NAMESPACE_BEGIN(mitsuba)
    template<typename Float, typename Spectrum>
    class IrradianceSample {
    public:
        IrradianceSample() {
        }

        IrradianceSample(const Point<Float, 3>& p, const Spectrum& E)
            : p(p), E(E) {
        }

        inline const Point<Float, 3>& getPosition() const {
            return p;
        }

        Point<Float, 3> p;
        Spectrum E;
        Float area; //!< total surface area represented by this sample
        uint8_t label; //!< used by the octree construction code
    };

    template<typename Float, typename Spectrum>
    class IrradianceSamplingProcess {
        MI_IMPORT_TYPES()
        MI_IMPORT_OBJECT_TYPES()

    public:
        IrradianceSamplingProcess(ref<Scene> scene,
                                  std::vector<PositionSample3f>* positions,
                                  size_t granularity,
                                  int irrSamples,
                                  bool irrIndirect,
                                  Float time
                                  /*void* data*/) : m_scene(scene),
                                                    m_positionSamples(positions),
                                                    m_granularity(granularity),
                                                    m_irrSamples(irrSamples),
                                                    m_irrIndirect(irrIndirect), m_time(time) {
            //m_resultMutex = new Mutex();
            m_irradianceSamples = new std::vector<IrradianceSample<Float, Spectrum>>();
            m_irradianceSamples->reserve(positions->size());
            m_samplesRequested = 0;
            m_integrator = m_scene->integrator();
            m_sampler = m_scene->sensors()[0]->sampler();
            m_sampler->seed(0, 64);
            //m_progress = new ProgressReporter("Sampling irradiance", data);
        }


        inline std::vector<PositionSample3f>* getIrradianceSampleVector() {
            return m_irradianceSamples.get();
        }

        inline std::vector<PositionSample3f>* getPositionSampleVector() {
            return m_positionSamples.get();
        }

        inline const BoundingBox3f& getAABB() const {
            return m_aabb;
        }

        /* ParallelProcess implementation */
        void prepare() {
            /*m_scene = static_cast<Scene*>(getResource("scene"));
            m_sampler = static_cast<Sampler*>(getResource("sampler"));
            m_integrator = static_cast<SamplingIntegrator*>(getResource("integrator"));*/
            /*m_scene->wakeup(NULL, m_resources);
            m_integrator->wakeup(NULL, m_resources);*/
        }

        void process(const std::vector<PositionSample3f>& positions,
                     std::vector<IrradianceSample<Float, Spectrum>>& result) {
            result.clear();
            SamplingIntegrator* integrator = (SamplingIntegrator*)m_integrator.get();
            for (size_t i = 0; i < positions.size(); ++i) {
                /* Create a fake intersection record */
                const PositionSample3f& sample = positions[i];
                SurfaceInteraction3f its;
                its.p = sample.p;
                its.n = sample.n;
                its.sh_frame = Frame3f(sample.n);
                //its.shape = m_scene->shapes()[sample.shape_index].get();
                its.time = m_time;
                Scene* scene = m_scene.get();
                Sampler* sampler = m_sampler.get();
                Spectrum irradiance = integrator->E(scene, its, sampler, m_irrSamples, m_irrIndirect);
                //Log(Debug, "Irradiance: %s", irradiance);
                result.push_back(IrradianceSample<Float, Spectrum>(its.p, irradiance));
            }
        }

        inline void processResult() {
        }

        //ParallelProcess::EStatus generateWork(WorkUnit* unit, int worker);


    protected:
        /// Virtual destructor
        ~IrradianceSamplingProcess() {
        };

    private:
        std::vector<PositionSample3f>* m_positionSamples;
        std::vector<IrradianceSample<Float, Spectrum>>* m_irradianceSamples;
        size_t m_samplesRequested, m_granularity;
        int m_irrSamples;
        bool m_irrIndirect;
        Float m_time;
        //ref<Mutex> m_resultMutex;
        ProgressReporter* m_progress;
        BoundingBox3f m_aabb;

        //Worker
        ref<Scene> m_scene;
        ref<Sampler> m_sampler;
        ref<Integrator> m_integrator;
        Float m_time_worker;
    };


NAMESPACE_END(mitsuba)

DRJIT_CALL_TEMPLATE_BEGIN(mitsuba::IrradianceSamplingProcess)
    DRJIT_CALL_METHOD(processResult)
DRJIT_CALL_END(mitsuba::IrradianceSampleProcess)
