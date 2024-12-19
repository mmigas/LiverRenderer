#include <mitsuba/core/platform.h>
#include <mitsuba/render/subsurface.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
/*#include "irrproc.h"
#include "bluenoise.hpp"
#include "irrtree.h"*/

NAMESPACE_BEGIN(mitsuba)
    /*template<typename Float, typename Point, typename Spectrum>
    struct IsotropicDipoleQuery {
        inline IsotropicDipoleQuery(const Spectrum& zr, const Spectrum& zv,
                                    const Spectrum& sigmaTr, const Point& p)
            : zr(zr), zv(zv), sigmaTr(sigmaTr), result(0.0f), p(p) {
        }

        inline void operator()(const IrradianceSample<Float, Point, Spectrum>& sample) {
            Float distance = dr::norm(p - sample.p);
            Spectrum rSqr = Spectrum(distance);

            /* Distance to the real source #1#
            Spectrum dr = dr::sqrt(rSqr + zr * zr);

            /* Distance to the image point source #1#
            Spectrum dv = dr::sqrt(rSqr + zv * zv);

            Spectrum C1 = zr * (sigmaTr + Spectrum(1.0f) / dr);
            Spectrum C2 = zv * (sigmaTr + Spectrum(1.0f) / dv);

            /* Do not include the reduced albedo - will be canceled out later #1#
            Spectrum dMo = Spectrum(dr::InvPi<Float>) *
                           (C1 * (dr::exp(-sigmaTr * dr)) / (dr * dr)
                            + C2 * (dr::exp(-sigmaTr * dv)) / (dv * dv));

            result += dMo * sample.E * sample.area;
        }

        inline const Spectrum& getResult() const {
            return result;
        }

        const Spectrum &zr, &zv, &sigmaTr;
        Spectrum result;
        Point p;
    };

    //static ref<Mutex> irrOctreeMutex = new Mutex();
    static int irrOctreeIndex = 0;*/

    template<typename Float, typename Spectrum>
    class Dipole final : public Subsurface<Float, Spectrum> {
    public:
        MI_IMPORT_BASE(Subsurface, m_id, m_active, Vector)
        MI_IMPORT_TYPES(Texture, Scene, Sampler, Sensor)

    private:
        Float m_radius, m_sampleMultiplier;
        Float m_Fdr, m_quality, m_eta;

        Spectrum m_D, m_de, m_sigmaT;
        Spectrum m_sigmaS, m_sigmaA, m_g;
        Spectrum m_sigmaTr, m_zr, m_zv;
        Spectrum m_sigmaSPrime, m_sigmaTPrime;
        //IrradianceOctree<Float, Point3f, Spectrum, BoundingBox3f>* m_octree;
        //ref<ParallelProcess> m_proc;
        int m_octreeResID, m_octreeIndex;
        int m_irrSamples;
        bool m_irrIndirect;

    public:
        explicit Dipole(const Properties& props) : Base(props) {
           // m_octreeIndex = irrOctreeIndex++;

            /* How many samples should be taken when estimating
               the irradiance at a given point in the scene? */
            m_irrSamples = props.get<int32_t>("irr_samples", 16);

            /* When estimating the irradiance at a given point,
               should indirect illumination be included in the final estimate? */
            m_irrIndirect = props.get<bool>("irr_indirect", true);

            /* Multiplicative factor, which can be used to adjust the number of
               irradiance samples */
            m_sampleMultiplier = props.get<float>("sample_multiplier", 1.0f);

            /* Error threshold - lower means better quality */
            m_quality = props.get<float>("quality", 0.2f);

            /* Asymmetry parameter of the phase function */
            m_octreeResID = -1;

            /* Refractive index of the medium */
            m_eta = props.get<float>("eta", 1.3f);
            m_sigmaS = Spectrum(0.74f);
            m_sigmaA = Spectrum(0.32f);
            m_g = Spectrum(0.0f);

            m_sigmaSPrime = m_sigmaS * (Spectrum(1.0f) - m_g);
            m_sigmaTPrime = m_sigmaSPrime + m_sigmaA;

            /* Find the smallest mean-free path over all wavelengths */
            Spectrum mfp = Spectrum(1.0f) / m_sigmaTPrime;
            m_radius = std::numeric_limits<Float>::max();
            for (int lambda = 0; lambda < 3; lambda++) {
                m_radius = drjit::minimum(m_radius, mfp[lambda]);
            }
            m_radius = 0.006711f;
            //m_octree = nullptr;

            /* Dipole boundary condition distance term */
            Float A = (1 + m_Fdr) / (1 - m_Fdr);

            /* Compute the reduced scattering coefficient */
            m_Fdr = fresnel_diffuse_reflectance(1 / m_eta);

            m_sigmaTr = drjit::sqrt(m_sigmaA * m_sigmaTPrime * 3.0f);
            m_zr = mfp;
            m_zv = mfp * (1.0f + 4.0f / 3.0f * A);
        }

        void preprocess(const ref<Scene> scene) override {
            /*if (m_octree)
                return; // Octree is already initialized


            // Start a timer for profiling
            Timer timer;

            BoundingBox3f aabb; // Bounding box for the scene
            Float sa; // Surface area for sample normalization

            // Step 1: Generate sampling points
            std::vector<PositionSample3f>* points = new std::vector<PositionSample3f>();
            Float actualRadius = m_radius / dr::sqrt(m_sampleMultiplier * 20);
            blueNoisePointSet<Float, Spectrum>(this->m_shapes, actualRadius, points, sa, aabb);
            Log(Info, "Blue noise point set generated in %s ms.", util::time_string(timer.reset()));
            /*if (drjit::any_or<true>(active)) {
                return Spectrum(0.0f);
            }#1#

            // Step 2: Sample irradiance
            const ref<Sensor> sensor = scene->sensors()[0];
            Float samplingTime = sensor->shutter_open() + 0.5f * sensor->shutter_open_time();
            Log(Info, "Gathering irradiance samples ..");
            IrradianceSamplingProcess<Float, Spectrum>* proc = new IrradianceSamplingProcess<Float, Spectrum>(scene,
                                                                                                              points,
                                                                                                              1024,
                                                                                                              m_irrSamples,
                                                                                                              m_irrIndirect,
                                                                                                              samplingTime);
            Log(Info, "Irradiance samples gathered in %s ms.", util::time_string(timer.reset()));
            ref<Sampler> sampler = PluginManager::instance()->create_object<Sampler>(Properties("independent"));

            // Perform irradiance sampling
            std::vector<IrradianceSample<Float, Point3f, Spectrum>> samples;
            proc->process(*points, samples);
            Log(Info, "Irradiance samples processed in %s ms.", util::time_string(timer.reset()));
            // Step 3: Normalize areas
            sa /= samples.size();
            for (auto& sample: samples) {
                sample.area = sa;
            }
            Log(Info, "Areas normalized in %s ms.", util::time_string(timer.reset()));
            samples[0].p = Point3f(0.506945f, -0.464751f, 0.960198f);
            samples[0].area = 3.032247;
            samples[0].label = 0;
            samples[1].p = Point3f(0.506945, -0.464751, 0.960198);
            samples[1].area = 3.032247;
            samples[1].label = 0;
            samples[2].p = Point3f(-0.628801, -0.705428, 0.629747);
            samples[2].area = 3.032247;
            samples[2].label = 0;
            samples[3].p = Point3f(-0.350456, -0.680434, 0.610963);
            samples[3].area = 3.032247;
            samples[3].label = 0;
            samples[4].p = Point3f(-0.100300, 0.846256, 1.232089);
            samples[4].area = 3.032247;
            samples[4].label = 0;
            samples[5].p = Point3f(0.427005, 0.065343, 0.270798);
            samples[5].area = 3.032247;
            samples[5].label = 0;
            samples[6].p = Point3f(0.686444, 0.539888, 0.164518);
            samples[6].area = 3.032247;
            samples[6].label = 0;
            samples[7].p = Point3f(0.960193, 0.445440, 0.056461);
            samples[7].area = 3.032247;
            samples[7].label = 0;
            for (auto sample: samples) {
                Log(Info, "Sample Position: %f %f %f", sample.p[0], sample.p[                             1], sample.p[2]);
            }
            // Step 4: Build the irradiance octree
            m_octree = new IrradianceOctree(aabb, m_quality, samples);
            Log(Debug, "Irradiance octree built in %i ms.", util::time_string(timer.reset()));*/
        }

        Spectrum sample(const Scene* scene, Sampler* sampler,
                        const SurfaceInteraction3f& si, const Vector& d, UInt32 depth) const override {
            /*if (!m_active || drjit::any_or<true>(drjit::dot(si.sh_frame.n, d) < 0.0f)) {
                return Spectrum(1.0f);
            }

            IsotropicDipoleQuery<Float, Point3f, Spectrum> query(m_zr, m_zv, m_sigmaTr, si.p);
            m_octree->performQuery(query);
            Spectrum result(query.getResult() * dr::InvPi<float>);

            //Log(Info, "Dipole result: %s", result);
            /*if (m_eta != 1.0f)
                result *= 1.0f - fresnelDielectricExt(dot(si.sh_Frame.n, d), m_eta);
                #1#*/
            return Spectrum(0.5f, 0.5f, 0.5f);
        }

        std::string to_string() const override {
            std::ostringstream oss;
            oss << "Dipole[" << std::endl
                    << "  id = " << m_id << std::endl
                    << "]";
            return oss.str();
        }

    protected:
        ~Dipole() override {
        }

        MI_DECLARE_CLASS()
    };

    MI_IMPLEMENT_CLASS_VARIANT(Dipole, Subsurface)
    MI_EXPORT_PLUGIN(Dipole, "Dipole Subsurface Scattering")
NAMESPACE_END(mitsuba)
