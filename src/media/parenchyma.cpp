#include <mitsuba/core/frame.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/volume.h>

#include "organic_material.h"
#include <random>
#include <thread>
#define LAYER2_QTD_ELEMENTS 4

NAMESPACE_BEGIN(mitsuba)
    /**!
    
    .. _medium-Parenchyma:
    
    Parenchyma medium (:monosp:`Parenchyma`)
    -----------------------------------------------
    
    .. pluginparameters::
    
     * - albedo
       - |float|, |spectrum| or |volume|
       - Single-scattering albedo of the medium (Default: 0.75).
       - |exposed|, |differentiable|
    
     * - sigma_t
       - |float| or |spectrum|
       - Extinction coefficient in inverse scene units (Default: 1).
       - |exposed|, |differentiable|
    
     * - scale
       - |float|
       - Optional scale factor that will be applied to the extinction parameter.
         It is provided for convenience when accommodating data based on different
         units, or to simply tweak the density of the medium. (Default: 1)
       - |exposed|
    
     * - sample_emitters
       - |bool|
       - Flag to specify whether shadow rays should be cast from inside the volume (Default: |true|)
         If the medium is enclosed in a :ref:`dielectric <bsdf-dielectric>` boundary,
         shadow rays are ineffective and turning them off will significantly reduce
         render time. This can reduce render time up to 50% when rendering objects
         with subsurface scattering.
    
     * - (Nested plugin)
       - |phase|
       - A nested phase function that describes the directional scattering properties of
         the medium. When none is specified, the renderer will automatically use an instance of
         isotropic.
       - |exposed|, |differentiable|
    
    This class implements a Parenchyma participating medium with support for arbitrary
    phase functions. This medium can be used to model effects such as fog or subsurface scattering.
    
    The medium is parametrized by the single scattering albedo and the extinction coefficient
    :math:`\sigma_t`. The extinction coefficient should be provided in inverse scene units.
    For instance, when a world-space distance of 1 unit corresponds to a meter, the
    extinction coefficient should have units of inverse meters. For convenience,
    the scale parameter can be used to correct the units. For instance, when the scene is in
    meters and the coefficients are in inverse millimeters, set scale to 1000.
    
    .. subfigstart::
    .. subfigure:: ../../resources/data/docs/images/render/medium_Parenchyma_sss.jpg
       :caption: Parenchyma medium with constant albedo
    .. subfigure:: ../../resources/data/docs/images/render/medium_Parenchyma_sss_textured.jpg
       :caption: Parenchyma medium with spatially varying albedo
    .. subfigend::
       :label: fig-Parenchyma
    
    
    The Parenchyma medium assumes the extinction coefficient to be constant throughout the medium.
    However, it supports the use of a spatially varying albedo.
    
    .. tabs::
        .. code-tab:: xml
            :name: lst-Parenchyma
    
            <medium id="myMedium" type="Parenchyma">
                <rgb name="albedo" value="0.99, 0.9, 0.96"/>
                <float name="sigma_t" value="5"/>
    
                <!-- The extinction is also allowed to be spectrally varying
                     Since RGB values have to be in the [0, 1]
                    <rgb name="sigma_t" value="0.5, 0.25, 0.8"/>
                -->
    
                <!-- A Parenchyma medium needs to have a constant extinction,
                    but can have a spatially varying albedo:
    
                    <volume name="albedo" type="gridvolume">
                        <string name="filename" value="albedo.vol"/>
                    </volume>
                -->
    
                <phase type="hg">
                    <float name="g" value="0.7"/>
                </phase>
            </medium>
    
        .. code-tab:: python
    
            'type': 'Parenchyma',
            'albedo': {
                'type': 'rgb',
                'value': [0.99, 0.9, 0.96]
            },
            'sigma_t': 5,
            # The extinction is also allowed to be spectrally varying
            # since RGB values have to be in the [0, 1]
            # 'sigma_t': {
            #     'value': [0.5, 0.25, 0.8]
            # }
    
            # A Parenchyma medium needs to have a constant extinction,
            # but can have a spatially varying albedo:
            # 'albedo': {
            #     'type': 'gridvolume',
            #     'filename': 'albedo.vol'
            # }
    
            'phase': {
                'type': 'hg',
                'g': 0.7
            }
    */

    template<typename Float, typename Spectrum>
    class ParenchymaMedium final : public Medium<Float, Spectrum> {
    public:
        MI_IMPORT_BASE(Medium, m_is_homogeneous, m_has_spectral_extinction, m_phase_function)
        MI_IMPORT_TYPES(Scene, Sampler, Texture, Volume)

        ParenchymaMedium(const Properties& props) : Base(props) {
            m_is_homogeneous = true;
            m_albedo = props.volume<Volume>("albedo", 0.75f);
            m_sigmat = props.volume<Volume>("sigma_t", 1.f);

            m_scale = props.get<ScalarFloat>("scale", 1.0f);
            m_has_spectral_extinction = props.get<bool>("has_spectral_extinction", true);
        }

        void traverse(TraversalCallback* callback) override {
            callback->put_parameter("scale", m_scale, +ParamFlags::NonDifferentiable);
            callback->put_object("albedo", m_albedo.get(), +ParamFlags::Differentiable);
            callback->put_object("sigma_t", m_sigmat.get(), +ParamFlags::Differentiable);
            Base::traverse(callback);
        }

        MI_INLINE auto eval_sigmat(const MediumInteraction3f& mi, Mask active) const {
            auto sigmat = m_sigmat->eval(mi) * m_scale;
            if (has_flag(m_phase_function->flags(), PhaseFunctionFlags::Microflake))
                sigmat *= m_phase_function->projected_area(mi, active);
            return sigmat;
        }

        UnpolarizedSpectrum
        get_majorant(const MediumInteraction3f& mi,
                     Mask active) const override {
            MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);
            return eval_sigmat(mi, active) & active;
        }

        std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum, UnpolarizedSpectrum>
        get_scattering_coefficients(const MediumInteraction3f& mi,
                                    Mask active) const override {
            MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);
            auto sigmat = eval_sigmat(mi, active);
            auto sigmas = sigmat * m_albedo->eval(mi, active);
            UnpolarizedSpectrum sigman = 0.f;

            return {sigmas & active, sigman, sigmat & active};
        }

        std::tuple<Mask, Float, Float>
        intersect_aabb(const Ray3f& /* ray */) const override {
            return {true, 0.f, dr::Infinity<Float>};
        }

        Float computeDistance(Int32& bioType, Float& sigma) const {
            Float distance = std::numeric_limits<Float>::infinity();
            Float criteria = std::numeric_limits<Float>::infinity();
            Int32 elementIndex = 0;
            Float sigmaA = 1.0f;
            Float maxAttIndex = 0.0f;

            //Segundo o artigo, escolhemos o elemento com menor distancia para a intersecção do raio
            //O elemento com maior attIndex tem mais chance de ser atingido

            // Create a random number generator and a uniform distribution
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            for (int i = 0; i < LAYER2_QTD_ELEMENTS; i++) {
                double r = dis(gen);
                if (r == 0.0f) r = 0.5f;

                dr::masked(sigmaA, true) = (float)layer2_sigmaA[i];
                dr::masked(bioType, true) = (int)layer2_types[i];
                Float attIndex = sigmaA;
                dr::if_stmt(std::make_tuple(attIndex),
                            attIndex > 0.0f,
                            [&](auto attIndex) {
                                Float aux = -(1.0 / attIndex) * (Float)log(r);
                                dr::masked(aux, bioType == (int)EAbsorber) = -(dr::log(attIndex + 1.0f) * dr::log(r));
                                dr::masked(elementIndex, i == 0 || aux < distance) = i;
                                dr::masked(sigma, i == 0 || aux < distance) = attIndex;
                                dr::masked(distance, i == 0 || aux < distance) = aux;
                                return aux;
                            },
                            [&](const auto) {
                                return 0.0f;
                            });
            }
            dr::masked(bioType, elementIndex == 0) = (int)layer2_types[0];
            dr::masked(bioType, elementIndex == 1) = (int)layer2_types[1];
            dr::masked(bioType, elementIndex == 2) = (int)layer2_types[2];
            dr::masked(bioType, elementIndex == 3) = (int)layer2_types[3];
            return distance;
        }

        MediumInteraction3f sample_interaction(const Ray3f& ray, Float sample,
                                               UInt32 channel, Mask active) const override {
            MI_MASKED_FUNCTION(ProfilerPhase::MediumSample, active);

            // initialize basic medium interaction fields
            MediumInteraction3f mei = dr::zeros<MediumInteraction3f>();
            mei.wi = -ray.d;
            mei.sh_frame = Frame3f(mei.wi);
            mei.time = ray.time;
            mei.wavelengths = ray.wavelengths;

            auto [aabb_its, mint, maxt] = intersect_aabb(ray);
            aabb_its &= (dr::isfinite(mint) || dr::isfinite(maxt));
            active &= aabb_its;
            dr::masked(mint, !active) = 0.f;
            dr::masked(maxt, !active) = dr::Infinity<Float>;

            mint = dr::maximum(0.f, mint);
            maxt = dr::minimum(ray.maxt, maxt);

            auto combined_extinction = get_majorant(mei, active);
            Float m = combined_extinction[0];
            if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
                dr::masked(m, channel == 1u) = combined_extinction[1];
                dr::masked(m, channel == 2u) = combined_extinction[2];
            } else {
                DRJIT_MARK_USED(channel);
            }

            Int32 bioType = 0.0f;
            Float sigma = 0.0f;
            Float sampled_t = mint + computeDistance(bioType, sigma);
            Mask valid_mi = active && (sampled_t <= maxt);
            mei.t = dr::select(valid_mi, sampled_t, dr::Infinity<Float>);
            mei.p = ray(sampled_t);
            mei.medium = this;
            mei.mint = mint;
            dr::masked(active, bioType == (int)EAbsorber) = false;
            dr::masked(active, bioType == (int)EAttenuator) = true;
            active = dr::select(sampled_t < 0.0025, false, true);

            dr::masked(mei.transmittance, sampled_t > 0.0f && sampled_t < valid_mi && active) =
                    UnpolarizedSpectrum(0.5f);
            dr::masked(mei.transmittance, sampled_t > 0.0f && sampled_t < valid_mi && !active) =
                    UnpolarizedSpectrum(0.5f);
            dr::masked(mei.transmittance, !(sampled_t > 0.0f && sampled_t < valid_mi)) = UnpolarizedSpectrum(0.5f);
            dr::masked(active, !(sampled_t > 0.0f && sampled_t < valid_mi)) = false;

            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) =
                    get_scattering_coefficients(mei, valid_mi);
            mei.combined_extinction = combined_extinction;
            return mei;
        }

        std::string to_string() const override {
            std::ostringstream oss;
            oss << "ParenchymaMedium[" << std::endl
                    << "  albedo = " << string::indent(m_albedo) << "," << std::endl
                    << "  sigma_t = " << string::indent(m_sigmat) << "," << std::endl
                    << "  scale = " << string::indent(m_scale) << std::endl
                    << "]";
            return oss.str();
        }

        MI_DECLARE_CLASS()

    private:
        ref<Volume> m_sigmat, m_albedo;
        ScalarFloat m_scale;
        float layer2_sigmaA[LAYER2_QTD_ELEMENTS] = {0.2500f, 0.2765f, 0.2776f, 269.2618f};
        EBioType layer2_types[LAYER2_QTD_ELEMENTS] = {EAbsorber, EAbsorber, EAbsorber, EAbsorberAndAttenuator};
    };

    MI_IMPLEMENT_CLASS_VARIANT(ParenchymaMedium, Medium)
    MI_EXPORT_PLUGIN(ParenchymaMedium, "Parenchyma Medium")
NAMESPACE_END(mitsuba)
