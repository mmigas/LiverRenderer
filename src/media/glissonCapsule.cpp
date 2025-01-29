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

#define LAYER1_QTD_ELEMENTS 2
#define LAYER2_QTD_ELEMENTS 2
#define LAYER3_QTD_ELEMENTS 2
#define LAYER4_QTD_ELEMENTS 2
#define LAYER5_QTD_ELEMENTS 3

NAMESPACE_BEGIN(mitsuba)
    /**!
    
    .. _medium-homogeneous:
    
    Homogeneous medium (:monosp:`homogeneous`)
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
    
    This class implements a homogeneous participating medium with support for arbitrary
    phase functions. This medium can be used to model effects such as fog or subsurface scattering.
    
    The medium is parametrized by the single scattering albedo and the extinction coefficient
    :math:`\sigma_t`. The extinction coefficient should be provided in inverse scene units.
    For instance, when a world-space distance of 1 unit corresponds to a meter, the
    extinction coefficient should have units of inverse meters. For convenience,
    the scale parameter can be used to correct the units. For instance, when the scene is in
    meters and the coefficients are in inverse millimeters, set scale to 1000.
    
    .. subfigstart::
    .. subfigure:: ../../resources/data/docs/images/render/medium_homogeneous_sss.jpg
       :caption: Homogeneous medium with constant albedo
    .. subfigure:: ../../resources/data/docs/images/render/medium_homogeneous_sss_textured.jpg
       :caption: Homogeneous medium with spatially varying albedo
    .. subfigend::
       :label: fig-homogeneous
    
    
    The homogeneous medium assumes the extinction coefficient to be constant throughout the medium.
    However, it supports the use of a spatially varying albedo.
    
    .. tabs::
        .. code-tab:: xml
            :name: lst-homogeneous
    
            <medium id="myMedium" type="homogeneous">
                <rgb name="albedo" value="0.99, 0.9, 0.96"/>
                <float name="sigma_t" value="5"/>
    
                <!-- The extinction is also allowed to be spectrally varying
                     Since RGB values have to be in the [0, 1]
                    <rgb name="sigma_t" value="0.5, 0.25, 0.8"/>
                -->
    
                <!-- A homogeneous medium needs to have a constant extinction,
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
    
            'type': 'homogeneous',
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
    
            # A homogeneous medium needs to have a constant extinction,
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
    class GlissonCapsuleMedium final : public Medium<Float, Spectrum> {
    public:
        MI_IMPORT_BASE(Medium, m_is_homogeneous, m_has_spectral_extinction, m_phase_function)
        MI_IMPORT_TYPES(Scene, Sampler, Texture, Volume)

        GlissonCapsuleMedium(const Properties& props) : Base(props) {
            m_is_homogeneous = true;
            m_albedo = props.volume<Volume>("albedo", 0.75f);
            m_sigmat = props.volume<Volume>("sigma_t", 1.f);

            layer1Limit = props.get<ScalarFloat>("layer1Limit", 0.0065f);
            layer2Limit = props.get<ScalarFloat>("layer2Limit", 0.0072f);
            layer3Limit = props.get<ScalarFloat>("layer3Limit", 0.0083f);
            layer4Limit = props.get<ScalarFloat>("layer4Limit", 0.01f);

            m_sigma_collagen_layer1 = props.volume<Volume>("sigma_collagen_layer1", 1.f);
            m_sigma_collagen_layer2 = props.volume<Volume>("sigma_collagen_layer2", 1.f);
            m_sigma_collagen_layer3 = props.volume<Volume>("sigma_collagen_layer3", 1.f);
            m_sigma_collagen_layer4 = props.volume<Volume>("sigma_collagen_layer4", 1.f);

            m_sigma_elastin_layer1 = props.volume<Volume>("sigma_elastin_layer1", 1.f);
            m_sigma_elastin_layer2 = props.volume<Volume>("sigma_elastin_layer2", 1.f);
            m_sigma_elastin_layer3 = props.volume<Volume>("sigma_elastin_layer3", 1.f);
            m_sigma_elastin_layer4 = props.volume<Volume>("sigma_elastin_layer4", 1.f);


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

        Float computeDistance(MediumInteraction3f& mei, Float surfaceDistance) const {
            Float distance = dr::Infinity<Float>;
            Int32 elementIndex = dr::zeros<Int32>();
            UnpolarizedSpectrum sigmaA = UnpolarizedSpectrum(1.0f);
            Int32 layer = dr::zeros<Int32>();
            struct LoopState {
                Float distance;
                Int32 elementIndex;
                UnpolarizedSpectrum sigmaA;
                DRJIT_STRUCT(LoopState, distance, elementIndex, sigmaA);
            } ls = {
                        distance,
                        elementIndex,
                        sigmaA,
                    };

            //Segundo o artigo, escolhemos o elemento com menor distancia para a intersecção do raio
            //O elemento com maior attIndex tem mais chance de ser atingido
            if (dr::any_or<true>(surfaceDistance > layer4Limit)) {
                return 0.0f;
            } else {
                dr::masked(layer, surfaceDistance <= layer1Limit) = 0;
                dr::masked(layer, surfaceDistance <= layer2Limit) = 1;
                dr::masked(layer, surfaceDistance <= layer3Limit) = 2;
                dr::masked(layer, surfaceDistance <= layer4Limit) = 3;
            }

            UnpolarizedSpectrum sigma_collagen;
            dr::masked(sigma_collagen, layer == 0) = m_sigma_collagen_layer1->eval(mei);
            dr::masked(sigma_collagen, layer == 1) = m_sigma_collagen_layer2->eval(mei);
            dr::masked(sigma_collagen, layer == 2) = m_sigma_collagen_layer3->eval(mei);
            dr::masked(sigma_collagen, layer == 3) = m_sigma_collagen_layer4->eval(mei);

            UnpolarizedSpectrum sigma_elastin;
            dr::masked(sigma_elastin, layer == 0) = m_sigma_elastin_layer1->eval(mei);
            dr::masked(sigma_elastin, layer == 1) = m_sigma_elastin_layer2->eval(mei);
            dr::masked(sigma_elastin, layer == 2) = m_sigma_elastin_layer3->eval(mei);
            dr::masked(sigma_elastin, layer == 3) = m_sigma_elastin_layer4->eval(mei);


            Int32 i = dr::zeros<Int32>();
            drjit::tie(i, ls) = dr::while_loop(
                dr::make_tuple(i, ls),

                [](const Int32& i, const LoopState& /*ls*/) {
                    return i < 2;
                },

                [this, &sigma_collagen, &sigma_elastin](Int32& i, LoopState& ls) {
                    Float& distance = ls.distance;
                    Int32& elementIndex = ls.elementIndex;
                    UnpolarizedSpectrum& sigmaA = ls.sigmaA;

                    auto rng = dr::PCG32<Float>();
                    Float r = rng.next_float32();
                    dr::masked(r, r == 0.0f) = 0.5f;

                    dr::masked(sigmaA, i == 0) = sigma_collagen;
                    dr::masked(sigmaA, i == 1) = sigma_elastin;

                    Float attIndex = dr::mean(sigmaA);
                    dr::if_stmt(std::make_tuple(attIndex),
                                attIndex > 0.0f,
                                [&](auto attIndex) {
                                    Float aux = -(1.0 / attIndex) * (Float)log(r);
                                    dr::masked(elementIndex, i == 0 || aux < distance) = i;
                                    dr::masked(distance, i == 0 || aux < distance) = aux;
                                    return aux;
                                },
                                [&](const auto) {
                                    return 0.0f;
                                });
                    i += 1;
                    return ls;
                }
            );

            return ls.distance;
        }

        MediumInteraction3f sample_interaction(const Ray3f& ray, Float sample,
                                               UInt32 channel, Mask active, Float tissueDepth) const override {
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

            Float sampled_t = mint + computeDistance(mei, tissueDepth);
            Mask valid_mi = active && (sampled_t <= maxt);
            mei.t = dr::select(valid_mi, sampled_t, dr::Infinity<Float>);
            mei.p = ray(sampled_t);
            mei.medium = this;
            mei.mint = mint;
            active = dr::select(sampled_t < 0.0025, false, true);

            dr::masked(mei.transmittance, sampled_t > 0.0f && sampled_t < (valid_mi) && active) = Spectrum(1.f);
            dr::masked(mei.transmittance, sampled_t > 0.0f && sampled_t < (valid_mi) && !active) = Spectrum(0);
            dr::masked(mei.transmittance, !(sampled_t > 0.0f && sampled_t < (valid_mi))) = Spectrum(1.f);
            dr::masked(active, !(sampled_t > 0.0f && sampled_t < (valid_mi))) = false;
            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) =
                    get_scattering_coefficients(mei, valid_mi);
            mei.combined_extinction = combined_extinction;
            return mei;
        }


        std::string to_string() const override {
            std::ostringstream oss;
            oss << "GlissonCapsuleMedium[" << std::endl
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
        // Four layers of collagen
        ref<Volume> m_sigma_collagen_layer1;
        ref<Volume> m_sigma_collagen_layer2;
        ref<Volume> m_sigma_collagen_layer3;
        ref<Volume> m_sigma_collagen_layer4;
        // Four layers of elastin
        ref<Volume> m_sigma_elastin_layer1;
        ref<Volume> m_sigma_elastin_layer2;
        ref<Volume> m_sigma_elastin_layer3;
        ref<Volume> m_sigma_elastin_layer4;
        Float layer1Limit, layer2Limit, layer3Limit, layer4Limit, layer5Limit;
        mutable EBioType layer_types[LAYER1_QTD_ELEMENTS] = {EAttenuator, EAttenuator};
    };

    MI_IMPLEMENT_CLASS_VARIANT(GlissonCapsuleMedium, Medium)
    MI_EXPORT_PLUGIN(GlissonCapsuleMedium, "GlissonCapsule Medium")
NAMESPACE_END(mitsuba)
