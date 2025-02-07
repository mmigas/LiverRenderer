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
        MI_IMPORT_BASE(Medium, m_is_homogeneous, m_has_spectral_extinction, m_phase_function, m_sample_emitters)
        MI_IMPORT_TYPES(Scene, Sampler, Texture, Volume)

        ParenchymaMedium(const Properties& props) : Base(props) {
            m_is_homogeneous = true;
            //m_sigmat = UnpolarizedSpectrum(77.2f / 255, 105.0f / 255, 149.0f / 255);
            m_albedo = props.volume<Volume>("albedo", 0.75f);
            m_sigmat = props.volume<Volume>("sigma_t", 1.f);
            m_sigma_blood = props.volume<Volume>("sigma_blood", 1.f);
            m_sigma_bile = props.volume<Volume>("sigma_bile", 1.f);
            m_sigma_hepatocity = props.volume<Volume>("sigma_hepatocity", 1.f);
            m_sigma_lipid_water = props.volume<Volume>("sigma_lipid_water", 1.f);

            m_scale = props.get<ScalarFloat>("scale", 1.0f);
            m_has_spectral_extinction = props.get<bool>("has_spectral_extinction", true);
            m_sample_emitters = props.get<bool>("sample_emitters", false);
        }

        void traverse(TraversalCallback* callback) override {
            callback->put_parameter("scale", m_scale, +ParamFlags::NonDifferentiable);
            callback->put_object("sigma_blood", m_sigma_blood.get(), +ParamFlags::Differentiable);
            callback->put_object("sigma_bile", m_sigma_bile.get(), +ParamFlags::Differentiable);
            callback->put_object("sigma_hepatocity", m_sigma_hepatocity.get(), +ParamFlags::Differentiable);
            callback->put_object("sigma_lipid_water", m_sigma_lipid_water.get(), +ParamFlags::Differentiable);
            Base::traverse(callback);
        }

        MI_INLINE auto eval_sigmat(const MediumInteraction3f& mi, Mask active) const {
            //auto sigmat = m_sigmat->eval(mi) * m_scale;
            auto sigmat = UnpolarizedSpectrum(77.2f / 255, 105.0f / 255, 149.0f / 255);
            /*dr::masked(sigmat, mi.bioType == 0) = m_sigma_blood->eval(mi, active);
            dr::masked(sigmat, mi.bioType == 1) = m_sigma_bile->eval(mi, active);
            dr::masked(sigmat, mi.bioType == 2) = m_sigma_lipid_water->eval(mi, active);
            dr::masked(sigmat, mi.bioType == 3) = m_sigma_hepatocity->eval(mi, active);*/
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
            auto sigmat = UnpolarizedSpectrum(77.2f / 255, 105.0f / 255, 149.0f / 255);
            auto sigmas = UnpolarizedSpectrum(74.0f / 255, 88.0f / 255, 101.0f / 255);
            /*auto sigmat = eval_sigmat(mi, active);
            auto sigmas = sigmat * m_albedo->eval(mi, active);*/
            UnpolarizedSpectrum sigman = 0.f;

            return {sigmas & active, sigman, sigmat & active};
        }

        std::tuple<Mask, Float, Float>
        intersect_aabb(const Ray3f& /* ray */) const override {
            return {true, 0.f, dr::Infinity<Float>};
        }

        dr::tuple<Int32, Float> computeDistance(MediumInteraction3f& mei, const UInt32& channel) const {
            Float distance = dr::Infinity<Float>;
            Int32 elementIndex = dr::zeros<Int32>();
            UnpolarizedSpectrum sigmaA = UnpolarizedSpectrum(1.0f);
            Int32 bioType = dr::zeros<Int32>();
            dr::PCG32<Float> rng = m_rng;
            struct LoopState {
                Float distance;
                Int32 elementIndex;
                UnpolarizedSpectrum sigmaA;
                Int32 bioType;
                dr::PCG32<Float> rng;
                DRJIT_STRUCT(LoopState, distance, elementIndex, sigmaA, bioType, rng);
            } ls = {
                        distance,
                        elementIndex,
                        sigmaA,
                        bioType,
                        rng
                    };

            //Segundo o artigo, escolhemos o elemento com menor distancia para a intersecção do raio
            //O elemento com maior attIndex tem mais chance de ser atingido
            Float sigma_blood = 0.0046f;
            Float sigma_bile = 0.0022f;
            Float sigma_lipid_water = 0.0044f;
            Float sigma_hepatocity = 269.0f;
            //           UnpolarizedSpectrum sigma_blood = m_sigma_blood->eval(mei);
            //           UnpolarizedSpectrum sigma_bile = m_sigma_bile->eval(mei);
            //           UnpolarizedSpectrum sigma_lipid_water = m_sigma_lipid_water->eval(mei);
            //           UnpolarizedSpectrum sigma_hepatocity = m_sigma_hepatocity->eval(mei);
            Int32 i = dr::zeros<Int32>();
            //Float log10Hepatocity = dr::array_t<Float>(log10(269.0f + 1.0f));
            drjit::tie(i, ls) = dr::while_loop(
                dr::make_tuple(i, ls),

                [](const Int32& i, const LoopState& ls) {
                    return i < 4;
                },

                [this, &sigma_blood, &sigma_bile, &sigma_lipid_water, &sigma_hepatocity, channel ](
            Int32& i, LoopState& ls) {
                    Float& distance = ls.distance;
                    Int32& elementIndex = ls.elementIndex;
                    UnpolarizedSpectrum& sigmaA = ls.sigmaA;
                    Int32& bioType = ls.bioType;

                    dr::PCG32<Float> rng;
                    double random = ((double)rand() / (RAND_MAX + 1));
                    Float r = dr::array_t<Float>(random);
                    dr::masked(r, r == 0.0f) = 0.5f;

                    dr::masked(sigmaA, i == 0) = sigma_blood;
                    dr::masked(bioType, i == 0) = (int)layer2_types[0];
                    dr::masked(sigmaA, i == 1) = sigma_bile;
                    dr::masked(bioType, i == 1) = (int)layer2_types[1];
                    dr::masked(sigmaA, i == 2) = sigma_lipid_water;
                    dr::masked(bioType, i == 2) = (int)layer2_types[2];
                    dr::masked(sigmaA, i == 3) = sigma_hepatocity;
                    dr::masked(bioType, i == 3) = (int)layer2_types[3];
                    Float attIndex = sigmaA[0];
                    dr::masked(attIndex, channel == 1u) = sigmaA[1];
                    dr::masked(attIndex, channel == 2u) = sigmaA[2];
                    dr::if_stmt(std::make_tuple(attIndex),
                                attIndex > 0.0f,
                                [&](Float attIndex) {
                                    Float aux = -(1.0 / attIndex) * (Float)log(r);
                                    if constexpr (is_rgb_v<Spectrum>) {
                                        dr::masked(aux, bioType == (int)EAbsorberAndAttenuator) = -(dr::log(r));
                                    }
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

            dr::masked(bioType, ls.elementIndex == 0) = (int)layer2_types[0];
            dr::masked(bioType, ls.elementIndex == 1) = (int)layer2_types[1];
            dr::masked(bioType, ls.elementIndex == 2) = (int)layer2_types[2];
            dr::masked(bioType, ls.elementIndex == 3) = (int)layer2_types[3];
            //Log(Info, "ElementIndex %d", ls.elementIndex);
            //Log(Info, "Distance %f", ls.distance);
            return {ls.bioType, ls.distance};
        }

        MediumInteraction3f sample_interaction(const Ray3f& ray, Float sample,
                                               UInt32 channel, Mask active, Float tissueDepth) const override {
            MI_MASKED_FUNCTION(ProfilerPhase::MediumSample, active);
            DRJIT_MARK_USED(tissueDepth);
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

            auto [bioType, distance] = computeDistance(mei, channel);
            //Log(Info, "Distance is %f", distance);
            //Log(Info, "Distance Sample %f", (-dr::log(1 - sample) / m));
            Float sampled_t = mint + distance;
            Mask valid_mi = active && (sampled_t <= maxt);
            mei.t = dr::select(valid_mi, sampled_t, dr::Infinity<Float>);
            mei.p = ray(sampled_t);
            mei.medium = this;
            mei.mint = mint;
            mei.transmittance = Spectrum(1.0f);
            mei.bioType = bioType;
            /*dr::masked(active, bioType == (int)EAbsorber) = false;
            dr::masked(active, bioType == (int)EAttenuator) = true;*/
            //active = dr::select(sampled_t < 0.0025, false, true);

            /*dr::masked(mei.transmittance, sampled_t > 0.0f && sampled_t < (maxt - mint) && active) = Spectrum(0.f);
            dr::masked(mei.transmittance, sampled_t > 0.0f && sampled_t < (maxt - mint) && !active) = Spectrum(0);
            dr::masked(mei.transmittance,
                       !(sampled_t > 0.0f && sampled_t < (maxt - mint) && active) && !(
                           sampled_t > 0.0f && sampled_t < (maxt - mint) && !active)) = Spectrum(1.f);
            dr::masked(mei.t, !(sampled_t > 0.0f && sampled_t < (maxt - mint) && active) && !(
                                  sampled_t > 0.0f && sampled_t < (maxt - mint) && !active)) = dr::Infinity<Float>;*/

            if (dr::all_nested(bioType == 0)) {
                active = false;
                mei.t = dr::Infinity<Float>;
                //test for absortion: No
            } else if (dr::all_nested(bioType == 1)) {
                active = true;
            }
            //test for absortion: Yes
            else {
                //propabilistic test for absorption(mockup)
                //double r = ((double) rand() / (RAND_MAX+1));
                /*double r = 0.0025; //hepatocity mean diameter
                if (dr::all_nested(sampled_t < r)) {
                    //Absorbed: Yes
                    active = false;
                    mei.t = dr::Infinity<Float>;
                } else {
                    //Absorbed: No
                    active = true;
                }*/
                
            }

            if (dr::all_nested(sampled_t > 0.0f && sampled_t < (maxt - mint) && active)) {
                mei.transmittance = Spectrum(1.0f); //(Spectrum(sigma) * (-sampled_t)).exp();		
            } else if (dr::all_nested(sampled_t > 0.0f && sampled_t < (maxt - mint) && !active)) {
                mei.transmittance = Spectrum(0.0f);
                mei.t = dr::Infinity<Float>;
            } else { //hit layer boundary
                mei.transmittance = Spectrum(1.0f);
                mei.t = dr::Infinity<Float>;
            }

            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) =
                    get_scattering_coefficients(mei, valid_mi);
            mei.combined_extinction = combined_extinction;
            return mei;
        }

        std::string to_string() const override {
            std::ostringstream oss;
            oss << "ParenchymaMedium[" << std::endl
                    << "  sigma_blood = " << string::indent(m_sigma_blood) << "," << std::endl
                    << "  scale = " << string::indent(m_scale) << std::endl
                    << "]";
            return oss.str();
        }

        MI_DECLARE_CLASS()

    private:
        dr::PCG32<Float> m_rng;


        /*UnpolarizedSpectrum m_sigmat/*, m_albedo#1#;*/
        ref<Volume> m_sigmat, m_albedo;
        ref<Volume> m_sigma_blood;
        ref<Volume> m_sigma_bile;
        ref<Volume> m_sigma_hepatocity;
        ref<Volume> m_sigma_lipid_water;
        ScalarFloat m_scale;
        EBioType layer2_types[LAYER2_QTD_ELEMENTS] = {EAbsorber, EAbsorber, EAbsorber, EAbsorberAndAttenuator};
    };

    MI_IMPLEMENT_CLASS_VARIANT(ParenchymaMedium, Medium)
    MI_EXPORT_PLUGIN(ParenchymaMedium, "Parenchyma Medium")
NAMESPACE_END(mitsuba)
