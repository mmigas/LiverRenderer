#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT Medium<Float, Spectrum>::Medium() :
    m_is_homogeneous(false),
    m_has_spectral_extinction(true) {

    MI_REGISTRY_PUT("Medium", this);
}

MI_VARIANT Medium<Float, Spectrum>::Medium(const Properties &props) : m_id(props.id()) {
    for (auto &[name, obj] : props.objects(false)) {
        auto *phase = dynamic_cast<PhaseFunction *>(obj.get());
        if (phase) {
            if (m_phase_function)
                Throw("Only a single phase function can be specified per medium");
            m_phase_function = phase;
            props.mark_queried(name);
        }
    }
    if (!m_phase_function) {
        // Create a default isotropic phase function
        m_phase_function =
            PluginManager::instance()->create_object<PhaseFunction>(Properties("isotropic"));
    }

    m_sample_emitters = props.get<bool>("sample_emitters", true);

    MI_REGISTRY_PUT("Medium", this);
}

MI_VARIANT Medium<Float, Spectrum>::~Medium() {
    if constexpr (dr::is_jit_v<Float>)
        jit_registry_remove(this);
}

MI_VARIANT void Medium<Float, Spectrum>::traverse(TraversalCallback *callback) {
    callback->put_object("phase_function", m_phase_function.get(), +ParamFlags::Differentiable);
}


MI_VARIANT
std::pair<typename Medium<Float, Spectrum>::UnpolarizedSpectrum,
          typename Medium<Float, Spectrum>::UnpolarizedSpectrum>
Medium<Float, Spectrum>::transmittance_eval_pdf(const MediumInteraction3f &mi,
                                                const SurfaceInteraction3f &si,
                                                Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);

    Float t      = dr::minimum(mi.t, si.t) - mi.mint;
    UnpolarizedSpectrum tr  = dr::exp(-t * mi.combined_extinction);
    UnpolarizedSpectrum pdf = dr::select(si.t < mi.t, tr, tr * mi.combined_extinction);
    return { tr, pdf };
}

MI_IMPLEMENT_CLASS_VARIANT(Medium, Object, "medium")
MI_INSTANTIATE_CLASS(Medium)
NAMESPACE_END(mitsuba)
