#pragma once

#include <mitsuba/render/subsurface.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)
MI_VARIANT Subsurface<Float, Spectrum>::Subsurface(const Properties &props)
    : m_id(props.id()), m_active(false) {
    MI_REGISTRY_PUT("Subsurface", this);
}

MI_VARIANT Subsurface<Float, Spectrum>::~Subsurface() {
    if constexpr (dr::is_jit_v<Float>)
        jit_registry_remove(this);
}


MI_IMPLEMENT_CLASS_VARIANT(Subsurface, Object, "subsurface")
MI_INSTANTIATE_CLASS(Subsurface)
NAMESPACE_END(mitsuba)