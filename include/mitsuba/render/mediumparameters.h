#pragma once

#include <mitsuba/mitsuba.h>

NAMESPACE_BEGIN(mitsuba)
    template<typename Float, typename Spectrum>
    struct MediumParameters {
        Spectrum albedo;
        Spectrum sigmaT;
        Float g;
        Float eta;

        MediumParameters() {
        }

        MediumParameters(const Spectrum& albedo, Float g, Float eta, const Spectrum& sigmaT)
            : albedo(albedo), sigmaT(sigmaT), g(g), eta(eta) {
        }

        inline bool isRgb() const {
            if constexpr (is_rgb_v<Spectrum>)
                return true;
            else
                return false;
        }
    };

NAMESPACE_END(mitsuba)
