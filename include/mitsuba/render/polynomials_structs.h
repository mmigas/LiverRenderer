#pragma once

#include <mitsuba/render/mediumparameters.h>

NAMESPACE_BEGIN(mitsuba)
    template<typename Float>
    struct Polynomial {
        std::vector<float> coeffs;
        Point<Float, 3> refPos;
        Vector<Float, 3> refDir;
        bool useLocalDir;
        float scaleFactor;
        int order;
        std::vector<float> normalHistogram;
    };

    struct PolyFitConfig {
        float regularization = 0.0001f;
        bool useSvd = false;
        bool useLightspace = true;
        int order = 3;
        bool hardSurfaceConstraint = true;
        float globalConstraintWeight = 0.01f;
        float kdTreeThreshold = 0.0f;
        bool extractNormalHistogram = false;
        bool useSimilarityKernel = true;
        float kernelEpsScale = 1.0f;
    };

    template<typename Float, typename Spectrum>
    struct PolyFitRecord {
        PolyFitConfig config;
        Point<Float, 3> p;
        Vector<Float, 3> d;
        Vector<Float, 3> n;
        MediumParameters<Float, Spectrum> medium;
        float kernelEps;
    };

NAMESPACE_END(mitsuba)
