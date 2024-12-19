#pragma once

#include <mitsuba/core/profiler.h>
#include <mitsuba/render/interaction.h>
#include <drjit/call.h>


NAMESPACE_BEGIN(mitsuba)
template <typename Float, typename Spectrum> class MI_EXPORT_LIB Subsurface
    : public Object {
public:
    static constexpr size_t Size = dr::size_v<Point<Float, 3>>;
    using Vector                 = Vector<Float, Size>;

public:
    MI_IMPORT_TYPES(Scene, Sampler, Texture)

    ~Subsurface();

    /// Evaluate the subsurface scattering model
    virtual Spectrum sample(const Scene *scene, Sampler *sampler,
                            const SurfaceInteraction3f &si, const Vector &d,
                            UInt32 depth) const = 0;

    void add_shape(const ref<Shape<Float, Spectrum>> &shape) {
        m_shapes.push_back(shape);
    }

    virtual void wake_up() {
        m_active = true;
    }

    virtual void preprocess(const ref<Scene> scene) = 0;

    /// Return a string identifier
    std::string id() const override {
        return m_id;
    }

    /// Set a string identifier
    void set_id(const std::string &id) override {
        m_id = id;
    };

    std::string to_string() const override = 0;

    MI_DECLARE_CLASS()

protected:
    explicit Subsurface(const Properties &props);

    std::string m_id;
    bool m_active;
    std::vector<ref<Shape<Float, Spectrum>>> m_shapes;
};

template <typename Float, typename Spectrum> std::ostream &operator<<(
    std::ostream &os, const Subsurface<Float, Spectrum> &s) {
    os << "Subsurface[" << std::endl
        << "  id = " << s.id() << std::endl
        << "]";
    return os;
}

MI_EXTERN_CLASS(Subsurface)
NAMESPACE_END(mitsuba)

// -----------------------------------------------------------------------
//! @{ \name Dr.Jit support for vectorized function calls
// -----------------------------------------------------------------------

DRJIT_CALL_TEMPLATE_BEGIN(mitsuba::Subsurface)
    DRJIT_CALL_METHOD(sample)
    DRJIT_CALL_METHOD(add_shape)
    DRJIT_CALL_METHOD(preprocess)
DRJIT_CALL_END(mitsuba::Subsurface)

//! @}
// -----------------------------------------------------------------------