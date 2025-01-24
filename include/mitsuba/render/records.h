#pragma once

#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>

NAMESPACE_BEGIN(mitsuba)
    /**
     * \brief Generic sampling record for positions
     *
     * This sampling record is used to implement techniques that draw a position
     * from a point, line, surface, or volume domain in 3D and furthermore provide
     * auxiliary information about the sample.
     *
     * Apart from returning the position and (optionally) the surface normal, the
     * responsible sampling method must annotate the record with the associated
     * probability density and delta.
     */
    template<typename Float_, typename Spectrum_>
    struct PositionSample {
        // =============================================================
        //! @{ \name Type declarations
        // =============================================================

        using Float = Float_;
        using Spectrum = Spectrum_;
        MI_IMPORT_RENDER_BASIC_TYPES()
        using SurfaceInteraction3f = typename RenderAliases::SurfaceInteraction3f;

        //! @}
        // =============================================================

        // =============================================================
        //! @{ \name Fields
        // =============================================================

        /// Sampled position
        Point3f p;

        /// Sampled surface normal (if applicable)
        Normal3f n;

        /**
         * \brief Optional: 2D sample position associated with the record
         *
         * In some uses of this record, a sampled position may be associated with
         * an important 2D quantity, such as the texture coordinates on a triangle
         * mesh or a position on the aperture of a sensor. When applicable, such
         * positions are stored in the \c uv attribute.
         */
        Point2f uv;

        /// Associated time value
        Float time;

        /// Probability density at the sample
        Float pdf;

        /// Set if the sample was drawn from a degenerate (Dirac delta) distribution
        Mask delta;

        //! @}
        // =============================================================

        // =============================================================
        //! @{ \name Constructors, methods, etc.
        // =============================================================

        /**
         * \brief Create a position sampling record from a surface intersection
         *
         * This is useful to determine the hypothetical sampling density on a
         * surface after hitting it using standard ray tracing. This happens for
         * instance in path tracing with multiple importance sampling.
         */
        PositionSample(const SurfaceInteraction3f& si)
            : p(si.p), n(si.sh_frame.n), uv(si.uv), time(si.time), pdf(0.f),
              delta(false) {
        }

        /// Basic field constructor
        PositionSample(const Point3f& p, const Normal3f& n, const Point2f& uv,
                       Float time, Float pdf, Mask delta)
            : p(p), n(n), uv(uv), time(time), pdf(pdf), delta(delta) {
        }

        //! @}
        // =============================================================

        DRJIT_STRUCT(PositionSample, p, n, uv, time, pdf, delta)
    };

    // -----------------------------------------------------------------------------

    /**
     * \brief Record for solid-angle based area sampling techniques
     *
     * This data structure is used in techniques that sample positions relative to
     * a fixed reference position in the scene. For instance, <em>direct
     * illumination strategies</em> importance sample the incident radiance
     * received by a given surface location. Mitsuba uses this approach in a wider
     * bidirectional sense: sampling the incident importance due to a sensor also
     * uses the same data structures and strategies, which are referred to as
     * <em>direct sampling</em>.
     *
     * This record inherits all fields from \ref PositionSample and extends it with
     * two useful quantities that are cached so that they don't need to be
     * recomputed: the unit direction and distance from the reference position to
     * the sampled point.
     */
    template<typename Float_, typename Spectrum_>
    struct DirectionSample : public PositionSample<Float_, Spectrum_> {
        // =============================================================
        //! @{ \name Type declarations
        // =============================================================
        using Float = Float_;
        using Spectrum = Spectrum_;

        MI_IMPORT_BASE(PositionSample, p, n, uv, time, pdf, delta)
        MI_IMPORT_RENDER_BASIC_TYPES()

        using Interaction3f = typename RenderAliases::Interaction3f;
        using SurfaceInteraction3f = typename RenderAliases::SurfaceInteraction3f;
        using EmitterPtr = typename RenderAliases::EmitterPtr;

        //! @}
        // =============================================================

        // =============================================================
        //! @{ \name Fields
        // =============================================================

        /// Unit direction from the reference point to the target shape
        Vector3f d;

        /// Distance from the reference point to the target shape
        Float dist;

        /**
          * \brief Optional: pointer to an associated object
          *
          * In some uses of this record, sampling a position also involves choosing
          * one of several objects (shapes, emitters, ..) on which the position
          * lies. In that case, the \c object attribute stores a pointer to this
          * object.
          */
        EmitterPtr emitter = nullptr;

        //! @}
        // =============================================================

        // =============================================================
        //! @{ \name Constructors, methods, etc.
        // =============================================================

        /**
         * \brief Create a direct sampling record, which can be used to \a query
         * the density of a surface position with respect to a given reference
         * position.
         *
         * Direction `s` is set so that it points from the reference surface to
         * the intersected surface, as required when using e.g. the \ref Endpoint
         * interface to compute PDF values.
         *
         * \param scene
         *     Pointer to the scene, which is needed to extract information
         *     about the environment emitter (if applicable)
         *
         * \param it
         *     Surface interaction
         *
         * \param ref
         *     Reference position
         */
        DirectionSample(const Scene<Float, Spectrum>* scene,
                        const SurfaceInteraction3f& si,
                        const Interaction3f& ref) : Base(si) {
            Vector3f rel = si.p - ref.p;
            dist = dr::norm(rel);
            d = select(si.is_valid(), rel / dist, -si.wi);
            emitter = si.emitter(scene);
        }

        /// Element-by-element constructor
        DirectionSample(const Point3f& p, const Normal3f& n, const Point2f& uv,
                        const Float& time, const Float& pdf, const Mask& delta,
                        const Vector3f& d, const Float& dist, const EmitterPtr& emitter)
            : Base(p, n, uv, time, pdf, delta), d(d), dist(dist), emitter(emitter) {
        }

        /// Construct from a position sample
        DirectionSample(const Base& base) : Base(base) {
        }

        /// Convenience operator for masking
        template<typename Array, drjit::enable_if_mask_t<Array>  = 0>
        auto operator[](const Array& array) {
            return drjit::masked(*this, array);
        }

        //! @}
        // =============================================================

        DRJIT_STRUCT(DirectionSample, p, n, uv, time, pdf, delta, d, dist, emitter)
    };

    /**
     *Mitusba 0.6
     * \brief Record for solid-angle based area sampling techniques
     *
     * This sampling record is used to implement techniques that randomly pick
     * a position on the surface of an object with the goal of importance sampling
     * a quantity that is defined over the sphere seen from a given reference point.
     *
     * This general approach for sampling positions is named "direct" sampling
     * throughout Mitsuba motivated by direct illumination rendering techniques,
     * which represent the most important application.
     *
     * This record inherits all fields from \ref PositionSamplingRecord and
     * extends it with two useful quantities that are cached so that they don't
     * need to be recomputed many times: the unit direction and length from the
     * reference position to the sampled point.
     *
     * \ingroup librender
     */
    template<typename Float_, typename Spectrum_>
    struct DirectSamplingRecord : public PositionSample<Float_, Spectrum_> {
        // =============================================================
        //! @{ \name Type declarations
        // =============================================================
        using Float = Float_;
        using Spectrum = Spectrum_;
        MI_IMPORT_BASE(PositionSample, p, n, uv, time, pdf, delta)
        MI_IMPORT_RENDER_BASIC_TYPES()

        using Interaction3f = typename RenderAliases::Interaction3f;
        using SurfaceInteraction3f = typename RenderAliases::SurfaceInteraction3f;
        using EmitterPtr = typename RenderAliases::EmitterPtr;
        /// Reference point for direct sampling
        Point3f ref;

        /**
         * \brief Optional: normal vector associated with the reference point
         *
         * When nonzero, the direct sampling method can use the normal vector
         * to sample according to the projected solid angle at \c ref.
         */
        Normal3f refN;

        /// Unit direction from the reference point to the target direction
        Vector3f d;

        /// Distance from the reference point to the target direction
        Float dist;

    public:
        /**
         * \brief Create an new direct sampling record for a reference point
         * \c ref located somewhere in space (i.e. \a not on a surface)
         *
         * \param ref
         *     The reference point
         * \param time
         *     An associated time value
         */
        inline DirectSamplingRecord(const Point3f& ref, Float time) : PositionSample<Float, Spectrum>(time), ref(ref),
                                                                      refN(0.0f) {
        }

        /**
         * \brief Create an new direct sampling record for a reference point
         * \c ref located on a surface.
         *
         * \param its
         *     The reference point specified using an intersection record
         */
        inline explicit DirectSamplingRecord(const SurfaceInteraction3f& refIts) {
        }

        /*
        /**
         * \brief Create an new direct sampling record for a reference point
         * \c ref located in a medium
         *
         * \param mRec
         *     The reference point specified using an medium sampling record
         #1#
        inline DirectSamplingRecord(const MediumSamplingRecord& mRec);
        */

        /**
         * \brief Create a direct sampling record, which can be used to \a query
         * the density of a surface position (where there reference point lies on
         * a \a surface)
         *
         * \param ray
         *     Reference to the ray that generated the intersection \c its.
         *     The ray origin must be located at \c refIts.p
         *
         * \param its
         *     A surface intersection record (usually on an emitter)
         */

        /*inline void setQuery(
            const Ray& ray,
            const Intersection& its,
            EMeasure measure = ESolidAngle);*/

        /// Return a human-readable description of the record
        std::string toString() const {
        }

        DRJIT_STRUCT(DirectSamplingRecord, p, n, uv, time, pdf, delta, ref, refN, d, dist)
    };

    /**
     *  Mitsuba 0.6
     * \brief Radiance query record data structure used by \ref SamplingIntegrator
     * \ingroup librender
     */
    MI_VARIANT
    struct RadianceQueryRecord {
    public:
        MI_IMPORT_RENDER_BASIC_TYPES()
        using SurfaceInteraction3f = typename RenderAliases::SurfaceInteraction3f;

        /// List of suported query types. These can be combined by a binary OR.
        enum ERadianceQuery {
            /// Emitted radiance from a luminaire intersected by the ray
            EEmittedRadiance = 0x0001,

            /// Emitted radiance from a subsurface integrator */
            ESubsurfaceRadiance = 0x0002,

            /// Direct (surface) radiance */
            EDirectSurfaceRadiance = 0x0004,

            /*! \brief Indirect (surface) radiance, where the last bounce did not go
                through a Dirac delta BSDF */
            EIndirectSurfaceRadiance = 0x0008,

            /*! \brief Indirect (surface) radiance, where the last bounce went
               through a Dirac delta BSDF */
            ECausticRadiance = 0x0010,

            /// In-scattered radiance due to volumetric scattering (direct)
            EDirectMediumRadiance = 0x0020,

            /// In-scattered radiance due to volumetric scattering (indirect)
            EIndirectMediumRadiance = 0x0040,

            /// Distance to the next surface intersection
            EDistance = 0x0080,

            /*! \brief Store an opacity value, which is equal to 1 when a shape
               was intersected and 0 when the ray passes through empty space.
               When there is a participating medium, it can also take on fractional
               values. */
            EOpacity = 0x0100,

            /*! \brief A ray intersection may need to be performed. This can be set to
               zero if the caller has already provided the intersection */
            EIntersection = 0x0200,

            /* Radiance from volumes */
            EVolumeRadiance = EDirectMediumRadiance | EIndirectMediumRadiance,

            /// Radiance query without emitted radiance, ray intersection required
            ERadianceNoEmission = ESubsurfaceRadiance | EDirectSurfaceRadiance
                                  | EIndirectSurfaceRadiance | ECausticRadiance | EDirectMediumRadiance
                                  | EIndirectMediumRadiance | EIntersection,

            /// Default radiance query, ray intersection required
            ERadiance = ERadianceNoEmission | EEmittedRadiance,

            /// Radiance + opacity
            ESensorRay = ERadiance | EOpacity
        };

        /// Additional flags that can be specified in the \ref extra field
        enum EExtraFlags {
            /// This is a query by an irradiance cache
            ECacheQuery = 0x01,
            /// This is a query by an adaptive integrator
            EAdaptiveQuery = 0x02
        };

        /// Construct an invalid radiance query record
        inline RadianceQueryRecord()
            : type(0), scene(NULL), sampler(NULL), medium(NULL),
              depth(0), alpha(0), dist(-1), extra(0) {
        }

        /// Construct a radiance query record for the given scene and sampler
        inline RadianceQueryRecord(const Scene<Float, Spectrum>* scene, Sampler<Float, Spectrum>* sampler)
            : type(0), scene(scene), sampler(sampler), medium(NULL),
              depth(0), alpha(0), dist(-1), extra(0) {
        }

        /// Copy constructor
        inline RadianceQueryRecord(const RadianceQueryRecord& rRec)
            : type(rRec.type), scene(rRec.scene), sampler(rRec.sampler), medium(rRec.medium),
              depth(rRec.depth), alpha(rRec.alpha), dist(rRec.dist), extra(rRec.extra) {
        }

        /// Begin a new query of the given type
        inline void newQuery(int _type, const Medium<Float, Spectrum>* _medium) {
            type = _type;
            medium = _medium;
            depth = 1;
            extra = 0;
            alpha = 1;
        }

        /// Initialize the query record for a recursive query
        inline void recursiveQuery(const RadianceQueryRecord& parent, int _type) {
            type = _type;
            scene = parent.scene;
            sampler = parent.sampler;
            depth = parent.depth + 1;
            medium = parent.medium;
            extra = parent.extra;
        }

        /// Initialize the query record for a recursive query
        inline void recursiveQuery(const RadianceQueryRecord& parent) {
            type = parent.type | EIntersection;
            scene = parent.scene;
            sampler = parent.sampler;
            depth = parent.depth + 1;
            medium = parent.medium;
            extra = parent.extra;
        }

        /**
         * \brief Search for a ray intersection
         *
         * This function does several things at once: if the
         * intersection has already been provided, it returns.
         *
         * Otherwise, it
         * 1. performs the ray intersection
         * 2. computes the transmittance due to participating media
         *   and stores it in \c transmittance.
         * 3. sets the alpha value (if \c EAlpha is set in \c type)
         * 4. sets the distance value (if \c EDistance is set in \c type)
         * 5. clears the \c EIntersection flag in \c type
         *
         * \return \c true if there is a valid intersection.
         */
        /*
        inline bool rayIntersect(const RayDifferential<Point3f, Spectrum>& ray);
        */

        /// Retrieve a 2D sample
        inline Point<Float, 2> nextSample2D();

        /*
        /// Retrieve a 1D sample
        inline Float nextSample1D();
        */

        /*/// Return a string representation
        std::string toString() const;*/

    public:
        // An asterisk (*) marks entries, which may be overwritten
        // by the callee.

        /// Query type (*)
        int type;

        /// Pointer to the associated scene
        const Scene<Float, Spectrum>* scene;

        /// Sample generator
        Sampler<Float, Spectrum>* sampler;

        /// Pointer to the current medium (*)
        const Medium<Float, Spectrum>* medium;

        /// Current depth value (# of light bounces) (*)
        int depth;

        /// Surface interaction data structure (*)
        SurfaceInteraction3f its;

        /// Opacity value of the associated pixel (*)
        Float alpha;

        /**
         * Ray distance to the first surface interaction
         * (if requested by the query type EDistance) (*)
         */
        Float dist;

        /**
         * Internal flag, which can be used to pass additional information
         * amonst recursive calls inside an integrator. The use
         * is dependent on the particular integrator implementation. (*)
         */
        int extra;
    };

    template<typename Float, typename Spectrum>
    Point<Float, 2> RadianceQueryRecord<Float, Spectrum>::nextSample2D() {
        return sampler->next_2d();
    }


    // -----------------------------------------------------------------------------

    template<typename Float, typename Spectrum>
    std::ostream& operator<<(std::ostream& os,
                             const PositionSample<Float, Spectrum>& ps) {
        os << "PositionSample" << type_suffix<Point<Float, 3>>() << "[" << std::endl
                << "  p = " << string::indent(ps.p, 6) << "," << std::endl
                << "  n = " << string::indent(ps.n, 6) << "," << std::endl
                << "  uv = " << string::indent(ps.uv, 7) << "," << std::endl
                << "  time = " << ps.time << "," << std::endl
                << "  pdf = " << ps.pdf << "," << std::endl
                << "  delta = " << ps.delta << "," << std::endl
                << "]";
        return os;
    }

    template<typename Float, typename Spectrum>
    std::ostream& operator<<(std::ostream& os,
                             const DirectionSample<Float, Spectrum>& ds) {
        os << "DirectionSample" << type_suffix<Point<Float, 3>>() << "[" << std::endl
                << "  p = " << string::indent(ds.p, 6) << "," << std::endl
                << "  n = " << string::indent(ds.n, 6) << "," << std::endl
                << "  uv = " << string::indent(ds.uv, 7) << "," << std::endl
                << "  time = " << ds.time << "," << std::endl
                << "  pdf = " << ds.pdf << "," << std::endl
                << "  delta = " << ds.delta << "," << std::endl
                << "  emitter = " << string::indent(ds.emitter) << "," << std::endl
                << "  d = " << string::indent(ds.d, 6) << "," << std::endl
                << "  dist = " << ds.dist << std::endl
                << "]";
        return os;
    }

NAMESPACE_END(mitsuba)
