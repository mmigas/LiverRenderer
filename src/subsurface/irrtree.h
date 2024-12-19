#pragma once
#include "irrproc.h"
#include "mitsuba/core/octree.h"


NAMESPACE_BEGIN(mitsuba)
    template<typename Float, typename P, typename Spectrum, typename BoundingBox>
    class IrradianceOctree : public StaticOctree<IrradianceSample<Float, P, Spectrum>, IrradianceSample<Float, P, Spectrum>, BoundingBox, P> {
    public:
        IrradianceOctree(const BoundingBox& bounds,
                         Float solidAngleThreshold,
                         std::vector<IrradianceSample<Float, P, Spectrum>>& records)
            : StaticOctree<IrradianceSample<Float, P, Spectrum>, IrradianceSample<Float, P, Spectrum>, BoundingBox, P>(bounds), m_solidAngleThreshold(solidAngleThreshold) {
            this->m_items.swap(records);
            this->build();
            this->propagate(this->m_root);

            for (size_t i = 0; i < this->m_items.size(); ++i) {
                Log(Info, "IrradianceOctree::performQuery: m_items[%d] = Position: %f %f %f, Spectrum %f %f %f, Area %f, Label %d", i, this->m_items[i].p[0],
                    this->m_items[i].p[1],
                    this->m_items[i].p[2],
                    this->m_items[i].E[0],
                    this->m_items[i].E[1],
                    this->m_items[i].E[2],
                    this->m_items[i].area,
                    this->m_items[i].label);
            }
        }

        /// Query the octree using a customizable functor, while representatives for distant nodes
        template<typename QueryType>
        inline void performQuery(QueryType& query) const {
            performQuery(this->m_aabb, this->m_root, query);
        }

    protected:
        inline void propagate(OctreeNode<IrradianceSample<Float, P, Spectrum>>* node) {
            IrradianceSample<Float, P, Spectrum>& repr = node->data;

            /* Initialize the cluster values */
            repr.E = Spectrum(0.0f);
            repr.area = 0.0f;
            repr.p = P(0.0f, 0.0f, 0.0f);
            Float weightSum = 0.0f;
            if (node->leaf) {
                /* Inner node */
                for (uint32_t i = 0; i < node->count; ++i) {
                    const IrradianceSample<Float, P, Spectrum>& sample = this->m_items[i + node->offset];
                    repr.E += sample.E * sample.area;
                    repr.area += sample.area;
                    Float luminance = sample.E.x() * 0.212671f + sample.E.y() * 0.715160f + sample.E.z() * 0.072169f;
                    Float weight = luminance * sample.area;
                    repr.p += sample.p * weight;
                    weightSum += weight;
                }
            } else {
                /* Inner node */
                for (int i = 0; i < 8; i++) {
                    OctreeNode<IrradianceSample<Float, P, Spectrum>>* child = node->children[i];
                    if (!child)
                        continue;
                    propagate(child);
                    repr.E += child->data.E * child->data.area;
                    repr.area += child->data.area;
                    Float luminance = child->data.E.x() * 0.212671f + child->data.E.y() * 0.715160f + child->data.E.z() * 0.072169f;
                    Float weight = luminance * child->data.area;
                    repr.p += child->data.p * weight;
                    weightSum += weight;
                }
            }

            drjit::if_stmt(std::make_tuple(repr.area),
                           repr.area != 0,
                           [&repr](auto) {
                               return true;
                           },
                           [](auto) {
                               return false;
                           });
            drjit::if_stmt(std::make_tuple(repr.area, weightSum),
                           weightSum != 0,
                           [&repr, &weightSum](auto, auto) {
                               return true;
                           },
                           [](auto, auto) {
                               return false;
                           });
        }


        /// Query the octree using a customizable functor, while representatives for distant nodes
        template<typename QueryType>
        void performQuery(const BoundingBox& aabb, OctreeNode<IrradianceSample<Float, P, Spectrum>>* node, QueryType& query) const {
            /* Compute the approximate solid angle subtended by samples within this node */
            P distance = query.p - node->data.p;
            Float legth_squared = drjit::norm(distance);
            Float approxSolidAngle = node->data.area / legth_squared;

            /* Use the representative if this is a distant node */
            auto contains = !aabb.contains(query.p);
            dr::if_stmt(std::make_tuple(contains, approxSolidAngle, m_solidAngleThreshold),
                        contains && approxSolidAngle < m_solidAngleThreshold,
                        [this, &query, &node](auto, auto, auto) {
                            query(node->data);
                            return true;
                        },
                        [this, &query, &node, &aabb](auto, auto, auto) {
                            if (node->leaf) {
                                for (uint32_t i = 0; i < node->count; ++i)
                                    query(this->m_items[node->offset + i]);
                            } else {
                                P center = aabb.center();
                                for (int i = 0; i < 8; i++) {
                                    if (!node->children[i])
                                        continue;

                                    BoundingBox childAABB = childBounds(i, aabb, center);
                                    performQuery(childAABB, node->children[i], query);
                                }
                            }
                            return false;
                        });
        }

    private:
        Float m_solidAngleThreshold;
    };


NAMESPACE_END(mitsuba)
