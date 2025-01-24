#pragma once

NAMESPACE_BEGIN(mitsuba)
    template<typename DataType, typename IndexType>
    void permute_inplace(
        DataType* data, std::vector<IndexType>& perm) {
        for (size_t i = 0; i < perm.size(); i++) {
            if (perm[i] != i) {
                /* The start of a new cycle has been found. Save
                   the value at this position, since it will be
                   overwritten */
                IndexType j = (IndexType)i;
                DataType curval = data[i];

                do {
                    /* Shuffle backwards */
                    IndexType k = perm[j];
                    data[j] = data[k];

                    /* Also fix the permutations on the way */
                    perm[j] = j;
                    j = k;

                    /* Until the end of the cycle has been found */
                } while (perm[j] != i);

                /* Fix the final position with the saved value */
                data[j] = curval;
                perm[j] = j;
            }
        }
    }


    template<typename NodeData>
    struct OctreeNode {
        bool leaf = false;
        NodeData data;

        union {
            struct {
                OctreeNode* children[8];
            };

            struct {
                uint32_t offset;
                uint32_t count;
            };
        };

        ~OctreeNode() {
            if (!leaf) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i])
                        delete children[i];
                }
            }
        }
    };

    template<typename Item, typename NodeData, typename Float>
    class StaticOctree {
    public:
        inline StaticOctree(const BoundingBox<Point<Float, 3>>& aabb, uint32_t maxDepth = 24,
                            uint32_t maxItems = 8) : m_aabb
                                                     (aabb),
                                                     m_maxDepth(maxDepth), m_maxItems(maxItems), m_root(NULL) {
        }

        /// Release all memory
        ~StaticOctree() {
            if (m_root)
                delete m_root;
        }

        void build() {
            Log(Debug, "Building an octree over  %Iu  data points",
                m_items.size());

            ref<Timer> timer = new Timer();
            std::vector<uint32_t> perm(m_items.size()), temp(m_items.size());

            for (uint32_t i = 0; i < m_items.size(); ++i)
                perm[i] = i;

            /* Build the kd-tree and compute a suitable permutation of the elements */
            m_root = build(m_aabb, 0, &perm[0], &temp[0], &perm[0], &perm[0] + m_items.size());

            /* Apply the permutation */
            permute_inplace(&m_items[0], perm);

            Log(Debug, "Done (took %i ms)", timer->value());
        }

        inline StaticOctree() : m_root(NULL) {
        }

    protected:
        /// Return the AABB for a child of the specified index
        inline BoundingBox<Point<Float, 3>> childBounds(int child, const BoundingBox<Point<Float, 3>>& nodeAABB,
                                                        const Point<Float, 3>& center) const {
            BoundingBox<Point<Float, 3>> childAABB;
            Float x = (child & 4) ? center.x() : nodeAABB.min.x();
            Float y = (child & 2) ? center.y() : nodeAABB.min.y();
            Float z = (child & 1) ? center.z() : nodeAABB.min.z();
            childAABB.min = Point<Float, 3>(x, y, z);
            x = (child & 4) ? nodeAABB.max.x() : center.x();
            y = (child & 2) ? nodeAABB.max.y() : center.y();
            z = (child & 1) ? nodeAABB.max.z() : center.z();
            childAABB.max = Point<Float, 3>(x, y, z);
            return childAABB;
        }

        OctreeNode<NodeData>* build(const BoundingBox<Point<Float, 3>>& aabb, uint32_t depth, uint32_t* base,
                                    uint32_t* temp, uint32_t* start, uint32_t* end) {
            if (start == end) {
                return NULL;
            } else if ((uint32_t)(end - start) < m_maxItems || depth > m_maxDepth) {
                OctreeNode<NodeData>* result = new OctreeNode<NodeData>();
                result->count = (uint32_t)(end - start);
                result->offset = (uint32_t)(start - base);
                result->leaf = true;
                return result;
            }

            Point<Float, 3> center = aabb.center();
            uint32_t nestedCounts[8];
            memset(nestedCounts, 0, sizeof(uint32_t) * 8);

            /* Label all items */
            for (uint32_t* it = start; it != end; ++it) {
                Item& item = m_items[*it];
                const Point<Float, 3>& p = item.getPosition();

                uint8_t label = 0;

                if (drjit::any_or<true>(p.x() > center.x())) {
                    label |= 4;
                }
                if (drjit::any_or<true>(p.y() > center.y())) {
                    label |= 2;
                }
                if (drjit::any_or<true>(p.z() > center.z())) {
                    label |= 1;
                }


                BoundingBox<Point<Float, 3>> bounds = childBounds(label, aabb, center);
                //Assert(drjit::any_or<true>(bounds.contains(p)));
                item.label = label;
                nestedCounts[label]++;
            }

            uint32_t nestedOffsets[9];
            nestedOffsets[0] = 0;
            for (int i = 1; i <= 8; ++i)
                nestedOffsets[i] = nestedOffsets[i - 1] + nestedCounts[i - 1];

            /* Sort by label */
            for (uint32_t* it = start; it != end; ++it) {
                int offset = nestedOffsets[m_items[*it].label]++;
                temp[offset] = *it;
            }
            memcpy(start, temp, (end - start) * sizeof(uint32_t));

            /* Recurse */
            OctreeNode<NodeData>* result = new OctreeNode<NodeData>();
            for (int i = 0; i < 8; i++) {
                BoundingBox<Point<Float, 3>> bounds = childBounds(i, aabb, center);

                uint32_t* it = start + nestedCounts[i];
                result->children[i] = build(bounds, depth + 1, base, temp, start, it);
                start = it;
            }

            result->leaf = false;

            return result;
        }

        BoundingBox<Point<Float, 3>> m_aabb;
        std::vector<Item> m_items;
        uint32_t m_maxDepth;
        uint32_t m_maxItems;
        OctreeNode<NodeData>* m_root;
    };

NAMESPACE_END(mitsuba)
