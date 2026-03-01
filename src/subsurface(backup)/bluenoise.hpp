/*
This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "irrproc.h"
#include <mitsuba/core/fwd.h>

NAMESPACE_BEGIN(mitsuba)
    /**
     * \brief Generate a point set with blue noise properties
     *
     * Based on the paper "Parallel Poisson Disk Sampling with
     * Spectrum Analysis on Surfaces" by John Bowers, Rui Wang,
     * Li-Yi Wei and David Maletz.
     *
     * \param shapes
     *    A list of input shapes on which samples should be placed
     * \param radius
     *    The Poisson radius of the point set to be generated
     * \param target
     *    A position sample vector (which will be populated with the result)
     * \param sa
     *    Will be used to return the combined surface area of the shapes
     * \param aabb
     *    Will be used to store an an axis-aligned box containing all samples
     * \param data
     *    Custom pointer that will be sent along with progress messages
     *    (usually contains a pointer to the \ref RenderJob instance)
     */
    template<typename Float, typename Spectrum>
    void blueNoisePointSet(
        std::vector<ref<Shape<Float, Spectrum>>> shapes,
        Float radius,
        std::vector<PositionSample<Float, Spectrum>>* target,
        Float& surface_area,
        BoundingBox<Point<Float, 3>>& aabb
        /*void* /*data#1#*/) {
        MI_IMPORT_TYPES()
        MI_IMPORT_OBJECT_TYPES()

        Timer timer;

        // Create and seed the sampler
        Log(Info, "Step 1: Creating and seeding the sampler");
        ref<Shape> shape = shapes[0];
        ref<Sampler> sampler = PluginManager::instance()->create_object<Sampler>(Properties("independent"));
        sampler->seed(42);
        Log(Info, "Step 1 complete (took %s)", util::time_string(timer.value()));
        // Compute the surface area of the shape
        Log(Info, "Step 2: Calculating surface area");
        surface_area = shape->surface_area();
        Log(Info, "Step 2 completed in %s", util::time_string(timer.reset()));

        // Estimate the number of initial samples based on radius
        float radius_float = drjit::slice(radius, 0);
        float surface_area_float = drjit::slice(surface_area, 0);
        int nsamples = static_cast<int>(15 * surface_area_float / (drjit::Pi<Float> * radius_float * radius_float));
        Log(Info, "Step 3: Estimating initial samples (radius=%i, surface area=%i, samples=%i)", radius_float, surface_area_float, nsamples);
        Log(Info, "Step 3 completed in %s", util::time_string(timer.reset()));

        nsamples = 10;
        // Generate white noise samples
        Log(Info, "Step 4: Generating white noise samples");
        std::vector<PositionSample3f> samples(nsamples);
        aabb.reset();

    
        struct LoopState {
            int i;
            int nsamples;
        } ls = {0, nsamples};
        drjit::while_loop(dr::make_tuple(ls),
                          [](const LoopState& ls) {
                              return ls.i < ls.nsamples;
                          },
                          [&shape, &sampler, &samples,&aabb ](LoopState& ls) {
                              PositionSample3f sample = shape->sample_position(0.0f, sampler->next_2d());
                              samples[ls.i] = sample;
                              aabb.expand(sample.p);
                              ls.i++;
                          });
        Log(Info, "Step 4 completed in %s", util::time_string(timer.reset()));

        // Cell-based filtering
        Log(Info, "Step 5: Filtering using a cell-based approach");
        Float cell_width = radius / std::sqrt(3.0f);
        Float inv_cell_width = 1.0f / cell_width;

        Vector3i cell_count = drjit::maximum(Vector3i(1, 1, 1), aabb.extents() * inv_cell_width);
        Vector3f extents = aabb.extents();

        Log(Info, "Cell count computed: x=%i, y=%i, z=%i", cell_count.x(), cell_count.y(), cell_count.z());
        Log(Info, "Step 5 completed in %s", util::time_string(timer.reset()));
        struct Vector3Int {
            int x, y, z;

            Vector3Int(Vector3i v) {
                x = drjit::slice(v.x(), 0);
                y = drjit::slice(v.y(), 0);
                z = drjit::slice(v.z(), 0);
            }

            Vector3Int(int x, int y, int z) : x(x), y(y), z(z) {
            }

            bool operator==(const Vector3Int& other) const {
                return x == other.x && y == other.y && z == other.z;
            }

            Vector3Int operator+(const Vector3Int& other) {
                x += other.x;
                y += other.y;
                z += other.z;
                return *this;
            }
        };

        struct Vector3Hasher {
            size_t operator()(const Vector3Int& v) const {
                return std::hash<int>()(v.x) ^ std::hash<int>()(v.y) ^ std::hash<int>()(v.z);
            }
        };

        Log(Info, "Step 6: Filtering samples to ensure blue noise properties");
        Vector3i cell_id = drjit::minimum((samples[0].p - aabb.min) * inv_cell_width, Vector3f(cell_count - 1));
        Vector3Int cell_id_int(cell_id);

        std::unordered_map<Vector3Int, PositionSample3f, Vector3Hasher> grid;
        for (const auto& sample: samples) {
            bool conflict = false;

            // Check for conflicts in neighboring cells
            for (int x = -1; x <= 1 && !conflict; ++x) {
                for (int y = -1; y <= 1 && !conflict; ++y) {
                    for (int z = -1; z <= 1 && !conflict; ++z) {
                        Vector3Int neighbor_cell = cell_id_int + Vector3Int(x, y, z);
                        if (grid.find(neighbor_cell) != grid.end()) {
                            const auto& neighbor = grid[neighbor_cell];
                            Point3f distance = sample.p - neighbor.p;
                            auto length = drjit::norm(distance);
                            conflict = drjit::if_stmt(std::make_tuple(length, radius),
                                                      length < radius * radius,
                                                      [](auto length, auto radius) {
                                                          Log(Info, "Conflict detected: distance=%f, radius=%f", length, radius);
                                                          return true;
                                                      },
                                                      [](auto, auto) {
                                                          return false;
                                                      });
                        }
                    }
                }
            }

            if (!conflict) {
                grid[cell_id_int] = sample;
                target->push_back(sample);
            }
        }
        Log(Info, "Step 6 completed in %s", util::time_string(timer.reset()));

        Log(Info, "Filtered down to %i blue noise samples", target->size());
        Log(Info, "Step 7: Blue noise sampling complete. Target contains %i samples.", target->size());
        Log(Info, "Total time: %s", util::time_string(timer.value()));
    }

NAMESPACE_END(mitsuba)
