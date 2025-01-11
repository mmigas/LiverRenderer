#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <vector>

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/mitsuba.h>
#include "interaction.h"
#include <Eigen/Core>
#include <Eigen/src/SVD/BDCSVD.h>
#include <mitsuba/render/polynomials_structs.h>

#include "sss_particle_tracer.h"


NAMESPACE_BEGIN(mitsuba)
    class ConstraintKdTree;
    /*
    class MTS_EXPORT_RENDER ConstraintKdTree {
    
    public:
        struct ExtraData {
            Point p, avgP;
            Vector n, avgN;
            size_t sampleCount;
        };
    
        typedef SimpleKDNode<Point3, ExtraData> TreeNode;
        typedef PointKDTree<SimpleKDNode<Point3, ExtraData>> CTree;
    
        void build(const std::vector<Point3> &sampledP, const std::vector<Vector3> &sampledN) {
            auto nPoints = sampledP.size();
            m_tree       = CTree(nPoints);
    
            for (size_t i = 0; i < nPoints; ++i) {
                m_tree[i].setPosition(sampledP[i]);
                ExtraData d;
                d.p           = sampledP[i];
                d.n           = sampledN[i];
                d.sampleCount = 1;
                d.avgN        = Vector(-10, -10, -10);
                d.avgP        = Point(-100, -100, -100);
                m_tree[i].setData(d);
            }
            m_tree.build(true);
    
            // For each node the tree: Traverse children recursively and get average position, normal as well as sample
            // count sum out
            avgValues(m_tree[0], 0);
            std::cout << "Sample Count " << m_tree[0].getData().sampleCount << std::endl;
            // Gather a random subset of points
            for (size_t i = 0; i < std::min(size_t(32), sampledP.size()); ++i) {
                m_globalPoints.push_back(sampledP[i]);
                m_globalNormals.push_back(sampledN[i]);
            }
        }
    
        std::tuple<std::vector<Point>, std::vector<Vector>, std::vector<Float>>
        getConstraints(const Point &p, Float kernelEps, const std::function<Float(Float, Float)> &kernel) const {
            // Extract constraints from KD Tree by traversing from the top
            std::vector<Point> positions;
            std::vector<Vector> normals;
            std::vector<Float> sampleWeights;
            getConstraints(p, m_tree[0], 0, m_tree.getAABB(), positions, normals, sampleWeights, kernelEps, kernel);
    
            // Add the super constraints
            for (size_t i = 0; i < m_globalPoints.size(); ++i) {
                positions.push_back(m_globalPoints[i]);
                normals.push_back(m_globalNormals[i]);
                sampleWeights.push_back(-1.0f);
            }
            return std::make_tuple(positions, normals, sampleWeights);
        }
    
    private:
        std::vector<Point> m_globalPoints;
        std::vector<Vector> m_globalNormals;
        CTree m_tree;
        std::tuple<Point, Vector, size_t> avgValues(TreeNode &node, TreeNode::IndexType index);
        std::pair<Float, Float> getMinMaxDistanceSquared(const Point &p, const AABB &bounds) const;
        void getConstraints(const Point &p, TreeNode node, TreeNode::IndexType index, const AABB &aabb,
                                std::vector<Point> &points, std::vector<Vector> &normals, std::vector<Float> &sampleWeights,
                                Float kernelEps, const std::function<Float(Float, Float)> &kernel) const;
    };
    */

    const static int PERMX2[10] = {1, 4, 5, 6, -1, -1, -1, -1, -1, -1};
    const static int PERMY2[10] = {2, 5, 7, 8, -1, -1, -1, -1, -1, -1};
    const static int PERMZ2[10] = {3, 6, 8, 9, -1, -1, -1, -1, -1, -1};
    const static int PERMX3[20] = {1, 4, 5, 6, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    const static int PERMY3[20] = {2, 5, 7, 8, 11, 13, 14, 16, 17, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    const static int PERMZ3[20] = {3, 6, 8, 9, 12, 14, 15, 17, 18, 19, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};


    inline size_t numPolynomialCoefficients(size_t deg) {
        return (deg + 1) * (deg + 2) * (deg + 3) / 6;
    }


    template<typename Float, typename Spectrum>
    class PolyUtils {
    protected:
        ~PolyUtils() = default;

    public:
        MI_IMPORT_TYPES(Scene, Sampler, Texture)

        PolyUtils() {
        };

        static constexpr int nChooseK(int n, int k) {
            return (k == 0 || n == k) ? 1 : nChooseK(n - 1, k - 1) + nChooseK(n - 1, k);
        }

        static constexpr int nPolyCoeffs(int polyOrder) {
            return nChooseK(3 + polyOrder, polyOrder);
        }

        static constexpr size_t powerToIndex(size_t dx, size_t dy, size_t dz) {
            // Converts a polynomial degree to a linear coefficient index
            auto d = dx + dy + dz;
            auto i = d - dx;
            auto j = d - dx - dy;
            return i * (i + 1) / 2 + j + d * (d + 1) * (d + 2) / 6;
        }

        static int multinomial(int a, int b, int c) {
            int res = 1;
            int i = 1;
            int denom = 1;
            for (int j = 1; j <= a; ++j) {
                res *= i;
                denom *= j;
                i++;
            }
            for (int j = 1; j <= b; ++j) {
                res *= i;
                denom *= j;
                i++;
            }
            for (int j = 1; j <= c; ++j) {
                res *= i;
                denom *= j;
                i++;
            }
            return res / denom;
        }

        static inline float powi(float f, int n) {
            float ret = 1.0f;
            for (int i = 0; i < n; ++i) {
                ret *= f;
            }
            return ret;
        }


        static Eigen::VectorXi derivPermutationEigen(size_t degree, size_t axis) {
            auto numCoeffs = numPolynomialCoefficients(degree);
            Eigen::VectorXi permutation = Eigen::VectorXi::Constant(numCoeffs, -1);
            for (size_t d = 0; d <= degree; ++d) {
                for (size_t i = 0; i <= d; ++i) {
                    size_t dx = d - i;
                    for (size_t j = 0; j <= i; ++j) {
                        size_t dy = d - dx - j;
                        size_t dz = d - dx - dy;
                        Vector3i deg(dx, dy, dz);
                        Vector3i derivDeg = deg;
                        derivDeg[axis] -= 1;
                        if (derivDeg[0] < 0 || derivDeg[1] < 0 || derivDeg[2] < 0) {
                            continue;
                        }
                        // For a valid derivative: add entry to matrix
                        permutation[PolyUtils::powerToIndex(derivDeg[0], derivDeg[1], derivDeg[2])] =
                                PolyUtils::powerToIndex(dx, dy, dz);
                    }
                }
            }
            return permutation;
        }


        static std::tuple<std::vector<std::vector<Point3f>>, std::vector<std::vector<Vector3f>>, std::vector<std::vector
            <float>>>
        getLocalPoints(const std::vector<Point<Float, 3>>& queryLocations, Float kernelEps, const std::string& kernel,
                       const ConstraintKdTree* kdtree) {
            std::function<Float(Float, Float)> kernelFun;
            kernelFun = gaussianKernel;
            std::vector<std::vector<Point<Float, 3>>> allPositionConstraints;
            std::vector<std::vector<Vector<Float, 3>>> allNormalConstraints;
            std::vector<std::vector<float>> allConstraintWeights;

            for (size_t k = 0; k < queryLocations.size(); ++k) {
                std::vector<Point<Float, 3>> positionConstraints;
                std::vector<Vector<Float, 3>> normalConstraints;
                std::vector<Float> sampleWeights;
                std::tie(positionConstraints, normalConstraints, sampleWeights) = kdtree->getConstraints(
                    queryLocations[k], kernelEps, kernelFun);

                allPositionConstraints.push_back(positionConstraints);
                allNormalConstraints.push_back(normalConstraints);
                allConstraintWeights.push_back(sampleWeights);
            }
            return std::make_tuple(allPositionConstraints, allNormalConstraints, allConstraintWeights);
        }

        template<size_t polyOrder, bool hardSurfaceConstraint = true>
        static void basisFunFitBuildA(const Point3f& pos, const Vector3f& inDir, const std::vector<Point3f>& evalP,
                                      const Eigen::VectorXi& permX, const Eigen::VectorXi& permY,
                                      const Eigen::VectorXi& permZ,
                                      const Eigen::VectorXf* weights,
                                      Eigen::Matrix<float, Eigen::Dynamic,
                                          nPolyCoeffs(polyOrder, hardSurfaceConstraint)>& A,
                                      const Eigen::VectorXf& weightedB, Float scaleFactor) {
            size_t n = evalP.size();
            Eigen::Matrix<float, Eigen::Dynamic, 3 * (polyOrder + 1)> relPosPow(n, 3 * (polyOrder + 1));
            for (size_t i = 0; i < n; ++i) {
                Vector rel = (evalP[i] - pos) * scaleFactor;
                for (size_t d = 0; d <= polyOrder; ++d) {
                    relPosPow(i, d * 3 + 0) = PolyUtils::powi(rel.x, d);
                    relPosPow(i, d * 3 + 1) = PolyUtils::powi(rel.y, d);
                    relPosPow(i, d * 3 + 2) = PolyUtils::powi(rel.z, d);
                }
            }

            constexpr int nCoeffs = nPolyCoeffs(polyOrder, false);
            Eigen::Matrix<float, Eigen::Dynamic, nCoeffs> fullA =
                    Eigen::Matrix<float, Eigen::Dynamic, nCoeffs>::Zero(n * 4, nCoeffs);
            size_t termIdx = 0;
            for (size_t d = 0; d <= polyOrder; ++d) {
                for (size_t i = 0; i <= d; ++i) {
                    size_t dx = d - i;
                    for (size_t j = 0; j <= i; ++j) {
                        size_t dy = d - dx - j;
                        size_t dz = d - dx - dy;
                        Eigen::VectorXf col = relPosPow.col(0 + 3 * dx).array() * relPosPow.col(1 + 3 * dy).array() *
                                              relPosPow.col(2 + 3 * dz).array();
                        if (weights) {
                            col = col.array() * (*weights).array();
                        }
                        fullA.block(0, termIdx, n, 1) = col;
                        const int pX = permX[termIdx];
                        const int pY = permY[termIdx];
                        const int pZ = permZ[termIdx];
                        if (pX > 0) {
                            fullA.block(n, pX, n, 1) = (dx + 1) * col;
                        }
                        if (pY > 0) {
                            fullA.block(2 * n, pY, n, 1) = (dy + 1) * col;
                        }
                        if (pZ > 0) {
                            fullA.block(3 * n, pZ, n, 1) = (dz + 1) * col;
                        }
                        ++termIdx;
                    }
                }
            }
            A = fullA.block(0, 1, fullA.rows(), fullA.cols() - 1);
        }


        /*template<size_t polyOrder, bool hardSurfaceConstraint = true>
        std::tuple<Polynomial<Float>,
            std::vector<Point3f>,
            std::vector<Vector3f>>
        fitPolynomialsImpl(const PolyFitRecord<Float, Spectrum>& pfRec, const ConstraintKdTree* kdtree) {
            float kernelEps = pfRec.kernelEps;
            std::function<Float(Float, Float)> kernelFun = gaussianKernel;

            std::vector<Point3f> positionConstraints;
            std::vector<Vector3f> normalConstraints;
            std::vector<Float> sampleWeights;
            std::tie(positionConstraints, normalConstraints, sampleWeights) = kdtree->getConstraints(
                pfRec.p, kernelEps, kernelFun);

            size_t n = positionConstraints.size();
            float invSqrtN = 1.0f / std::sqrt(n);
            Eigen::VectorXf weights(n);

            Vector3f s, t;
            Volpath3D<Float, Spectrum>::onbDuff(pfRec.d, s, t);
            Frame local(s, t, pfRec.d);
            bool useLightSpace = pfRec.config.useLightspace;
            for (size_t i = 0; i < n; ++i) {
                Float d2 = distanceSquared(pfRec.p, positionConstraints[i]);
                Float w;
                if (sampleWeights[i] < 0) { // Special constraint
                    // w = std::sqrt(kernelFun(d2, kernelEps * 4.0) * 1.0f) * 4.0f;
                    w = pfRec.config.globalConstraintWeight * std::sqrt(1.0f / 32.0f);
                } else {
                    w = std::sqrt(kernelFun(d2, kernelEps) * sampleWeights[i]) * invSqrtN;
                }
                weights[i] = w;
                if (useLightSpace) {
                    auto localPos = local.toLocal(positionConstraints[i] - pfRec.p);
                    positionConstraints[i] = Point(localPos.x, localPos.y, localPos.z);
                }
            }
            Eigen::VectorXf weightedB(4 * n);
            for (size_t i = 0; i < n; ++i) {
                Vector normal = normalConstraints[i];
                if (useLightSpace) {
                    normal = local.toLocal(normal);
                }
                weightedB[i + 0 * n] = 0.0f;
                weightedB[i + 1 * n] = normal.x * weights[i];
                weightedB[i + 2 * n] = normal.y * weights[i];
                weightedB[i + 3 * n] = normal.z * weights[i];
            }

            // Evaluate derivatives
            Eigen::VectorXi pX = derivPermutationEigen(polyOrder, 0);
            Eigen::VectorXi pY = derivPermutationEigen(polyOrder, 1);
            Eigen::VectorXi pZ = derivPermutationEigen(polyOrder, 2);

            constexpr size_t nCoeffs = nPolyCoeffs(polyOrder, hardSurfaceConstraint);
            Eigen::Matrix<float, Eigen::Dynamic, nCoeffs> A(4 * n, nCoeffs);
            Eigen::Matrix<float, nCoeffs, nCoeffs> AtA(nCoeffs, nCoeffs);

            // This scale factor seems to lead to a well behaved fit in many different settings
            float fitScaleFactor = PolyUtils::getFitScaleFactor(kernelEps);
            Vector usedRefDir = useLightSpace ? local.toLocal(pfRec.n) : pfRec.n;

            if (useLightSpace) {
                basisFunFitBuildA<polyOrder, hardSurfaceConstraint>(Point3f(0.0f), usedRefDir, positionConstraints,
                                                                    pX, pY, pZ, &weights, A, weightedB, fitScaleFactor);
            } else {
                basisFunFitBuildA<polyOrder, hardSurfaceConstraint>(pfRec.p, usedRefDir, positionConstraints, pX, pY,
                                                                    pZ, &weights, A, weightedB, fitScaleFactor);
            }
            Eigen::Matrix<float, nCoeffs, 1> Atb = A.transpose() * weightedB;

            Eigen::VectorXf coeffs;
            if (pfRec.config.useSvd) {
                Eigen::MatrixXf ADyn = A;
                Eigen::BDCSVD<Eigen::MatrixXf> svd = ADyn.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
                const Eigen::VectorXf& sVal = svd.singularValues();
                float eps = 0.01f;
                coeffs = svd.matrixV() * ((sVal.array() > eps).select(sVal.array().inverse(), 0)).matrix().asDiagonal()
                         * svd.matrixU().transpose() * weightedB;
            } else {
                Eigen::MatrixXf reg = Eigen::MatrixXf::Identity(A.cols(), A.cols()) * pfRec.config.regularization;
                reg(0, 0) = 0.0f;
                reg(1, 1) = 0.0f;
                reg(2, 2) = 0.0f;
                AtA = A.transpose() * A + reg;
                coeffs = AtA.ldlt().solve(Atb);
                // coeffs = A.householderQr().solve(weightedB);
            }

            std::vector<Float> coeffsVec(numPolynomialCoefficients(polyOrder));
            if (hardSurfaceConstraint) {
                coeffsVec[0] = 0.0f;
                for (size_t i = 1; i < coeffs.size(); ++i)
                    coeffsVec[i] = coeffs[i - 1];
            } else {
                for (size_t i = 0; i < coeffs.size(); ++i)
                    coeffsVec[i] = coeffs[i];
            }

            Polynomial<Float> poly;
            poly.coeffs = coeffsVec;
            poly.refPos = pfRec.p;
            poly.refDir = pfRec.d;
            poly.useLocalDir = pfRec.config.useLightspace;
            poly.scaleFactor = PolyUtils::getFitScaleFactor(kernelEps);
            poly.order = polyOrder;
            return std::make_tuple(poly, positionConstraints, normalConstraints);
        }*/

        /*
        static std::tuple<Polynomial<Float>, std::vector<Point3f>, std::vector<Vector3f>>
        fitPolynomial(const PolyFitRecord<Float, Spectrum>& pfRec, const ConstraintKdTree* kdtree) {
           if (pfRec.config.hardSurfaceConstraint) {
                if (pfRec.config.order == 2)
                    return fitPolynomialsImpl<2, true>(pfRec, kdtree);
                else
                    return fitPolynomialsImpl<3, true>(pfRec, kdtree);
            } else {
                std::cout << "UNSUPPORTED: hardSurfaceConstraint = false\n";
                return std::make_tuple(Polynomial<Float>(), std::vector<Point3f>(), std::vector<Vector3f>());
            }
        }
        */

        static Vector3f evalGradient(const Point3f& pos,
                                     const Eigen::VectorX<Float>& coeffs,
                                     const ScatterSamplingRecord<Float, Spectrum>& sRec, size_t degree,
                                     Float scaleFactor,
                                     bool useLocalDir,
                                     const Vector3f& refDir) {
            const int* permX = derivPermutation(degree, 0);
            const int* permY = derivPermutation(degree, 1);
            const int* permZ = derivPermutation(degree, 2);
            Float polyValue;
            Vector3f gradient;
            std::tie(polyValue, gradient) = evalPolyGrad(pos, sRec.p, degree, permX, permY, permZ, scaleFactor,
                                                         useLocalDir, refDir, coeffs);
            if (useLocalDir) {
                Vector3f s, t;
                Volpath3D<Float, Spectrum>::onbDuff(refDir, s, t);
                Frame3f local(s, t, refDir);
                gradient = local.to_world(gradient);
            }
            return gradient;
        }

        static void projectPointsToSurface(const Scene* scene, const Point3f& refPoint, const Vector<Float, 3>& refDir,
                                           ScatterSamplingRecord<Float, Spectrum>& sRec,
                                           const Eigen::VectorX<Float>& polyCoefficients,
                                           size_t polyOrder, bool useLocalDir, Float scaleFactor, Float kernelEps) {
            if (!sRec.isValid)
                return;

            Vector3f dir = evalGradient(refPoint, polyCoefficients, sRec, polyOrder, scaleFactor, useLocalDir, refDir);
            dir = normalize(dir);
            // Float dists[5] = {0.0, kernelEps, 2 * kernelEps, 3 * kernelEps, std::numeric_limits<Float>::infinity()};
            // Float dists[3] = {0.0, kernelEps, std::numeric_limits<Float>::infinity()};
            Float dists[2] = {2 * kernelEps, std::numeric_limits<Float>::infinity()};
            for (int i = 0; i < 2; ++i) {
                Float maxProjDist = dists[i];
                Ray3f ray1 = Ray3f(sRec.p, dir, maxProjDist);
                SurfaceInteraction3f its;
                Point<Float, 3> projectedP;
                Vector<Float, 3> normal;
                Float pointDist = -1;
                bool itsFoundCurrent = false;
                Mask active = true;
                its = scene->ray_intersect(ray1, active);
                if (dr::any_or<true>(its.is_valid())) {
                    projectedP = its.p;
                    normal = its.sh_frame.n;
                    pointDist = its.t;
                    itsFoundCurrent = true;
                }
                Float maxT = itsFoundCurrent ? its.t : maxProjDist;
                Ray3f ray2 = Ray3f(sRec.p, -dir, -maxT);
                SurfaceInteraction3f its2 = scene->ray_intersect(ray2, true);
                if (drjit::any_or<true>(its2.is_valid())) {
                    if (drjit::any_or<true>(pointDist < 0 || pointDist > its2.t)) {
                        projectedP = its2.p;
                        normal = its2.sh_frame.n;
                    }
                    itsFoundCurrent = true;
                }
                sRec.isValid = itsFoundCurrent;
                if (itsFoundCurrent) {
                    sRec.p = projectedP;
                    sRec.n = normal;
                }
                if (itsFoundCurrent)
                    return;
            }
        }

        static const int* derivPermutation(size_t degree, size_t axis) {
            switch (axis) {
                case 0:
                    return degree == 2 ? PERMX2 : PERMX3;
                case 1:
                    return degree == 2 ? PERMY2 : PERMY3;
                case 2:
                    return degree == 2 ? PERMZ2 : PERMZ3;
                default:
                    return nullptr;
            }
        }

        static Float getKernelEps(const MediumParameters<Float, Spectrum>& medium, int channel = 0,
                                  Float kernel_multiplier = 1.0f) {
            Float sigmaT = medium.sigmaT[channel];
            Float albedo = medium.albedo[channel];
            Float g = medium.g;
            Float sigmaS = albedo * sigmaT;
            Float sigmaa = sigmaT - sigmaS;
            Float sigmaSp = (1 - g) * sigmaS;
            Float sigmaTp = sigmaSp + sigmaa;
            Float alphaP = sigmaSp / sigmaTp;
            Float effAlphaP = Volpath3D<Float, Spectrum>::effectiveAlbedo(alphaP);
            Float val = 0.25f * g + 0.25 * alphaP + 1.0 * effAlphaP;
            // Float d         = alphaP - 0.5f;
            // Float val = 0.474065f * alphaP + 0.578414f * g + 0.000028448817f * std::exp(d * d / 0.025f); //
            // Submission version
            return kernel_multiplier * 4.0f * val * val / (sigmaTp * sigmaTp);
        }

        template<size_t degree>
        static float evalPolyImpl(const Point3f& pos,
                                  const Point3f& evalP, Float scaleFactor, bool useLocalDir, const Vector3f& refDir,
                                  const std::vector<float>& coeffs) {
            assert(degree <= 3);

            Vector3f relPos;
            if (useLocalDir) {
                Vector3f s, t;
                Volpath3D<Float, Spectrum>::onbDuff(refDir, s, t);
                Frame local(s, t, refDir);
                relPos = local.toLocal(evalP - pos) * scaleFactor;
            } else {
                relPos = (evalP - pos) * scaleFactor;
            }

            size_t termIdx = 4;
            float value = coeffs[0] + coeffs[1] * relPos.x + coeffs[2] * relPos.y + coeffs[3] * relPos.z;
            float xPowers[4] = {1.0, relPos.x, relPos.x * relPos.x, relPos.x * relPos.x * relPos.x};
            float yPowers[4] = {1.0, relPos.y, relPos.y * relPos.y, relPos.y * relPos.y * relPos.y};
            float zPowers[4] = {1.0, relPos.z, relPos.z * relPos.z, relPos.z * relPos.z * relPos.z};

            for (size_t d = 2; d <= degree; ++d) {
                for (size_t i = 0; i <= d; ++i) {
                    size_t dx = d - i;
                    for (size_t j = 0; j <= i; ++j) {
                        size_t dy = d - dx - j;
                        size_t dz = d - dx - dy;
                        value += coeffs[termIdx] * xPowers[dx] * yPowers[dy] * zPowers[dz];
                        ++termIdx;
                    }
                }
            }
            return value;
        }


        static float evalPoly(const Point3f& pos,
                              const Point3f& evalP, size_t degree,
                              Float scaleFactor, bool useLocalDir, const Vector3f& refDir,
                              const std::vector<float>& coeffs) {
            assert(degree <= 3 && degree >= 2);
            if (degree == 3)
                return evalPolyImpl<3>(pos, evalP, scaleFactor, useLocalDir, refDir, coeffs);
            else
                return evalPolyImpl<2>(pos, evalP, scaleFactor, useLocalDir, refDir, coeffs);
        }

        template<typename T>
        static std::pair<Float, Vector3f> evalPolyGrad(const Point3f& pos,
                                                       const Point3f& evalP, size_t degree,
                                                       const int* permX, const int* permY, const int* permZ,
                                                       Float scaleFactor, bool useLocalDir, const Vector3f& refDir,
                                                       const T& coeffs) {
            Vector3f relPos;
            if (useLocalDir) {
                Vector3f s, t;
                Volpath3D<Float, Spectrum>::onbDuff(refDir, s, t);
                Frame3f local(s, t, refDir);
                relPos = local.to_local(evalP - pos) * scaleFactor;
            } else {
                relPos = (evalP - pos) * scaleFactor;
            }

            size_t termIdx = 0;
            Vector3f deriv(0.0f, 0.0f, 0.0f);
            Float value = 0.0f;

            Float xPowers[4] = {1.0, relPos.x(), relPos.x() * relPos.x(), relPos.x() * relPos.x() * relPos.x()};
            Float yPowers[4] = {1.0, relPos.y(), relPos.y() * relPos.y(), relPos.y() * relPos.y() * relPos.y()};
            Float zPowers[4] = {1.0, relPos.z(), relPos.z() * relPos.z(), relPos.z() * relPos.z() * relPos.z()};
            for (size_t d = 0; d <= degree; ++d) {
                for (size_t i = 0; i <= d; ++i) {
                    size_t dx = d - i;
                    for (size_t j = 0; j <= i; ++j) {
                        size_t dy = d - dx - j;
                        size_t dz = d - dx - dy;
                        Float t = xPowers[dx] * yPowers[dy] * zPowers[dz];
                        value += coeffs[termIdx] * t;

                        int pX = permX[termIdx];
                        int pY = permY[termIdx];
                        int pZ = permZ[termIdx];
                        if (pX > 0)
                            deriv[0] += (dx + 1) * t * coeffs[pX];
                        if (pY > 0)
                            deriv[1] += (dy + 1) * t * coeffs[pY];
                        if (pZ > 0)
                            deriv[2] += (dz + 1) * t * coeffs[pZ];
                        ++termIdx;
                    }
                }
            }
            return std::make_pair(value, deriv);
        }

        static Float getFitScaleFactor(const MediumParameters<Float, Spectrum>& medium, int channel = 0,
                                       Float kernel_multiplier = 1.0f) {
            return getFitScaleFactor(getKernelEps(medium, channel, kernel_multiplier));
        }

        static Float getFitScaleFactor(Float kernelEps) {
            return 1.0f / drjit::sqrt(kernelEps);
        }

        static inline Float gaussianKernel(Float dist2, Float sigma2) {
            return drjit::exp(-dist2 / (2 * sigma2)); // Dont do any normalization since number of constraint points is
            // adjusted based on sigma2
        }

        template<size_t PolyOrder = 3>
        static void legendreTransform(Eigen::Matrix<Float, nPolyCoeffs(PolyOrder), 1>& coeffs) {
            if (PolyOrder == 2) {
                float t[4] = {2.8284271247461903, 0.9428090415820634, 0.9428090415820634, 0.9428090415820634};
                int indices[4] = {0, 4, 7, 9};

                float t2[9] = {
                    1.632993161855452, 1.632993161855452, 1.632993161855452,
                    0.8432740427115678, 0.9428090415820634, 0.9428090415820634,
                    0.8432740427115678, 0.9428090415820634, 0.8432740427115678
                };
                Float val = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    val += coeffs[indices[i]] * t[i];
                }
                coeffs[0] = val;
                for (int i = 1; i < coeffs.size(); ++i) {
                    coeffs[i] = t2[i - 1] * coeffs[i];
                }
            } else if (PolyOrder == 3) {
                float t[4][4] = {
                    {2.8284271247461903, 0.9428090415820634, 0.9428090415820634, 0.9428090415820634},
                    {1.632993161855452, 0.9797958971132712, 0.5443310539518174, 0.5443310539518174},
                    {1.632993161855452, 0.5443310539518174, 0.9797958971132712, 0.5443310539518174},
                    {1.632993161855452, 0.5443310539518174, 0.5443310539518174, 0.9797958971132712}
                };
                int indices[4][4] = {{0, 4, 7, 9}, {1, 10, 13, 15}, {2, 11, 16, 18}, {3, 12, 17, 19}};
                float t2[16] = {
                    0.8432740427115678, 0.9428090415820634, 0.9428090415820634, 0.8432740427115678,
                    0.9428090415820634, 0.8432740427115678, 0.427617987059879, 0.48686449556014766,
                    0.48686449556014766, 0.48686449556014766, 0.5443310539518174, 0.48686449556014766,
                    0.427617987059879, 0.48686449556014766, 0.48686449556014766, 0.427617987059879
                };

                for (int i = 0; i < 4; ++i) {
                    Float val = 0.0f;
                    for (int j = 0; j < 4; ++j) {
                        val += coeffs[indices[i][j]] * t[i][j];
                    }
                    coeffs[i] = val;
                }
                for (int i = 4; i < coeffs.size(); ++i) {
                    coeffs[i] = t2[i - 4] * coeffs[i];
                }
            } else {
                std::cout << "CANNOT HANDLE POLYNOMIALS OF ORDER != 2 or 3!!\n";
            }
        }

        template<int order>
        static void rotatePolynomial(std::vector<float>& c,
                                     std::vector<float>& c2,
                                     const Vector<Float, 3>& s,
                                     const Vector<Float, 3>& t,
                                     const Vector<Float, 3>& n) {
            c2[0] = c[0];
            c2[1] = c[1] * s[0] + c[2] * s[1] + c[3] * s[2];
            c2[2] = c[1] * t[0] + c[2] * t[1] + c[3] * t[2];
            c2[3] = c[1] * n[0] + c[2] * n[1] + c[3] * n[2];
            c2[4] = c[4] * drjit::pow(s[0], 2) + c[5] * s[0] * s[1] + c[6] * s[0] * s[2] + c[7] * drjit::pow(s[1], 2) +
                    c[8] * s[1]
                    * s[2] + c[9] * drjit::pow(s[2], 2);
            c2[5] = 2 * c[4] * s[0] * t[0] + c[5] * (s[0] * t[1] + s[1] * t[0]) + c[6] * (s[0] * t[2] + s[2] * t[0]) + 2
                    * c[7] * s[1] * t[1] + c[8] * (s[1] * t[2] + s[2] * t[1]) + 2 * c[9] * s[2] * t[2];
            c2[6] = 2 * c[4] * n[0] * s[0] + c[5] * (n[0] * s[1] + n[1] * s[0]) + c[6] * (n[0] * s[2] + n[2] * s[0]) + 2
                    * c[7] * n[1] * s[1] + c[8] * (n[1] * s[2] + n[2] * s[1]) + 2 * c[9] * n[2] * s[2];
            c2[7] = c[4] * drjit::pow(t[0], 2) + c[5] * t[0] * t[1] + c[6] * t[0] * t[2] + c[7] * drjit::pow(t[1], 2) +
                    c[8] * t[1]
                    * t[2] + c[9] * drjit::pow(t[2], 2);
            c2[8] = 2 * c[4] * n[0] * t[0] + c[5] * (n[0] * t[1] + n[1] * t[0]) + c[6] * (n[0] * t[2] + n[2] * t[0]) + 2
                    * c[7] * n[1] * t[1] + c[8] * (n[1] * t[2] + n[2] * t[1]) + 2 * c[9] * n[2] * t[2];
            c2[9] = c[4] * drjit::pow(n[0], 2) + c[5] * n[0] * n[1] + c[6] * n[0] * n[2] + c[7] * drjit::pow(n[1], 2) +
                    c[8] * n[1]
                    * n[2] + c[9] * drjit::pow(n[2], 2);
            if (order > 2) {
                c2[10] = c[10] * drjit::pow(s[0], 3) + c[11] * drjit::pow(s[0], 2) * s[1] + c[12] * drjit::pow(s[0], 2)
                         * s[2] + c[13] * s
                         [0] * drjit::pow(s[1], 2) + c[14] * s[0] * s[1] * s[2] + c[15] * s[0] * drjit::pow(s[2], 2) + c
                         [16] *
                         drjit::pow(s[1], 3) + c[17] * drjit::pow(s[1], 2) * s[2] + c[18] * s[1] * drjit::pow(s[2], 2) +
                         c[19] * drjit::pow(
                             s[2], 3);
                c2[11] = 3 * c[10] * drjit::pow(s[0], 2) * t[0] + c[11] * (
                             drjit::pow(s[0], 2) * t[1] + 2 * s[0] * s[1] * t[0]) + c[
                             12] * (drjit::pow(s[0], 2) * t[2] + 2 * s[0] * s[2] * t[0]) + c[13] * (
                             2 * s[0] * s[1] * t[1] + drjit::pow(s[1], 2) * t[0]) + c[14] * (
                             s[0] * s[1] * t[2] + s[0] * s[2] * t[1] + s[1] * s[2] * t[0]) + c[15] * (
                             2 * s[0] * s[2] * t[2] + drjit::pow(s[2], 2) * t[0]) + 3 * c[16] *
                         drjit::pow(s[1], 2) * t[1] + c[17] * (drjit::pow(s[1], 2) * t[2] + 2 * s[1] * s[2] * t[1]) + c[
                             18] * (
                             2 * s[1] * s[2] * t[2] + drjit::pow(s[2], 2) * t[1]) + 3 * c[19] * drjit::pow(s[2], 2) * t[
                             2];
                c2[12] = 3 * c[10] * n[0] * drjit::pow(s[0], 2) + c[11] * (
                             2 * n[0] * s[0] * s[1] + n[1] * drjit::pow(s[0], 2)) + c[
                             12] * (2 * n[0] * s[0] * s[2] + n[2] * drjit::pow(s[0], 2)) + c[13] * (
                             n[0] * drjit::pow(s[1], 2) + 2 * n[1] * s[0] * s[1]) + c[14] * (
                             n[0] * s[1] * s[2] + n[1] * s[0] * s[2] + n[2] * s[0] * s[1]) + c[15] * (
                             n[0] * drjit::pow(s[2], 2) + 2 * n[2] * s[0] * s[2]) + 3 * c[16] * n[
                             1] * drjit::pow(s[1], 2) + c[17] * (2 * n[1] * s[1] * s[2] + n[2] * drjit::pow(s[1], 2)) +
                         c[18] * (
                             n[1] * drjit::pow(s[2], 2) + 2 * n[2] * s[1] * s[2]) + 3 * c[19] * n[2] * drjit::pow(
                             s[2], 2);
                c2[13] = 3 * c[10] * s[0] * drjit::pow(t[0], 2) + c[11] * (
                             2 * s[0] * t[0] * t[1] + s[1] * drjit::pow(t[0], 2)) + c[
                             12] * (2 * s[0] * t[0] * t[2] + s[2] * drjit::pow(t[0], 2)) + c[13] * (
                             s[0] * drjit::pow(t[1], 2) + 2 * s[1] * t[0] * t[1]) + c[14] * (
                             s[0] * t[1] * t[2] + s[1] * t[0] * t[2] + s[2] * t[0] * t[1]) + c[15] * (
                             s[0] * drjit::pow(t[2], 2) + 2 * s[2] * t[0] * t[2]) + 3 * c[16] * s[
                             1] * drjit::pow(t[1], 2) + c[17] * (2 * s[1] * t[1] * t[2] + s[2] * drjit::pow(t[1], 2)) +
                         c[18] * (
                             s[1] * drjit::pow(t[2], 2) + 2 * s[2] * t[1] * t[2]) + 3 * c[19] * s[2] * drjit::pow(
                             t[2], 2);
                c2[14] = 6 * c[10] * n[0] * s[0] * t[0] + c[11] * (
                             2 * n[0] * s[0] * t[1] + 2 * n[0] * s[1] * t[0] + 2 * n[1] * s[0] * t[0]) + c[12] * (
                             2 * n[0] * s[0] * t[2] + 2 * n[0] * s[2] * t[0] + 2 * n[2] * s[0] * t[0]) + c[13] * (
                             2 * n[0] * s[1] * t[1] + 2 * n[1] * s[0] * t[1] + 2 * n[1] * s[1] * t[0]) + c[14] * (
                             n[0] * s[1] * t[2] + n[0] * s[2] * t[1] + n[1] * s[0] * t[2] + n[1] * s[2] * t[0] + n[2] *
                             s[0] * t[1] + n[2] * s[1] * t[0]) + c[15] * (
                             2 * n[0] * s[2] * t[2] + 2 * n[2] * s[0] * t[2] + 2 * n[2] * s[2] * t[0]) + 6 * c[16] * n[
                             1] * s[1] * t[1] + c[17] * (
                             2 * n[1] * s[1] * t[2] + 2 * n[1] * s[2] * t[1] + 2 * n[2] * s[1] * t[1]) + c[18] * (
                             2 * n[1] * s[2] * t[2] + 2 * n[2] * s[1] * t[2] + 2 * n[2] * s[2] * t[1]) + 6 * c[19] * n[
                             2] * s[2] * t[2];
                c2[15] = 3 * c[10] * drjit::pow(n[0], 2) * s[0] + c[11] * (
                             drjit::pow(n[0], 2) * s[1] + 2 * n[0] * n[1] * s[0]) + c[
                             12] * (drjit::pow(n[0], 2) * s[2] + 2 * n[0] * n[2] * s[0]) + c[13] * (
                             2 * n[0] * n[1] * s[1] + drjit::pow(n[1], 2) * s[0]) + c[14] * (
                             n[0] * n[1] * s[2] + n[0] * n[2] * s[1] + n[1] * n[2] * s[0]) + c[15] * (
                             2 * n[0] * n[2] * s[2] + drjit::pow(n[2], 2) * s[0]) + 3 * c[16] *
                         drjit::pow(n[1], 2) * s[1] + c[17] * (drjit::pow(n[1], 2) * s[2] + 2 * n[1] * n[2] * s[1]) + c[
                             18] * (
                             2 * n[1] * n[2] * s[2] + drjit::pow(n[2], 2) * s[1]) + 3 * c[19] * drjit::pow(n[2], 2) * s[
                             2];
                c2[16] = c[10] * drjit::pow(t[0], 3) + c[11] * drjit::pow(t[0], 2) * t[1] + c[12] * drjit::pow(t[0], 2)
                         * t[2] + c[13] * t
                         [0] * drjit::pow(t[1], 2) + c[14] * t[0] * t[1] * t[2] + c[15] * t[0] * drjit::pow(t[2], 2) + c
                         [16] *
                         drjit::pow(t[1], 3) + c[17] * drjit::pow(t[1], 2) * t[2] + c[18] * t[1] * drjit::pow(t[2], 2) +
                         c[19] * drjit::pow(
                             t[2], 3);
                c2[17] = 3 * c[10] * n[0] * drjit::pow(t[0], 2) + c[11] * (
                             2 * n[0] * t[0] * t[1] + n[1] * drjit::pow(t[0], 2)) + c[
                             12] * (2 * n[0] * t[0] * t[2] + n[2] * drjit::pow(t[0], 2)) + c[13] * (
                             n[0] * drjit::pow(t[1], 2) + 2 * n[1] * t[0] * t[1]) + c[14] * (
                             n[0] * t[1] * t[2] + n[1] * t[0] * t[2] + n[2] * t[0] * t[1]) + c[15] * (
                             n[0] * drjit::pow(t[2], 2) + 2 * n[2] * t[0] * t[2]) + 3 * c[16] * n[
                             1] * drjit::pow(t[1], 2) + c[17] * (2 * n[1] * t[1] * t[2] + n[2] * drjit::pow(t[1], 2)) +
                         c[18] * (
                             n[1] * drjit::pow(t[2], 2) + 2 * n[2] * t[1] * t[2]) + 3 * c[19] * n[2] * drjit::pow(
                             t[2], 2);
                c2[18] = 3 * c[10] * drjit::pow(n[0], 2) * t[0] + c[11] * (
                             drjit::pow(n[0], 2) * t[1] + 2 * n[0] * n[1] * t[0]) + c[
                             12] * (drjit::pow(n[0], 2) * t[2] + 2 * n[0] * n[2] * t[0]) + c[13] * (
                             2 * n[0] * n[1] * t[1] + drjit::pow(n[1], 2) * t[0]) + c[14] * (
                             n[0] * n[1] * t[2] + n[0] * n[2] * t[1] + n[1] * n[2] * t[0]) + c[15] * (
                             2 * n[0] * n[2] * t[2] + drjit::pow(n[2], 2) * t[0]) + 3 * c[16] *
                         drjit::pow(n[1], 2) * t[1] + c[17] * (drjit::pow(n[1], 2) * t[2] + 2 * n[1] * n[2] * t[1]) + c[
                             18] * (
                             2 * n[1] * n[2] * t[2] + drjit::pow(n[2], 2) * t[1]) + 3 * c[19] * drjit::pow(n[2], 2) * t[
                             2];
                c2[19] = c[10] * drjit::pow(n[0], 3) + c[11] * drjit::pow(n[0], 2) * n[1] + c[12] * drjit::pow(n[0], 2)
                         * n[2] + c[13] * n
                         [0] * drjit::pow(n[1], 2) + c[14] * n[0] * n[1] * n[2] + c[15] * n[0] * drjit::pow(n[2], 2) + c
                         [16] *
                         drjit::pow(n[1], 3) + c[17] * drjit::pow(n[1], 2) * n[2] + c[18] * n[1] * drjit::pow(n[2], 2) +
                         c[19] * drjit::pow(
                             n[2], 3);
            }
            for (size_t i = 0; i < c2.size(); ++i) {
                c[i] = c2[i];
            }
        }

        static Vector3f adjustRayDirForPolynomialTracing(Vector3f& inDir,
                                                         const SurfaceInteraction3f& its, int polyOrder,
                                                         Float polyScaleFactor, int channel = 0) {
            const int* pX = derivPermutation(polyOrder, 0);
            const int* pY = derivPermutation(polyOrder, 1);
            const int* pZ = derivPermutation(polyOrder, 2);

            Float polyValue;
            Vector3f polyNormal;
            std::tie(polyValue, polyNormal) = evalPolyGrad(its.p, its.p, polyOrder, pX, pY, pZ,
                                                           polyScaleFactor, false,
                                                           inDir, its.polyCoeffs[channel]);
            polyNormal = drjit::normalize(polyNormal);
            Vector3f rotationAxis = drjit::cross(its.sh_frame.n, polyNormal);
            /*if (rotationAxis.length() < 1e-8f) { // If the normal and the polynomial normal are parallel
                return polyNormal;
            }*/
            Vector3f normalizedTarget = drjit::normalize(its.sh_frame.n);
            Float angle = acos(drjit::maximum(drjit::minimum(drjit::dot(polyNormal, normalizedTarget), 1.0f), -1.0f));
            Transform4f transf = Transform4f::rotate(rotationAxis, drjit::rad_to_deg(angle));
            // Not sure if transform4f is the right one
            inDir = transf * inDir;
            return polyNormal;
        }

        /*static Vector3f adjustRayForPolynomialTracing(Ray<Float, Spectrum>& ray, const Polynomial<Float>& polynomial,
                                                      const Vector<Float, 3>& targetNormal) {
            // Transforms a given ray such that it always comes from the upper hemisphere wrt. to the polynomial normal
            const int* pX = derivPermutation(polynomial.order, 0);
            const int* pY = derivPermutation(polynomial.order, 1);
            const int* pZ = derivPermutation(polynomial.order, 2);
            float polyValue;
            Vector3f polyNormal;
            assert(pX && pY && pZ);
            assert(polynomial.coeffs.size() > 0);

            std::tie(polyValue, polyNormal) = evalPolyGrad(polynomial.refPos, ray.o, polynomial.order, pX, pY, pZ,
                                                           polynomial.scaleFactor, polynomial.useLocalDir,
                                                           polynomial.refDir, polynomial.coeffs);
            polyNormal = normalize(polyNormal);
            Vector rotationAxis = cross(targetNormal, polyNormal);
            if (rotationAxis.length() < 1e-8f)
                return polyNormal;
            Vector normalizedTarget = normalize(targetNormal);
            float angle = acos(std::max(std::min(dot(polyNormal, normalizedTarget), 1.0f), -1.0f));
            Transform3f transf = Transform3f::rotate(rotationAxis, drjit::rad_to_deg(angle));
            ray.d = transf(ray.d);
            return polyNormal;
        }
        */

        /*
        static bool adjustRayForPolynomialTracingFull(Ray<Float, Spectrum>& ray, const Polynomial<Float>& polynomial,
                                                      const Vector3f& targetNormal) {
            const int* pX = derivPermutation(polynomial.order, 0);
            const int* pY = derivPermutation(polynomial.order, 1);
            const int* pZ = derivPermutation(polynomial.order, 2);

            float polyValue;
            Vector3f polyGradient;
            std::tie(polyValue, polyGradient) = evalPolyGrad(polynomial.refPos, ray.o, polynomial.order, pX, pY, pZ,
                                                             polynomial.scaleFactor, polynomial.useLocalDir,
                                                             polynomial.refDir, polynomial.coeffs);
            polyGradient = drjit::normalize(polyGradient);
            Vector3f rayD = polyValue > 0 ? -polyGradient : polyGradient;

            // 1. Trace ray along
            float polyStepSize = 0.1f; // TODO: Set the stepsize based on the bounding box of the object
            Ray projectionRay(ray.o - rayD * 0.5f * polyStepSize, rayD, 0.0f);
            Interaction3f polyIts = intersectPolynomial(projectionRay, polynomial, polyStepSize, false);

            if (!polyIts.isValid()) {
                std::cout << "polyValue: " << polyValue << std::endl;
                std::cout << "polyGradient.toString(): " << polyGradient.toString() << std::endl;
                std::cout << "rayD.toString(): " << rayD.toString() << std::endl;
                return false;
            }
            // ray.o = polyIts.p + (polyValue > 0 ? rayD * ShadowEpsilon : -rayD * ShadowEpsilon); // ray now has a new origin
            ray.o = polyIts.p; // ray now has a new origin

            // 2. Evaluate the normal
            Vector3f polyNormal;
            std::tie(polyValue, polyNormal) = evalPolyGrad(polynomial.refPos, ray.o, polynomial.order, pX, pY, pZ,
                                                           polynomial.scaleFactor, polynomial.useLocalDir,
                                                           polynomial.refDir, polynomial.coeffs);

            polyNormal = drjit::normalize(polyNormal);
            Vector rotationAxis = cross(targetNormal, polyNormal);
            if (rotationAxis.length() < 1e-8f) {
                return true;
            }

            Vector normalizedTarget = normalize(targetNormal);
            float angle = acos(std::max(std::min(dot(polyNormal, normalizedTarget), 1.0f), -1.0f));

            Transform3f transf = Transform3f::rotate(rotationAxis, drjit::rad_to_deg(angle));
            ray.d = transf(ray.d);
            return true;
        }
        */

        template<int order, typename T>
        static Eigen::Matrix<Float, nPolyCoeffs(order), 1> rotatePolynomialEigen(
            const T& c,
            const Vector3f& s,
            const Vector3f& t,
            const Vector3f& n) {
            Eigen::Matrix<Float, nPolyCoeffs(order), 1> c2;
            c2[0] = c[0];
            c2[1] = c[1] * s[0] + c[2] * s[1] + c[3] * s[2];
            c2[2] = c[1] * t[0] + c[2] * t[1] + c[3] * t[2];
            c2[3] = c[1] * n[0] + c[2] * n[1] + c[3] * n[2];
            c2[4] = c[4] * drjit::pow(s[0], 2) + c[5] * s[0] * s[1] + c[6] * s[0] * s[2] + c[7] * drjit::pow(s[1], 2) +
                    c[8] *
                    s[1]
                    * s[2] + c[9] * drjit::pow(s[2], 2);
            c2[5] = 2 * c[4] * s[0] * t[0] + c[5] * (s[0] * t[1] + s[1] * t[0]) + c[6] * (s[0] * t[2] + s[2] * t[0]) + 2
                    * c[7] * s[1] * t[1] + c[8] * (s[1] * t[2] + s[2] * t[1]) + 2 * c[9] * s[2] * t[2];
            c2[6] = 2 * c[4] * n[0] * s[0] + c[5] * (n[0] * s[1] + n[1] * s[0]) + c[6] * (n[0] * s[2] + n[2] * s[0]) + 2
                    * c[7] * n[1] * s[1] + c[8] * (n[1] * s[2] + n[2] * s[1]) + 2 * c[9] * n[2] * s[2];
            c2[7] = c[4] * drjit::pow(t[0], 2) + c[5] * t[0] * t[1] + c[6] * t[0] * t[2] + c[7] * drjit::pow(t[1], 2) +
                    c[8] * t[1]
                    * t[2] + c[9] * drjit::pow(t[2], 2);
            c2[8] = 2 * c[4] * n[0] * t[0] + c[5] * (n[0] * t[1] + n[1] * t[0]) + c[6] * (n[0] * t[2] + n[2] * t[0]) + 2
                    * c[7] * n[1] * t[1] + c[8] * (n[1] * t[2] + n[2] * t[1]) + 2 * c[9] * n[2] * t[2];
            c2[9] = c[4] * drjit::pow(n[0], 2) + c[5] * n[0] * n[1] + c[6] * n[0] * n[2] + c[7] * drjit::pow(n[1], 2) +
                    c[8] * n[1]
                    * n[2] + c[9] * drjit::pow(n[2], 2);
            if (order > 2) {
                c2[10] = c[10] * drjit::pow(s[0], 3) + c[11] * drjit::pow(s[0], 2) * s[1] + c[12] * drjit::pow(s[0], 2)
                         * s[2] + c[13] * s
                         [0] * drjit::pow(s[1], 2) + c[14] * s[0] * s[1] * s[2] + c[15] * s[0] * drjit::pow(s[2], 2) + c
                         [16] *
                         drjit::pow(s[1], 3) + c[17] * drjit::pow(s[1], 2) * s[2] + c[18] * s[1] * drjit::pow(s[2], 2) +
                         c[19] * drjit::pow(
                             s[2], 3);
                c2[11] = 3 * c[10] * drjit::pow(s[0], 2) * t[0] + c[11] * (
                             drjit::pow(s[0], 2) * t[1] + 2 * s[0] * s[1] * t[0]) + c[
                             12] * (drjit::pow(s[0], 2) * t[2] + 2 * s[0] * s[2] * t[0]) + c[13] * (
                             2 * s[0] * s[1] * t[1] + drjit::pow(s[1], 2) * t[0]) + c[14] * (
                             s[0] * s[1] * t[2] + s[0] * s[2] * t[1] + s[1] * s[2] * t[0]) + c[15] * (
                             2 * s[0] * s[2] * t[2] + drjit::pow(s[2], 2) * t[0]) + 3 * c[16] *
                         drjit::pow(s[1], 2) * t[1] + c[17] * (drjit::pow(s[1], 2) * t[2] + 2 * s[1] * s[2] * t[1]) + c[
                             18] * (
                             2 * s[1] * s[2] * t[2] + drjit::pow(s[2], 2) * t[1]) + 3 * c[19] * drjit::pow(s[2], 2) * t[
                             2];
                c2[12] = 3 * c[10] * n[0] * drjit::pow(s[0], 2) + c[11] * (
                             2 * n[0] * s[0] * s[1] + n[1] * drjit::pow(s[0], 2)) + c[
                             12] * (2 * n[0] * s[0] * s[2] + n[2] * drjit::pow(s[0], 2)) + c[13] * (
                             n[0] * drjit::pow(s[1], 2) + 2 * n[1] * s[0] * s[1]) + c[14] * (
                             n[0] * s[1] * s[2] + n[1] * s[0] * s[2] + n[2] * s[0] * s[1]) + c[15] * (
                             n[0] * drjit::pow(s[2], 2) + 2 * n[2] * s[0] * s[2]) + 3 * c[16] * n[
                             1] * drjit::pow(s[1], 2) + c[17] * (2 * n[1] * s[1] * s[2] + n[2] * drjit::pow(s[1], 2)) +
                         c[18] * (
                             n[1] * drjit::pow(s[2], 2) + 2 * n[2] * s[1] * s[2]) + 3 * c[19] * n[2] * drjit::pow(
                             s[2], 2);
                c2[13] = 3 * c[10] * s[0] * drjit::pow(t[0], 2) + c[11] * (
                             2 * s[0] * t[0] * t[1] + s[1] * drjit::pow(t[0], 2)) + c[
                             12] * (2 * s[0] * t[0] * t[2] + s[2] * drjit::pow(t[0], 2)) + c[13] * (
                             s[0] * drjit::pow(t[1], 2) + 2 * s[1] * t[0] * t[1]) + c[14] * (
                             s[0] * t[1] * t[2] + s[1] * t[0] * t[2] + s[2] * t[0] * t[1]) + c[15] * (
                             s[0] * drjit::pow(t[2], 2) + 2 * s[2] * t[0] * t[2]) + 3 * c[16] * s[
                             1] * drjit::pow(t[1], 2) + c[17] * (2 * s[1] * t[1] * t[2] + s[2] * drjit::pow(t[1], 2)) +
                         c[18] * (
                             s[1] * drjit::pow(t[2], 2) + 2 * s[2] * t[1] * t[2]) + 3 * c[19] * s[2] * drjit::pow(
                             t[2], 2);
                c2[14] = 6 * c[10] * n[0] * s[0] * t[0] + c[11] * (
                             2 * n[0] * s[0] * t[1] + 2 * n[0] * s[1] * t[0] + 2 * n[1] * s[0] * t[0]) + c[12] * (
                             2 * n[0] * s[0] * t[2] + 2 * n[0] * s[2] * t[0] + 2 * n[2] * s[0] * t[0]) + c[13] * (
                             2 * n[0] * s[1] * t[1] + 2 * n[1] * s[0] * t[1] + 2 * n[1] * s[1] * t[0]) + c[14] * (
                             n[0] * s[1] * t[2] + n[0] * s[2] * t[1] + n[1] * s[0] * t[2] + n[1] * s[2] * t[0] + n[2] *
                             s[0] * t[1] + n[2] * s[1] * t[0]) + c[15] * (
                             2 * n[0] * s[2] * t[2] + 2 * n[2] * s[0] * t[2] + 2 * n[2] * s[2] * t[0]) + 6 * c[16] * n[
                             1] * s[1] * t[1] + c[17] * (
                             2 * n[1] * s[1] * t[2] + 2 * n[1] * s[2] * t[1] + 2 * n[2] * s[1] * t[1]) + c[18] * (
                             2 * n[1] * s[2] * t[2] + 2 * n[2] * s[1] * t[2] + 2 * n[2] * s[2] * t[1]) + 6 * c[19] * n[
                             2] * s[2] * t[2];
                c2[15] = 3 * c[10] * drjit::pow(n[0], 2) * s[0] + c[11] * (
                             drjit::pow(n[0], 2) * s[1] + 2 * n[0] * n[1] * s[0]) + c[
                             12] * (drjit::pow(n[0], 2) * s[2] + 2 * n[0] * n[2] * s[0]) + c[13] * (
                             2 * n[0] * n[1] * s[1] + drjit::pow(n[1], 2) * s[0]) + c[14] * (
                             n[0] * n[1] * s[2] + n[0] * n[2] * s[1] + n[1] * n[2] * s[0]) + c[15] * (
                             2 * n[0] * n[2] * s[2] + drjit::pow(n[2], 2) * s[0]) + 3 * c[16] *
                         drjit::pow(n[1], 2) * s[1] + c[17] * (drjit::pow(n[1], 2) * s[2] + 2 * n[1] * n[2] * s[1]) + c[
                             18] * (
                             2 * n[1] * n[2] * s[2] + drjit::pow(n[2], 2) * s[1]) + 3 * c[19] * drjit::pow(n[2], 2) * s[
                             2];
                c2[16] = c[10] * drjit::pow(t[0], 3) + c[11] * drjit::pow(t[0], 2) * t[1] + c[12] * drjit::pow(t[0], 2)
                         * t[2] + c[13] * t
                         [0] * drjit::pow(t[1], 2) + c[14] * t[0] * t[1] * t[2] + c[15] * t[0] * drjit::pow(t[2], 2) + c
                         [16] *
                         drjit::pow(t[1], 3) + c[17] * drjit::pow(t[1], 2) * t[2] + c[18] * t[1] * drjit::pow(t[2], 2) +
                         c[19] * drjit::pow(
                             t[2], 3);
                c2[17] = 3 * c[10] * n[0] * drjit::pow(t[0], 2) + c[11] * (
                             2 * n[0] * t[0] * t[1] + n[1] * drjit::pow(t[0], 2)) + c[
                             12] * (2 * n[0] * t[0] * t[2] + n[2] * drjit::pow(t[0], 2)) + c[13] * (
                             n[0] * drjit::pow(t[1], 2) + 2 * n[1] * t[0] * t[1]) + c[14] * (
                             n[0] * t[1] * t[2] + n[1] * t[0] * t[2] + n[2] * t[0] * t[1]) + c[15] * (
                             n[0] * drjit::pow(t[2], 2) + 2 * n[2] * t[0] * t[2]) + 3 * c[16] * n[
                             1] * drjit::pow(t[1], 2) + c[17] * (2 * n[1] * t[1] * t[2] + n[2] * drjit::pow(t[1], 2)) +
                         c[18] * (
                             n[1] * drjit::pow(t[2], 2) + 2 * n[2] * t[1] * t[2]) + 3 * c[19] * n[2] * drjit::pow(
                             t[2], 2);
                c2[18] = 3 * c[10] * drjit::pow(n[0], 2) * t[0] + c[11] * (
                             drjit::pow(n[0], 2) * t[1] + 2 * n[0] * n[1] * t[0]) + c[
                             12] * (drjit::pow(n[0], 2) * t[2] + 2 * n[0] * n[2] * t[0]) + c[13] * (
                             2 * n[0] * n[1] * t[1] + drjit::pow(n[1], 2) * t[0]) + c[14] * (
                             n[0] * n[1] * t[2] + n[0] * n[2] * t[1] + n[1] * n[2] * t[0]) + c[15] * (
                             2 * n[0] * n[2] * t[2] + drjit::pow(n[2], 2) * t[0]) + 3 * c[16] *
                         drjit::pow(n[1], 2) * t[1] + c[17] * (drjit::pow(n[1], 2) * t[2] + 2 * n[1] * n[2] * t[1]) + c[
                             18] * (
                             2 * n[1] * n[2] * t[2] + drjit::pow(n[2], 2) * t[1]) + 3 * c[19] * drjit::pow(n[2], 2) * t[
                             2];
                c2[19] = c[10] * drjit::pow(n[0], 3) + c[11] * drjit::pow(n[0], 2) * n[1] + c[12] * drjit::pow(n[0], 2)
                         * n[2] + c[13] * n
                         [0] * drjit::pow(n[1], 2) + c[14] * n[0] * n[1] * n[2] + c[15] * n[0] * drjit::pow(n[2], 2) + c
                         [16] *
                         drjit::pow(n[1], 3) + c[17] * drjit::pow(n[1], 2) * n[2] + c[18] * n[1] * drjit::pow(n[2], 2) +
                         c[19] * drjit::pow(
                             n[2], 3);
            }
            return c2;
        }


        /*
        // Slow version of polynomial rotation. Retained for understandability.
        template<int order>
        static void rotatePolynomialOld(std::vector<float>& coeffs, std::vector<float>& tmpCoeffs,
                                        const Vector<Float, 3>& s, const Vector<Float, 3>& t,
                                        const Vector<Float, 3>& n) {
            assert(tmpCoeffs.size() == coeffs.size());
            for (size_t i = 0; i < tmpCoeffs.size(); ++i) {
                tmpCoeffs[i] = 0;
            }
            for (int l = 0; l <= order; ++l) {
                for (int m = 0; m <= order - l; ++m) {
                    for (int k = 0; k <= order - l - m; ++k) {
                        const int pi = powerToIndex(l, m, k);
                        for (int a = 0; a <= order; ++a) {
                            for (int b = 0; b <= order - a; ++b) {
                                for (int c = 0; c <= order - a - b; ++c) {
                                    if (l + m + k != a + b + c)
                                        continue;
                                    const int pj = powerToIndex(a, b, c);
                                    float coeff = 0.0f;
                                    for (int i_1 = 0; i_1 <= l; ++i_1) {
                                        for (int j_1 = 0; j_1 <= l - i_1; ++j_1) {
                                            for (int i_2 = std::max(0, a - i_1 - k); i_2 <= std::min(m, a - i_1); ++
                                                 i_2) {
                                                for (int j_2 = std::max(0, b - j_1 - k + a - i_1 - i_2);
                                                     j_2 <= std::min(m - i_2, b - j_1); ++j_2) {
                                                    const int i_3 = a - i_1 - i_2;
                                                    const int j_3 = b - j_1 - j_2;
                                                    const int k_1 = l - i_1 - j_1;
                                                    const int k_2 = m - i_2 - j_2;
                                                    const int k_3 = k - i_3 - j_3;
                                                    float term =
                                                            multinomial(i_1, j_1, k_1) * multinomial(i_2, j_2, k_2) *
                                                            multinomial(i_3, j_3, k_3);
                                                    term *= powi(s[0], i_1);
                                                    term *= powi(t[0], j_1);
                                                    term *= powi(n[0], k_1);
                                                    term *= powi(s[1], i_2);
                                                    term *= powi(t[1], j_2);
                                                    term *= powi(n[1], k_2);
                                                    term *= powi(s[2], i_3);
                                                    term *= powi(t[2], j_3);
                                                    term *= powi(n[2], k_3);
                                                    coeff += term;
                                                }
                                            }
                                        }
                                    }
                                    tmpCoeffs[pj] += coeffs[pi] * coeff;
                                }
                            }
                        }
                    }
                }
            }
            for (size_t i = 0; i < tmpCoeffs.size(); ++i) {
                coeffs[i] = tmpCoeffs[i];
            }
        }*/
    };

NAMESPACE_END(mitsuba)
