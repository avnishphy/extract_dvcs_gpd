#include "../src/FixedScaleDDGPD.h"

#include <ElementaryUtils/parameters/Parameters.h>
#include <apfel/apfelxx.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/ServiceObjectRegistry.h>
#include <partons/beans/PerturbativeQCDOrderType.h>
#include <partons/beans/QuarkFlavor.h>
#include <partons/beans/gpd/GPDKinematic.h>
#include <partons/beans/gpd/GPDResult.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/modules/active_flavors_thresholds/ActiveFlavorsThresholdsConstant.h>
#include <partons/modules/active_flavors_thresholds/ActiveFlavorsThresholdsModule.h>
#include <partons/modules/evolution/gpd/GPDEvolutionApfel.h>
#include <partons/modules/gpd/GPDModule.h>
#include <partons/modules/running_alpha_strong/RunningAlphaStrongApfel.h>
#include <partons/modules/running_alpha_strong/RunningAlphaStrongModule.h>
#include <partons/services/GPDService.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct Values {
    double up;
    double plus;
    double minus;
    double gluon;
};

struct Moments {
    double m0;
    double m1;
    double m2;
};

std::vector<std::pair<double, double>> gaussLegendre(
        unsigned int order, double lower, double upper) {
    constexpr double tolerance = 1e-15;
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::vector<std::pair<double, double>> result(order);
    const unsigned int half = (order + 1U) / 2U;
    const double midpoint = 0.5 * (lower + upper);
    const double halfWidth = 0.5 * (upper - lower);
    for (unsigned int index = 0; index < half; ++index) {
        double root = std::cos(
            pi * (static_cast<double>(index) + 0.75) /
            (static_cast<double>(order) + 0.5));
        double previous = 0.0;
        double derivative = 0.0;
        do {
            double p0 = 1.0;
            double p1 = root;
            for (unsigned int degree = 2; degree <= order; ++degree) {
                const double p2 =
                    ((2.0 * degree - 1.0) * root * p1 -
                     (degree - 1.0) * p0) /
                    degree;
                p0 = p1;
                p1 = p2;
            }
            derivative = order * (root * p1 - p0) /
                (root * root - 1.0);
            previous = root;
            root = previous - p1 / derivative;
        } while (std::abs(root - previous) > tolerance);
        const double weight =
            2.0 / ((1.0 - root * root) * derivative * derivative);
        result[index] =
            {midpoint - halfWidth * root, halfWidth * weight};
        result[order - 1U - index] =
            {midpoint + halfWidth * root, halfWidth * weight};
    }
    return result;
}

Values fromResult(const PARTONS::GPDResult& result) {
    const PARTONS::PartonDistribution& h =
        result.getPartonDistribution(PARTONS::GPDType::H);
    const PARTONS::QuarkDistribution& up =
        h.getQuarkDistribution(PARTONS::QuarkFlavor::UP);
    return Values{up.getQuarkDistribution(),
        up.getQuarkDistributionPlus(),
        up.getQuarkDistributionMinus(),
        h.getGluonDistribution().getGluonDistribution()};
}

bool finite(const Values& values) {
    return std::isfinite(values.up) && std::isfinite(values.plus) &&
        std::isfinite(values.minus) && std::isfinite(values.gluon);
}

Values fromTable(const std::map<int, apfel::Distribution>& distributions,
        double x) {
    const double quark = distributions.at(2).Evaluate(x) / x;
    const double antiquark = distributions.at(-2).Evaluate(x) / x;
    return Values{quark, quark + antiquark, quark - antiquark,
        distributions.at(0).Evaluate(x)};
}

Moments evolvedMoments(
        const std::map<int, apfel::Distribution>& distributions,
        unsigned int order) {
    Moments moments{0.0, 0.0, 0.0};
    for (const auto& [x, weight] :
            gaussLegendre(order, 1e-7, 1.0)) {
        const Values values = fromTable(distributions, x);
        moments.m0 += weight * values.minus;
        moments.m1 += weight * x * values.plus;
        moments.m2 += weight * x * x * values.minus;
    }
    return moments;
}

Moments inputMoments(DVCSInference::FixedScaleDDGPD& model, double xi,
        unsigned int order) {
    Moments moments{0.0, 0.0, 0.0};
    for (const auto& [x, weight] : gaussLegendre(order, -1.0, 1.0)) {
        const double value = model.evaluateValue(x, xi, 0.0);
        moments.m0 += weight * value;
        moments.m1 += weight * x * value;
        moments.m2 += weight * x * x * value;
    }
    return moments;
}

double linearResidual(const std::vector<double>& x,
        const std::vector<double>& y) {
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    for (std::size_t index = 0; index < x.size(); ++index) {
        sx += x[index];
        sy += y[index];
        sxx += x[index] * x[index];
        sxy += x[index] * y[index];
    }
    const double count = static_cast<double>(x.size());
    const double denominator = count * sxx - sx * sx;
    const double slope = (count * sxy - sx * sy) / denominator;
    const double intercept = (sy - slope * sx) / count;
    double maximum = 0.0;
    for (std::size_t index = 0; index < x.size(); ++index) {
        maximum = std::max(maximum,
            std::abs(y[index] - intercept - slope * x[index]));
    }
    return maximum;
}

}  // namespace

int main(int argc, char** argv) {
    PARTONS::Partons* partons = nullptr;
    PARTONS::GPDModule* gpd = nullptr;
    PARTONS::GPDEvolutionModule* evolutionBase = nullptr;
    PARTONS::RunningAlphaStrongModule* alpha = nullptr;
    PARTONS::ActiveFlavorsThresholdsModule* flavors = nullptr;

    try {
        partons = PARTONS::Partons::getInstance();
        partons->init(argc, argv);
        PARTONS::ModuleObjectFactory* factory =
            partons->getModuleObjectFactory();
        PARTONS::GPDService* service =
            partons->getServiceObjectRegistry()->getGPDService();

        gpd = factory->newGPDModule(
            DVCSInference::FixedScaleDDGPD::classId);
        auto* dd = dynamic_cast<DVCSInference::FixedScaleDDGPD*>(gpd);
        if (dd == nullptr) {
            throw std::runtime_error(
                "factory did not return FixedScaleDDGPD");
        }
        dd->setParameters(
            DVCSInference::DDParameters{2.0, 0.55, 3.4, 2.0, 1.0});
        dd->setQuadratureOrder(128);

        const auto evaluate = [&](double x, double xi, double muF2) {
            return fromResult(service->computeSingleKinematic(
                PARTONS::GPDKinematic(x, xi, 0.0, muF2, muF2), gpd));
        };
        const Values directQ0 = evaluate(0.2, 0.1, 4.0);

        evolutionBase = factory->newGPDEvolutionModule(
            PARTONS::GPDEvolutionApfel::classId);
        auto* evolution =
            dynamic_cast<PARTONS::GPDEvolutionApfel*>(evolutionBase);
        if (evolution == nullptr) {
            throw std::runtime_error(
                "factory did not return GPDEvolutionApfel");
        }
        evolution->configure(ElemUtils::Parameters(
            PARTONS::PerturbativeQCDOrderType::
                PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::LO));

        alpha = factory->newRunningAlphaStrongModule(
            PARTONS::RunningAlphaStrongApfel::classId);
        ElemUtils::Parameters alphaParameters;
        alphaParameters.add(
            PARTONS::PerturbativeQCDOrderType::
                PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::LO);
        alphaParameters.add(
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_ALPHAS_REF,
            0.118);
        alphaParameters.add(
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_MU_REF,
            91.1876);
        alphaParameters.add(
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_THRESHOLDS,
            "0 0 0");
        alpha->configure(alphaParameters);

        flavors = factory->newActiveFlavorsThresholdsModule(
            PARTONS::ActiveFlavorsThresholdsConstant::classId);
        auto* constantFlavors =
            dynamic_cast<PARTONS::ActiveFlavorsThresholdsConstant*>(flavors);
        if (constantFlavors == nullptr) {
            throw std::runtime_error(
                "factory did not return ActiveFlavorsThresholdsConstant");
        }
        constantFlavors->setNFlavors(3);
        evolution->setRunningAlphaStrongModule(alpha);
        evolution->setActiveFlavorsModule(flavors);
        gpd->setEvolQcdModule(evolution);

        const Values attachedQ0 = evaluate(0.2, 0.1, 4.0);
        const Values nearQ0 = evaluate(0.2, 0.1, 4.0004);
        const Values evolved10 = evaluate(0.2, 0.1, 10.0);
        const auto table10 = apfel::QCDEvToPhys(
            evolution->getTabulatedGPDs()->Evaluate(std::sqrt(10.0))
                .GetObjects());
        const Values tableAnchor10 = fromTable(table10, 0.2);
        const Values evolved16 = evaluate(0.2, 0.1, 16.0);

        const double q0Identity = std::max({
            std::abs(directQ0.up - attachedQ0.up),
            std::abs(directQ0.plus - attachedQ0.plus),
            std::abs(directQ0.minus - attachedQ0.minus),
            std::abs(directQ0.gluon - attachedQ0.gluon)});
        const double nearQ0Relative =
            std::abs(nearQ0.up - directQ0.up) / std::abs(directQ0.up);
        const double tableAnchorDifference = std::max({
            std::abs(tableAnchor10.up - evolved10.up),
            std::abs(tableAnchor10.plus - evolved10.plus),
            std::abs(tableAnchor10.minus - evolved10.minus),
            std::abs(tableAnchor10.gluon - evolved10.gluon)});
        const bool scaleEffect =
            evolved10.up != directQ0.up && evolved16.up != directQ0.up;
        const bool generatedGluon =
            evolved10.gluon != 0.0 && evolved16.gluon != 0.0;

        const std::array<double, 4> xis{0.0, 0.15, 0.3, 0.45};
        const std::array<double, 3> scales{4.0, 10.0, 16.0};
        std::array<double, 3> sumRuleErrors{};
        std::array<double, 3> m0XiResiduals{};
        std::array<double, 3> m1PolynomialResiduals{};
        std::array<double, 3> m2PolynomialResiduals{};
        double integrationConvergence = 0.0;
        bool allFinite = finite(directQ0) && finite(attachedQ0) &&
            finite(nearQ0) && finite(evolved10) && finite(evolved16);

        for (std::size_t scaleIndex = 0;
                scaleIndex < scales.size(); ++scaleIndex) {
            std::vector<double> xiSquared;
            std::vector<double> m0;
            std::vector<double> m1;
            std::vector<double> m2;
            for (double xi : xis) {
                Moments moments{};
                if (scales[scaleIndex] == 4.0) {
                    moments = inputMoments(*dd, xi, 256);
                } else {
                    const Values anchor =
                        evaluate(0.2, xi, scales[scaleIndex]);
                    allFinite = allFinite && finite(anchor);
                    const auto distributions = apfel::QCDEvToPhys(
                        evolution->getTabulatedGPDs()
                            ->Evaluate(std::sqrt(scales[scaleIndex]))
                            .GetObjects());
                    moments = evolvedMoments(distributions, 256);
                    const Moments lowerOrder =
                        evolvedMoments(distributions, 128);
                    integrationConvergence = std::max(
                        integrationConvergence,
                        std::max({
                            std::abs(moments.m0 - lowerOrder.m0),
                            std::abs(moments.m1 - lowerOrder.m1),
                            std::abs(moments.m2 - lowerOrder.m2)}));
                    const Values lowerEdge =
                        fromTable(distributions, 1e-7);
                    const Values upperEdge =
                        fromTable(distributions, 0.999999);
                    allFinite = allFinite && finite(lowerEdge) &&
                        finite(upperEdge);
                }
                xiSquared.push_back(xi * xi);
                m0.push_back(moments.m0);
                m1.push_back(moments.m1);
                m2.push_back(moments.m2);
            }
            for (double value : m0) {
                sumRuleErrors[scaleIndex] = std::max(
                    sumRuleErrors[scaleIndex], std::abs(value - 2.0));
            }
            const auto [minimumM0, maximumM0] =
                std::minmax_element(m0.begin(), m0.end());
            m0XiResiduals[scaleIndex] = *maximumM0 - *minimumM0;
            m1PolynomialResiduals[scaleIndex] =
                linearResidual(xiSquared, m1);
            m2PolynomialResiduals[scaleIndex] =
                linearResidual(xiSquared, m2);
        }

        const double crossoverXi = 0.3;
        const double crossoverStep = 1e-6;
        evaluate(0.2, crossoverXi, 16.0);
        const auto crossoverTable = apfel::QCDEvToPhys(
            evolution->getTabulatedGPDs()->Evaluate(4.0).GetObjects());
        const Values crossoverLeft =
            fromTable(crossoverTable, crossoverXi - crossoverStep);
        const Values crossoverRight =
            fromTable(crossoverTable, crossoverXi + crossoverStep);
        const double crossoverJump = std::max({
            std::abs(crossoverLeft.up - crossoverRight.up),
            std::abs(crossoverLeft.plus - crossoverRight.plus),
            std::abs(crossoverLeft.minus - crossoverRight.minus),
            std::abs(crossoverLeft.gluon - crossoverRight.gluon)});

        const double maximumSumRuleError =
            *std::max_element(sumRuleErrors.begin(), sumRuleErrors.end());
        const double maximumM0XiResidual =
            *std::max_element(m0XiResiduals.begin(), m0XiResiduals.end());
        const double maximumM1PolynomialResidual =
            *std::max_element(m1PolynomialResiduals.begin(),
                m1PolynomialResiduals.end());
        const double maximumM2PolynomialResidual =
            *std::max_element(m2PolynomialResiduals.begin(),
                m2PolynomialResiduals.end());

        // Thresholds are declared before the accepted campaign.
        const bool passed =
            q0Identity == 0.0 &&
            nearQ0Relative < 5e-4 &&
            tableAnchorDifference < 5e-13 &&
            scaleEffect &&
            generatedGluon &&
            allFinite &&
            maximumSumRuleError < 5e-3 &&
            maximumM0XiResidual < 5e-3 &&
            maximumM1PolynomialResidual < 2e-3 &&
            maximumM2PolynomialResidual < 2e-3 &&
            integrationConvergence < 2e-4 &&
            crossoverJump < 5e-3;

        std::cout << std::setprecision(17)
                  << "{\"schema_version\":1,"
                  << "\"test\":\"stage06_evolution_physics\","
                  << "\"q0_identity_max_abs\":" << q0Identity << ","
                  << "\"near_q0_relative_change\":" << nearQ0Relative << ","
                  << "\"service_table_anchor_max_abs_difference\":"
                  << tableAnchorDifference << ","
                  << "\"scale_effect\":" << (scaleEffect ? "true" : "false")
                  << ",\"generated_gluon_observed\":"
                  << (generatedGluon ? "true" : "false")
                  << ",\"all_domain_values_finite\":"
                  << (allFinite ? "true" : "false")
                  << ",\"sum_rule_max_abs_error\":"
                  << maximumSumRuleError
                  << ",\"m0_xi_independence_max_abs\":"
                  << maximumM0XiResidual
                  << ",\"m1_polynomial_fit_max_abs_residual\":"
                  << maximumM1PolynomialResidual
                  << ",\"m2_polynomial_fit_max_abs_residual\":"
                  << maximumM2PolynomialResidual
                  << ",\"gauss_128_vs_256_max_abs_difference\":"
                  << integrationConvergence
                  << ",\"crossover_left_right_max_abs_difference\":"
                  << crossoverJump
                  << ",\"scales_squared_GeV2\":[4,10,16],"
                  << "\"xi_values\":[0,0.15,0.3,0.45],"
                  << "\"integration_domain\":[1e-7,1],"
                  << "\"passed\":" << (passed ? "true" : "false")
                  << "}\n";

        factory->updateModulePointerReference(gpd, 0);
        gpd = nullptr;
        factory->updateModulePointerReference(evolutionBase, 0);
        evolutionBase = nullptr;
        factory->updateModulePointerReference(alpha, 0);
        alpha = nullptr;
        factory->updateModulePointerReference(flavors, 0);
        flavors = nullptr;
        partons->close();
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }
    if (partons != nullptr) {
        partons->close();
    }
    return 2;
}
