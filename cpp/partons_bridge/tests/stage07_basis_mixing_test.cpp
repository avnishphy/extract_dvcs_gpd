#include "../src/BasisInjectionGPD.h"

#include <ElementaryUtils/parameters/Parameters.h>
#include <apfel/rotations.h>
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
#include <partons/modules/evolution/gpd/GPDEvolutionModule.h>
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

namespace {

struct Values {
    std::array<double, 6> quark{};
    std::array<double, 6> plus{};
    std::array<double, 6> minus{};
    double gluon = 0.0;
};

Values fromResult(const PARTONS::GPDResult& result) {
    const PARTONS::PartonDistribution& h =
        result.getPartonDistribution(PARTONS::GPDType::H);
    Values values;
    for (const auto& entry : h.getQuarkDistributions()) {
        const int index = static_cast<int>(entry.first) - 1;
        if (index >= 0 && index < 6) {
            values.quark[index] = entry.second.getQuarkDistribution();
            values.plus[index] = entry.second.getQuarkDistributionPlus();
            values.minus[index] = entry.second.getQuarkDistributionMinus();
        }
    }
    values.gluon = h.getGluonDistribution().getGluonDistribution();
    return values;
}

double maximumDifference(const Values& left, const Values& right) {
    double maximum = std::abs(left.gluon - right.gluon);
    for (std::size_t index = 0; index < left.quark.size(); ++index) {
        maximum = std::max(maximum,
            std::abs(left.quark[index] - right.quark[index]));
        maximum = std::max(maximum,
            std::abs(left.plus[index] - right.plus[index]));
        maximum = std::max(maximum,
            std::abs(left.minus[index] - right.minus[index]));
    }
    return maximum;
}

Values linearCombination(const Values& sigma, const Values& t3,
                         const Values& gluon, double t3Scale,
                         double gluonScale) {
    Values result;
    result.gluon =
        sigma.gluon + t3Scale * t3.gluon + gluonScale * gluon.gluon;
    for (std::size_t index = 0; index < result.quark.size(); ++index) {
        result.quark[index] = sigma.quark[index] +
            t3Scale * t3.quark[index] + gluonScale * gluon.quark[index];
        result.plus[index] = sigma.plus[index] +
            t3Scale * t3.plus[index] + gluonScale * gluon.plus[index];
        result.minus[index] = sigma.minus[index] +
            t3Scale * t3.minus[index] + gluonScale * gluon.minus[index];
    }
    return result;
}

std::map<int, double> physicalMap(const Values& values, double x) {
    std::map<int, double> physical{{0, values.gluon}};
    const std::array<int, 6> partonsToPdg{2, 1, 3, 4, 5, 6};
    for (std::size_t index = 0; index < values.quark.size(); ++index) {
        const int pdg = partonsToPdg[index];
        const double q = x * values.quark[index];
        const double qbar = x * (values.plus[index] - values.quark[index]);
        physical[pdg] = q;
        physical[-pdg] = qbar;
    }
    return physical;
}

double roundTripError(const Values& values, double x) {
    const auto physical = physicalMap(values, x);
    const auto reconstructed =
        apfel::QCDEvToPhys(apfel::PhysToQCDEv(physical));
    double maximum = 0.0;
    for (const auto& entry : physical) {
        maximum = std::max(maximum,
            std::abs(reconstructed.at(entry.first) - entry.second));
    }
    return maximum;
}

double heavyMaximum(const Values& values) {
    double maximum = 0.0;
    for (std::size_t index = 3; index < 6; ++index) {
        maximum = std::max(maximum, std::abs(values.quark[index]));
        maximum = std::max(maximum, std::abs(values.plus[index]));
        maximum = std::max(maximum, std::abs(values.minus[index]));
    }
    return maximum;
}

bool finite(const Values& values) {
    if (!std::isfinite(values.gluon)) {
        return false;
    }
    for (std::size_t index = 0; index < values.quark.size(); ++index) {
        if (!std::isfinite(values.quark[index]) ||
                !std::isfinite(values.plus[index]) ||
                !std::isfinite(values.minus[index])) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    PARTONS::Partons* partons = nullptr;
    PARTONS::GPDModule* gpd = nullptr;
    PARTONS::GPDEvolutionModule* evolution = nullptr;
    PARTONS::RunningAlphaStrongModule* alpha = nullptr;
    PARTONS::ActiveFlavorsThresholdsModule* flavors = nullptr;

    try {
        constexpr double x = 0.2;
        constexpr double xi = 0.1;
        constexpr double t = -0.1;
        constexpr double q0Squared = 4.0;
        constexpr double evolvedScale = 16.0;
        constexpr double identityMaximum = 2e-10;
        constexpr double roundTripMaximum = 2e-14;
        constexpr double superpositionMaximum = 2e-9;
        constexpr double decoupledGluonMaximum = 2e-10;
        constexpr double heavyMaximumAllowed = 2e-10;
        constexpr double mixingResponseMinimum = 1e-6;

        partons = PARTONS::Partons::getInstance();
        partons->init(argc, argv);
        PARTONS::ModuleObjectFactory* factory =
            partons->getModuleObjectFactory();
        PARTONS::GPDService* service =
            partons->getServiceObjectRegistry()->getGPDService();

        gpd = factory->newGPDModule(
            DVCSInference::BasisInjectionGPD::classId);
        auto* basis =
            dynamic_cast<DVCSInference::BasisInjectionGPD*>(gpd);
        if (basis == nullptr) {
            throw std::runtime_error(
                "factory did not return BasisInjectionGPD");
        }
        basis->setShapeParameters(
            DVCSInference::DDParameters{2.0, 0.55, 3.4, 2.0, 1.0});
        basis->setQuadratureOrder(128);

        const auto evaluate = [&](const DVCSInference::BasisAmplitudes& input,
                                  double muF2) {
            basis->setAmplitudes(input);
            return fromResult(service->computeSingleKinematic(
                PARTONS::GPDKinematic(x, xi, t, muF2, muF2), gpd));
        };

        const DVCSInference::BasisAmplitudes sigmaInput{1.0, 0.0, 0.0};
        const DVCSInference::BasisAmplitudes t3Input{0.0, 1.0, 0.0};
        const DVCSInference::BasisAmplitudes gluonInput{0.0, 0.0, 1.0};
        const DVCSInference::BasisAmplitudes combinedInput{1.0, 0.5, 0.8};

        const Values directSigmaQ0 = evaluate(sigmaInput, q0Squared);
        const Values directT3Q0 = evaluate(t3Input, q0Squared);
        const Values directGluonQ0 = evaluate(gluonInput, q0Squared);

        evolution = factory->newGPDEvolutionModule(
            PARTONS::GPDEvolutionApfel::classId);
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
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_ALPHAS_REF, 0.118);
        alphaParameters.add(
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_MU_REF, 91.1876);
        alphaParameters.add(
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_THRESHOLDS, "0 0 0");
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

        const Values attachedSigmaQ0 = evaluate(sigmaInput, q0Squared);
        const Values attachedT3Q0 = evaluate(t3Input, q0Squared);
        const Values attachedGluonQ0 = evaluate(gluonInput, q0Squared);
        const Values evolvedSigma = evaluate(sigmaInput, evolvedScale);
        const Values evolvedT3 = evaluate(t3Input, evolvedScale);
        const Values evolvedGluon = evaluate(gluonInput, evolvedScale);
        const Values evolvedCombined = evaluate(combinedInput, evolvedScale);

        const double q0Identity = std::max({
            maximumDifference(directSigmaQ0, attachedSigmaQ0),
            maximumDifference(directT3Q0, attachedT3Q0),
            maximumDifference(directGluonQ0, attachedGluonQ0)});
        const Values expectedCombined = linearCombination(
            evolvedSigma, evolvedT3, evolvedGluon, 0.5, 0.8);
        const double superposition =
            maximumDifference(evolvedCombined, expectedCombined);
        const double maximumRoundTrip = std::max({
            roundTripError(attachedSigmaQ0, x),
            roundTripError(attachedT3Q0, x),
            roundTripError(attachedGluonQ0, x),
            roundTripError(evolvedSigma, x),
            roundTripError(evolvedT3, x),
            roundTripError(evolvedGluon, x)});
        const double t3Gluon = std::abs(evolvedT3.gluon);
        const double sigmaToGluon = std::abs(evolvedSigma.gluon);
        double gluonToSigma = 0.0;
        for (std::size_t index = 0; index < 3; ++index) {
            gluonToSigma += x * evolvedGluon.plus[index];
        }
        gluonToSigma = std::abs(gluonToSigma);
        const double maximumHeavy = std::max({
            heavyMaximum(evolvedSigma), heavyMaximum(evolvedT3),
            heavyMaximum(evolvedGluon), heavyMaximum(evolvedCombined)});
        const bool allFinite = finite(directSigmaQ0) &&
            finite(directT3Q0) && finite(directGluonQ0) &&
            finite(evolvedSigma) && finite(evolvedT3) &&
            finite(evolvedGluon) && finite(evolvedCombined);

        const bool passed = allFinite &&
            q0Identity <= identityMaximum &&
            maximumRoundTrip <= roundTripMaximum &&
            superposition <= superpositionMaximum &&
            t3Gluon <= decoupledGluonMaximum &&
            maximumHeavy <= heavyMaximumAllowed &&
            sigmaToGluon >= mixingResponseMinimum &&
            gluonToSigma >= mixingResponseMinimum;

        std::cout << std::setprecision(17)
                  << "{\"status\":\"" << (passed ? "ok" : "error")
                  << "\",\"all_finite\":" << (allFinite ? "true" : "false")
                  << ",\"q0_identity_max_abs\":" << q0Identity
                  << ",\"basis_round_trip_max_abs\":" << maximumRoundTrip
                  << ",\"superposition_max_abs\":" << superposition
                  << ",\"t3_to_gluon_abs\":" << t3Gluon
                  << ",\"sigma_to_gluon_abs\":" << sigmaToGluon
                  << ",\"gluon_to_sigma_abs\":" << gluonToSigma
                  << ",\"heavy_flavor_max_abs\":" << maximumHeavy
                  << ",\"thresholds\":{\"q0_identity_max\":"
                  << identityMaximum << ",\"basis_round_trip_max\":"
                  << roundTripMaximum << ",\"superposition_max\":"
                  << superpositionMaximum << ",\"t3_gluon_max\":"
                  << decoupledGluonMaximum << ",\"heavy_flavor_max\":"
                  << heavyMaximumAllowed << ",\"mixing_response_min\":"
                  << mixingResponseMinimum << "}}\n";

        factory->updateModulePointerReference(gpd, 0);
        gpd = nullptr;
        factory->updateModulePointerReference(evolution, 0);
        evolution = nullptr;
        factory->updateModulePointerReference(alpha, 0);
        alpha = nullptr;
        factory->updateModulePointerReference(flavors, 0);
        flavors = nullptr;
        partons->close();
        partons = nullptr;
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (partons != nullptr) {
            PARTONS::ModuleObjectFactory* factory =
                partons->getModuleObjectFactory();
            if (gpd != nullptr) {
                factory->updateModulePointerReference(gpd, 0);
            }
            if (evolution != nullptr) {
                factory->updateModulePointerReference(evolution, 0);
            }
            if (alpha != nullptr) {
                factory->updateModulePointerReference(alpha, 0);
            }
            if (flavors != nullptr) {
                factory->updateModulePointerReference(flavors, 0);
            }
            partons->close();
        }
        return 2;
    }
}
