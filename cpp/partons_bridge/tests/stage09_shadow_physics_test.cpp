#include "../src/ShadowInjectionGPD.h"

#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/ServiceObjectRegistry.h>
#include <partons/beans/QuarkFlavor.h>
#include <partons/beans/gpd/GPDKinematic.h>
#include <partons/beans/gpd/GPDResult.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/modules/gpd/GPDBDMMS21.h>
#include <partons/modules/gpd/GPDGK16.h>
#include <partons/modules/gpd/GPDModule.h>
#include <partons/services/GPDService.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

double quark(const PARTONS::PartonDistribution& distribution,
        PARTONS::QuarkFlavor::Type flavor) {
    return distribution.getQuarkDistribution(flavor)
        .getQuarkDistribution();
}
}  // namespace

int main(int argc, char** argv) {
    PARTONS::Partons* partons = nullptr;
    PARTONS::GPDModule* baseline = nullptr;
    PARTONS::GPDModule* shadow = nullptr;
    PARTONS::GPDModule* combined = nullptr;
    try {
        partons = PARTONS::Partons::getInstance();
        partons->init(argc, argv);
        PARTONS::ModuleObjectFactory* factory =
            partons->getModuleObjectFactory();
        baseline = factory->newGPDModule(PARTONS::GPDGK16::classId);
        shadow = factory->newGPDModule(PARTONS::GPDBDMMS21::classId);
        combined = factory->newGPDModule(
            DVCSInference::ShadowInjectionGPD::classId);
        auto* model =
            dynamic_cast<DVCSInference::ShadowInjectionGPD*>(combined);
        if (model == nullptr) {
            throw std::runtime_error(
                "factory did not return ShadowInjectionGPD");
        }
        model->setBaselineModule(baseline);
        model->setShadowModule(shadow);
        model->setCoefficients({0.91, -0.7});

        double maximumLinearityResidual = 0.0;
        double maximumOddResidual = 0.0;
        double maximumShadowDeformation = 0.0;
        double maximumForbiddenContent = 0.0;
        for (const double xi : {0.05, 0.1, 0.2}) {
            for (const double x : {0.25, 0.45, 0.7}) {
                const PARTONS::GPDKinematic positive(
                    x, xi, -0.1, 4.0, 4.0);
                const PARTONS::GPDKinematic negative(
                    -x, xi, -0.1, 4.0, 4.0);
                const PARTONS::PartonDistribution b =
                    baseline->compute(positive, PARTONS::GPDType::H);
                const PARTONS::PartonDistribution s =
                    shadow->compute(positive, PARTONS::GPDType::H);
                const PARTONS::PartonDistribution c =
                    combined->compute(positive, PARTONS::GPDType::H);
                const PARTONS::PartonDistribution sNegative =
                    shadow->compute(negative, PARTONS::GPDType::H);

                maximumLinearityResidual = std::max(
                    maximumLinearityResidual,
                    std::abs(quark(c, PARTONS::QuarkFlavor::UP) -
                        (0.91 * quark(b, PARTONS::QuarkFlavor::UP) -
                         0.7 * quark(s, PARTONS::QuarkFlavor::UP))));
                maximumOddResidual = std::max(maximumOddResidual,
                    std::abs(quark(s, PARTONS::QuarkFlavor::UP) +
                        quark(sNegative, PARTONS::QuarkFlavor::UP)));
                maximumShadowDeformation = std::max(
                    maximumShadowDeformation,
                    std::abs(0.7 * quark(s, PARTONS::QuarkFlavor::UP)));
                maximumForbiddenContent = std::max(
                    maximumForbiddenContent,
                    std::abs(quark(s, PARTONS::QuarkFlavor::DOWN)));
                maximumForbiddenContent = std::max(
                    maximumForbiddenContent,
                    std::abs(quark(s, PARTONS::QuarkFlavor::STRANGE)));
                maximumForbiddenContent = std::max(
                    maximumForbiddenContent,
                    std::abs(
                        s.getGluonDistribution().getGluonDistribution()));
            }
        }

        const PARTONS::GPDKinematic endpoint(1.0, 0.1, -0.1, 4.0, 4.0);
        const double endpointAbs = std::abs(
            quark(shadow->compute(endpoint, PARTONS::GPDType::H),
                PARTONS::QuarkFlavor::UP));

        PARTONS::GPDService* service =
            partons->getServiceObjectRegistry()->getGPDService();
        const PARTONS::GPDKinematic serviceKinematics(
            0.35, 0.1, -0.1, 4.0, 4.0);
        const double serviceValue = quark(
            service->computeSingleKinematic(serviceKinematics, combined)
                .getPartonDistribution(PARTONS::GPDType::H),
            PARTONS::QuarkFlavor::UP);
        const double directValue = quark(
            combined->compute(serviceKinematics, PARTONS::GPDType::H),
            PARTONS::QuarkFlavor::UP);
        const double serviceResidual = std::abs(serviceValue - directValue);

        bool rejectedBaseline = false;
        bool rejectedShadow = false;
        try {
            model->setCoefficients({0.79, 0.0});
        } catch (const std::invalid_argument&) {
            rejectedBaseline = true;
        }
        try {
            model->setCoefficients({1.0, 1.01});
        } catch (const std::invalid_argument&) {
            rejectedShadow = true;
        }

        const bool passed =
            maximumLinearityResidual < 1e-14 &&
            maximumOddResidual < 1e-14 &&
            maximumShadowDeformation > 1e-3 &&
            maximumForbiddenContent == 0.0 &&
            endpointAbs == 0.0 &&
            serviceResidual < 1e-14 &&
            rejectedBaseline && rejectedShadow;

        std::cout << std::setprecision(17)
                  << "{\"schema_version\":1,"
                  << "\"test\":\"stage09_shadow_physics\","
                  << "\"factory_module\":\"ShadowInjectionGPD\","
                  << "\"baseline_module\":\"GPDGK16\","
                  << "\"shadow_module\":\"GPDBDMMS21\","
                  << "\"maximum_linearity_abs_residual\":"
                  << maximumLinearityResidual << ","
                  << "\"maximum_odd_symmetry_abs_residual\":"
                  << maximumOddResidual << ","
                  << "\"maximum_abs_shadow_deformation\":"
                  << maximumShadowDeformation << ","
                  << "\"maximum_forbidden_content_abs\":"
                  << maximumForbiddenContent << ","
                  << "\"endpoint_abs\":" << endpointAbs << ","
                  << "\"service_direct_abs_residual\":"
                  << serviceResidual << ","
                  << "\"rejected_baseline_out_of_bounds\":"
                  << (rejectedBaseline ? "true" : "false") << ","
                  << "\"rejected_shadow_out_of_bounds\":"
                  << (rejectedShadow ? "true" : "false") << ","
                  << "\"passed\":" << (passed ? "true" : "false")
                  << "}\n";

        factory->updateModulePointerReference(combined, nullptr);
        combined = nullptr;
        factory->updateModulePointerReference(shadow, nullptr);
        shadow = nullptr;
        factory->updateModulePointerReference(baseline, nullptr);
        baseline = nullptr;
        partons->close();
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        if (partons != nullptr) {
            PARTONS::ModuleObjectFactory* factory =
                partons->getModuleObjectFactory();
            if (combined != nullptr) {
                factory->updateModulePointerReference(combined, nullptr);
            }
            if (shadow != nullptr) {
                factory->updateModulePointerReference(shadow, nullptr);
            }
            if (baseline != nullptr) {
                factory->updateModulePointerReference(baseline, nullptr);
            }
            partons->close();
        }
        std::cerr << error.what() << '\n';
        return 2;
    }
}
