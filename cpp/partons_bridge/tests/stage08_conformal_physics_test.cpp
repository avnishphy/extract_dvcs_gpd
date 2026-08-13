#include "../src/ConformalMomentGPD.h"

#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/ServiceObjectRegistry.h>
#include <partons/beans/QuarkFlavor.h>
#include <partons/beans/gpd/GPDKinematic.h>
#include <partons/beans/gpd/GPDResult.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/modules/gpd/GPDModule.h>
#include <partons/services/GPDService.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

double midpointMoment(DVCSInference::ConformalMomentGPD& model,
        unsigned int power, double t, unsigned int intervals) {
    const double step = 1.0 / intervals;
    double result = 0.0;
    for (unsigned int index = 0; index < intervals; ++index) {
        const double x = (static_cast<double>(index) + 0.5) * step;
        result += std::pow(x, static_cast<double>(power)) *
            model.evaluateValue(x, 0.0, t);
    }
    return result * step;
}

}  // namespace

int main(int argc, char** argv) {
    PARTONS::Partons* partons = nullptr;
    PARTONS::GPDModule* gpd = nullptr;
    try {
        partons = PARTONS::Partons::getInstance();
        partons->init(argc, argv);
        PARTONS::ModuleObjectFactory* factory =
            partons->getModuleObjectFactory();
        PARTONS::GPDService* service =
            partons->getServiceObjectRegistry()->getGPDService();
        gpd = factory->newGPDModule(
            DVCSInference::ConformalMomentGPD::classId);
        auto* model =
            dynamic_cast<DVCSInference::ConformalMomentGPD*>(gpd);
        if (model == nullptr) {
            throw std::runtime_error(
                "factory did not return ConformalMomentGPD");
        }
        model->setParameters(
            DVCSInference::ConformalMomentParameters{-0.55, 3.4, 1.0});
        model->setIntegrationLimit(1024);

        double maximumMomentError = 0.0;
        for (double t : {0.0, -0.2}) {
            for (unsigned int j = 0; j < 4; ++j) {
                maximumMomentError = std::max(maximumMomentError,
                    std::abs(midpointMoment(*model, j, t, 12000) -
                             model->conformalMoment(j, t)));
            }
        }

        const double forwardNormalization =
            midpointMoment(*model, 0, 0.0, 100000);
        const double endpoint = std::max(
            std::abs(model->evaluateValue(0.0, 0.0, -0.2)),
            std::abs(model->evaluateValue(1.0, 0.0, -0.2)));
        const double outsideSupport = std::max(
            std::abs(model->evaluateValue(-0.1, 0.0, -0.2)),
            std::abs(model->evaluateValue(1.1, 0.0, -0.2)));

        bool rejectedNonzeroXi = false;
        try {
            (void)model->evaluateValue(0.2, 0.1, -0.2);
        } catch (const std::invalid_argument&) {
            rejectedNonzeroXi = true;
        }

        double maximumLimitDifference = 0.0;
        for (double x : {0.04, 0.2, 0.5, 0.82}) {
            model->setIntegrationLimit(256);
            const double low = model->evaluateValue(x, 0.0, -0.4);
            model->setIntegrationLimit(2048);
            const double high = model->evaluateValue(x, 0.0, -0.4);
            maximumLimitDifference = std::max(
                maximumLimitDifference, std::abs(low - high));
        }

        const PARTONS::GPDResult serviceResult =
            service->computeSingleKinematic(
                PARTONS::GPDKinematic(0.2, 0.0, -0.2, 4.0, 4.0), gpd);
        const double serviceValue = serviceResult
            .getPartonDistribution(PARTONS::GPDType::H)
            .getQuarkDistribution(PARTONS::QuarkFlavor::UP)
            .getQuarkDistribution();
        const double directValue =
            model->evaluateValue(0.2, 0.0, -0.2);
        const double serviceDifference =
            std::abs(serviceValue - directValue);

        const bool passed =
            maximumMomentError < 2e-6 &&
            std::abs(forwardNormalization - 2.0) < 2e-7 &&
            endpoint == 0.0 &&
            outsideSupport == 0.0 &&
            rejectedNonzeroXi &&
            maximumLimitDifference < 1e-12 &&
            serviceDifference < 1e-14;

        std::cout << std::setprecision(17)
                  << "{\"schema_version\":1,"
                  << "\"test\":\"stage08_conformal_physics\","
                  << "\"factory_module\":\"ConformalMomentGPD\","
                  << "\"maximum_moment_abs_error\":"
                  << maximumMomentError << ","
                  << "\"forward_normalization_abs_error\":"
                  << std::abs(forwardNormalization - 2.0) << ","
                  << "\"endpoint_max_abs\":" << endpoint << ","
                  << "\"outside_support_max_abs\":" << outsideSupport << ","
                  << "\"rejected_nonzero_xi\":"
                  << (rejectedNonzeroXi ? "true" : "false") << ","
                  << "\"integration_limit_max_abs_difference\":"
                  << maximumLimitDifference << ","
                  << "\"service_direct_abs_difference\":"
                  << serviceDifference << ","
                  << "\"passed\":" << (passed ? "true" : "false")
                  << "}\n";

        factory->updateModulePointerReference(gpd, 0);
        gpd = nullptr;
        partons->close();
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        if (partons != nullptr) {
            if (gpd != nullptr) {
                partons->getModuleObjectFactory()
                    ->updateModulePointerReference(gpd, 0);
            }
            partons->close();
        }
        std::cerr << error.what() << '\n';
        return 2;
    }
}
