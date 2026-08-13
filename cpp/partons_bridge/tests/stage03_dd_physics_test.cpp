#include "../src/FixedScaleDDGPD.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

double simpsonProfile(DVCSInference::FixedScaleDDGPD& model, double beta) {
    constexpr unsigned int intervals = 20000;
    const double lower = -(1.0 - beta);
    const double upper = 1.0 - beta;
    const double step = (upper - lower) / intervals;
    double sum = model.profileValue(beta, lower) +
        model.profileValue(beta, upper);
    for (unsigned int index = 1; index < intervals; ++index) {
        sum += (index % 2U == 0U ? 2.0 : 4.0) *
            model.profileValue(beta, lower + index * step);
    }
    return sum * step / 3.0;
}

std::vector<double> midpointMoments(DVCSInference::FixedScaleDDGPD& model,
        double xi, double t, unsigned int intervals) {
    const double step = 2.0 / intervals;
    std::vector<double> moments(3, 0.0);
    for (unsigned int index = 0; index < intervals; ++index) {
        const double x = -1.0 + (index + 0.5) * step;
        const double value = model.evaluateValue(x, xi, t);
        moments[0] += value;
        moments[1] += x * value;
        moments[2] += x * x * value;
    }
    for (double& moment : moments) {
        moment *= step;
    }
    return moments;
}

double maximumAbsolute(const std::vector<double>& values) {
    double maximum = 0.0;
    for (double value : values) {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

}  // namespace

int main() {
    DVCSInference::FixedScaleDDGPD model("Stage03DDPhysicsTest");
    const DVCSInference::DDParameters parameters{2.0, 0.5, 3.0, 2.0, 1.0};
    model.setParameters(parameters);
    model.setQuadratureOrder(128);

    std::vector<double> profileErrors;
    for (double beta : {0.1, 0.5, 0.9}) {
        profileErrors.push_back(simpsonProfile(model, beta) - 1.0);
    }

    constexpr unsigned int forwardIntervals = 200000;
    const double forwardStep = 1.0 / forwardIntervals;
    double forwardIntegral = 0.0;
    for (unsigned int index = 0; index < forwardIntervals; ++index) {
        forwardIntegral += model.forwardValue(
            (index + 0.5) * forwardStep, 0.0);
    }
    forwardIntegral *= forwardStep;

    const double forwardPoint =
        std::abs(model.evaluateValue(0.37, 0.0, -0.2) -
                 model.forwardValue(0.37, -0.2));
    const double xiSymmetry =
        std::abs(model.evaluateValue(0.05, 0.3, -0.2) -
                 model.evaluateValue(0.05, -0.3, -0.2));
    const double outsideSupport = std::max(
        std::abs(model.evaluateValue(-1.01, 0.2, -0.2)),
        std::abs(model.evaluateValue(1.01, 0.2, -0.2)));
    const double endpoint = std::max(
        std::abs(model.evaluateValue(-1.0, 0.2, -0.2)),
        std::abs(model.evaluateValue(1.0, 0.2, -0.2)));

    constexpr double continuityStep = 1e-6;
    const double continuityAtXi = std::abs(
        model.evaluateValue(0.3 - continuityStep, 0.3, -0.2) -
        model.evaluateValue(0.3 + continuityStep, 0.3, -0.2));
    const double continuityAtMinusXi = std::abs(
        model.evaluateValue(-0.3 - continuityStep, 0.3, -0.2) -
        model.evaluateValue(-0.3 + continuityStep, 0.3, -0.2));
    const double maximumContinuityJump =
        std::max(continuityAtXi, continuityAtMinusXi);

    const double t = -0.2;
    const double tFactor = std::exp(parameters.tSlope * t);
    const double expectedM0 = 2.0 * tFactor;
    const double expectedM1 = expectedM0 *
        (parameters.a + 1.0) /
        (parameters.a + parameters.c + 2.0);
    const double betaSecondMoment =
        (parameters.a + 1.0) * (parameters.a + 2.0) /
        ((parameters.a + parameters.c + 2.0) *
         (parameters.a + parameters.c + 3.0));
    const double oneMinusBetaSecondMoment =
        (parameters.c + 1.0) * (parameters.c + 2.0) /
        ((parameters.a + parameters.c + 2.0) *
         (parameters.a + parameters.c + 3.0));
    const double alphaSecondMoment =
        oneMinusBetaSecondMoment / (2.0 * parameters.profileB + 3.0);

    std::vector<double> moment0Errors;
    std::vector<double> moment1Errors;
    std::vector<double> moment2Errors;
    for (double xi : {0.0, 0.15, 0.35, 0.6}) {
        const std::vector<double> moments =
            midpointMoments(model, xi, t, 16000);
        moment0Errors.push_back(moments[0] - expectedM0);
        moment1Errors.push_back(moments[1] - expectedM1);
        const double expectedM2 = expectedM0 *
            (betaSecondMoment + xi * xi * alphaSecondMoment);
        moment2Errors.push_back(moments[2] - expectedM2);
    }

    std::vector<double> convergenceMaxByOrder;
    const std::vector<unsigned int> orders{32, 64, 128, 256};
    const std::vector<double> xs{-0.15, 0.0, 0.12, 0.45, 0.8};
    std::vector<double> referenceValues;
    model.setQuadratureOrder(256);
    for (double x : xs) {
        referenceValues.push_back(model.evaluateValue(x, 0.35, -0.3));
    }
    for (unsigned int order : orders) {
        model.setQuadratureOrder(order);
        double maximumForOrder = 0.0;
        for (std::size_t index = 0; index < xs.size(); ++index) {
            maximumForOrder = std::max(maximumForOrder,
                std::abs(model.evaluateValue(xs[index], 0.35, -0.3) -
                         referenceValues[index]));
        }
        convergenceMaxByOrder.push_back(maximumForOrder);
    }

    const double maxProfileError = maximumAbsolute(profileErrors);
    const double sumRuleError = std::abs(forwardIntegral - 2.0);
    const double maxMoment0Error = maximumAbsolute(moment0Errors);
    const double maxMoment1Error = maximumAbsolute(moment1Errors);
    const double maxMoment2Error = maximumAbsolute(moment2Errors);
    const bool convergencePassed =
        convergenceMaxByOrder[1] < convergenceMaxByOrder[0] &&
        convergenceMaxByOrder[2] < convergenceMaxByOrder[1] &&
        convergenceMaxByOrder[2] < 5e-7 &&
        convergenceMaxByOrder[3] == 0.0;

    const bool passed =
        maxProfileError < 2e-11 &&
        sumRuleError < 2e-8 &&
        forwardPoint < 1e-15 &&
        xiSymmetry < 1e-15 &&
        outsideSupport == 0.0 &&
        endpoint < 1e-14 &&
        maximumContinuityJump < 2e-4 &&
        maxMoment0Error < 3e-6 &&
        maxMoment1Error < 3e-6 &&
        maxMoment2Error < 3e-6 &&
        convergencePassed;

    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,"
              << "\"test\":\"stage03_dd_physics\","
              << "\"profile_normalization_max_abs_error\":"
              << maxProfileError
              << ",\"forward_sum_rule_abs_error\":" << sumRuleError
              << ",\"forward_limit_abs_error\":" << forwardPoint
              << ",\"xi_evenness_abs_error\":" << xiSymmetry
              << ",\"outside_support_max_abs\":" << outsideSupport
              << ",\"endpoint_max_abs\":" << endpoint
              << ",\"continuity_max_jump\":" << maximumContinuityJump
              << ",\"moment0_max_abs_error\":" << maxMoment0Error
              << ",\"moment1_max_abs_error\":" << maxMoment1Error
              << ",\"moment2_polynomial_max_abs_error\":"
              << maxMoment2Error
              << ",\"quadrature_max_abs_difference_from_order256\":{"
              << "\"32\":" << convergenceMaxByOrder[0]
              << ",\"64\":" << convergenceMaxByOrder[1]
              << ",\"128\":" << convergenceMaxByOrder[2]
              << ",\"256\":" << convergenceMaxByOrder[3] << "}"
              << ",\"quadrature_convergence_passed\":"
              << (convergencePassed ? "true" : "false")
              << ",\"passed\":" << (passed ? "true" : "false")
              << "}\n";
    return passed ? 0 : 1;
}
