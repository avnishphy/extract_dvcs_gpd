#include "ConformalMomentGPD.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/beans/QuarkFlavor.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/parton_distribution/PartonDistribution.h>
#include <partons/beans/parton_distribution/QuarkDistribution.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace DVCSInference {

namespace {

constexpr double kUpValenceNumber = 2.0;

}  // namespace

const unsigned int ConformalMomentGPD::classId =
    PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
        new ConformalMomentGPD("ConformalMomentGPD"));

ConformalMomentGPD::ConformalMomentGPD(const std::string& className)
    : PARTONS::GPDModule(className),
      m_parameters{-0.5, 3.0, 1.0},
      m_integrationLimit(1024),
      m_workspace(nullptr) {
    gsl_set_error_handler_off();
    m_MuF2_ref = referenceMuF2;
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::H, &PARTONS::GPDModule::computeH));
    allocateWorkspace();
}

ConformalMomentGPD::ConformalMomentGPD(const ConformalMomentGPD& other)
    : PARTONS::GPDModule(other),
      m_parameters(other.m_parameters),
      m_integrationLimit(other.m_integrationLimit),
      m_workspace(nullptr) {
    allocateWorkspace();
}

ConformalMomentGPD::~ConformalMomentGPD() {
    gsl_integration_workspace_free(m_workspace);
    m_workspace = nullptr;
}

ConformalMomentGPD* ConformalMomentGPD::clone() const {
    return new ConformalMomentGPD(*this);
}

void ConformalMomentGPD::allocateWorkspace() {
    m_workspace = gsl_integration_workspace_alloc(m_integrationLimit);
    if (m_workspace == nullptr) {
        throw std::runtime_error(
            "unable to allocate conformal reconstruction workspace");
    }
}

void ConformalMomentGPD::validateParameters(
        const ConformalMomentParameters& parameters) {
    if (!std::isfinite(parameters.alpha) ||
            parameters.alpha < -1.0 || parameters.alpha > -0.1) {
        throw std::invalid_argument(
            "conformal alpha must be in [-1, -0.1]");
    }
    if (!std::isfinite(parameters.beta) ||
            parameters.beta < 2.0 || parameters.beta > 6.0) {
        throw std::invalid_argument(
            "conformal beta must be in [2, 6]");
    }
    if (!std::isfinite(parameters.trajectorySlope) ||
            parameters.trajectorySlope < 0.0 ||
            parameters.trajectorySlope > 2.0) {
        throw std::invalid_argument(
            "trajectory_slope must be in [0, 2] GeV^-2");
    }
}

void ConformalMomentGPD::setParameters(
        const ConformalMomentParameters& parameters) {
    validateParameters(parameters);
    m_parameters = parameters;
}

const ConformalMomentParameters&
ConformalMomentGPD::getParameters() const {
    return m_parameters;
}

bool ConformalMomentGPD::isAllowedLimit(unsigned int limit) {
    return limit == 256U || limit == 512U || limit == 1024U ||
        limit == 2048U;
}

void ConformalMomentGPD::setIntegrationLimit(unsigned int limit) {
    if (!isAllowedLimit(limit)) {
        throw std::invalid_argument(
            "integration_limit must be one of 256, 512, 1024, 2048");
    }
    if (limit == m_integrationLimit) {
        return;
    }
    gsl_integration_workspace* replacement =
        gsl_integration_workspace_alloc(limit);
    if (replacement == nullptr) {
        throw std::runtime_error(
            "unable to allocate conformal reconstruction workspace");
    }
    gsl_integration_workspace_free(m_workspace);
    m_workspace = replacement;
    m_integrationLimit = limit;
}

unsigned int ConformalMomentGPD::getIntegrationLimit() const {
    return m_integrationLimit;
}

double ConformalMomentGPD::normalization() const {
    const double logBeta =
        std::lgamma(1.0 - m_parameters.alpha) +
        std::lgamma(m_parameters.beta + 1.0) -
        std::lgamma(m_parameters.beta + 2.0 - m_parameters.alpha);
    return kUpValenceNumber * std::exp(-logBeta);
}

double ConformalMomentGPD::conformalMoment(
        unsigned int j, double t) const {
    if (!std::isfinite(t) || t > 0.0 || t < -0.5) {
        throw std::invalid_argument(
            "conformal moment t must be in [-0.5, 0] GeV2");
    }
    const double s = static_cast<double>(j) + 1.0;
    const double shifted = s - m_parameters.alpha;
    const double delta = -m_parameters.trajectorySlope * t;
    const double logBeta =
        std::lgamma(shifted) +
        std::lgamma(m_parameters.beta + 1.0) -
        std::lgamma(shifted + m_parameters.beta + 1.0);
    return normalization() * std::exp(logBeta) *
        shifted / (shifted + delta);
}

double ConformalMomentGPD::convolutionIntegrand(double y, void* raw) {
    const auto& parameters = *static_cast<IntegralParameters*>(raw);
    return std::pow(y, -parameters.delta - 1.0) *
        std::pow(1.0 - y, parameters.beta);
}

double ConformalMomentGPD::evaluateValue(
        double x, double xi, double t) {
    validateParameters(m_parameters);
    if (!std::isfinite(x) || !std::isfinite(xi) || !std::isfinite(t)) {
        throw std::invalid_argument(
            "conformal-family kinematics must be finite");
    }
    if (xi != 0.0) {
        throw std::invalid_argument(
            "Stage 08 conformal reconstruction requires xi=0");
    }
    if (t > 0.0 || t < -0.5) {
        throw std::invalid_argument(
            "Stage 08 conformal reconstruction requires "
            "t in [-0.5, 0] GeV2");
    }
    if (x <= 0.0 || x >= 1.0) {
        return 0.0;
    }

    const double coefficient = normalization();
    const double forward = coefficient *
        std::pow(x, -m_parameters.alpha) *
        std::pow(1.0 - x, m_parameters.beta);
    const double delta = -m_parameters.trajectorySlope * t;
    if (delta == 0.0) {
        return forward;
    }

    IntegralParameters parameters{delta, m_parameters.beta};
    gsl_function function{};
    function.function = &ConformalMomentGPD::convolutionIntegrand;
    function.params = &parameters;
    double integral = 0.0;
    double absoluteError = 0.0;
    const int status = gsl_integration_qag(
        &function, x, 1.0, 1e-12, 1e-10, m_integrationLimit,
        GSL_INTEG_GAUSS61, m_workspace, &integral, &absoluteError);
    if (status != GSL_SUCCESS) {
        throw std::runtime_error(
            "conformal inverse-moment quadrature failed with GSL status " +
            std::to_string(status));
    }
    const double value = forward - delta * coefficient *
        std::pow(x, delta - m_parameters.alpha) * integral;
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "conformal inverse-moment reconstruction was non-finite");
    }
    return value;
}

void ConformalMomentGPD::initModule() {
    PARTONS::GPDModule::initModule();
}

void ConformalMomentGPD::isModuleWellConfigured() {
    PARTONS::GPDModule::isModuleWellConfigured();
    validateParameters(m_parameters);
    if (!isAllowedLimit(m_integrationLimit)) {
        throw std::runtime_error(
            "invalid conformal reconstruction integration limit");
    }
}

PARTONS::PartonDistribution ConformalMomentGPD::computeH() {
    const double valueAtX = evaluateValue(m_x, m_xi, m_t);
    const double valueAtMinusX = evaluateValue(-m_x, m_xi, m_t);
    PARTONS::PartonDistribution result;
    result.addQuarkDistribution(PARTONS::QuarkDistribution(
        PARTONS::QuarkFlavor::UP,
        valueAtX,
        valueAtX - valueAtMinusX,
        valueAtX + valueAtMinusX));
    return result;
}

}  // namespace DVCSInference
