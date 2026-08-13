#include "FixedScaleDDGPD.h"

#include <NumA/functor/one_dimension/Functor1D.h>
#include <NumA/integration/one_dimension/GaussLegendreIntegrator1D.h>
#include <NumA/integration/one_dimension/Integrator1D.h>
#include <NumA/integration/one_dimension/IntegratorType1D.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/beans/QuarkFlavor.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/parton_distribution/PartonDistribution.h>
#include <partons/beans/parton_distribution/QuarkDistribution.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace DVCSInference {

namespace {

void validateParameters(const DDParameters& parameters) {
    if (!std::isfinite(parameters.normalization) ||
            parameters.normalization < -4.0 ||
            parameters.normalization > 4.0) {
        throw std::invalid_argument(
            "DD normalization must be in [-4, 4]");
    }
    if (!std::isfinite(parameters.a) || parameters.a < 0.1 ||
            parameters.a > 1.0) {
        throw std::invalid_argument("DD parameter a must be in [0.1, 1]");
    }
    if (!std::isfinite(parameters.c) || parameters.c < 2.0 ||
            parameters.c > 6.0) {
        throw std::invalid_argument("DD parameter c must be in [2, 6]");
    }
    if (!std::isfinite(parameters.profileB) ||
            parameters.profileB < 1.0 || parameters.profileB > 4.0) {
        throw std::invalid_argument(
            "DD parameter profile_b must be in [1, 4]");
    }
    if (!std::isfinite(parameters.tSlope) ||
            parameters.tSlope < 0.0 || parameters.tSlope > 2.0) {
        throw std::invalid_argument(
            "DD parameter t_slope must be in [0, 2] GeV^-2");
    }
}

}  // namespace

const unsigned int FixedScaleDDGPD::classId =
    PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
        new FixedScaleDDGPD("FixedScaleDDGPD"));

FixedScaleDDGPD::FixedScaleDDGPD(const std::string& className)
    : PARTONS::GPDModule(className),
      PARTONS::MathIntegratorModule(),
      m_parameters{2.0, 0.5, 3.0, 2.0, 1.0},
      m_quadratureOrder(128),
      m_integrationX(0.0),
      m_integrationXi(0.0),
      m_integrationT(0.0),
      m_integrandFunctor(nullptr) {
    m_MuF2_ref = referenceMuF2;
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::H, &PARTONS::GPDModule::computeH));
    initializeIntegrator();
}

FixedScaleDDGPD::FixedScaleDDGPD(const FixedScaleDDGPD& other)
    : PARTONS::GPDModule(other),
      PARTONS::MathIntegratorModule(other),
      m_parameters(other.m_parameters),
      m_quadratureOrder(other.m_quadratureOrder),
      m_integrationX(0.0),
      m_integrationXi(0.0),
      m_integrationT(0.0),
      m_integrandFunctor(nullptr) {
    m_integrandFunctor = NumA::Integrator1D::newIntegrationFunctor(
        this, &FixedScaleDDGPD::integrand);
    setQuadratureOrder(m_quadratureOrder);
}

FixedScaleDDGPD::~FixedScaleDDGPD() {
    delete m_integrandFunctor;
    m_integrandFunctor = nullptr;
}

FixedScaleDDGPD* FixedScaleDDGPD::clone() const {
    return new FixedScaleDDGPD(*this);
}

void FixedScaleDDGPD::initializeIntegrator() {
    setIntegrator(NumA::IntegratorType1D::GL);
    m_integrandFunctor = NumA::Integrator1D::newIntegrationFunctor(
        this, &FixedScaleDDGPD::integrand);
    setQuadratureOrder(m_quadratureOrder);
}

void FixedScaleDDGPD::setParameters(const DDParameters& parameters) {
    validateParameters(parameters);
    m_parameters = parameters;
}

const DDParameters& FixedScaleDDGPD::getParameters() const {
    return m_parameters;
}

bool FixedScaleDDGPD::isAllowedOrder(unsigned int order) {
    return order == 32U || order == 64U || order == 128U || order == 256U;
}

void FixedScaleDDGPD::setQuadratureOrder(unsigned int order) {
    if (!isAllowedOrder(order)) {
        throw std::invalid_argument(
            "quadrature_order must be one of 32, 64, 128, 256");
    }
    auto* integrator = dynamic_cast<NumA::GaussLegendreIntegrator1D*>(
        getMathIntegrator());
    if (integrator == nullptr) {
        throw std::runtime_error(
            "FixedScaleDDGPD requires NumA Gauss-Legendre integration");
    }
    integrator->setN(order);
    m_quadratureOrder = order;
}

unsigned int FixedScaleDDGPD::getQuadratureOrder() const {
    return m_quadratureOrder;
}

double FixedScaleDDGPD::forwardValue(double beta, double t) const {
    if (beta <= 0.0 || beta >= 1.0) {
        return 0.0;
    }
    const double logNormalization =
        std::lgamma(m_parameters.a + m_parameters.c + 2.0) -
        std::lgamma(m_parameters.a + 1.0) -
        std::lgamma(m_parameters.c + 1.0);
    return m_parameters.normalization * std::exp(
        logNormalization + m_parameters.a * std::log(beta) +
        m_parameters.c * std::log1p(-beta) +
        m_parameters.tSlope * t);
}

double FixedScaleDDGPD::profileValue(double beta, double alpha) const {
    if (beta < 0.0 || beta >= 1.0 ||
            std::abs(alpha) > 1.0 - beta) {
        return 0.0;
    }
    const double width = 1.0 - beta;
    const double shape =
        std::max(0.0, width * width - alpha * alpha);
    if (shape == 0.0) {
        return 0.0;
    }
    const double logCoefficient =
        std::lgamma(m_parameters.profileB + 1.5) -
        0.5 * std::log(std::acos(-1.0)) -
        std::lgamma(m_parameters.profileB + 1.0);
    return std::exp(
        logCoefficient +
        m_parameters.profileB * std::log(shape) -
        (2.0 * m_parameters.profileB + 1.0) * std::log(width));
}

double FixedScaleDDGPD::integrand(
        double beta, std::vector<double> parameters) {
    (void)parameters;
    const double alpha = (m_integrationX - beta) / m_integrationXi;
    return forwardValue(beta, m_integrationT) *
        profileValue(beta, alpha) / m_integrationXi;
}

double FixedScaleDDGPD::evaluateValue(double x, double xi, double t) {
    if (!std::isfinite(x) || !std::isfinite(xi) || !std::isfinite(t)) {
        throw std::invalid_argument("DD kinematics must be finite");
    }
    if (std::abs(x) > 1.0) {
        return 0.0;
    }
    if (std::abs(xi) > 1.0) {
        throw std::invalid_argument("|xi| must be <= 1");
    }
    if (t > 0.0) {
        throw std::invalid_argument("t must be non-positive GeV2");
    }

    const double absoluteXi = std::abs(xi);
    if (absoluteXi == 0.0) {
        return forwardValue(x, t);
    }

    double lower = 0.0;
    double upper = 0.0;
    if (absoluteXi == 1.0) {
        upper = std::min(1.0, (x + 1.0) / 2.0);
    } else {
        lower = std::max(0.0, (x - absoluteXi) / (1.0 - absoluteXi));
        upper = std::min(1.0, (x + absoluteXi) / (1.0 + absoluteXi));
    }
    if (upper <= lower) {
        return 0.0;
    }

    m_integrationX = x;
    m_integrationXi = absoluteXi;
    m_integrationT = t;
    std::vector<double> emptyParameters;
    const double result =
        integrate(m_integrandFunctor, lower, upper, emptyParameters);
    if (!std::isfinite(result)) {
        throw std::runtime_error("DD quadrature returned a non-finite value");
    }
    return result;
}

void FixedScaleDDGPD::initModule() {
    PARTONS::GPDModule::initModule();
}

void FixedScaleDDGPD::isModuleWellConfigured() {
    PARTONS::GPDModule::isModuleWellConfigured();
    validateParameters(m_parameters);
    if (!isAllowedOrder(m_quadratureOrder)) {
        throw std::runtime_error("invalid DD quadrature order");
    }
}

PARTONS::PartonDistribution FixedScaleDDGPD::computeH() {
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
