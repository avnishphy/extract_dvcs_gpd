#include "PseudodataInputGPD.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/beans/QuarkFlavor.h>
#include <partons/beans/gpd/GPDKinematic.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/parton_distribution/GluonDistribution.h>
#include <partons/beans/parton_distribution/PartonDistribution.h>
#include <partons/beans/parton_distribution/QuarkDistribution.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace DVCSInference {

namespace {

PseudodataInputParameters defaultParameters() {
    const std::array<DDParameters, 4> byType{
        DDParameters{2.0, 0.55, 3.8, 1.0, 0.8},
        DDParameters{1.0, 0.70, 4.2, 1.5, 1.0},
        DDParameters{1.27, 0.45, 3.2, 1.0, 0.7},
        DDParameters{0.8, 0.60, 4.5, 2.0, 1.2}};
    PseudodataInputParameters result{};
    for (std::size_t type = 0; type < 4; ++type) {
        result.ddByTypeAndChannel[type] = {
            byType[type], byType[type], byType[type], byType[type]};
        result.shadowAmplitudeByTypeAndChannel[type] = {1.0, 0.5, 0.25, 1.0};
    }
    result.shadowCoefficientByType = {0.0, 0.0, 0.0, 0.0};
    return result;
}

}  // namespace

const unsigned int PseudodataInputGPD::classId =
    PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
        new PseudodataInputGPD("PseudodataInputGPD"));

PseudodataInputGPD::PseudodataInputGPD(const std::string& className)
    : PARTONS::GPDModule(className),
      m_dd(nullptr),
      m_shadow(nullptr),
      m_parameters(defaultParameters()),
      m_quadratureOrder(128U) {
    m_MuF2_ref = defaultReferenceMuF2;
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::H, &PARTONS::GPDModule::computeH));
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::E, &PARTONS::GPDModule::computeE));
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::Ht, &PARTONS::GPDModule::computeHt));
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::Et, &PARTONS::GPDModule::computeEt));
}

PseudodataInputGPD::PseudodataInputGPD(
        const PseudodataInputGPD& other)
    : PARTONS::GPDModule(other),
      m_dd(nullptr),
      m_shadow(nullptr),
      m_parameters(other.m_parameters),
      m_quadratureOrder(other.m_quadratureOrder) {
    if (other.m_dd != nullptr) {
        m_dd = m_pModuleObjectFactory->cloneModuleObject(other.m_dd);
    }
    if (other.m_shadow != nullptr) {
        m_shadow =
            m_pModuleObjectFactory->cloneModuleObject(other.m_shadow);
    }
}

PseudodataInputGPD::~PseudodataInputGPD() {
    if (m_dd != nullptr) {
        setDDModule(nullptr);
    }
    if (m_shadow != nullptr) {
        setShadowModule(nullptr);
    }
}

PseudodataInputGPD* PseudodataInputGPD::clone() const {
    return new PseudodataInputGPD(*this);
}

void PseudodataInputGPD::setDDModule(FixedScaleDDGPD* module) {
    m_pModuleObjectFactory->updateModulePointerReference(m_dd, module);
    m_dd = module;
    if (m_dd != nullptr) {
        m_dd->setParameters(m_parameters.ddByTypeAndChannel[0][0]);
        m_dd->setQuadratureOrder(m_quadratureOrder);
    }
}

void PseudodataInputGPD::setShadowModule(PARTONS::GPDModule* module) {
    m_pModuleObjectFactory->updateModulePointerReference(m_shadow, module);
    m_shadow = module;
}

void PseudodataInputGPD::validateParameters(
        const PseudodataInputParameters& parameters) {
    for (const auto& channels : parameters.ddByTypeAndChannel) {
      for (const DDParameters& dd : channels) {
        if (!std::isfinite(dd.normalization) ||
                dd.normalization < -4.0 || dd.normalization > 4.0) {
            throw std::invalid_argument(
                "normalization must be finite in [-4, 4]");
        }
        if (!std::isfinite(dd.a) || dd.a <= 0.1 || dd.a >= 1.0) {
            throw std::invalid_argument("a must be finite in (0.1, 1.0)");
        }
        if (!std::isfinite(dd.c) || dd.c <= 2.0 || dd.c >= 6.0) {
            throw std::invalid_argument("c must be finite in (2.0, 6.0)");
        }
        if (!std::isfinite(dd.profileB) ||
                dd.profileB < 1.0 || dd.profileB > 4.0) {
            throw std::invalid_argument(
                "profile_b must be finite in [1, 4]");
        }
        if (!std::isfinite(dd.tSlope) ||
                dd.tSlope < 0.0 || dd.tSlope > 2.0) {
            throw std::invalid_argument(
                "t_slope must be finite in [0, 2] GeV^-2");
        }
      }
    }
    for (std::size_t type = 0; type < 4; ++type) {
        const double coefficient = parameters.shadowCoefficientByType[type];
        if (!std::isfinite(coefficient) || coefficient < -1.0 ||
                coefficient > 1.0) {
            throw std::invalid_argument(
                "shadow coefficients must be finite in [-1, 1]");
        }
        for (double amplitude :
                parameters.shadowAmplitudeByTypeAndChannel[type]) {
            if (!std::isfinite(amplitude) || amplitude < -4.0 ||
                    amplitude > 4.0) {
                throw std::invalid_argument(
                    "shadow channel amplitudes must be finite in [-4, 4]");
            }
        }
    }
}

void PseudodataInputGPD::setParameters(
        const PseudodataInputParameters& parameters) {
    validateParameters(parameters);
    m_parameters = parameters;
    if (m_dd != nullptr) {
        m_dd->setParameters(parameters.ddByTypeAndChannel[0][0]);
    }
}

const PseudodataInputParameters&
PseudodataInputGPD::getParameters() const {
    return m_parameters;
}

void PseudodataInputGPD::setQuadratureOrder(unsigned int order) {
    // The Stage 06 evolved route froze 128-point input quadrature. Stage 10
    // keeps that exact numerical contract so a user cannot silently trade
    // accuracy for speed through configuration.
    if (order != 128U) {
        throw std::invalid_argument("quadrature_order must equal 128");
    }
    m_quadratureOrder = order;
    if (m_dd != nullptr) {
        m_dd->setQuadratureOrder(order);
    }
}

unsigned int PseudodataInputGPD::getQuadratureOrder() const {
    return m_quadratureOrder;
}

void PseudodataInputGPD::setReferenceMuF2(double value) {
    // APFEL++ tabulates in Q=sqrt(Q2) from 1 GeV upward in the locally
    // audited wrapper.  The milestone declares Q0^2=1 GeV2 and rejects
    // backward evolution instead of evolving measurements.
    if (!std::isfinite(value) || value != defaultReferenceMuF2) {
        throw std::invalid_argument("q0_squared must equal 1 GeV2");
    }
    m_MuF2_ref = value;
}

void PseudodataInputGPD::initModule() {
    PARTONS::GPDModule::initModule();
}

void PseudodataInputGPD::isModuleWellConfigured() {
    PARTONS::GPDModule::isModuleWellConfigured();
    validateParameters(m_parameters);
    if (m_dd == nullptr || m_shadow == nullptr) {
        throw ElemUtils::CustomException(
            getClassName(), __func__, "component module not set");
    }
    if (m_dd->getQuadratureOrder() != m_quadratureOrder) {
        throw ElemUtils::CustomException(
            getClassName(), __func__, "DD quadrature contract mismatch");
    }
}

PARTONS::PartonDistribution PseudodataInputGPD::computeH() {
    return computeDD(0U, PARTONS::GPDType::H);
}

PARTONS::PartonDistribution PseudodataInputGPD::computeE() {
    return computeDD(1U, PARTONS::GPDType::E);
}

PARTONS::PartonDistribution PseudodataInputGPD::computeHt() {
    return computeDD(2U, PARTONS::GPDType::Ht);
}

PARTONS::PartonDistribution PseudodataInputGPD::computeEt() {
    return computeDD(3U, PARTONS::GPDType::Et);
}

double PseudodataInputGPD::evaluateProjectShadow(
        double x, double xi, double t) {
    // A difference of two unit-normalized DDs has a zero forward zeroth
    // moment. Support and polynomiality are inherited from the DD integral.
    // It is a stress-test deformation, not a verified DVCS CFF-null mode.
    const DDParameters first{1.0, 0.35, 5.2, 1.0, 0.8};
    const DDParameters second{1.0, 0.75, 3.0, 2.0, 0.8};
    m_dd->setParameters(first);
    const double firstValue = m_dd->evaluateValue(x, xi, t);
    m_dd->setParameters(second);
    return firstValue - m_dd->evaluateValue(x, xi, t);
}

PARTONS::PartonDistribution PseudodataInputGPD::computeDD(
        std::size_t parameterIndex, PARTONS::GPDType::Type type) {
    if (m_dd == nullptr || m_shadow == nullptr) {
        throw ElemUtils::CustomException(
            getClassName(), __func__, "component module not set");
    }

    // When APFEL++ requests the initial distribution, GPDModule supplies this
    // method with muF^2=Q0^2. Direct Q0 evaluation follows the same route.
    const PARTONS::GPDKinematic kinematics(
        m_x, m_xi, m_t, m_MuF2, m_MuR2);
    const double shadowCoefficient =
        m_parameters.shadowCoefficientByType.at(parameterIndex);
    const PARTONS::PartonDistribution nativeShadow = parameterIndex == 0U
        ? m_shadow->compute(kinematics, type)
        : PARTONS::PartonDistribution();

    PARTONS::PartonDistribution combined;
    const std::array<PARTONS::QuarkFlavor::Type, 3> flavors{
        PARTONS::QuarkFlavor::UP,
        PARTONS::QuarkFlavor::DOWN,
        PARTONS::QuarkFlavor::STRANGE};
    const auto& nativeQuarks = nativeShadow.getQuarkDistributions();
    for (std::size_t channel = 0; channel < flavors.size(); ++channel) {
        const PARTONS::QuarkFlavor::Type flavor = flavors[channel];
        m_dd->setParameters(
            m_parameters.ddByTypeAndChannel.at(parameterIndex).at(channel));
        double valueAtX = m_dd->evaluateValue(m_x, m_xi, m_t);
        double valueAtMinusX = m_dd->evaluateValue(-m_x, m_xi, m_t);
        const double amplitude = m_parameters
            .shadowAmplitudeByTypeAndChannel.at(parameterIndex).at(channel);
        if (!(parameterIndex == 0U && channel == 0U)) {
            valueAtX += shadowCoefficient * amplitude *
                evaluateProjectShadow(m_x, m_xi, m_t);
            valueAtMinusX += shadowCoefficient * amplitude *
                evaluateProjectShadow(-m_x, m_xi, m_t);
        }
        const auto nativeFound = nativeQuarks.find(flavor);
        const PARTONS::QuarkDistribution nativeInput =
            nativeFound == nativeQuarks.end()
                ? PARTONS::QuarkDistribution(flavor, 0.0, 0.0, 0.0)
                : nativeFound->second;
        combined.addQuarkDistribution(PARTONS::QuarkDistribution(
            flavor,
            valueAtX + shadowCoefficient * amplitude *
                nativeInput.getQuarkDistribution(),
            valueAtX - valueAtMinusX + shadowCoefficient * amplitude *
                nativeInput.getQuarkDistributionPlus(),
            valueAtX + valueAtMinusX + shadowCoefficient * amplitude *
                nativeInput.getQuarkDistributionMinus()));
    }

    const std::size_t gluonChannel = 3U;
    m_dd->setParameters(m_parameters.ddByTypeAndChannel
        .at(parameterIndex).at(gluonChannel));
    const double absoluteX = std::abs(m_x);
    double gluon = absoluteX *
        m_dd->evaluateValue(absoluteX, std::abs(m_xi), m_t);
    const double gluonAmplitude = m_parameters
        .shadowAmplitudeByTypeAndChannel.at(parameterIndex).at(gluonChannel);
    gluon += shadowCoefficient * gluonAmplitude * absoluteX *
        evaluateProjectShadow(absoluteX, std::abs(m_xi), m_t);
    combined.setGluonDistribution(PARTONS::GluonDistribution(
        gluon));
    return combined;
}

}  // namespace DVCSInference
