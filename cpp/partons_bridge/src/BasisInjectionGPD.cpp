#include "BasisInjectionGPD.h"

#include <partons/BaseObjectRegistry.h>
#include <partons/beans/QuarkFlavor.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/parton_distribution/GluonDistribution.h>
#include <partons/beans/parton_distribution/PartonDistribution.h>
#include <partons/beans/parton_distribution/QuarkDistribution.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace DVCSInference {

const unsigned int BasisInjectionGPD::classId =
    PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
        new BasisInjectionGPD("BasisInjectionGPD"));

BasisInjectionGPD::BasisInjectionGPD(const std::string& className)
    : PARTONS::GPDModule(className),
      m_shape("BasisInjectionDDShape"),
      m_amplitudes{1.0, 0.0, 0.0} {
    m_MuF2_ref = referenceMuF2;
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::H, &PARTONS::GPDModule::computeH));
}

BasisInjectionGPD::BasisInjectionGPD(const BasisInjectionGPD& other)
    : PARTONS::GPDModule(other),
      m_shape("BasisInjectionDDShape"),
      m_amplitudes(other.m_amplitudes) {
    m_shape.setParameters(other.m_shape.getParameters());
    m_shape.setQuadratureOrder(other.m_shape.getQuadratureOrder());
}

BasisInjectionGPD::~BasisInjectionGPD() = default;

BasisInjectionGPD* BasisInjectionGPD::clone() const {
    return new BasisInjectionGPD(*this);
}

void BasisInjectionGPD::setShapeParameters(const DDParameters& parameters) {
    m_shape.setParameters(parameters);
}

const DDParameters& BasisInjectionGPD::getShapeParameters() const {
    return m_shape.getParameters();
}

void BasisInjectionGPD::setQuadratureOrder(unsigned int order) {
    m_shape.setQuadratureOrder(order);
}

unsigned int BasisInjectionGPD::getQuadratureOrder() const {
    return m_shape.getQuadratureOrder();
}

void BasisInjectionGPD::validateAmplitudes(
        const BasisAmplitudes& amplitudes) {
    const auto valid = [](double value) {
        return std::isfinite(value) && value >= -4.0 && value <= 4.0;
    };
    if (!valid(amplitudes.sigma) || !valid(amplitudes.t3) ||
            !valid(amplitudes.gluon)) {
        throw std::invalid_argument(
            "basis amplitudes sigma, t3, and gluon must be finite in [-4, 4]");
    }
}

void BasisInjectionGPD::setAmplitudes(
        const BasisAmplitudes& amplitudes) {
    validateAmplitudes(amplitudes);
    m_amplitudes = amplitudes;
}

const BasisAmplitudes& BasisInjectionGPD::getAmplitudes() const {
    return m_amplitudes;
}

void BasisInjectionGPD::initModule() {
    PARTONS::GPDModule::initModule();
}

void BasisInjectionGPD::isModuleWellConfigured() {
    PARTONS::GPDModule::isModuleWellConfigured();
    validateAmplitudes(m_amplitudes);
}

PARTONS::PartonDistribution BasisInjectionGPD::computeH() {
    const double absoluteX = std::abs(m_x);
    const double shape =
        m_shape.evaluateValue(absoluteX, std::abs(m_xi), m_t);
    const double xSign = m_x < 0.0 ? -1.0 : 1.0;

    const double upPlusPositive =
        shape * (m_amplitudes.sigma / 3.0 + m_amplitudes.t3 / 2.0);
    const double downPlusPositive =
        shape * (m_amplitudes.sigma / 3.0 - m_amplitudes.t3 / 2.0);
    const double strangePlusPositive = shape * m_amplitudes.sigma / 3.0;

    PARTONS::PartonDistribution result;
    const auto addSeaFlavor =
        [&result, xSign](PARTONS::QuarkFlavor::Type flavor,
                        double positivePlus) {
            const double plus = xSign * positivePlus;
            result.addQuarkDistribution(PARTONS::QuarkDistribution(
                flavor, 0.5 * plus, plus, 0.0));
        };
    addSeaFlavor(PARTONS::QuarkFlavor::UP, upPlusPositive);
    addSeaFlavor(PARTONS::QuarkFlavor::DOWN, downPlusPositive);
    addSeaFlavor(PARTONS::QuarkFlavor::STRANGE, strangePlusPositive);

    // GPDEvolutionApfel passes the gluon field directly, while it multiplies
    // quark and antiquark fields by x. Store x*H_g so all injected evolution
    // basis components share the same shape convention.
    result.setGluonDistribution(PARTONS::GluonDistribution(
        m_amplitudes.gluon * absoluteX * shape));
    return result;
}

}  // namespace DVCSInference
