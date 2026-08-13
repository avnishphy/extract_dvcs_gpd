#include "ShadowInjectionGPD.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/ModuleObjectFactory.h>
#include <partons/beans/QuarkFlavor.h>
#include <partons/beans/gpd/GPDKinematic.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/parton_distribution/GluonDistribution.h>
#include <partons/beans/parton_distribution/PartonDistribution.h>
#include <partons/beans/parton_distribution/QuarkDistribution.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace DVCSInference {

const unsigned int ShadowInjectionGPD::classId =
    PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
        new ShadowInjectionGPD("ShadowInjectionGPD"));

ShadowInjectionGPD::ShadowInjectionGPD(const std::string& className)
    : PARTONS::GPDModule(className),
      m_baseline(nullptr),
      m_shadow(nullptr),
      m_coefficients{1.0, 0.0} {
    m_MuF2_ref = referenceMuF2;
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::H, &PARTONS::GPDModule::computeH));
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::E, &PARTONS::GPDModule::computeE));
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::Ht, &PARTONS::GPDModule::computeHt));
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::Et, &PARTONS::GPDModule::computeEt));
}

ShadowInjectionGPD::ShadowInjectionGPD(const ShadowInjectionGPD& other)
    : PARTONS::GPDModule(other),
      m_baseline(nullptr),
      m_shadow(nullptr),
      m_coefficients(other.m_coefficients) {
    if (other.m_baseline != nullptr) {
        m_baseline =
            m_pModuleObjectFactory->cloneModuleObject(other.m_baseline);
    }
    if (other.m_shadow != nullptr) {
        m_shadow = m_pModuleObjectFactory->cloneModuleObject(other.m_shadow);
    }
}

ShadowInjectionGPD::~ShadowInjectionGPD() {
    if (m_baseline != nullptr) {
        setBaselineModule(nullptr);
    }
    if (m_shadow != nullptr) {
        setShadowModule(nullptr);
    }
}

ShadowInjectionGPD* ShadowInjectionGPD::clone() const {
    return new ShadowInjectionGPD(*this);
}

void ShadowInjectionGPD::setBaselineModule(PARTONS::GPDModule* module) {
    m_pModuleObjectFactory->updateModulePointerReference(m_baseline, module);
    m_baseline = module;
}

void ShadowInjectionGPD::setShadowModule(PARTONS::GPDModule* module) {
    m_pModuleObjectFactory->updateModulePointerReference(m_shadow, module);
    m_shadow = module;
}

void ShadowInjectionGPD::validateCoefficients(
        const ShadowCoefficients& coefficients) {
    if (!std::isfinite(coefficients.baseline) ||
            coefficients.baseline < 0.8 || coefficients.baseline > 1.2) {
        throw std::invalid_argument(
            "baseline coefficient must be finite in [0.8, 1.2]");
    }
    if (!std::isfinite(coefficients.shadow) ||
            coefficients.shadow < -1.0 || coefficients.shadow > 1.0) {
        throw std::invalid_argument(
            "shadow coefficient must be finite in [-1, 1]");
    }
}

void ShadowInjectionGPD::setCoefficients(
        const ShadowCoefficients& coefficients) {
    validateCoefficients(coefficients);
    m_coefficients = coefficients;
}

const ShadowCoefficients& ShadowInjectionGPD::getCoefficients() const {
    return m_coefficients;
}

void ShadowInjectionGPD::initModule() {
    PARTONS::GPDModule::initModule();
}

void ShadowInjectionGPD::isModuleWellConfigured() {
    PARTONS::GPDModule::isModuleWellConfigured();
    validateCoefficients(m_coefficients);
    if (m_baseline == nullptr || m_shadow == nullptr) {
        throw ElemUtils::CustomException(
            getClassName(), __func__, "component module not set");
    }
}

PARTONS::PartonDistribution ShadowInjectionGPD::computeH() {
    if (m_baseline == nullptr || m_shadow == nullptr) {
        throw ElemUtils::CustomException(
            getClassName(), __func__, "component module not set");
    }
    const PARTONS::GPDKinematic kinematics(
        m_x, m_xi, m_t, m_MuF2, m_MuR2);
    const PARTONS::PartonDistribution baseline =
        m_baseline->compute(kinematics, PARTONS::GPDType::H);
    const PARTONS::PartonDistribution shadow =
        m_shadow->compute(kinematics, PARTONS::GPDType::H);

    PARTONS::PartonDistribution combined;
    for (const PARTONS::QuarkFlavor::Type flavor :
            baseline.listTypeOfQuarkFlavor()) {
        const PARTONS::QuarkDistribution& baseInput =
            baseline.getQuarkDistribution(flavor);
        const PARTONS::QuarkDistribution& shadowInput =
            shadow.getQuarkDistribution(flavor);
        combined.addQuarkDistribution(PARTONS::QuarkDistribution(
            flavor,
            m_coefficients.baseline * baseInput.getQuarkDistribution() +
                m_coefficients.shadow *
                    shadowInput.getQuarkDistribution(),
            m_coefficients.baseline *
                    baseInput.getQuarkDistributionPlus() +
                m_coefficients.shadow *
                    shadowInput.getQuarkDistributionPlus(),
            m_coefficients.baseline *
                    baseInput.getQuarkDistributionMinus() +
                m_coefficients.shadow *
                    shadowInput.getQuarkDistributionMinus()));
    }
    combined.setGluonDistribution(PARTONS::GluonDistribution(
        m_coefficients.baseline *
                baseline.getGluonDistribution().getGluonDistribution() +
            m_coefficients.shadow *
                shadow.getGluonDistribution().getGluonDistribution()));
    return combined;
}

PARTONS::PartonDistribution ShadowInjectionGPD::computeScaledBaseline(
        PARTONS::GPDType::Type type) {
    if (m_baseline == nullptr) {
        throw ElemUtils::CustomException(
            getClassName(), __func__, "baseline module not set");
    }
    const PARTONS::PartonDistribution native = m_baseline->compute(
        PARTONS::GPDKinematic(m_x, m_xi, m_t, m_MuF2, m_MuR2), type);

    PARTONS::PartonDistribution scaled;
    for (const PARTONS::QuarkFlavor::Type flavor :
            native.listTypeOfQuarkFlavor()) {
        const PARTONS::QuarkDistribution& input =
            native.getQuarkDistribution(flavor);
        scaled.addQuarkDistribution(PARTONS::QuarkDistribution(
            flavor,
            m_coefficients.baseline * input.getQuarkDistribution(),
            m_coefficients.baseline *
                input.getQuarkDistributionPlus(),
            m_coefficients.baseline *
                input.getQuarkDistributionMinus()));
    }
    scaled.setGluonDistribution(PARTONS::GluonDistribution(
        m_coefficients.baseline *
            native.getGluonDistribution().getGluonDistribution()));
    return scaled;
}

PARTONS::PartonDistribution ShadowInjectionGPD::computeE() {
    return computeScaledBaseline(PARTONS::GPDType::E);
}

PARTONS::PartonDistribution ShadowInjectionGPD::computeHt() {
    return computeScaledBaseline(PARTONS::GPDType::Ht);
}

PARTONS::PartonDistribution ShadowInjectionGPD::computeEt() {
    return computeScaledBaseline(PARTONS::GPDType::Et);
}

}  // namespace DVCSInference
