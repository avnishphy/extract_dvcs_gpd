#include "TabulatedPseudodataGPD.h"

#include <ElementaryUtils/logger/CustomException.h>
#include <partons/BaseObjectRegistry.h>
#include <partons/beans/QuarkFlavor.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/parton_distribution/GluonDistribution.h>
#include <partons/beans/parton_distribution/PartonDistribution.h>
#include <partons/beans/parton_distribution/QuarkDistribution.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace DVCSInference {

const unsigned int TabulatedPseudodataGPD::classId =
    PARTONS::BaseObjectRegistry::getInstance()->registerBaseObject(
        new TabulatedPseudodataGPD("TabulatedPseudodataGPD"));

TabulatedPseudodataGPD::TabulatedPseudodataGPD(const std::string& className)
    : PARTONS::GPDModule(className), m_tables{} {
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::H, &PARTONS::GPDModule::computeH));
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::E, &PARTONS::GPDModule::computeE));
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::Ht, &PARTONS::GPDModule::computeHt));
    m_listGPDComputeTypeAvailable.insert(
        std::make_pair(PARTONS::GPDType::Et, &PARTONS::GPDModule::computeEt));
}

TabulatedPseudodataGPD::TabulatedPseudodataGPD(
        const TabulatedPseudodataGPD& other)
    : PARTONS::GPDModule(other), m_tables(other.m_tables) {}

TabulatedPseudodataGPD* TabulatedPseudodataGPD::clone() const {
    return new TabulatedPseudodataGPD(*this);
}

void TabulatedPseudodataGPD::setTable(
        std::size_t typeIndex, std::shared_ptr<Table> table) {
    if (typeIndex >= m_tables.size() || table == nullptr) {
        throw std::invalid_argument("invalid native evolution table");
    }
    m_tables[typeIndex] = std::move(table);
}

void TabulatedPseudodataGPD::initModule() {
    PARTONS::GPDModule::initModule();
}

void TabulatedPseudodataGPD::isModuleWellConfigured() {
    PARTONS::GPDModule::isModuleWellConfigured();
    for (const auto& table : m_tables) {
        if (table == nullptr) {
            throw ElemUtils::CustomException(
                getClassName(), __func__, "native evolution table not set");
        }
    }
}

PARTONS::PartonDistribution TabulatedPseudodataGPD::computeH() {
    return computeType(0U);
}

PARTONS::PartonDistribution TabulatedPseudodataGPD::computeE() {
    return computeType(1U);
}

PARTONS::PartonDistribution TabulatedPseudodataGPD::computeHt() {
    return computeType(2U);
}

PARTONS::PartonDistribution TabulatedPseudodataGPD::computeEt() {
    return computeType(3U);
}

PARTONS::PartonDistribution TabulatedPseudodataGPD::computeType(
        std::size_t typeIndex) {
    const double positiveX = std::max(std::abs(m_x), 1.e-7);
    const auto evolved = m_tables.at(typeIndex)->Evaluate(std::sqrt(m_MuF2));
    const auto physical = apfel::QCDEvToPhys(evolved.GetObjects());
    std::map<PARTONS::QuarkFlavor::Type, PARTONS::QuarkDistribution> quarks;
    for (int apfelFlavor = 1; apfelFlavor <= 6; ++apfelFlavor) {
        int partonsFlavor = apfelFlavor;
        if (apfelFlavor == 1) {
            partonsFlavor = 2;
        } else if (apfelFlavor == 2) {
            partonsFlavor = 1;
        }
        const double q = physical.at(apfelFlavor).Evaluate(positiveX)
            / positiveX;
        const double qbar = physical.at(-apfelFlavor).Evaluate(positiveX)
            / positiveX;
        const auto flavor =
            static_cast<PARTONS::QuarkFlavor::Type>(partonsFlavor);
        if (m_x >= 0.) {
            quarks.emplace(flavor, PARTONS::QuarkDistribution(
                flavor, q, q + qbar, q - qbar));
        } else {
            quarks.emplace(flavor, PARTONS::QuarkDistribution(
                flavor, -qbar, -(q + qbar), q - qbar));
        }
    }
    PARTONS::PartonDistribution result;
    result.setQuarkDistributions(quarks);
    result.setGluonDistribution(PARTONS::GluonDistribution(
        physical.at(0).Evaluate(positiveX)));
    return result;
}

}  // namespace DVCSInference
