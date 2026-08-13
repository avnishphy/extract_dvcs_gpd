#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/modules/observable/DVCS/DVCSObservable.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

int main() {
    // These are source-label candidates, not convention equivalences. The
    // probe asks the installed PARTONS registry to construct each named class.
    const std::array<std::pair<const char*, const char*>, 34> candidates{{
        {"ALL", "DVCSAllMinus"},
        {"ALLCos0Phi", "DVCSAllMinusCos0Phi"},
        {"ALLCos1Phi", "DVCSAllMinusCos1Phi"},
        {"ALLCos2Phi", "DVCSAllMinusCos2Phi"},
        {"ALTDVCSCosPhiMPhisCos0Phi", "DVCSAltDVCSCosPhiMPhisCos0Phi"},
        {"ALTDVCSCosPhiMPhisCos1Phi", "DVCSAltDVCSCosPhiMPhisCos1Phi"},
        {"ALTDVCSSinPhiMPhisSin1Phi", "DVCSAltDVCSSinPhiMPhisSin1Phi"},
        {"ALTIntCosPhiMPhisCos0Phi", "DVCSAltIntCosPhiMPhisCos0Phi"},
        {"ALTIntCosPhiMPhisCos1Phi", "DVCSAltIntCosPhiMPhisCos1Phi"},
        {"ALTIntCosPhiMPhisCos2Phi", "DVCSAltIntCosPhiMPhisCos2Phi"},
        {"ALTIntSinPhiMPhisSin1Phi", "DVCSAltIntSinPhiMPhisSin1Phi"},
        {"ALTIntSinPhiMPhisSin2Phi", "DVCSAltIntSinPhiMPhisSin2Phi"},
        {"ALU", "DVCSAluMinus"},
        {"ALUDVCSSin1Phi", "DVCSAluDVCSSin1Phi"},
        {"ALUIntSin1Phi", "DVCSAluIntSin1Phi"},
        {"ALUIntSin2Phi", "DVCSAluIntSin2Phi"},
        {"ALUSin1Phi", "DVCSAluMinusSin1Phi"},
        {"ALUSin2Phi", "DVCSAluMinusSin2Phi"},
        {"AUL", "DVCSAulMinus"},
        {"AULSin1Phi", "DVCSAulMinusSin1Phi"},
        {"AULSin2Phi", "DVCSAulMinusSin2Phi"},
        {"AULSin3Phi", "DVCSAulMinusSin3Phi"},
        {"AUTDVCSSinPhiMPhisCos0Phi", "DVCSAutDVCSSinPhiMPhisCos0Phi"},
        {"AUTIntSinPhiMPhisCos0Phi", "DVCSAutIntSinPhiMPhisCos0Phi"},
        {"AUTIntSinPhiMPhisCos1Phi", "DVCSAutIntSinPhiMPhisCos1Phi"},
        {"AUTIntSinPhiMPhisSin1Phi", "DVCSAutIntSinPhiMPhisSin1Phi"},
        {"Ac", "DVCSAc"},
        {"AcCos0Phi", "DVCSAcCos0Phi"},
        {"AcCos1Phi", "DVCSAcCos1Phi"},
        {"AcCos2Phi", "DVCSAcCos2Phi"},
        {"AcCos3Phi", "DVCSAcCos3Phi"},
        {"CrossSectionDifferenceLU", "DVCSCrossSectionDifferenceLUMinus"},
        {"CrossSectionUU", "DVCSCrossSectionUUMinus"},
        {"CrossSectionUUVirtualPhotoProduction",
         "DVCSCrossSectionUUVirtualPhotoProduction"},
    }};

    PARTONS::Partons* partons = PARTONS::Partons::getInstance();
    // The probe is executed from the audited bridge build directory containing
    // partons.properties; pass that explicit location to native initialization.
    std::array<char, 2> programName{'p', '\0'};
    std::array<char, 2> workingDirectory{'.', '\0'};
    std::array<char*, 2> arguments{
        programName.data(), workingDirectory.data()};
    partons->init(static_cast<int>(arguments.size()), arguments.data());
    PARTONS::ModuleObjectFactory* factory =
        partons->getModuleObjectFactory();
    std::size_t constructed = 0;
    for (const auto& [sourceLabel, className] : candidates) {
        PARTONS::DVCSObservable* observable =
            factory->newDVCSObservable(std::string(className));
        if (observable == nullptr || observable->getClassName() != className) {
            throw std::runtime_error(
                std::string("factory mismatch for ") + sourceLabel);
        }
        std::cout << sourceLabel << " -> " << observable->getClassName()
                  << '\n';
        factory->updateModulePointerReference(observable, nullptr);
        ++constructed;
    }
    std::cout << "constructed=" << constructed
              << " source_labels=35 unmapped_pointwise=TSlope" << '\n';
    return constructed == candidates.size() ? 0 : 1;
}
