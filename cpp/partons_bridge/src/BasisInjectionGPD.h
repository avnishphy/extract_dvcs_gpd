#ifndef EXTRACT_DVCS_CFF_BASIS_INJECTION_GPD_H
#define EXTRACT_DVCS_CFF_BASIS_INJECTION_GPD_H

#include "FixedScaleDDGPD.h"

#include <partons/modules/gpd/GPDModule.h>

#include <string>

namespace DVCSInference {

struct BasisAmplitudes {
    double sigma;
    double t3;
    double gluon;
};

class BasisInjectionGPD final : public PARTONS::GPDModule {
public:
    static const unsigned int classId;
    static constexpr double referenceMuF2 = 4.0;

    explicit BasisInjectionGPD(const std::string& className);
    ~BasisInjectionGPD() override;

    BasisInjectionGPD* clone() const override;

    void setShapeParameters(const DDParameters& parameters);
    const DDParameters& getShapeParameters() const;
    void setQuadratureOrder(unsigned int order);
    unsigned int getQuadratureOrder() const;
    void setAmplitudes(const BasisAmplitudes& amplitudes);
    const BasisAmplitudes& getAmplitudes() const;

protected:
    BasisInjectionGPD(const BasisInjectionGPD& other);

    void initModule() override;
    void isModuleWellConfigured() override;
    PARTONS::PartonDistribution computeH() override;

private:
    static void validateAmplitudes(const BasisAmplitudes& amplitudes);

    FixedScaleDDGPD m_shape;
    BasisAmplitudes m_amplitudes;
};

}  // namespace DVCSInference

#endif
