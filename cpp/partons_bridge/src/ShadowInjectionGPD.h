#ifndef EXTRACT_DVCS_CFF_SHADOW_INJECTION_GPD_H
#define EXTRACT_DVCS_CFF_SHADOW_INJECTION_GPD_H

#include <partons/modules/gpd/GPDModule.h>

#include <string>

namespace DVCSInference {

struct ShadowCoefficients {
    double baseline;
    double shadow;
};

class ShadowInjectionGPD final : public PARTONS::GPDModule {
public:
    static const unsigned int classId;
    static constexpr double referenceMuF2 = 1.0;

    explicit ShadowInjectionGPD(const std::string& className);
    ~ShadowInjectionGPD() override;

    ShadowInjectionGPD* clone() const override;

    void setBaselineModule(PARTONS::GPDModule* module);
    void setShadowModule(PARTONS::GPDModule* module);
    void setCoefficients(const ShadowCoefficients& coefficients);
    const ShadowCoefficients& getCoefficients() const;

protected:
    ShadowInjectionGPD(const ShadowInjectionGPD& other);

    void initModule() override;
    void isModuleWellConfigured() override;
    PARTONS::PartonDistribution computeH() override;
    PARTONS::PartonDistribution computeE() override;
    PARTONS::PartonDistribution computeHt() override;
    PARTONS::PartonDistribution computeEt() override;

private:
    static void validateCoefficients(const ShadowCoefficients& coefficients);
    PARTONS::PartonDistribution computeScaledBaseline(
        PARTONS::GPDType::Type type);

    PARTONS::GPDModule* m_baseline;
    PARTONS::GPDModule* m_shadow;
    ShadowCoefficients m_coefficients;
};

}  // namespace DVCSInference

#endif
