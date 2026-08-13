#ifndef EXTRACT_DVCS_CFF_PSEUDODATA_INPUT_GPD_H
#define EXTRACT_DVCS_CFF_PSEUDODATA_INPUT_GPD_H

#include "FixedScaleDDGPD.h"

#include <partons/modules/gpd/GPDModule.h>

#include <array>
#include <cstddef>
#include <string>

namespace DVCSInference {

/**
 * Input-scale parameters for the full independent DD pseudodata model.
 *
 * The two array axes have stable orders (H,E,Htilde,Etilde) and
 * (u,d,s,gluon). Every channel has a complete independent DD shape. One
 * shadow coefficient is fixed per GPD type; configurable channel amplitudes
 * determine its flavor/gluon direction. The H/u component is the
 * installed GPDBDMMS21 shadow. All other components are project-defined DD
 * test directions and are not claimed to be CFF-null.
 */
struct PseudodataInputParameters {
    std::array<std::array<DDParameters, 4>, 4> ddByTypeAndChannel;
    std::array<double, 4> shadowCoefficientByType;
    std::array<std::array<double, 4>, 4> shadowAmplitudeByTypeAndChannel;
};

/**
 * Native input-scale composition used by the Stage 10 pseudodata simulator.
 *
 * All four twist-2 proton GPD types and u/d/s/gluon channels have separate
 * project DD parameters. Numerical DD integrals delegate to FixedScaleDDGPD.
 * This is a synthetic representation, not a native phenomenological shape.
 */
class PseudodataInputGPD final : public PARTONS::GPDModule {
public:
    static const unsigned int classId;
    static constexpr double defaultReferenceMuF2 = 1.0;

    explicit PseudodataInputGPD(const std::string& className);
    ~PseudodataInputGPD() override;

    PseudodataInputGPD* clone() const override;

    void setDDModule(FixedScaleDDGPD* module);
    void setShadowModule(PARTONS::GPDModule* module);
    void setParameters(const PseudodataInputParameters& parameters);
    const PseudodataInputParameters& getParameters() const;
    void setQuadratureOrder(unsigned int order);
    unsigned int getQuadratureOrder() const;
    void setReferenceMuF2(double value);

protected:
    PseudodataInputGPD(const PseudodataInputGPD& other);

    void initModule() override;
    void isModuleWellConfigured() override;
    PARTONS::PartonDistribution computeH() override;
    PARTONS::PartonDistribution computeE() override;
    PARTONS::PartonDistribution computeHt() override;
    PARTONS::PartonDistribution computeEt() override;

private:
    static void validateParameters(
        const PseudodataInputParameters& parameters);
    PARTONS::PartonDistribution computeDD(
        std::size_t parameterIndex, PARTONS::GPDType::Type type);
    double evaluateProjectShadow(double x, double xi, double t);

    FixedScaleDDGPD* m_dd;
    PARTONS::GPDModule* m_shadow;
    PseudodataInputParameters m_parameters;
    unsigned int m_quadratureOrder;
};

}  // namespace DVCSInference

#endif
