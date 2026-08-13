#ifndef EXTRACT_DVCS_CFF_FIXED_SCALE_DD_GPD_H
#define EXTRACT_DVCS_CFF_FIXED_SCALE_DD_GPD_H

#include <partons/modules/MathIntegratorModule.h>
#include <partons/modules/gpd/GPDModule.h>

#include <string>
#include <vector>

namespace NumA {
class FunctionType1D;
}

namespace DVCSInference {

struct DDParameters {
    double normalization;
    double a;
    double c;
    double profileB;
    double tSlope;
};

class FixedScaleDDGPD final : public PARTONS::GPDModule,
                              public PARTONS::MathIntegratorModule {
public:
    static const unsigned int classId;
    static constexpr double referenceMuF2 = 4.0;

    explicit FixedScaleDDGPD(const std::string& className);
    ~FixedScaleDDGPD() override;

    FixedScaleDDGPD* clone() const override;

    void setParameters(const DDParameters& parameters);
    const DDParameters& getParameters() const;
    void setQuadratureOrder(unsigned int order);
    unsigned int getQuadratureOrder() const;

    // Public only to support independent mathematical property tests. Normal
    // production evaluation enters through PARTONS GPDService/computeH.
    double evaluateValue(double x, double xi, double t);
    double forwardValue(double beta, double t) const;
    double profileValue(double beta, double alpha) const;

protected:
    FixedScaleDDGPD(const FixedScaleDDGPD& other);

    void initModule() override;
    void isModuleWellConfigured() override;
    PARTONS::PartonDistribution computeH() override;

private:
    void initializeIntegrator();
    double integrand(double beta, std::vector<double> parameters);
    static bool isAllowedOrder(unsigned int order);

    DDParameters m_parameters;
    unsigned int m_quadratureOrder;
    double m_integrationX;
    double m_integrationXi;
    double m_integrationT;
    NumA::FunctionType1D* m_integrandFunctor;
};

}  // namespace DVCSInference

#endif
