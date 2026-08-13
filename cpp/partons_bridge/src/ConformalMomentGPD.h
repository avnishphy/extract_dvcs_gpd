#ifndef EXTRACT_DVCS_CFF_CONFORMAL_MOMENT_GPD_H
#define EXTRACT_DVCS_CFF_CONFORMAL_MOMENT_GPD_H

#include <gsl/gsl_integration.h>
#include <partons/modules/gpd/GPDModule.h>

#include <cstddef>
#include <string>

namespace DVCSInference {

struct ConformalMomentParameters {
    double alpha;
    double beta;
    double trajectorySlope;
};

class ConformalMomentGPD final : public PARTONS::GPDModule {
public:
    static const unsigned int classId;
    static constexpr double referenceMuF2 = 4.0;

    explicit ConformalMomentGPD(const std::string& className);
    ~ConformalMomentGPD() override;

    ConformalMomentGPD* clone() const override;

    void setParameters(const ConformalMomentParameters& parameters);
    const ConformalMomentParameters& getParameters() const;
    void setIntegrationLimit(unsigned int limit);
    unsigned int getIntegrationLimit() const;

    // Public for independent moment and reconstruction tests. Production
    // evaluation enters through PARTONS GPDService/computeH.
    double evaluateValue(double x, double xi, double t);
    double conformalMoment(unsigned int j, double t) const;
    double normalization() const;

protected:
    ConformalMomentGPD(const ConformalMomentGPD& other);

    void initModule() override;
    void isModuleWellConfigured() override;
    PARTONS::PartonDistribution computeH() override;

private:
    struct IntegralParameters {
        double delta;
        double beta;
    };

    static void validateParameters(
        const ConformalMomentParameters& parameters);
    static bool isAllowedLimit(unsigned int limit);
    static double convolutionIntegrand(double y, void* raw);
    void allocateWorkspace();

    ConformalMomentParameters m_parameters;
    unsigned int m_integrationLimit;
    gsl_integration_workspace* m_workspace;
};

}  // namespace DVCSInference

#endif
