#ifndef EXTRACT_DVCS_CFF_TABULATED_PSEUDODATA_GPD_H
#define EXTRACT_DVCS_CFF_TABULATED_PSEUDODATA_GPD_H

#include <partons/modules/gpd/GPDModule.h>
#include <apfel/apfelxx.h>

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace DVCSInference {

/**
 * Read-only adapter for four tables produced by native GPDEvolutionApfel.
 *
 * The installed PARTONS wrapper rebuilds its APFEL++ table on every GPD
 * query. A DVCS convolution makes many queries at one fixed (xi,t,Q2), so the
 * bridge primes one authoritative native table per GPD type and serves those
 * same tables to DVCSCFFStandard. No evolution kernel, coefficient function,
 * observable formula, or interpolation is reimplemented here.
 */
class TabulatedPseudodataGPD final : public PARTONS::GPDModule {
public:
    using Table = apfel::TabulateObject<
        apfel::Set<apfel::Distribution>>;

    static const unsigned int classId;

    explicit TabulatedPseudodataGPD(const std::string& className);
    ~TabulatedPseudodataGPD() override = default;
    TabulatedPseudodataGPD* clone() const override;

    void setTable(std::size_t typeIndex, std::shared_ptr<Table> table);

protected:
    TabulatedPseudodataGPD(const TabulatedPseudodataGPD& other);
    void initModule() override;
    void isModuleWellConfigured() override;
    PARTONS::PartonDistribution computeH() override;
    PARTONS::PartonDistribution computeE() override;
    PARTONS::PartonDistribution computeHt() override;
    PARTONS::PartonDistribution computeEt() override;

private:
    PARTONS::PartonDistribution computeType(std::size_t typeIndex);
    std::array<std::shared_ptr<Table>, 4> m_tables;
};

}  // namespace DVCSInference

#endif
