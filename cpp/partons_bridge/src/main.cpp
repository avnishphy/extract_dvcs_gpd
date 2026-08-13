#include "BasisInjectionGPD.h"
#include "ConformalMomentGPD.h"
#include "FixedScaleDDGPD.h"
#include "PseudodataInputGPD.h"
#include "ShadowInjectionGPD.h"
#include "TabulatedPseudodataGPD.h"

#include <gsl/gsl_version.h>
#include <ElementaryUtils/ElementaryUtilsVersion.h>
#include <ElementaryUtils/logger/CustomException.h>
#include <ElementaryUtils/parameters/Parameter.h>
#include <LHAPDF/Version.h>
#include <NumA/NumAVersion.h>
#include <apfel/version.h>
#include <apfel/rotations.h>
#include <boost/json.hpp>
#include <partons/ModuleObjectFactory.h>
#include <partons/Partons.h>
#include <partons/PartonsVersion.h>
#include <partons/FundamentalPhysicalConstants.h>
#include <partons/ServiceObjectRegistry.h>
#include <partons/beans/QuarkFlavor.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionKinematic.h>
#include <partons/beans/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionResult.h>
#include <partons/beans/gpd/GPDKinematic.h>
#include <partons/beans/gpd/GPDResult.h>
#include <partons/beans/gpd/GPDType.h>
#include <partons/beans/observable/DVCS/DVCSObservableKinematic.h>
#include <partons/modules/active_flavors_thresholds/ActiveFlavorsThresholdsConstant.h>
#include <partons/modules/active_flavors_thresholds/ActiveFlavorsThresholdsModule.h>
#include <partons/modules/evolution/gpd/GPDEvolutionApfel.h>
#include <partons/modules/evolution/gpd/GPDEvolutionModule.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSCFFStandard.h>
#include <partons/modules/convol_coeff_function/DVCS/DVCSConvolCoeffFunctionModule.h>
#include <partons/modules/gpd/GPDBDMMS21.h>
#include <partons/modules/gpd/GPDGK11.h>
#include <partons/modules/gpd/GPDGK16.h>
#include <partons/modules/gpd/GPDGK19.h>
#include <partons/modules/gpd/GPDModule.h>
#include <partons/modules/gpd/GPDVGG99.h>
#include <partons/modules/observable/DVCS/DVCSObservable.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAc.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAllMinus.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAluMinus.h>
#include <partons/modules/observable/DVCS/asymmetry/DVCSAulMinus.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionDifferenceLUMinus.h>
#include <partons/modules/observable/DVCS/cross_section/DVCSCrossSectionUUMinus.h>
#include <partons/modules/process/DVCS/DVCSProcessGV08.h>
#include <partons/modules/process/DVCS/DVCSProcessModule.h>
#include <partons/modules/running_alpha_strong/RunningAlphaStrongApfel.h>
#include <partons/modules/running_alpha_strong/RunningAlphaStrongModule.h>
#include <partons/modules/scales/DVCS/DVCSScalesModule.h>
#include <partons/modules/scales/DVCS/DVCSScalesQ2Multiplier.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterModule.h>
#include <partons/modules/xi_converter/DVCS/DVCSXiConverterXBToXi.h>
#include <partons/services/DVCSConvolCoeffFunctionService.h>
#include <partons/services/DVCSObservableService.h>
#include <partons/services/GPDService.h>
#include <partons/beans/PerturbativeQCDOrderType.h>
#include <partons/utils/type/PhysicalUnit.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace json = boost::json;

constexpr int kExitUsage = 64;
constexpr int kExitParse = 65;
constexpr int kExitValidation = 66;
constexpr int kExitNumerical = 67;
constexpr int kExitSelfTest = 68;
constexpr int kExitBackend = 70;

class BridgeError : public std::runtime_error {
public:
    BridgeError(std::string code, std::string message, int exitCode)
        : std::runtime_error(std::move(message)),
          m_code(std::move(code)),
          m_exitCode(exitCode) {
    }

    const std::string& code() const {
        return m_code;
    }

    int exitCode() const {
        return m_exitCode;
    }

private:
    std::string m_code;
    int m_exitCode;
};

struct Kinematics {
    double x;
    double xi;
    double t;
    double muF2;
    double muR2;
};

struct Evaluation {
    std::string requestId;
    Kinematics kinematics;
};

struct GPDValues {
    double up;
    double gluon;
};

struct DDKinematics {
    double x;
    double xi;
    double t;
    double q0Squared;
    double muFSquared;
};

struct DDEvaluation {
    std::string requestId;
    DVCSInference::DDParameters parameters;
    unsigned int quadratureOrder;
    DDKinematics kinematics;
    bool evolved;
};

struct DDValues {
    double up;
    double upPlus;
    double upMinus;
    double gluon;
};

struct BasisEvaluation {
    std::string requestId;
    DVCSInference::DDParameters shapeParameters;
    DVCSInference::BasisAmplitudes amplitudes;
    unsigned int quadratureOrder;
    DDKinematics kinematics;
};

struct BasisValues {
    std::array<double, 6> quark;
    std::array<double, 6> plus;
    std::array<double, 6> minus;
    double gluon;
};

struct ConformalEvaluation {
    std::string requestId;
    DVCSInference::ConformalMomentParameters parameters;
    unsigned int integrationLimit;
    DDKinematics kinematics;
};

struct ConformalValues {
    double up;
    double upPlus;
    double upMinus;
};

struct ShadowGPDKinematics {
    double x;
    double xi;
    double t;
    double muF2;
    double muR2;
};

struct ShadowObservableKinematics {
    double xB;
    double t;
    double q2;
    double beamEnergy;
    double phi;
};

double requirePhysicalFixedTargetInelasticity(
        const ShadowObservableKinematics& kinematics,
        const std::string& context) {
    const double y = kinematics.q2 /
        (2.0 * PARTONS::Constant::PROTON_MASS *
            kinematics.beamEnergy * kinematics.xB);
    if (!std::isfinite(y) || y <= 0.0 || y >= 1.0) {
        std::ostringstream message;
        message.precision(12);
        message << context << " is not physical fixed-target DVCS: "
                << "y=Q2/(2*M_p*E*x_b)=" << y
                << " must lie in (0,1)";
        throw BridgeError("unphysical_fixed_target_kinematics",
            message.str(), kExitValidation);
    }
    return y;
}

struct ShadowEvaluation {
    std::string requestId;
    DVCSInference::ShadowCoefficients coefficients;
    ShadowGPDKinematics gpdKinematics;
    ShadowObservableKinematics observableKinematics;
};

struct ShadowValues {
    double hUpCombined;
    double hUpBaselineComponent;
    double hUpShadowComponent;
    std::complex<double> cffH;
    double cffXi;
    double observable;
    std::string observableUnit;
};

struct PseudodataEvaluation {
    std::string requestId;
    DVCSInference::PseudodataInputParameters parameters;
    unsigned int quadratureOrder;
    double q0Squared;
    ShadowObservableKinematics observableKinematics;
    std::vector<double> diagnosticX;
};

struct PseudodataGPDValues {
    double x;
    struct Distribution {
        std::array<DDValues, 3> quarks;
        double gluon;
    };
    std::array<Distribution, 4> byType;
};

struct PseudodataValues {
    std::array<std::complex<double>, 4> cffs;
    double cffXi;
    std::array<double, 6> observables;
    std::array<std::string, 6> observableUnits;
    std::vector<PseudodataGPDValues> gpdDiagnostics;
};

struct NativeModelHoldoutEvaluation {
    std::string requestId;
    std::string model;
    ShadowObservableKinematics observableKinematics;
    std::vector<double> diagnosticX;
};

json::array libraryProvenance() {
    json::array libraries;
    libraries.emplace_back(json::object{
        {"name", "PARTONS"}, {"path", BRIDGE_PARTONS_LIBRARY}});
    libraries.emplace_back(json::object{
        {"name", "APFEL++"}, {"path", BRIDGE_APFELXX_LIBRARY}});
    libraries.emplace_back(json::object{
        {"name", "NumA++"}, {"path", BRIDGE_NUMA_LIBRARY}});
    libraries.emplace_back(json::object{
        {"name", "ElementaryUtils"}, {"path", BRIDGE_ELEMENTARY_LIBRARY}});
    libraries.emplace_back(json::object{
        {"name", "LHAPDF"}, {"path", BRIDGE_LHAPDF_LIBRARY}});
    libraries.emplace_back(json::object{
        {"name", "GSL"}, {"path", BRIDGE_GSL_LIBRARY}});
    return libraries;
}

json::object backendProvenance() {
    return json::object{
        {"bridge_version", BRIDGE_VERSION},
        {"bridge_schema_version", BRIDGE_SCHEMA_VERSION},
        {"bridge_source_sha256", BRIDGE_SOURCE_SHA256},
        {"compiler",
         json::object{{"id", BRIDGE_COMPILER_ID},
                      {"version", BRIDGE_COMPILER_VERSION},
                      {"cxx_standard", 17}}},
        {"partons",
         json::object{
             {"version",
              std::to_string(PARTONS_VERSION_MAJOR) + "." +
                  std::to_string(PARTONS_VERSION_MINOR) + "." +
                  std::to_string(PARTONS_VERSION_PATCH)},
             {"git_revision", PARTONS_GIT_REVISION}}},
        {"apfelxx", json::object{{"version", VERSION}}},
        {"elementary_utils",
         json::object{
             {"version",
              std::to_string(ELEMENTARY_UTILS_VERSION_MAJOR) + "." +
                  std::to_string(ELEMENTARY_UTILS_VERSION_MINOR) + "." +
                  std::to_string(ELEMENTARY_UTILS_VERSION_PATCH)},
             {"git_revision", ELEMENTARY_UTILS_GIT_REVISION}}},
        {"numa",
         json::object{
             {"version",
              std::to_string(NUMA_VERSION_MAJOR) + "." +
                  std::to_string(NUMA_VERSION_MINOR) + "." +
                  std::to_string(NUMA_VERSION_PATCH)},
             {"git_revision", NUMA_GIT_REVISION}}},
        {"lhapdf", json::object{{"version", LHAPDF::version()}}},
        {"gsl", json::object{{"version", GSL_VERSION}}},
        {"native_libraries", libraryProvenance()},
        {"execution_mode", "single_process_single_worker"}};
}

json::object errorResponse(const std::string& code,
        const std::string& message) {
    return json::object{
        {"schema_version", BRIDGE_SCHEMA_VERSION},
        {"status", "error"},
        {"error", json::object{{"code", code}, {"message", message}}},
        {"backend", backendProvenance()}};
}

void requireOnlyKeys(const json::object& object,
        const std::set<std::string>& allowed, const std::string& context) {
    for (const auto& entry : object) {
        const std::string key(entry.key());
        if (allowed.count(key) == 0U) {
            throw BridgeError("unknown_field",
                    context + " contains unsupported field '" + key + "'",
                    kExitValidation);
        }
    }
}

json::value requireField(const json::object& object,
        const std::string& key, const std::string& context) {
    const auto iterator = object.find(key);
    if (iterator == object.end()) {
        throw BridgeError("missing_field",
                context + " is missing required field '" + key + "'",
                kExitValidation);
    }
    return iterator->value();
}

json::object requireObject(const json::value& value,
        const std::string& context) {
    if (!value.is_object()) {
        throw BridgeError("invalid_type", context + " must be an object",
                kExitValidation);
    }
    return value.as_object();
}

std::string requireString(const json::object& object, const std::string& key,
        const std::string& context) {
    const json::value value = requireField(object, key, context);
    if (!value.is_string()) {
        throw BridgeError("invalid_type",
                context + "." + key + " must be a string", kExitValidation);
    }
    return std::string(value.as_string());
}

double requireFiniteNumber(const json::object& object,
        const std::string& key, const std::string& context) {
    const json::value value = requireField(object, key, context);
    if (!value.is_number()) {
        throw BridgeError("invalid_type",
                context + "." + key + " must be a number", kExitValidation);
    }
    const double result = value.to_number<double>();
    if (!std::isfinite(result)) {
        throw BridgeError("non_finite_input",
                context + "." + key + " must be finite", kExitValidation);
    }
    return result;
}

double requireQuantity(const json::object& parent, const std::string& key,
        const std::string& expectedUnit) {
    const json::object quantity = requireObject(
        requireField(parent, key, "kinematics"), "kinematics." + key);
    requireOnlyKeys(quantity, {"value", "unit"}, "kinematics." + key);
    const std::string unit =
        requireString(quantity, "unit", "kinematics." + key);
    if (unit != expectedUnit) {
        throw BridgeError("unsupported_unit",
                "kinematics." + key + ".unit must be '" + expectedUnit + "'",
                kExitValidation);
    }
    return requireFiniteNumber(quantity, "value", "kinematics." + key);
}

unsigned int requireUnsignedInteger(const json::object& object,
        const std::string& key, const std::string& context) {
    const json::value value = requireField(object, key, context);
    if (!value.is_int64() || value.as_int64() < 0) {
        throw BridgeError("invalid_type",
                context + "." + key + " must be a non-negative integer",
                kExitValidation);
    }
    const std::int64_t integer = value.as_int64();
    if (integer > std::numeric_limits<unsigned int>::max()) {
        throw BridgeError("invalid_value",
                context + "." + key + " is too large", kExitValidation);
    }
    return static_cast<unsigned int>(integer);
}

void requireNull(const json::object& object, const std::string& key,
        const std::string& context) {
    const json::value value = requireField(object, key, context);
    if (!value.is_null()) {
        throw BridgeError("unsupported_perturbative_configuration",
            context + "." + key + " must be null in the Stage 06 "
                      "evolution-only contract",
            kExitValidation);
    }
}

void requireClosedInterval(double value, double minimum, double maximum,
        const std::string& path) {
    if (value < minimum || value > maximum) {
        throw BridgeError("invalid_parameter",
            path + " must be in [" + std::to_string(minimum) + ", " +
                std::to_string(maximum) + "]",
            kExitValidation);
    }
}

double requireParameterQuantity(const json::object& parameters,
        const std::string& key, const std::string& expectedUnit,
        const std::string& context) {
    const json::object quantity = requireObject(
        requireField(parameters, key, context), context + "." + key);
    requireOnlyKeys(quantity, {"value", "unit"}, context + "." + key);
    const std::string unit = requireString(quantity, "unit",
        context + "." + key);
    if (unit != expectedUnit) {
        throw BridgeError("unsupported_unit",
            context + "." + key + ".unit must be '" + expectedUnit + "'",
            kExitValidation);
    }
    return requireFiniteNumber(quantity, "value", context + "." + key);
}

DDKinematics parseDDKinematics(const json::object& object,
        const std::string& context, bool evolved) {
    requireOnlyKeys(object, {"x", "xi", "t", "q0_squared", "mu_f_squared"},
        context);
    DDKinematics kinematics{
        requireFiniteNumber(object, "x", context),
        requireFiniteNumber(object, "xi", context),
        requireQuantity(object, "t", "GeV2"),
        requireQuantity(object, "q0_squared", "GeV2"),
        requireQuantity(object, "mu_f_squared", "GeV2")};
    if ((!evolved && (kinematics.x < -1.0 || kinematics.x > 1.0)) ||
            (evolved && (kinematics.x < 1e-5 || kinematics.x > 0.95))) {
        throw BridgeError("outside_support",
            evolved ? context + ".x must be in [1e-5, 0.95] for the "
                              "audited positive-x APFEL++ interface"
                    : context + ".x must be in [-1, 1]",
            kExitValidation);
    }
    if ((!evolved && (kinematics.xi < -1.0 || kinematics.xi > 1.0)) ||
            (evolved && (kinematics.xi < 0.0 ||
                         kinematics.xi > 0.8))) {
        throw BridgeError("invalid_kinematics",
            evolved ? context + ".xi must be in [0, 0.8]"
                    : context + ".xi must be in [-1, 1]",
            kExitValidation);
    }
    if (kinematics.t > 0.0 || (evolved && kinematics.t < -1.0)) {
        throw BridgeError("invalid_kinematics",
            evolved ? context + ".t must be in [-1, 0] GeV2"
                    : context + ".t must be non-positive GeV2",
            kExitValidation);
    }
    if (kinematics.q0Squared !=
            DVCSInference::FixedScaleDDGPD::referenceMuF2) {
        throw BridgeError("unsupported_scale",
            context + ".q0_squared must be exactly 4 GeV2",
            kExitValidation);
    }
    if ((!evolved && kinematics.muFSquared !=
            DVCSInference::FixedScaleDDGPD::referenceMuF2) ||
            (evolved && (kinematics.muFSquared <
                             DVCSInference::FixedScaleDDGPD::referenceMuF2 ||
                         kinematics.muFSquared > 16.0))) {
        throw BridgeError("unsupported_scale",
            evolved ? context + ".mu_f_squared must be in [4, 16] GeV2"
                    : context + ".mu_f_squared must be exactly 4 GeV2; "
                              "Stage 03 has no evolution",
            kExitValidation);
    }
    return kinematics;
}

void parseEvolutionConfiguration(const json::object& object,
        const std::string& context) {
    requireOnlyKeys(object,
        {"module", "gpd_evolution_order", "factorization_scheme",
            "alpha_s", "active_flavors", "coefficient_function_order",
            "renormalization_scale", "twist"},
        context);
    if (requireString(object, "module", context) != "GPDEvolutionApfel" ||
            requireString(object, "gpd_evolution_order", context) != "LO" ||
            requireString(object, "factorization_scheme", context) !=
                "MSbar") {
        throw BridgeError("unsupported_perturbative_configuration",
            context + " must select GPDEvolutionApfel, LO, and MSbar",
            kExitValidation);
    }
    requireNull(object, "coefficient_function_order", context);
    requireNull(object, "renormalization_scale", context);
    requireNull(object, "twist", context);

    const json::object alpha = requireObject(
        requireField(object, "alpha_s", context), context + ".alpha_s");
    requireOnlyKeys(alpha,
        {"module", "order", "scheme", "reference_value",
            "reference_scale", "thresholds"},
        context + ".alpha_s");
    if (requireString(alpha, "module", context + ".alpha_s") !=
            "RunningAlphaStrongApfel" ||
            requireString(alpha, "order", context + ".alpha_s") != "LO" ||
            requireString(alpha, "scheme", context + ".alpha_s") !=
                "MSbar" ||
            requireFiniteNumber(alpha, "reference_value",
                context + ".alpha_s") != 0.118 ||
            requireQuantity(alpha, "reference_scale", "GeV") != 91.1876) {
        throw BridgeError("unsupported_perturbative_configuration",
            context + ".alpha_s does not match the audited fixed contract",
            kExitValidation);
    }
    const json::value thresholdsValue =
        requireField(alpha, "thresholds", context + ".alpha_s");
    if (!thresholdsValue.is_array()) {
        throw BridgeError("invalid_type",
            context + ".alpha_s.thresholds must be an array",
            kExitValidation);
    }
    const json::array thresholds = thresholdsValue.as_array();
    if (thresholds.size() != 3U) {
        throw BridgeError("unsupported_perturbative_configuration",
            context + ".alpha_s.thresholds must be [0, 0, 0]",
            kExitValidation);
    }
    for (const json::value& threshold : thresholds) {
        if (!threshold.is_number() ||
                threshold.to_number<double>() != 0.0) {
            throw BridgeError("unsupported_perturbative_configuration",
                context + ".alpha_s.thresholds must be [0, 0, 0]",
                kExitValidation);
        }
    }

    const json::object flavors = requireObject(
        requireField(object, "active_flavors", context),
        context + ".active_flavors");
    requireOnlyKeys(flavors, {"module", "n_flavors"},
        context + ".active_flavors");
    if (requireString(flavors, "module",
            context + ".active_flavors") !=
            "ActiveFlavorsThresholdsConstant" ||
            requireUnsignedInteger(flavors, "n_flavors",
                context + ".active_flavors") != 3U) {
        throw BridgeError("unsupported_perturbative_configuration",
            context + ".active_flavors must select constant n_flavors=3",
            kExitValidation);
    }
}

DDEvaluation parseDDEvaluation(const json::object& object,
        const std::string& context, bool topLevel = false,
        bool evolved = false) {
    std::set<std::string> allowed{"request_id", "representation",
        "gpd_type", "flavor", "parton_content", "parameters",
        "quadrature_order", "kinematics"};
    if (evolved) {
        allowed.insert("evolution_configuration");
    }
    if (topLevel) {
        allowed.insert("schema_version");
        allowed.insert("operation");
    }
    requireOnlyKeys(object, allowed, context);
    const std::string requestId = requireString(object, "request_id", context);
    if (requestId.empty()) {
        throw BridgeError("invalid_value",
            context + ".request_id must not be empty", kExitValidation);
    }
    const std::string expectedRepresentation = evolved
        ? "stage06_lo_apfel_evolved_reduced_dd_v1"
        : "stage03_fixed_scale_up_valence_v1";
    if (requireString(object, "representation", context) !=
            expectedRepresentation) {
        throw BridgeError("unsupported_representation",
            context + ".representation must be '" +
                expectedRepresentation + "'",
            kExitValidation);
    }
    if (requireString(object, "gpd_type", context) != "H") {
        throw BridgeError("unsupported_gpd_type",
            context + ".gpd_type must be 'H'", kExitValidation);
    }
    if (requireString(object, "flavor", context) != "up") {
        throw BridgeError("unsupported_flavor",
            context + ".flavor must be 'up'", kExitValidation);
    }
    const std::string expectedContent = evolved
        ? "reduced_up_valence_input_evolved_fixed_nf3"
        : "reduced_up_valence_forward_input_no_gluon_no_sea";
    if (requireString(object, "parton_content", context) !=
            expectedContent) {
        throw BridgeError("unsupported_parton_content",
            context + ".parton_content must be the declared reduced "
                      "Stage-specific reduced content",
            kExitValidation);
    }

    const json::object parameters = requireObject(
        requireField(object, "parameters", context), context + ".parameters");
    requireOnlyKeys(parameters, {"a", "c", "profile_b", "t_slope"},
        context + ".parameters");
    DVCSInference::DDParameters parsedParameters{
        2.0,
        requireFiniteNumber(parameters, "a", context + ".parameters"),
        requireFiniteNumber(parameters, "c", context + ".parameters"),
        requireFiniteNumber(parameters, "profile_b",
            context + ".parameters"),
        requireParameterQuantity(parameters, "t_slope", "GeV-2",
            context + ".parameters")};
    requireClosedInterval(parsedParameters.a, 0.1, 1.0,
        context + ".parameters.a");
    requireClosedInterval(parsedParameters.c, 2.0, 6.0,
        context + ".parameters.c");
    requireClosedInterval(parsedParameters.profileB, 1.0, 4.0,
        context + ".parameters.profile_b");
    requireClosedInterval(parsedParameters.tSlope, 0.0, 2.0,
        context + ".parameters.t_slope.value");

    const unsigned int order =
        requireUnsignedInteger(object, "quadrature_order", context);
    if ((!evolved && order != 32U && order != 64U &&
            order != 128U && order != 256U) ||
            (evolved && order != 128U)) {
        throw BridgeError("unsupported_quadrature_order",
            evolved ? context + ".quadrature_order must be 128"
                    : context + ".quadrature_order must be one of "
                              "32, 64, 128, 256",
            kExitValidation);
    }
    if (evolved) {
        const json::object configuration = requireObject(
            requireField(object, "evolution_configuration", context),
            context + ".evolution_configuration");
        parseEvolutionConfiguration(configuration,
            context + ".evolution_configuration");
    }
    const json::object kinematics = requireObject(
        requireField(object, "kinematics", context), context + ".kinematics");
    return DDEvaluation{requestId, parsedParameters, order,
        parseDDKinematics(kinematics, context + ".kinematics", evolved),
        evolved};
}

ConformalEvaluation parseConformalEvaluation(
        const json::object& object, const std::string& context,
        bool topLevel = false) {
    std::set<std::string> allowed{"request_id", "representation",
        "gpd_type", "flavor", "parton_content", "parameters",
        "integration_limit", "kinematics"};
    if (topLevel) {
        allowed.insert("schema_version");
        allowed.insert("operation");
    }
    requireOnlyKeys(object, allowed, context);
    const std::string requestId = requireString(object, "request_id", context);
    if (requestId.empty()) {
        throw BridgeError("invalid_value",
            context + ".request_id must not be empty", kExitValidation);
    }
    if (requireString(object, "representation", context) !=
            "stage08_zero_skewness_conformal_moment_v1") {
        throw BridgeError("unsupported_representation",
            context + ".representation must be "
                      "'stage08_zero_skewness_conformal_moment_v1'",
            kExitValidation);
    }
    if (requireString(object, "gpd_type", context) != "H") {
        throw BridgeError("unsupported_gpd_type",
            context + ".gpd_type must be 'H'", kExitValidation);
    }
    if (requireString(object, "flavor", context) != "up") {
        throw BridgeError("unsupported_flavor",
            context + ".flavor must be 'up'", kExitValidation);
    }
    if (requireString(object, "parton_content", context) !=
            "reduced_up_valence_zero_skewness_no_gluon_no_sea") {
        throw BridgeError("unsupported_parton_content",
            context + ".parton_content must be the declared Stage 08 "
                      "reduced content",
            kExitValidation);
    }

    const json::object parameters = requireObject(
        requireField(object, "parameters", context), context + ".parameters");
    requireOnlyKeys(parameters, {"alpha", "beta", "trajectory_slope"},
        context + ".parameters");
    DVCSInference::ConformalMomentParameters parsedParameters{
        requireFiniteNumber(parameters, "alpha", context + ".parameters"),
        requireFiniteNumber(parameters, "beta", context + ".parameters"),
        requireParameterQuantity(parameters, "trajectory_slope", "GeV-2",
            context + ".parameters")};
    requireClosedInterval(parsedParameters.alpha, -1.0, -0.1,
        context + ".parameters.alpha");
    requireClosedInterval(parsedParameters.beta, 2.0, 6.0,
        context + ".parameters.beta");
    requireClosedInterval(parsedParameters.trajectorySlope, 0.0, 2.0,
        context + ".parameters.trajectory_slope.value");

    const unsigned int limit =
        requireUnsignedInteger(object, "integration_limit", context);
    if (limit != 256U && limit != 512U &&
            limit != 1024U && limit != 2048U) {
        throw BridgeError("unsupported_integration_limit",
            context + ".integration_limit must be one of "
                      "256, 512, 1024, 2048",
            kExitValidation);
    }
    const json::object kinematicsObject = requireObject(
        requireField(object, "kinematics", context), context + ".kinematics");
    const DDKinematics kinematics = parseDDKinematics(
        kinematicsObject, context + ".kinematics", false);
    if (kinematics.x < 1e-5 || kinematics.x > 0.95) {
        throw BridgeError("outside_support",
            context + ".kinematics.x must be in [1e-5, 0.95]",
            kExitValidation);
    }
    if (kinematics.xi != 0.0) {
        throw BridgeError("unsupported_skewness",
            context + ".kinematics.xi must be exactly 0 for the "
                      "Stage 08 conformal-moment slice",
            kExitValidation);
    }
    if (kinematics.t < -0.5) {
        throw BridgeError("invalid_kinematics",
            context + ".kinematics.t must be in [-0.5, 0] GeV2",
            kExitValidation);
    }
    return ConformalEvaluation{
        requestId, parsedParameters, limit, kinematics};
}

BasisEvaluation parseBasisEvaluation(const json::object& object,
        const std::string& context, bool topLevel = false) {
    std::set<std::string> allowed{"request_id", "representation",
        "gpd_type", "input_basis", "parton_content", "parameters",
        "quadrature_order", "kinematics", "evolution_configuration"};
    if (topLevel) {
        allowed.insert("schema_version");
        allowed.insert("operation");
    }
    requireOnlyKeys(object, allowed, context);
    const std::string requestId = requireString(object, "request_id", context);
    if (requestId.empty()) {
        throw BridgeError("invalid_value",
            context + ".request_id must not be empty", kExitValidation);
    }
    if (requireString(object, "representation", context) !=
            "stage07_coupled_basis_dd_v1") {
        throw BridgeError("unsupported_representation",
            context + ".representation must be "
                      "'stage07_coupled_basis_dd_v1'",
            kExitValidation);
    }
    if (requireString(object, "gpd_type", context) != "H") {
        throw BridgeError("unsupported_gpd_type",
            context + ".gpd_type must be 'H'", kExitValidation);
    }
    if (requireString(object, "input_basis", context) !=
            "apfel_nf3_sigma_t3_gluon") {
        throw BridgeError("unsupported_basis",
            context + ".input_basis must be "
                      "'apfel_nf3_sigma_t3_gluon'",
            kExitValidation);
    }
    if (requireString(object, "parton_content", context) !=
            "explicit_sigma_t3_gluon_input_fixed_nf3") {
        throw BridgeError("unsupported_parton_content",
            context + ".parton_content must be "
                      "'explicit_sigma_t3_gluon_input_fixed_nf3'",
            kExitValidation);
    }

    const json::object parameters = requireObject(
        requireField(object, "parameters", context), context + ".parameters");
    requireOnlyKeys(parameters, {"shape", "amplitudes"},
        context + ".parameters");
    const json::object shape = requireObject(
        requireField(parameters, "shape", context + ".parameters"),
        context + ".parameters.shape");
    requireOnlyKeys(shape, {"a", "c", "profile_b", "t_slope"},
        context + ".parameters.shape");
    DVCSInference::DDParameters shapeParameters{
        2.0,
        requireFiniteNumber(shape, "a", context + ".parameters.shape"),
        requireFiniteNumber(shape, "c", context + ".parameters.shape"),
        requireFiniteNumber(shape, "profile_b",
            context + ".parameters.shape"),
        requireParameterQuantity(shape, "t_slope", "GeV-2",
            context + ".parameters.shape")};
    requireClosedInterval(shapeParameters.a, 0.1, 1.0,
        context + ".parameters.shape.a");
    requireClosedInterval(shapeParameters.c, 2.0, 6.0,
        context + ".parameters.shape.c");
    requireClosedInterval(shapeParameters.profileB, 1.0, 4.0,
        context + ".parameters.shape.profile_b");
    requireClosedInterval(shapeParameters.tSlope, 0.0, 2.0,
        context + ".parameters.shape.t_slope.value");

    const json::object amplitudes = requireObject(
        requireField(parameters, "amplitudes", context + ".parameters"),
        context + ".parameters.amplitudes");
    requireOnlyKeys(amplitudes, {"sigma", "t3", "gluon"},
        context + ".parameters.amplitudes");
    DVCSInference::BasisAmplitudes basisAmplitudes{
        requireFiniteNumber(amplitudes, "sigma",
            context + ".parameters.amplitudes"),
        requireFiniteNumber(amplitudes, "t3",
            context + ".parameters.amplitudes"),
        requireFiniteNumber(amplitudes, "gluon",
            context + ".parameters.amplitudes")};
    requireClosedInterval(basisAmplitudes.sigma, -4.0, 4.0,
        context + ".parameters.amplitudes.sigma");
    requireClosedInterval(basisAmplitudes.t3, -4.0, 4.0,
        context + ".parameters.amplitudes.t3");
    requireClosedInterval(basisAmplitudes.gluon, -4.0, 4.0,
        context + ".parameters.amplitudes.gluon");

    const unsigned int order =
        requireUnsignedInteger(object, "quadrature_order", context);
    if (order != 128U) {
        throw BridgeError("unsupported_quadrature_order",
            context + ".quadrature_order must be 128", kExitValidation);
    }
    const json::object configuration = requireObject(
        requireField(object, "evolution_configuration", context),
        context + ".evolution_configuration");
    parseEvolutionConfiguration(configuration,
        context + ".evolution_configuration");
    const json::object kinematics = requireObject(
        requireField(object, "kinematics", context), context + ".kinematics");
    return BasisEvaluation{requestId, shapeParameters, basisAmplitudes, order,
        parseDDKinematics(kinematics, context + ".kinematics", true)};
}

void parseShadowTheoryConfiguration(const json::object& object,
        const std::string& context) {
    requireOnlyKeys(object,
        {"cff_module", "process_module", "observable_module",
            "xi_converter_module", "scales_module",
            "coefficient_function_order", "gpd_evolution_order", "alpha_s",
            "factorization_scale_squared", "renormalization_scale_squared",
            "twist", "analysis_order_label"},
        context);
    if (requireString(object, "cff_module", context) !=
                "DVCSCFFStandard" ||
            requireString(object, "process_module", context) !=
                "DVCSProcessGV08" ||
            requireString(object, "observable_module", context) !=
                "DVCSCrossSectionUUMinus" ||
            requireString(object, "xi_converter_module", context) !=
                "DVCSXiConverterXBToXi" ||
            requireString(object, "scales_module", context) !=
                "DVCSScalesQ2Multiplier" ||
            requireString(object, "coefficient_function_order", context) !=
                "NLO" ||
            requireString(object, "factorization_scale_squared", context) !=
                "Q2" ||
            requireString(object, "renormalization_scale_squared", context) !=
                "Q2") {
        throw BridgeError("unsupported_perturbative_configuration",
            context + " does not match the frozen Stage 09 native route",
            kExitValidation);
    }
    for (const std::string key :
            {"gpd_evolution_order", "twist", "analysis_order_label"}) {
        if (!requireField(object, key, context).is_null()) {
            throw BridgeError("unsupported_perturbative_configuration",
                context + "." + key + " must be null",
                kExitValidation);
        }
    }

    const json::object alpha = requireObject(
        requireField(object, "alpha_s", context), context + ".alpha_s");
    requireOnlyKeys(alpha,
        {"module", "implementation_order", "scheme", "reference_value",
            "reference_scale", "flavor_threshold_source",
            "threshold_matching_verified"},
        context + ".alpha_s");
    if (requireString(alpha, "module", context + ".alpha_s") !=
                "RunningAlphaStrongStandard" ||
            requireString(alpha, "implementation_order",
                context + ".alpha_s") != "four_loop" ||
            requireString(alpha, "scheme", context + ".alpha_s") !=
                "MSbar" ||
            requireFiniteNumber(alpha, "reference_value",
                context + ".alpha_s") != 0.1184 ||
            requireParameterQuantity(alpha, "reference_scale", "GeV",
                context + ".alpha_s") != 91.1876 ||
            requireString(alpha, "flavor_threshold_source",
                context + ".alpha_s") != "installed_quark_masses") {
        throw BridgeError("unsupported_perturbative_configuration",
            context + ".alpha_s does not match the audited Stage 09 route",
            kExitValidation);
    }
    const json::value matching = requireField(
        alpha, "threshold_matching_verified", context + ".alpha_s");
    if (!matching.is_bool() || matching.as_bool()) {
        throw BridgeError("unsupported_perturbative_configuration",
            context + ".alpha_s.threshold_matching_verified must be false",
            kExitValidation);
    }
}

ShadowEvaluation parseShadowEvaluation(const json::object& object,
        const std::string& context, bool topLevel = false) {
    std::set<std::string> allowed{"request_id", "representation",
        "parameters", "gpd_diagnostic_kinematics",
        "observable_kinematics", "theory_configuration"};
    if (topLevel) {
        allowed.insert("schema_version");
        allowed.insert("operation");
    }
    requireOnlyKeys(object, allowed, context);

    const std::string requestId = requireString(object, "request_id", context);
    if (requestId.empty()) {
        throw BridgeError("invalid_value",
            context + ".request_id must not be empty", kExitValidation);
    }
    if (requireString(object, "representation", context) !=
            "stage09_gk16_plus_bdmms21_shadow_v1") {
        throw BridgeError("unsupported_representation",
            context + ".representation must be "
                      "'stage09_gk16_plus_bdmms21_shadow_v1'",
            kExitValidation);
    }

    const json::object parameters = requireObject(
        requireField(object, "parameters", context), context + ".parameters");
    requireOnlyKeys(parameters,
        {"baseline_coefficient", "shadow_coefficient"},
        context + ".parameters");
    const DVCSInference::ShadowCoefficients coefficients{
        requireFiniteNumber(parameters, "baseline_coefficient",
            context + ".parameters"),
        requireFiniteNumber(parameters, "shadow_coefficient",
            context + ".parameters")};
    requireClosedInterval(coefficients.baseline, 0.8, 1.2,
        context + ".parameters.baseline_coefficient");
    requireClosedInterval(coefficients.shadow, -1.0, 1.0,
        context + ".parameters.shadow_coefficient");

    const json::object gpd = requireObject(
        requireField(object, "gpd_diagnostic_kinematics", context),
        context + ".gpd_diagnostic_kinematics");
    requireOnlyKeys(gpd, {"x", "xi", "t", "mu_f_squared", "mu_r_squared"},
        context + ".gpd_diagnostic_kinematics");
    const ShadowGPDKinematics gpdKinematics{
        requireFiniteNumber(gpd, "x",
            context + ".gpd_diagnostic_kinematics"),
        requireFiniteNumber(gpd, "xi",
            context + ".gpd_diagnostic_kinematics"),
        requireParameterQuantity(gpd, "t", "GeV2",
            context + ".gpd_diagnostic_kinematics"),
        requireParameterQuantity(gpd, "mu_f_squared", "GeV2",
            context + ".gpd_diagnostic_kinematics"),
        requireParameterQuantity(gpd, "mu_r_squared", "GeV2",
            context + ".gpd_diagnostic_kinematics")};
    if (std::abs(gpdKinematics.x) < 0.05 ||
            std::abs(gpdKinematics.x) > 0.8 ||
            gpdKinematics.xi < 0.05 || gpdKinematics.xi > 0.2 ||
            std::abs(gpdKinematics.x) < gpdKinematics.xi ||
            gpdKinematics.t < -0.2 || gpdKinematics.t > -0.08) {
        throw BridgeError("invalid_kinematics",
            context + ".gpd_diagnostic_kinematics is outside the frozen "
                      "Stage 09 domain",
            kExitValidation);
    }
    if (gpdKinematics.muF2 != 4.0 || gpdKinematics.muR2 != 4.0) {
        throw BridgeError("unsupported_scale",
            context + ".gpd_diagnostic_kinematics requires "
                      "mu_f_squared=mu_r_squared=4 GeV2",
            kExitValidation);
    }

    const json::object observable = requireObject(
        requireField(object, "observable_kinematics", context),
        context + ".observable_kinematics");
    requireOnlyKeys(observable,
        {"x_b", "t", "Q2", "beam_energy", "phi"},
        context + ".observable_kinematics");
    const ShadowObservableKinematics observableKinematics{
        requireFiniteNumber(observable, "x_b",
            context + ".observable_kinematics"),
        requireParameterQuantity(observable, "t", "GeV2",
            context + ".observable_kinematics"),
        requireParameterQuantity(observable, "Q2", "GeV2",
            context + ".observable_kinematics"),
        requireParameterQuantity(observable, "beam_energy", "GeV",
            context + ".observable_kinematics"),
        requireParameterQuantity(observable, "phi", "rad",
            context + ".observable_kinematics")};
    if (observableKinematics.xB < 0.12 ||
            observableKinematics.xB > 0.28 ||
            observableKinematics.t < -0.2 ||
            observableKinematics.t > -0.08 ||
            observableKinematics.q2 != 2.0 ||
            observableKinematics.beamEnergy != 6.0 ||
            observableKinematics.phi < 0.0 ||
            observableKinematics.phi > 2.0 * std::acos(-1.0)) {
        throw BridgeError("invalid_kinematics",
            context + ".observable_kinematics is outside the frozen "
                      "Stage 09 domain",
            kExitValidation);
    }

    const json::object theory = requireObject(
        requireField(object, "theory_configuration", context),
        context + ".theory_configuration");
    parseShadowTheoryConfiguration(theory,
        context + ".theory_configuration");
    return ShadowEvaluation{
        requestId, coefficients, gpdKinematics, observableKinematics};
}

void parsePseudodataTheoryConfiguration(const json::object& object,
        const std::string& context) {
    requireOnlyKeys(object,
        {"cff_module", "process_module", "observable_modules",
            "xi_converter_module", "scales_module",
            "coefficient_function_order", "factorization_scale_squared",
            "renormalization_scale_squared", "twist",
            "analysis_order_label"},
        context);
    if (requireString(object, "cff_module", context) !=
                "DVCSCFFStandard" ||
            requireString(object, "process_module", context) !=
                "DVCSProcessGV08" ||
            requireString(object, "xi_converter_module", context) !=
                "DVCSXiConverterXBToXi" ||
            requireString(object, "scales_module", context) !=
                "DVCSScalesQ2Multiplier" ||
            requireString(object, "coefficient_function_order", context) !=
                "LO" ||
            requireString(object, "factorization_scale_squared", context) !=
                "Q2" ||
            requireString(object, "renormalization_scale_squared", context) !=
                "Q2") {
        throw BridgeError("unsupported_perturbative_configuration",
            context + " does not match the frozen Stage 10 native route",
            kExitValidation);
    }
    const json::value observableValue =
        requireField(object, "observable_modules", context);
    if (!observableValue.is_array()) {
        throw BridgeError("invalid_type",
            context + ".observable_modules must be an array",
            kExitValidation);
    }
    const std::array<std::string, 6> expected{
        "DVCSCrossSectionUUMinus",
        "DVCSCrossSectionDifferenceLUMinus",
        "DVCSAc",
        "DVCSAluMinus",
        "DVCSAulMinus",
        "DVCSAllMinus"};
    const json::array modules = observableValue.as_array();
    if (modules.size() != expected.size()) {
        throw BridgeError("unsupported_observable_configuration",
            context + ".observable_modules does not match the verified "
                      "Stage 10 list",
            kExitValidation);
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!modules[index].is_string() ||
                std::string(modules[index].as_string()) != expected[index]) {
            throw BridgeError("unsupported_observable_configuration",
                context + ".observable_modules has the wrong order or name",
                kExitValidation);
        }
    }
    requireNull(object, "twist", context);
    requireNull(object, "analysis_order_label", context);
}

DVCSInference::DDParameters parsePseudodataDD(
        const json::object& shapes, const std::string& name,
        const std::string& context) {
    const json::object shape = requireObject(
        requireField(shapes, name, context), context + "." + name);
    requireOnlyKeys(shape,
        {"normalization", "a", "c", "profile_b", "t_slope"},
        context + "." + name);
    DVCSInference::DDParameters result{
        requireFiniteNumber(shape, "normalization", context + "." + name),
        requireFiniteNumber(shape, "a", context + "." + name),
        requireFiniteNumber(shape, "c", context + "." + name),
        requireFiniteNumber(shape, "profile_b", context + "." + name),
        requireParameterQuantity(
            shape, "t_slope", "GeV-2", context + "." + name)};
    requireClosedInterval(result.normalization, -4.0, 4.0,
        context + "." + name + ".normalization");
    if (result.a <= 0.1 || result.a >= 1.0 ||
            result.c <= 2.0 || result.c >= 6.0) {
        throw BridgeError("invalid_parameter",
            context + "." + name +
                " requires 0.1<a<1 and 2<c<6",
            kExitValidation);
    }
    requireClosedInterval(result.profileB, 1.0, 4.0,
        context + "." + name + ".profile_b");
    requireClosedInterval(result.tSlope, 0.0, 2.0,
        context + "." + name + ".t_slope.value");
    return result;
}

PseudodataEvaluation parsePseudodataEvaluation(
        const json::object& object, const std::string& context,
        bool topLevel = false, bool postTrainingComparison = false) {
    std::set<std::string> allowed{
        "request_id", "representation", "parameters", "quadrature_order",
        "q0_squared", "observable_kinematics", "evolution_configuration",
        "theory_configuration", "gpd_diagnostic_x"};
    if (topLevel) {
        allowed.insert("schema_version");
        allowed.insert("operation");
    }
    requireOnlyKeys(object, allowed, context);

    const std::string requestId = requireString(object, "request_id", context);
    if (requestId.empty()) {
        throw BridgeError("invalid_value",
            context + ".request_id must not be empty", kExitValidation);
    }
    if (requireString(object, "representation", context) !=
            "stage11_lo_multiq2_full_independent_dd_v1") {
        throw BridgeError("unsupported_representation",
            context + ".representation must be "
                      "'stage11_lo_multiq2_full_independent_dd_v1'",
            kExitValidation);
    }

    const json::object parameters = requireObject(
        requireField(object, "parameters", context), context + ".parameters");
    requireOnlyKeys(parameters,
        {"gpd_shapes", "shadow_coefficients", "shadow_channel_amplitudes"},
        context + ".parameters");
    const json::object shapes = requireObject(
        requireField(parameters, "gpd_shapes", context + ".parameters"),
        context + ".parameters.gpd_shapes");
    requireOnlyKeys(shapes, {"H", "E", "Htilde", "Etilde"},
        context + ".parameters.gpd_shapes");
    const json::object shadowCoefficients = requireObject(
        requireField(parameters, "shadow_coefficients",
            context + ".parameters"),
        context + ".parameters.shadow_coefficients");
    const json::object shadowAmplitudes = requireObject(
        requireField(parameters, "shadow_channel_amplitudes",
            context + ".parameters"),
        context + ".parameters.shadow_channel_amplitudes");
    const std::array<std::string, 4> typeNames{
        "H", "E", "Htilde", "Etilde"};
    const std::array<std::string, 4> channelNames{
        "u", "d", "s", "gluon"};
    requireOnlyKeys(shadowCoefficients,
        {"H", "E", "Htilde", "Etilde"},
        context + ".parameters.shadow_coefficients");
    requireOnlyKeys(shadowAmplitudes,
        {"H", "E", "Htilde", "Etilde"},
        context + ".parameters.shadow_channel_amplitudes");
    DVCSInference::PseudodataInputParameters nativeParameters{};
    for (std::size_t typeIndex = 0; typeIndex < typeNames.size(); ++typeIndex) {
        const std::string& typeName = typeNames[typeIndex];
        const json::object typeShapes = requireObject(
            requireField(shapes, typeName,
                context + ".parameters.gpd_shapes"),
            context + ".parameters.gpd_shapes." + typeName);
        const json::object typeAmplitudes = requireObject(
            requireField(shadowAmplitudes, typeName,
                context + ".parameters.shadow_channel_amplitudes"),
            context + ".parameters.shadow_channel_amplitudes." + typeName);
        requireOnlyKeys(typeShapes, {"u", "d", "s", "gluon"},
            context + ".parameters.gpd_shapes." + typeName);
        requireOnlyKeys(typeAmplitudes, {"u", "d", "s", "gluon"},
            context + ".parameters.shadow_channel_amplitudes." + typeName);
        for (std::size_t channel = 0; channel < channelNames.size(); ++channel) {
            nativeParameters.ddByTypeAndChannel[typeIndex][channel] =
                parsePseudodataDD(typeShapes, channelNames[channel],
                    context + ".parameters.gpd_shapes." + typeName);
            const double amplitude = requireFiniteNumber(
                typeAmplitudes, channelNames[channel],
                context + ".parameters.shadow_channel_amplitudes." + typeName);
            requireClosedInterval(amplitude, -4.0, 4.0,
                context + ".parameters.shadow_channel_amplitudes." +
                    typeName + "." + channelNames[channel]);
            nativeParameters.shadowAmplitudeByTypeAndChannel[typeIndex]
                [channel] = amplitude;
        }
        const double coefficient = requireFiniteNumber(
            shadowCoefficients, typeName,
            context + ".parameters.shadow_coefficients");
        requireClosedInterval(coefficient, -1.0, 1.0,
            context + ".parameters.shadow_coefficients." + typeName);
        nativeParameters.shadowCoefficientByType[typeIndex] = coefficient;
    }

    const unsigned int order =
        requireUnsignedInteger(object, "quadrature_order", context);
    if (order != 128U) {
        throw BridgeError("unsupported_quadrature_order",
            context + ".quadrature_order must be 128", kExitValidation);
    }
    const double q0Squared = requireParameterQuantity(
        object, "q0_squared", "GeV2", context);
    if (q0Squared !=
            DVCSInference::PseudodataInputGPD::defaultReferenceMuF2) {
        throw BridgeError("unsupported_scale",
            context + ".q0_squared must equal 1 GeV2", kExitValidation);
    }

    const json::object observable = requireObject(
        requireField(object, "observable_kinematics", context),
        context + ".observable_kinematics");
    requireOnlyKeys(observable,
        {"x_b", "t", "Q2", "beam_energy", "phi"},
        context + ".observable_kinematics");
    const ShadowObservableKinematics kinematics{
        requireFiniteNumber(
            observable, "x_b", context + ".observable_kinematics"),
        requireParameterQuantity(
            observable, "t", "GeV2", context + ".observable_kinematics"),
        requireParameterQuantity(
            observable, "Q2", "GeV2", context + ".observable_kinematics"),
        requireParameterQuantity(observable, "beam_energy", "GeV",
            context + ".observable_kinematics"),
        requireParameterQuantity(
            observable, "phi", "rad", context + ".observable_kinematics")};
    const bool commonDomain =
        kinematics.q2 >= q0Squared && kinematics.q2 <= 80.0 &&
        kinematics.phi >= 0.0 &&
        kinematics.phi <= 2.0 * std::acos(-1.0);
    if (postTrainingComparison) {
        if (!commonDomain || kinematics.xB <= 0.0 || kinematics.xB >= 1.0 ||
                kinematics.t < -1.0 || kinematics.t >= 0.0 ||
                kinematics.beamEnergy < 3.0 ||
                kinematics.beamEnergy > 200.0) {
            throw BridgeError("invalid_kinematics",
                context + ".observable_kinematics is outside the audited "
                          "post-training comparison domain",
                kExitValidation);
        }
    } else if (!commonDomain ||
            kinematics.xB < 0.12 || kinematics.xB > 0.28 ||
            kinematics.t < -0.2 || kinematics.t > -0.08 ||
            kinematics.beamEnergy < 3.0 ||
            kinematics.beamEnergy > 200.0) {
        throw BridgeError("invalid_kinematics",
            context + ".observable_kinematics is outside the frozen "
                      "milestone multi-Q2 domain",
            kExitValidation);
    }
    requirePhysicalFixedTargetInelasticity(
        kinematics, context + ".observable_kinematics");

    const json::object evolution = requireObject(
        requireField(object, "evolution_configuration", context),
        context + ".evolution_configuration");
    parseEvolutionConfiguration(
        evolution, context + ".evolution_configuration");
    const json::object theory = requireObject(
        requireField(object, "theory_configuration", context),
        context + ".theory_configuration");
    parsePseudodataTheoryConfiguration(
        theory, context + ".theory_configuration");
    std::vector<double> diagnosticX;
    const auto diagnosticIterator = object.find("gpd_diagnostic_x");
    if (diagnosticIterator != object.end()) {
        if (!diagnosticIterator->value().is_array()) {
            throw BridgeError("invalid_type",
                context + ".gpd_diagnostic_x must be an array",
                kExitValidation);
        }
        const json::array values = diagnosticIterator->value().as_array();
        if (values.empty() || values.size() > 129U) {
            throw BridgeError("invalid_value",
                context + ".gpd_diagnostic_x must contain 1..129 values",
                kExitValidation);
        }
        for (const json::value& value : values) {
            if (!value.is_number()) {
                throw BridgeError("invalid_type",
                    context + ".gpd_diagnostic_x values must be numbers",
                    kExitValidation);
            }
            const double x = value.to_number<double>();
            if (!std::isfinite(x) || x < -1.0 || x > 1.0) {
                throw BridgeError("outside_support",
                    context + ".gpd_diagnostic_x values must be in [-1,1]",
                    kExitValidation);
            }
            if (!diagnosticX.empty() && x <= diagnosticX.back()) {
                throw BridgeError("invalid_value",
                    context +
                        ".gpd_diagnostic_x must be strictly increasing",
                    kExitValidation);
            }
            diagnosticX.push_back(x);
        }
    }
    return PseudodataEvaluation{
        requestId, nativeParameters, order, q0Squared, kinematics,
        diagnosticX};
}

NativeModelHoldoutEvaluation parseNativeModelHoldoutEvaluation(
        const json::object& object, const std::string& context,
        bool topLevel = false) {
    std::set<std::string> allowed{
        "request_id", "representation", "model", "observable_kinematics",
        "theory_configuration", "gpd_diagnostic_x"};
    if (topLevel) {
        allowed.insert("schema_version");
        allowed.insert("operation");
    }
    requireOnlyKeys(object, allowed, context);
    const std::string requestId = requireString(object, "request_id", context);
    if (requestId.empty()) {
        throw BridgeError("invalid_value",
            context + ".request_id must not be empty", kExitValidation);
    }
    if (requireString(object, "representation", context) !=
            "stage11_native_model_holdout_v1") {
        throw BridgeError("unsupported_representation",
            context + ".representation must be "
                      "'stage11_native_model_holdout_v1'",
            kExitValidation);
    }
    const std::string model = requireString(object, "model", context);
    if (model != "GPDGK11" && model != "GPDGK16" &&
            model != "GPDGK19" && model != "GPDVGG99") {
        throw BridgeError("unsupported_module",
            context + ".model must be one of GPDGK11, GPDGK16, GPDGK19, "
                      "or GPDVGG99",
            kExitValidation);
    }
    const json::object observable = requireObject(
        requireField(object, "observable_kinematics", context),
        context + ".observable_kinematics");
    requireOnlyKeys(observable,
        {"x_b", "t", "Q2", "beam_energy", "phi"},
        context + ".observable_kinematics");
    const ShadowObservableKinematics kinematics{
        requireFiniteNumber(
            observable, "x_b", context + ".observable_kinematics"),
        requireParameterQuantity(
            observable, "t", "GeV2", context + ".observable_kinematics"),
        requireParameterQuantity(
            observable, "Q2", "GeV2", context + ".observable_kinematics"),
        requireParameterQuantity(observable, "beam_energy", "GeV",
            context + ".observable_kinematics"),
        requireParameterQuantity(
            observable, "phi", "rad", context + ".observable_kinematics")};
    if (kinematics.q2 < 1.0 || kinematics.q2 > 80.0 ||
            kinematics.xB <= 0.0 || kinematics.xB >= 1.0 ||
            kinematics.t < -1.0 || kinematics.t >= 0.0 ||
            kinematics.beamEnergy < 3.0 ||
            kinematics.beamEnergy > 200.0 ||
            kinematics.phi < 0.0 ||
            kinematics.phi > 2.0 * std::acos(-1.0)) {
        throw BridgeError("invalid_kinematics",
            context + ".observable_kinematics is outside the frozen "
                      "output-blind multi-Q2 holdout domain",
            kExitValidation);
    }
    requirePhysicalFixedTargetInelasticity(
        kinematics, context + ".observable_kinematics");
    const json::object theory = requireObject(
        requireField(object, "theory_configuration", context),
        context + ".theory_configuration");
    parsePseudodataTheoryConfiguration(
        theory, context + ".theory_configuration");

    std::vector<double> diagnosticX;
    if (const json::value* diagnosticValue =
            object.if_contains("gpd_diagnostic_x")) {
        if (!diagnosticValue->is_array()) {
            throw BridgeError("invalid_type",
                context + ".gpd_diagnostic_x must be an array",
                kExitValidation);
        }
        const json::array values = diagnosticValue->as_array();
        if (values.empty() || values.size() > 129U) {
            throw BridgeError("invalid_value",
                context + ".gpd_diagnostic_x must contain 1..129 values",
                kExitValidation);
        }
        for (const json::value& value : values) {
            if (!value.is_number()) {
                throw BridgeError("invalid_type",
                    context + ".gpd_diagnostic_x values must be numbers",
                    kExitValidation);
            }
            const double x = value.to_number<double>();
            if (!std::isfinite(x) || x < -1.0 || x > 1.0) {
                throw BridgeError("outside_support",
                    context + ".gpd_diagnostic_x values must be in [-1,1]",
                    kExitValidation);
            }
            if (!diagnosticX.empty() && x <= diagnosticX.back()) {
                throw BridgeError("invalid_value",
                    context + ".gpd_diagnostic_x must be strictly increasing",
                    kExitValidation);
            }
            diagnosticX.push_back(x);
        }
    }
    return NativeModelHoldoutEvaluation{
        requestId, model, kinematics, diagnosticX};
}

Evaluation parseEvaluation(const json::object& object,
        const std::string& context, bool topLevel = false) {
    std::set<std::string> allowed{
        "request_id", "module", "gpd_type", "kinematics"};
    if (topLevel) {
        allowed.insert("schema_version");
        allowed.insert("operation");
    }
    requireOnlyKeys(object, allowed, context);
    const std::string requestId =
        requireString(object, "request_id", context);
    if (requestId.empty()) {
        throw BridgeError("invalid_value",
                context + ".request_id must not be empty", kExitValidation);
    }
    if (requireString(object, "module", context) != "GPDGK16") {
        throw BridgeError("unsupported_module",
                context + ".module must be 'GPDGK16'", kExitValidation);
    }
    if (requireString(object, "gpd_type", context) != "H") {
        throw BridgeError("unsupported_gpd_type",
                context + ".gpd_type must be 'H'", kExitValidation);
    }

    const json::object input = requireObject(
        requireField(object, "kinematics", context),
        context + ".kinematics");
    requireOnlyKeys(input, {"x", "xi", "t", "mu_f2", "mu_r2"},
        context + ".kinematics");

    Kinematics kinematics{
        requireFiniteNumber(input, "x", context + ".kinematics"),
        requireFiniteNumber(input, "xi", context + ".kinematics"),
        requireQuantity(input, "t", "GeV2"),
        requireQuantity(input, "mu_f2", "GeV2"),
        requireQuantity(input, "mu_r2", "GeV2")};

    if (kinematics.x < -1.0 || kinematics.x > 1.0) {
        throw BridgeError("outside_support",
                context + ".kinematics.x must be in [-1, 1]",
                kExitValidation);
    }
    if (kinematics.xi < 0.0 || kinematics.xi > 1.0) {
        throw BridgeError("invalid_kinematics",
                context + ".kinematics.xi must be in [0, 1]",
                kExitValidation);
    }
    if (kinematics.t > 0.0) {
        throw BridgeError("invalid_kinematics",
                context + ".kinematics.t must use the non-positive GeV2 "
                          "PARTONS convention",
                kExitValidation);
    }
    if (kinematics.muF2 <= 0.0 || kinematics.muR2 <= 0.0) {
        throw BridgeError("invalid_kinematics",
                context + " factorization and renormalization scales must "
                          "be positive",
                kExitValidation);
    }

    return Evaluation{requestId, kinematics};
}

json::object kinematicsJson(const Kinematics& input) {
    return json::object{
        {"x", input.x},
        {"xi", input.xi},
        {"t", json::object{{"value", input.t}, {"unit", "GeV2"}}},
        {"mu_f2", json::object{{"value", input.muF2}, {"unit", "GeV2"}}},
        {"mu_r2", json::object{{"value", input.muR2}, {"unit", "GeV2"}}}};
}

class PartonsSession {
public:
    explicit PartonsSession(const char* executablePath)
        : m_partons(PARTONS::Partons::getInstance()) {
        const std::filesystem::path executable =
            std::filesystem::canonical(executablePath);
        std::filesystem::current_path(executable.parent_path());
        m_executablePath = executable.string();
        char* arguments[] = {m_executablePath.data(), nullptr};
        m_partons->init(1, arguments);
    }

    PartonsSession(const PartonsSession&) = delete;
    PartonsSession& operator=(const PartonsSession&) = delete;

    ~PartonsSession() {
        if (m_partons != nullptr) {
            m_partons->close();
        }
    }

    PARTONS::Partons* get() const {
        return m_partons;
    }

private:
    PARTONS::Partons* m_partons;
    std::string m_executablePath;
};

GPDValues evaluateGPD(PARTONS::Partons* partons,
        const Kinematics& input) {
    PARTONS::ModuleObjectFactory* factory =
        partons->getModuleObjectFactory();
    PARTONS::GPDModule* module =
        factory->newGPDModule(PARTONS::GPDGK16::classId);

    try {
        PARTONS::GPDService* service =
            partons->getServiceObjectRegistry()->getGPDService();
        const PARTONS::GPDResult result = service->computeSingleKinematic(
            PARTONS::GPDKinematic(input.x, input.xi, input.t,
                input.muF2, input.muR2),
            module);
        const PARTONS::PartonDistribution& h =
            result.getPartonDistribution(PARTONS::GPDType::H);
        const GPDValues values{
            h.getQuarkDistribution(PARTONS::QuarkFlavor::UP)
                .getQuarkDistribution(),
            h.getGluonDistribution().getGluonDistribution()};
        factory->updateModulePointerReference(module, 0);
        module = nullptr;

        if (!std::isfinite(values.up) || !std::isfinite(values.gluon)) {
            throw BridgeError("non_finite_backend_output",
                "PARTONS returned a non-finite GPD value", kExitNumerical);
        }
        return values;
    } catch (...) {
        if (module != nullptr) {
            factory->updateModulePointerReference(module, 0);
        }
        throw;
    }
}

DDValues evaluateDDGPD(PARTONS::Partons* partons,
        const DDEvaluation& evaluation) {
    PARTONS::ModuleObjectFactory* factory =
        partons->getModuleObjectFactory();
    PARTONS::GPDModule* module =
        factory->newGPDModule(DVCSInference::FixedScaleDDGPD::classId);
    PARTONS::GPDEvolutionModule* evolution = nullptr;
    PARTONS::RunningAlphaStrongModule* alpha = nullptr;
    PARTONS::ActiveFlavorsThresholdsModule* flavors = nullptr;

    try {
        auto* ddModule =
            dynamic_cast<DVCSInference::FixedScaleDDGPD*>(module);
        if (ddModule == nullptr) {
            throw BridgeError("backend_type_error",
                "PARTONS factory did not return FixedScaleDDGPD",
                kExitBackend);
        }
        ddModule->setParameters(evaluation.parameters);
        ddModule->setQuadratureOrder(evaluation.quadratureOrder);

        if (evaluation.evolved) {
            evolution = factory->newGPDEvolutionModule(
                PARTONS::GPDEvolutionApfel::classId);
            evolution->configure(ElemUtils::Parameters(
                PARTONS::PerturbativeQCDOrderType::
                    PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
                PARTONS::PerturbativeQCDOrderType::LO));

            alpha = factory->newRunningAlphaStrongModule(
                PARTONS::RunningAlphaStrongApfel::classId);
            ElemUtils::Parameters alphaParameters;
            alphaParameters.add(
                PARTONS::PerturbativeQCDOrderType::
                    PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
                PARTONS::PerturbativeQCDOrderType::LO);
            alphaParameters.add(
                PARTONS::RunningAlphaStrongApfel::PARAM_NAME_ALPHAS_REF,
                0.118);
            alphaParameters.add(
                PARTONS::RunningAlphaStrongApfel::PARAM_NAME_MU_REF,
                91.1876);
            alphaParameters.add(
                PARTONS::RunningAlphaStrongApfel::PARAM_NAME_THRESHOLDS,
                "0 0 0");
            alpha->configure(alphaParameters);

            flavors = factory->newActiveFlavorsThresholdsModule(
                PARTONS::ActiveFlavorsThresholdsConstant::classId);
            auto* constantFlavors =
                dynamic_cast<PARTONS::ActiveFlavorsThresholdsConstant*>(
                    flavors);
            if (constantFlavors == nullptr) {
                throw BridgeError("backend_type_error",
                    "PARTONS factory did not return "
                    "ActiveFlavorsThresholdsConstant",
                    kExitBackend);
            }
            constantFlavors->setNFlavors(3);
            evolution->setRunningAlphaStrongModule(alpha);
            evolution->setActiveFlavorsModule(flavors);
            module->setEvolQcdModule(evolution);
        }

        PARTONS::GPDService* service =
            partons->getServiceObjectRegistry()->getGPDService();
        const PARTONS::GPDResult result = service->computeSingleKinematic(
            PARTONS::GPDKinematic(evaluation.kinematics.x,
                std::abs(evaluation.kinematics.xi),
                evaluation.kinematics.t,
                evaluation.kinematics.muFSquared,
                evaluation.kinematics.muFSquared),
            module);
        const PARTONS::PartonDistribution& h =
            result.getPartonDistribution(PARTONS::GPDType::H);
        const PARTONS::QuarkDistribution& up =
            h.getQuarkDistribution(PARTONS::QuarkFlavor::UP);
        const DDValues values{up.getQuarkDistribution(),
            up.getQuarkDistributionPlus(),
            up.getQuarkDistributionMinus(),
            h.getGluonDistribution().getGluonDistribution()};
        factory->updateModulePointerReference(module, 0);
        module = nullptr;
        if (evolution != nullptr) {
            factory->updateModulePointerReference(evolution, 0);
            evolution = nullptr;
        }
        if (alpha != nullptr) {
            factory->updateModulePointerReference(alpha, 0);
            alpha = nullptr;
        }
        if (flavors != nullptr) {
            factory->updateModulePointerReference(flavors, 0);
            flavors = nullptr;
        }

        if (!std::isfinite(values.up) || !std::isfinite(values.upPlus) ||
                !std::isfinite(values.upMinus) ||
                !std::isfinite(values.gluon)) {
            throw BridgeError("non_finite_backend_output",
                "native DD evolution route returned a non-finite value",
                kExitNumerical);
        }
        return values;
    } catch (...) {
        if (module != nullptr) {
            factory->updateModulePointerReference(module, 0);
        }
        if (evolution != nullptr) {
            factory->updateModulePointerReference(evolution, 0);
        }
        if (alpha != nullptr) {
            factory->updateModulePointerReference(alpha, 0);
        }
        if (flavors != nullptr) {
            factory->updateModulePointerReference(flavors, 0);
        }
        throw;
    }
}

ConformalValues evaluateConformalGPD(PARTONS::Partons* partons,
        const ConformalEvaluation& evaluation) {
    PARTONS::ModuleObjectFactory* factory =
        partons->getModuleObjectFactory();
    PARTONS::GPDModule* module =
        factory->newGPDModule(DVCSInference::ConformalMomentGPD::classId);
    try {
        auto* conformalModule =
            dynamic_cast<DVCSInference::ConformalMomentGPD*>(module);
        if (conformalModule == nullptr) {
            throw BridgeError("backend_type_error",
                "PARTONS factory did not return ConformalMomentGPD",
                kExitBackend);
        }
        conformalModule->setParameters(evaluation.parameters);
        conformalModule->setIntegrationLimit(evaluation.integrationLimit);

        PARTONS::GPDService* service =
            partons->getServiceObjectRegistry()->getGPDService();
        const PARTONS::GPDResult result = service->computeSingleKinematic(
            PARTONS::GPDKinematic(evaluation.kinematics.x, 0.0,
                evaluation.kinematics.t,
                evaluation.kinematics.muFSquared,
                evaluation.kinematics.muFSquared),
            module);
        const PARTONS::PartonDistribution& h =
            result.getPartonDistribution(PARTONS::GPDType::H);
        const PARTONS::QuarkDistribution& up =
            h.getQuarkDistribution(PARTONS::QuarkFlavor::UP);
        const ConformalValues values{
            up.getQuarkDistribution(),
            up.getQuarkDistributionPlus(),
            up.getQuarkDistributionMinus()};
        factory->updateModulePointerReference(module, 0);
        module = nullptr;

        if (!std::isfinite(values.up) || !std::isfinite(values.upPlus) ||
                !std::isfinite(values.upMinus)) {
            throw BridgeError("non_finite_backend_output",
                "native conformal-moment route returned a non-finite value",
                kExitNumerical);
        }
        return values;
    } catch (...) {
        if (module != nullptr) {
            factory->updateModulePointerReference(module, 0);
        }
        throw;
    }
}

BasisValues evaluateBasisGPD(PARTONS::Partons* partons,
        const BasisEvaluation& evaluation) {
    PARTONS::ModuleObjectFactory* factory =
        partons->getModuleObjectFactory();
    PARTONS::GPDModule* module =
        factory->newGPDModule(DVCSInference::BasisInjectionGPD::classId);
    PARTONS::GPDEvolutionModule* evolution = nullptr;
    PARTONS::RunningAlphaStrongModule* alpha = nullptr;
    PARTONS::ActiveFlavorsThresholdsModule* flavors = nullptr;

    try {
        auto* basisModule =
            dynamic_cast<DVCSInference::BasisInjectionGPD*>(module);
        if (basisModule == nullptr) {
            throw BridgeError("backend_type_error",
                "PARTONS factory did not return BasisInjectionGPD",
                kExitBackend);
        }
        basisModule->setShapeParameters(evaluation.shapeParameters);
        basisModule->setQuadratureOrder(evaluation.quadratureOrder);
        basisModule->setAmplitudes(evaluation.amplitudes);

        evolution = factory->newGPDEvolutionModule(
            PARTONS::GPDEvolutionApfel::classId);
        evolution->configure(ElemUtils::Parameters(
            PARTONS::PerturbativeQCDOrderType::
                PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::LO));

        alpha = factory->newRunningAlphaStrongModule(
            PARTONS::RunningAlphaStrongApfel::classId);
        ElemUtils::Parameters alphaParameters;
        alphaParameters.add(
            PARTONS::PerturbativeQCDOrderType::
                PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::LO);
        alphaParameters.add(
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_ALPHAS_REF, 0.118);
        alphaParameters.add(
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_MU_REF, 91.1876);
        alphaParameters.add(
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_THRESHOLDS,
            "0 0 0");
        alpha->configure(alphaParameters);

        flavors = factory->newActiveFlavorsThresholdsModule(
            PARTONS::ActiveFlavorsThresholdsConstant::classId);
        auto* constantFlavors =
            dynamic_cast<PARTONS::ActiveFlavorsThresholdsConstant*>(flavors);
        if (constantFlavors == nullptr) {
            throw BridgeError("backend_type_error",
                "PARTONS factory did not return "
                "ActiveFlavorsThresholdsConstant",
                kExitBackend);
        }
        constantFlavors->setNFlavors(3);
        evolution->setRunningAlphaStrongModule(alpha);
        evolution->setActiveFlavorsModule(flavors);
        module->setEvolQcdModule(evolution);

        PARTONS::GPDService* service =
            partons->getServiceObjectRegistry()->getGPDService();
        const PARTONS::GPDResult result = service->computeSingleKinematic(
            PARTONS::GPDKinematic(evaluation.kinematics.x,
                std::abs(evaluation.kinematics.xi),
                evaluation.kinematics.t,
                evaluation.kinematics.muFSquared,
                evaluation.kinematics.muFSquared),
            module);
        const PARTONS::PartonDistribution& h =
            result.getPartonDistribution(PARTONS::GPDType::H);
        BasisValues values{};
        const std::array<PARTONS::QuarkFlavor::Type, 6> flavorTypes{
            PARTONS::QuarkFlavor::UP, PARTONS::QuarkFlavor::DOWN,
            PARTONS::QuarkFlavor::STRANGE, PARTONS::QuarkFlavor::CHARM,
            PARTONS::QuarkFlavor::BOTTOM, PARTONS::QuarkFlavor::TOP};
        const auto& distributions = h.getQuarkDistributions();
        for (std::size_t index = 0; index < flavorTypes.size(); ++index) {
            const auto found = distributions.find(flavorTypes[index]);
            if (found != distributions.end()) {
                values.quark[index] =
                    found->second.getQuarkDistribution();
                values.plus[index] =
                    found->second.getQuarkDistributionPlus();
                values.minus[index] =
                    found->second.getQuarkDistributionMinus();
            }
        }
        values.gluon =
            h.getGluonDistribution().getGluonDistribution();

        factory->updateModulePointerReference(module, 0);
        module = nullptr;
        factory->updateModulePointerReference(evolution, 0);
        evolution = nullptr;
        factory->updateModulePointerReference(alpha, 0);
        alpha = nullptr;
        factory->updateModulePointerReference(flavors, 0);
        flavors = nullptr;

        if (!std::isfinite(values.gluon)) {
            throw BridgeError("non_finite_backend_output",
                "native basis evolution route returned a non-finite gluon",
                kExitNumerical);
        }
        for (std::size_t index = 0; index < values.quark.size(); ++index) {
            if (!std::isfinite(values.quark[index]) ||
                    !std::isfinite(values.plus[index]) ||
                    !std::isfinite(values.minus[index])) {
                throw BridgeError("non_finite_backend_output",
                    "native basis evolution route returned a non-finite "
                    "quark value",
                    kExitNumerical);
            }
        }
        return values;
    } catch (...) {
        if (module != nullptr) {
            factory->updateModulePointerReference(module, 0);
        }
        if (evolution != nullptr) {
            factory->updateModulePointerReference(evolution, 0);
        }
        if (alpha != nullptr) {
            factory->updateModulePointerReference(alpha, 0);
        }
        if (flavors != nullptr) {
            factory->updateModulePointerReference(flavors, 0);
        }
        throw;
    }
}

ShadowValues evaluateShadowDVCS(PARTONS::Partons* partons,
        const ShadowEvaluation& evaluation) {
    PARTONS::ModuleObjectFactory* factory =
        partons->getModuleObjectFactory();
    PARTONS::GPDModule* baseline = nullptr;
    PARTONS::GPDModule* nativeShadow = nullptr;
    PARTONS::GPDModule* combined = nullptr;
    PARTONS::DVCSConvolCoeffFunctionModule* cff = nullptr;
    PARTONS::DVCSXiConverterModule* xi = nullptr;
    PARTONS::DVCSScalesModule* scales = nullptr;
    PARTONS::DVCSProcessModule* process = nullptr;
    PARTONS::DVCSObservable* observable = nullptr;

    try {
        baseline = factory->newGPDModule(PARTONS::GPDGK16::classId);
        nativeShadow =
            factory->newGPDModule(PARTONS::GPDBDMMS21::classId);
        combined = factory->newGPDModule(
            DVCSInference::ShadowInjectionGPD::classId);
        auto* shadowModule =
            dynamic_cast<DVCSInference::ShadowInjectionGPD*>(combined);
        if (shadowModule == nullptr) {
            throw BridgeError("backend_type_error",
                "PARTONS factory did not return ShadowInjectionGPD",
                kExitBackend);
        }
        shadowModule->setBaselineModule(baseline);
        shadowModule->setShadowModule(nativeShadow);
        shadowModule->setCoefficients(evaluation.coefficients);

        const PARTONS::GPDKinematic diagnosticKinematics(
            evaluation.gpdKinematics.x,
            evaluation.gpdKinematics.xi,
            evaluation.gpdKinematics.t,
            evaluation.gpdKinematics.muF2,
            evaluation.gpdKinematics.muR2);
        const auto upValue = [](const PARTONS::PartonDistribution& input) {
            return input
                .getQuarkDistribution(PARTONS::QuarkFlavor::UP)
                .getQuarkDistribution();
        };
        const double hUpBaseline = upValue(
            baseline->compute(diagnosticKinematics, PARTONS::GPDType::H));
        const double hUpShadow = upValue(
            nativeShadow->compute(diagnosticKinematics, PARTONS::GPDType::H));
        const double hUpCombined = upValue(
            combined->compute(diagnosticKinematics, PARTONS::GPDType::H));

        cff = factory->newDVCSConvolCoeffFunctionModule(
            PARTONS::DVCSCFFStandard::classId);
        cff->configure(ElemUtils::Parameter(
            PARTONS::PerturbativeQCDOrderType::
                PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::NLO));
        cff->setGPDModule(combined);
        xi = factory->newDVCSXiConverterModule(
            PARTONS::DVCSXiConverterXBToXi::classId);
        scales = factory->newDVCSScalesModule(
            PARTONS::DVCSScalesQ2Multiplier::classId);
        scales->configure(ElemUtils::Parameter(
            PARTONS::DVCSScalesQ2Multiplier::PARAMETER_NAME_LAMBDA,
            1.0));
        process = factory->newDVCSProcessModule(
            PARTONS::DVCSProcessGV08::classId);
        observable = factory->newDVCSObservable(
            PARTONS::DVCSCrossSectionUUMinus::classId);
        observable->setProcessModule(process);
        process->setScaleModule(scales);
        process->setXiConverterModule(xi);
        process->setConvolCoeffFunctionModule(cff);

        const PARTONS::DVCSObservableKinematic observableKinematics(
            evaluation.observableKinematics.xB,
            evaluation.observableKinematics.t,
            evaluation.observableKinematics.q2,
            evaluation.observableKinematics.beamEnergy,
            evaluation.observableKinematics.phi);
        const double cffXi = xi->compute(observableKinematics).getValue();
        PARTONS::DVCSConvolCoeffFunctionService* cffService =
            partons->getServiceObjectRegistry()
                ->getDVCSConvolCoeffFunctionService();
        const std::complex<double> cffH =
            cffService->computeSingleKinematic(
                PARTONS::DVCSConvolCoeffFunctionKinematic(
                    cffXi,
                    evaluation.observableKinematics.t,
                    evaluation.observableKinematics.q2,
                    evaluation.observableKinematics.q2,
                    evaluation.observableKinematics.q2),
                cff)
                .getResult(PARTONS::GPDType::H);

        PARTONS::DVCSObservableService* observableService =
            partons->getServiceObjectRegistry()->getDVCSObservableService();
        const auto observableResult =
            observableService->computeSingleKinematic(
                observableKinematics, observable);
        const double observableValue =
            observableResult.getValue().getValue();
        const std::string observableUnit =
            PARTONS::PhysicalUnit(observableResult.getValue().getUnit())
                .toString();

        const ShadowValues values{hUpCombined, hUpBaseline, hUpShadow,
            cffH, cffXi, observableValue, observableUnit};

        factory->updateModulePointerReference(observable, nullptr);
        observable = nullptr;
        factory->updateModulePointerReference(process, nullptr);
        process = nullptr;
        factory->updateModulePointerReference(scales, nullptr);
        scales = nullptr;
        factory->updateModulePointerReference(xi, nullptr);
        xi = nullptr;
        factory->updateModulePointerReference(cff, nullptr);
        cff = nullptr;
        factory->updateModulePointerReference(combined, nullptr);
        combined = nullptr;
        factory->updateModulePointerReference(nativeShadow, nullptr);
        nativeShadow = nullptr;
        factory->updateModulePointerReference(baseline, nullptr);
        baseline = nullptr;

        if (!std::isfinite(values.hUpCombined) ||
                !std::isfinite(values.hUpBaselineComponent) ||
                !std::isfinite(values.hUpShadowComponent) ||
                !std::isfinite(values.cffH.real()) ||
                !std::isfinite(values.cffH.imag()) ||
                !std::isfinite(values.cffXi) ||
                !std::isfinite(values.observable)) {
            throw BridgeError("non_finite_backend_output",
                "native shadow DVCS route returned a non-finite value",
                kExitNumerical);
        }
        return values;
    } catch (...) {
        if (observable != nullptr) {
            factory->updateModulePointerReference(observable, nullptr);
        }
        if (process != nullptr) {
            factory->updateModulePointerReference(process, nullptr);
        }
        if (scales != nullptr) {
            factory->updateModulePointerReference(scales, nullptr);
        }
        if (xi != nullptr) {
            factory->updateModulePointerReference(xi, nullptr);
        }
        if (cff != nullptr) {
            factory->updateModulePointerReference(cff, nullptr);
        }
        if (combined != nullptr) {
            factory->updateModulePointerReference(combined, nullptr);
        }
        if (nativeShadow != nullptr) {
            factory->updateModulePointerReference(nativeShadow, nullptr);
        }
        if (baseline != nullptr) {
            factory->updateModulePointerReference(baseline, nullptr);
        }
        throw;
    }
}

PseudodataValues evaluatePseudodataDVCS(PARTONS::Partons* partons,
        const PseudodataEvaluation& evaluation) {
    PARTONS::ModuleObjectFactory* factory =
        partons->getModuleObjectFactory();
    PARTONS::GPDModule* combined = nullptr;
    PARTONS::GPDModule* tabulated = nullptr;
    PARTONS::GPDModule* dd = nullptr;
    PARTONS::GPDModule* nativeShadow = nullptr;
    PARTONS::GPDEvolutionModule* evolution = nullptr;
    std::array<PARTONS::GPDEvolutionModule*, 4> evolutionTables{};
    PARTONS::RunningAlphaStrongModule* alpha = nullptr;
    PARTONS::ActiveFlavorsThresholdsModule* flavors = nullptr;
    PARTONS::DVCSConvolCoeffFunctionModule* cff = nullptr;
    PARTONS::DVCSXiConverterModule* xi = nullptr;
    PARTONS::DVCSScalesModule* scales = nullptr;
    PARTONS::DVCSProcessModule* process = nullptr;
    std::array<PARTONS::DVCSObservable*, 6> observables{};

    try {
        combined = factory->newGPDModule(
            DVCSInference::PseudodataInputGPD::classId);
        dd = factory->newGPDModule(
            DVCSInference::FixedScaleDDGPD::classId);
        nativeShadow =
            factory->newGPDModule(PARTONS::GPDBDMMS21::classId);
        auto* input =
            dynamic_cast<DVCSInference::PseudodataInputGPD*>(combined);
        auto* ddInput = dynamic_cast<DVCSInference::FixedScaleDDGPD*>(dd);
        if (input == nullptr || ddInput == nullptr) {
            throw BridgeError("backend_type_error",
                "PARTONS factory did not return the Stage 10 input modules",
                kExitBackend);
        }
        input->setDDModule(ddInput);
        input->setShadowModule(nativeShadow);
        input->setParameters(evaluation.parameters);
        input->setQuadratureOrder(evaluation.quadratureOrder);
        input->setReferenceMuF2(evaluation.q0Squared);

        // Parameterize once at Q0 and attach the audited native APFEL++ route.
        // PARTONS evaluates directly at Q0 and evolves the GPD—not data—to
        // each datum's factorization scale when Q2 differs from Q0^2.
        evolution = factory->newGPDEvolutionModule(
            PARTONS::GPDEvolutionApfel::classId);
        evolution->configure(ElemUtils::Parameter(
            PARTONS::PerturbativeQCDOrderType::
                PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::LO));
        alpha = factory->newRunningAlphaStrongModule(
            PARTONS::RunningAlphaStrongApfel::classId);
        ElemUtils::Parameters alphaParameters;
        alphaParameters.add(
            PARTONS::PerturbativeQCDOrderType::
                PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::LO);
        alphaParameters.add(
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_ALPHAS_REF, 0.118);
        alphaParameters.add(
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_MU_REF, 91.1876);
        alphaParameters.add(
            PARTONS::RunningAlphaStrongApfel::PARAM_NAME_THRESHOLDS,
            "0 0 0");
        alpha->configure(alphaParameters);
        flavors = factory->newActiveFlavorsThresholdsModule(
            PARTONS::ActiveFlavorsThresholdsConstant::classId);
        auto* fixedFlavors =
            dynamic_cast<PARTONS::ActiveFlavorsThresholdsConstant*>(flavors);
        if (fixedFlavors == nullptr) {
            throw BridgeError("backend_type_error",
                "PARTONS factory did not return constant flavors",
                kExitBackend);
        }
        fixedFlavors->setNFlavors(3);
        evolution->setRunningAlphaStrongModule(alpha);
        evolution->setActiveFlavorsModule(flavors);
        evolutionTables[0] = evolution;
        for (std::size_t index = 1; index < evolutionTables.size(); ++index) {
            evolutionTables[index] = factory->newGPDEvolutionModule(
                PARTONS::GPDEvolutionApfel::classId);
            evolutionTables[index]->configure(ElemUtils::Parameter(
                PARTONS::PerturbativeQCDOrderType::
                    PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
                PARTONS::PerturbativeQCDOrderType::LO));
            evolutionTables[index]->setRunningAlphaStrongModule(alpha);
            evolutionTables[index]->setActiveFlavorsModule(flavors);
        }
        combined->setEvolQcdModule(evolution);

        cff = factory->newDVCSConvolCoeffFunctionModule(
            PARTONS::DVCSCFFStandard::classId);
        cff->configure(ElemUtils::Parameter(
            PARTONS::PerturbativeQCDOrderType::
                PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::LO));
        auto* standard = dynamic_cast<PARTONS::DVCSCFFStandard*>(cff);
        if (standard == nullptr) {
            throw BridgeError("backend_type_error",
                "PARTONS factory did not return DVCSCFFStandard",
                kExitBackend);
        }
        standard->setRunningAlphaStrongModule(alpha);

        xi = factory->newDVCSXiConverterModule(
            PARTONS::DVCSXiConverterXBToXi::classId);
        scales = factory->newDVCSScalesModule(
            PARTONS::DVCSScalesQ2Multiplier::classId);
        scales->configure(ElemUtils::Parameter(
            PARTONS::DVCSScalesQ2Multiplier::PARAMETER_NAME_LAMBDA,
            1.0));
        process = factory->newDVCSProcessModule(
            PARTONS::DVCSProcessGV08::classId);
        process->setScaleModule(scales);
        process->setXiConverterModule(xi);
        process->setConvolCoeffFunctionModule(cff);
        const std::array<unsigned int, 6> observableClassIds{
            PARTONS::DVCSCrossSectionUUMinus::classId,
            PARTONS::DVCSCrossSectionDifferenceLUMinus::classId,
            PARTONS::DVCSAc::classId,
            PARTONS::DVCSAluMinus::classId,
            PARTONS::DVCSAulMinus::classId,
            PARTONS::DVCSAllMinus::classId};
        for (std::size_t index = 0; index < observables.size(); ++index) {
            observables[index] =
                factory->newDVCSObservable(observableClassIds[index]);
            observables[index]->setProcessModule(process);
        }

        const PARTONS::DVCSObservableKinematic observableKinematics(
            evaluation.observableKinematics.xB,
            evaluation.observableKinematics.t,
            evaluation.observableKinematics.q2,
            evaluation.observableKinematics.beamEnergy,
            evaluation.observableKinematics.phi);
        const double cffXi = xi->compute(observableKinematics).getValue();
        PARTONS::GPDModule* evaluationGPD = combined;
        if (evaluation.observableKinematics.q2 != evaluation.q0Squared) {
            tabulated = factory->newGPDModule(
                DVCSInference::TabulatedPseudodataGPD::classId);
            auto* tableAdapter = dynamic_cast<
                DVCSInference::TabulatedPseudodataGPD*>(tabulated);
            if (tableAdapter == nullptr) {
                throw BridgeError("backend_type_error",
                    "native evolution table adapter construction failed",
                    kExitBackend);
            }
            const std::array<PARTONS::GPDType::Type, 4> evolvedTypes{
                PARTONS::GPDType::H, PARTONS::GPDType::E,
                PARTONS::GPDType::Ht, PARTONS::GPDType::Et};
            std::array<PARTONS::PartonDistribution, 4> directAnchors{};
            for (std::size_t typeIndex = 0;
                    typeIndex < evolvedTypes.size(); ++typeIndex) {
                combined->setEvolQcdModule(evolutionTables[typeIndex]);
                auto* nativeEvolution =
                    dynamic_cast<PARTONS::GPDEvolutionApfel*>(
                        evolutionTables[typeIndex]);
                if (nativeEvolution == nullptr) {
                    throw BridgeError("backend_type_error",
                        "PARTONS evolution module type changed",
                        kExitBackend);
                }
                directAnchors[typeIndex] = combined->compute(
                    PARTONS::GPDKinematic(
                    cffXi, cffXi, evaluation.observableKinematics.t,
                    evaluation.observableKinematics.q2,
                    evaluation.observableKinematics.q2),
                    evolvedTypes[typeIndex]);
                tableAdapter->setTable(
                    typeIndex, nativeEvolution->getTabulatedGPDs());
            }
            const std::array<PARTONS::QuarkFlavor::Type, 3> anchorFlavors{
                PARTONS::QuarkFlavor::UP, PARTONS::QuarkFlavor::DOWN,
                PARTONS::QuarkFlavor::STRANGE};
            const auto close = [](double first, double second) {
                return std::abs(first - second) <= 1.e-11 *
                    std::max({1., std::abs(first), std::abs(second)});
            };
            for (std::size_t typeIndex = 0;
                    typeIndex < evolvedTypes.size(); ++typeIndex) {
                const PARTONS::PartonDistribution cached = tabulated->compute(
                    PARTONS::GPDKinematic(
                        cffXi, cffXi, evaluation.observableKinematics.t,
                        evaluation.observableKinematics.q2,
                        evaluation.observableKinematics.q2),
                    evolvedTypes[typeIndex]);
                for (const auto flavor : anchorFlavors) {
                    const auto& expected = directAnchors[typeIndex]
                        .getQuarkDistribution(flavor);
                    const auto& observed = cached.getQuarkDistribution(flavor);
                    if (!close(expected.getQuarkDistribution(),
                               observed.getQuarkDistribution()) ||
                            !close(expected.getQuarkDistributionPlus(),
                                   observed.getQuarkDistributionPlus()) ||
                            !close(expected.getQuarkDistributionMinus(),
                                   observed.getQuarkDistributionMinus())) {
                        throw BridgeError("evolution_cache_mismatch",
                            "tabulated native evolution anchor changed",
                            kExitNumerical);
                    }
                }
                if (!close(directAnchors[typeIndex].getGluonDistribution()
                               .getGluonDistribution(),
                           cached.getGluonDistribution()
                               .getGluonDistribution())) {
                    throw BridgeError("evolution_cache_mismatch",
                        "tabulated native gluon evolution anchor changed",
                        kExitNumerical);
                }
            }
            evaluationGPD = tabulated;
        }
        cff->setGPDModule(evaluationGPD);
        PARTONS::DVCSConvolCoeffFunctionService* cffService =
            partons->getServiceObjectRegistry()
                ->getDVCSConvolCoeffFunctionService();
        const PARTONS::DVCSConvolCoeffFunctionResult cffResult =
            cffService->computeSingleKinematic(
                PARTONS::DVCSConvolCoeffFunctionKinematic(
                    cffXi,
                    evaluation.observableKinematics.t,
                    evaluation.observableKinematics.q2,
                    evaluation.observableKinematics.q2,
                    evaluation.observableKinematics.q2),
                cff);
        const std::array<std::complex<double>, 4> cffs{
            cffResult.getResult(PARTONS::GPDType::H),
            cffResult.getResult(PARTONS::GPDType::E),
            cffResult.getResult(PARTONS::GPDType::Ht),
            cffResult.getResult(PARTONS::GPDType::Et)};

        std::vector<PseudodataGPDValues> gpdDiagnostics;
        if (!evaluation.diagnosticX.empty()) {
            PARTONS::GPDService* gpdService =
                partons->getServiceObjectRegistry()->getGPDService();
            const std::array<PARTONS::GPDType::Type, 4> gpdTypes{
                PARTONS::GPDType::H, PARTONS::GPDType::E,
                PARTONS::GPDType::Ht, PARTONS::GPDType::Et};
            const std::array<PARTONS::QuarkFlavor::Type, 3> quarkFlavors{
                PARTONS::QuarkFlavor::UP,
                PARTONS::QuarkFlavor::DOWN,
                PARTONS::QuarkFlavor::STRANGE};
            for (double xValue : evaluation.diagnosticX) {
                const PARTONS::GPDResult result =
                    gpdService->computeSingleKinematic(
                        PARTONS::GPDKinematic(
                            xValue, cffXi,
                            evaluation.observableKinematics.t,
                            evaluation.observableKinematics.q2,
                            evaluation.observableKinematics.q2),
                        evaluationGPD);
                PseudodataGPDValues point{xValue, {}};
                for (std::size_t index = 0; index < gpdTypes.size(); ++index) {
                    const PARTONS::PartonDistribution& distribution =
                        result.getPartonDistribution(gpdTypes[index]);
                    for (std::size_t flavorIndex = 0;
                            flavorIndex < quarkFlavors.size(); ++flavorIndex) {
                        const PARTONS::QuarkDistribution& quark =
                            distribution.getQuarkDistribution(
                                quarkFlavors[flavorIndex]);
                        point.byType[index].quarks[flavorIndex] = DDValues{
                            quark.getQuarkDistribution(),
                            quark.getQuarkDistributionPlus(),
                            quark.getQuarkDistributionMinus(), 0.0};
                    }
                    point.byType[index].gluon = distribution
                        .getGluonDistribution().getGluonDistribution();
                }
                gpdDiagnostics.push_back(point);
            }
        }

        PARTONS::DVCSObservableService* observableService =
            partons->getServiceObjectRegistry()->getDVCSObservableService();
        std::array<double, 6> observableValues{};
        std::array<std::string, 6> observableUnits{};
        for (std::size_t index = 0; index < observables.size(); ++index) {
            const auto result = observableService->computeSingleKinematic(
                observableKinematics, observables[index]);
            observableValues[index] = result.getValue().getValue();
            observableUnits[index] =
                PARTONS::PhysicalUnit(result.getValue().getUnit()).toString();
        }
        const PseudodataValues values{
            cffs, cffXi, observableValues, observableUnits, gpdDiagnostics};

        for (PARTONS::DVCSObservable*& observable : observables) {
            factory->updateModulePointerReference(observable, nullptr);
            observable = nullptr;
        }
        factory->updateModulePointerReference(process, nullptr);
        process = nullptr;
        factory->updateModulePointerReference(scales, nullptr);
        scales = nullptr;
        factory->updateModulePointerReference(xi, nullptr);
        xi = nullptr;
        factory->updateModulePointerReference(cff, nullptr);
        cff = nullptr;
        if (tabulated != nullptr) {
            factory->updateModulePointerReference(tabulated, nullptr);
            tabulated = nullptr;
        }
        factory->updateModulePointerReference(combined, nullptr);
        combined = nullptr;
        for (PARTONS::GPDEvolutionModule*& item : evolutionTables) {
            factory->updateModulePointerReference(item, nullptr);
            item = nullptr;
        }
        evolution = nullptr;
        factory->updateModulePointerReference(alpha, nullptr);
        alpha = nullptr;
        factory->updateModulePointerReference(flavors, nullptr);
        flavors = nullptr;
        factory->updateModulePointerReference(dd, nullptr);
        dd = nullptr;
        factory->updateModulePointerReference(nativeShadow, nullptr);
        nativeShadow = nullptr;

        const auto requireFinite = [&](double value,
                const std::string& component) {
            if (std::isfinite(value)) {
                return;
            }
            std::ostringstream message;
            message << "native pseudodata DVCS component '" << component
                    << "' is non-finite for request '"
                    << evaluation.requestId << "' at xB="
                    << evaluation.observableKinematics.xB << ", t="
                    << evaluation.observableKinematics.t << " GeV2, Q2="
                    << evaluation.observableKinematics.q2
                    << " GeV2, beam_energy="
                    << evaluation.observableKinematics.beamEnergy
                    << " GeV, phi="
                    << evaluation.observableKinematics.phi << " rad";
            throw BridgeError("non_finite_backend_output",
                message.str(), kExitNumerical);
        };
        requireFinite(values.cffXi, "cff_xi");
        const std::array<std::string, 4> gpdTypeNames{
            "H", "E", "Htilde", "Etilde"};
        for (std::size_t index = 0; index < values.cffs.size(); ++index) {
            requireFinite(values.cffs[index].real(),
                "cff." + gpdTypeNames[index] + ".real");
            requireFinite(values.cffs[index].imag(),
                "cff." + gpdTypeNames[index] + ".imaginary");
        }
        const std::array<std::string, 6> observableNames{
            "DVCSCrossSectionUUMinus",
            "DVCSCrossSectionDifferenceLUMinus", "DVCSAc",
            "DVCSAluMinus", "DVCSAulMinus", "DVCSAllMinus"};
        for (std::size_t index = 0;
                index < values.observables.size(); ++index) {
            requireFinite(values.observables[index],
                "observable." + observableNames[index]);
        }
        const std::array<std::string, 3> flavorNames{"u", "d", "s"};
        for (std::size_t pointIndex = 0;
                pointIndex < values.gpdDiagnostics.size(); ++pointIndex) {
            const PseudodataGPDValues& point =
                values.gpdDiagnostics[pointIndex];
            requireFinite(point.x, "gpd_diagnostics[" +
                std::to_string(pointIndex) + "].x");
            for (std::size_t typeIndex = 0;
                    typeIndex < point.byType.size(); ++typeIndex) {
                const auto& distribution = point.byType[typeIndex];
                const std::string prefix = "gpd_diagnostics[" +
                    std::to_string(pointIndex) + "]." +
                    gpdTypeNames[typeIndex] + ".";
                requireFinite(distribution.gluon, prefix + "gluon");
                for (std::size_t flavorIndex = 0;
                        flavorIndex < distribution.quarks.size();
                        ++flavorIndex) {
                    const DDValues& value =
                        distribution.quarks[flavorIndex];
                    const std::string flavorPrefix = prefix +
                        flavorNames[flavorIndex];
                    requireFinite(value.up, flavorPrefix + ".value");
                    requireFinite(value.upPlus, flavorPrefix + ".plus");
                    requireFinite(value.upMinus, flavorPrefix + ".minus");
                }
            }
        }
        return values;
    } catch (...) {
        for (PARTONS::DVCSObservable*& observable : observables) {
            if (observable != nullptr) {
                factory->updateModulePointerReference(observable, nullptr);
            }
        }
        if (process != nullptr) {
            factory->updateModulePointerReference(process, nullptr);
        }
        if (scales != nullptr) {
            factory->updateModulePointerReference(scales, nullptr);
        }
        if (xi != nullptr) {
            factory->updateModulePointerReference(xi, nullptr);
        }
        if (cff != nullptr) {
            factory->updateModulePointerReference(cff, nullptr);
        }
        if (tabulated != nullptr) {
            factory->updateModulePointerReference(tabulated, nullptr);
        }
        if (combined != nullptr) {
            factory->updateModulePointerReference(combined, nullptr);
        }
        for (PARTONS::GPDEvolutionModule*& item : evolutionTables) {
            if (item != nullptr) {
                factory->updateModulePointerReference(item, nullptr);
                item = nullptr;
            }
        }
        evolution = nullptr;
        if (alpha != nullptr) {
            factory->updateModulePointerReference(alpha, nullptr);
        }
        if (flavors != nullptr) {
            factory->updateModulePointerReference(flavors, nullptr);
        }
        if (dd != nullptr) {
            factory->updateModulePointerReference(dd, nullptr);
        }
        if (nativeShadow != nullptr) {
            factory->updateModulePointerReference(nativeShadow, nullptr);
        }
        throw;
    }
}

PseudodataValues evaluateNativeModelHoldout(PARTONS::Partons* partons,
        const NativeModelHoldoutEvaluation& evaluation) {
    PARTONS::ModuleObjectFactory* factory = partons->getModuleObjectFactory();
    PARTONS::GPDModule* model = nullptr;
    PARTONS::DVCSConvolCoeffFunctionModule* cff = nullptr;
    PARTONS::DVCSXiConverterModule* xi = nullptr;
    PARTONS::DVCSScalesModule* scales = nullptr;
    PARTONS::DVCSProcessModule* process = nullptr;
    std::array<PARTONS::DVCSObservable*, 6> observables{};
    try {
        if (evaluation.model == "GPDGK11") {
            model = factory->newGPDModule(PARTONS::GPDGK11::classId);
        } else if (evaluation.model == "GPDGK16") {
            model = factory->newGPDModule(PARTONS::GPDGK16::classId);
        } else if (evaluation.model == "GPDGK19") {
            model = factory->newGPDModule(PARTONS::GPDGK19::classId);
        } else {
            model = factory->newGPDModule(PARTONS::GPDVGG99::classId);
        }
        if (evaluation.model == "GPDVGG99") {
            // PARTONS commit ccd142ab replaced the model's historical
            // mstw2008nlo.00.dat input with caller-configurable LHAPDF access.
            // Pin the corresponding central LHAPDF6 member for a reproducible
            // named-model holdout; do not rely on the module's UNDEFINED
            // constructor state or on a machine-global PDF selection.
            auto* vgg = dynamic_cast<PARTONS::GPDVGG99*>(model);
            if (vgg == nullptr) {
                throw std::runtime_error(
                    "PARTONS factory did not return GPDVGG99 for its class ID");
            }
            ElemUtils::Parameters vggParameters;
            vggParameters.add(ElemUtils::Parameter(
                PARTONS::GPDVGG99::PARAM_NAME_SET_NAME,
                "MSTW2008nlo68cl"));
            vggParameters.add(ElemUtils::Parameter(
                PARTONS::GPDVGG99::PARAM_NAME_MEMBER, 0U));
            vgg->configure(vggParameters);
        }
        cff = factory->newDVCSConvolCoeffFunctionModule(
            PARTONS::DVCSCFFStandard::classId);
        cff->configure(ElemUtils::Parameter(
            PARTONS::PerturbativeQCDOrderType::
                PARAMETER_NAME_PERTURBATIVE_QCD_ORDER_TYPE,
            PARTONS::PerturbativeQCDOrderType::LO));
        cff->setGPDModule(model);
        xi = factory->newDVCSXiConverterModule(
            PARTONS::DVCSXiConverterXBToXi::classId);
        scales = factory->newDVCSScalesModule(
            PARTONS::DVCSScalesQ2Multiplier::classId);
        scales->configure(ElemUtils::Parameter(
            PARTONS::DVCSScalesQ2Multiplier::PARAMETER_NAME_LAMBDA, 1.0));
        process = factory->newDVCSProcessModule(
            PARTONS::DVCSProcessGV08::classId);
        process->setScaleModule(scales);
        process->setXiConverterModule(xi);
        process->setConvolCoeffFunctionModule(cff);
        const std::array<unsigned int, 6> observableClassIds{
            PARTONS::DVCSCrossSectionUUMinus::classId,
            PARTONS::DVCSCrossSectionDifferenceLUMinus::classId,
            PARTONS::DVCSAc::classId,
            PARTONS::DVCSAluMinus::classId,
            PARTONS::DVCSAulMinus::classId,
            PARTONS::DVCSAllMinus::classId};
        for (std::size_t index = 0; index < observables.size(); ++index) {
            observables[index] =
                factory->newDVCSObservable(observableClassIds[index]);
            observables[index]->setProcessModule(process);
        }
        const PARTONS::DVCSObservableKinematic observableKinematics(
            evaluation.observableKinematics.xB,
            evaluation.observableKinematics.t,
            evaluation.observableKinematics.q2,
            evaluation.observableKinematics.beamEnergy,
            evaluation.observableKinematics.phi);
        const double cffXi = xi->compute(observableKinematics).getValue();
        const auto cffResult = partons->getServiceObjectRegistry()
            ->getDVCSConvolCoeffFunctionService()->computeSingleKinematic(
                PARTONS::DVCSConvolCoeffFunctionKinematic(
                    cffXi, evaluation.observableKinematics.t,
                    evaluation.observableKinematics.q2,
                    evaluation.observableKinematics.q2,
                    evaluation.observableKinematics.q2),
                cff);
        const std::array<std::complex<double>, 4> cffs{
            cffResult.getResult(PARTONS::GPDType::H),
            cffResult.getResult(PARTONS::GPDType::E),
            cffResult.getResult(PARTONS::GPDType::Ht),
            cffResult.getResult(PARTONS::GPDType::Et)};

        std::vector<PseudodataGPDValues> gpdDiagnostics;
        PARTONS::GPDService* gpdService =
            partons->getServiceObjectRegistry()->getGPDService();
        const std::array<PARTONS::GPDType::Type, 4> gpdTypes{
            PARTONS::GPDType::H, PARTONS::GPDType::E,
            PARTONS::GPDType::Ht, PARTONS::GPDType::Et};
        const std::array<PARTONS::QuarkFlavor::Type, 3> quarkFlavors{
            PARTONS::QuarkFlavor::UP, PARTONS::QuarkFlavor::DOWN,
            PARTONS::QuarkFlavor::STRANGE};
        for (double xValue : evaluation.diagnosticX) {
            const auto result = gpdService->computeSingleKinematic(
                PARTONS::GPDKinematic(
                    xValue, cffXi, evaluation.observableKinematics.t,
                    evaluation.observableKinematics.q2,
                    evaluation.observableKinematics.q2),
                model);
            PseudodataGPDValues point{xValue, {}};
            for (std::size_t index = 0; index < gpdTypes.size(); ++index) {
                const auto& distribution =
                    result.getPartonDistribution(gpdTypes[index]);
                for (std::size_t flavor = 0; flavor < quarkFlavors.size();
                        ++flavor) {
                    const auto& quark =
                        distribution.getQuarkDistribution(quarkFlavors[flavor]);
                    point.byType[index].quarks[flavor] = DDValues{
                        quark.getQuarkDistribution(),
                        quark.getQuarkDistributionPlus(),
                        quark.getQuarkDistributionMinus(), 0.0};
                }
                point.byType[index].gluon = distribution.getGluonDistribution()
                    .getGluonDistribution();
            }
            gpdDiagnostics.push_back(point);
        }
        std::array<double, 6> observableValues{};
        std::array<std::string, 6> observableUnits{};
        auto* observableService =
            partons->getServiceObjectRegistry()->getDVCSObservableService();
        for (std::size_t index = 0; index < observables.size(); ++index) {
            const auto result = observableService->computeSingleKinematic(
                observableKinematics, observables[index]);
            observableValues[index] = result.getValue().getValue();
            observableUnits[index] =
                PARTONS::PhysicalUnit(result.getValue().getUnit()).toString();
        }
        const PseudodataValues values{
            cffs, cffXi, observableValues, observableUnits, gpdDiagnostics};
        bool finite = std::isfinite(values.cffXi);
        for (const auto& value : values.cffs) {
            finite = finite && std::isfinite(value.real()) &&
                std::isfinite(value.imag());
        }
        for (double value : values.observables) {
            finite = finite && std::isfinite(value);
        }
        for (const auto& point : values.gpdDiagnostics) {
            for (const auto& distribution : point.byType) {
                finite = finite && std::isfinite(distribution.gluon);
                for (const auto& quark : distribution.quarks) {
                    finite = finite && std::isfinite(quark.up) &&
                        std::isfinite(quark.upPlus) &&
                        std::isfinite(quark.upMinus);
                }
            }
        }
        if (!finite) {
            throw BridgeError("non_finite_backend_output",
                "native named-model holdout returned a non-finite value",
                kExitNumerical);
        }
        for (auto*& observable : observables) {
            factory->updateModulePointerReference(observable, nullptr);
            observable = nullptr;
        }
        factory->updateModulePointerReference(process, nullptr);
        process = nullptr;
        factory->updateModulePointerReference(scales, nullptr);
        scales = nullptr;
        factory->updateModulePointerReference(xi, nullptr);
        xi = nullptr;
        factory->updateModulePointerReference(cff, nullptr);
        cff = nullptr;
        factory->updateModulePointerReference(model, nullptr);
        model = nullptr;
        return values;
    } catch (...) {
        for (auto*& observable : observables) {
            if (observable != nullptr) {
                factory->updateModulePointerReference(observable, nullptr);
            }
        }
        if (process != nullptr) factory->updateModulePointerReference(process, nullptr);
        if (scales != nullptr) factory->updateModulePointerReference(scales, nullptr);
        if (xi != nullptr) factory->updateModulePointerReference(xi, nullptr);
        if (cff != nullptr) factory->updateModulePointerReference(cff, nullptr);
        if (model != nullptr) factory->updateModulePointerReference(model, nullptr);
        throw;
    }
}

json::object evaluationJson(const Evaluation& evaluation,
        const GPDValues& values) {
    return json::object{
        {"request_id", evaluation.requestId},
        {"module", "GPDGK16"},
        {"gpd_type", "H"},
        {"scale_treatment",
         "native_GPDGK16_internal_scale_dependence_no_GPDEvolutionModule"},
        {"model_reference_mu_f2",
         json::object{{"value", 4.0}, {"unit", "GeV2"}}},
        {"mu_r2_role", "passed_to_GPDKinematic_not_separately_benchmarked"},
        {"kinematics", kinematicsJson(evaluation.kinematics)},
        {"result",
         json::object{{"h_up", values.up},
                      {"h_gluon", values.gluon},
                      {"finite", true}}}};
}

json::object shadowEvaluationJson(const ShadowEvaluation& evaluation,
        const ShadowValues& values) {
    return json::object{
        {"request_id", evaluation.requestId},
        {"representation", "stage09_gk16_plus_bdmms21_shadow_v1"},
        {"representation_schema_version", 1},
        {"composition_module", "ShadowInjectionGPD"},
        {"baseline_module", "GPDGK16"},
        {"shadow_module", "GPDBDMMS21"},
        {"parameters",
            json::object{
                {"baseline_coefficient", evaluation.coefficients.baseline},
                {"shadow_coefficient", evaluation.coefficients.shadow}}},
        {"gpd_diagnostic_kinematics",
            json::object{
                {"x", evaluation.gpdKinematics.x},
                {"xi", evaluation.gpdKinematics.xi},
                {"t",
                    json::object{
                        {"value", evaluation.gpdKinematics.t},
                        {"unit", "GeV2"}}},
                {"mu_f_squared",
                    json::object{
                        {"value", evaluation.gpdKinematics.muF2},
                        {"unit", "GeV2"}}},
                {"mu_r_squared",
                    json::object{
                        {"value", evaluation.gpdKinematics.muR2},
                        {"unit", "GeV2"}}}}},
        {"observable_kinematics",
            json::object{
                {"x_b", evaluation.observableKinematics.xB},
                {"xi_from_native_converter", values.cffXi},
                {"t",
                    json::object{
                        {"value", evaluation.observableKinematics.t},
                        {"unit", "GeV2"}}},
                {"Q2",
                    json::object{
                        {"value", evaluation.observableKinematics.q2},
                        {"unit", "GeV2"}}},
                {"beam_energy",
                    json::object{
                        {"value", evaluation.observableKinematics.beamEnergy},
                        {"unit", "GeV"}}},
                {"phi",
                    json::object{
                        {"value", evaluation.observableKinematics.phi},
                        {"unit", "rad"}}}}},
        {"result",
            json::object{
                {"gpd",
                    json::object{
                        {"h_up_combined", values.hUpCombined},
                        {"h_up_baseline_component",
                            values.hUpBaselineComponent},
                        {"h_up_shadow_component",
                            values.hUpShadowComponent}}},
                {"cff_h",
                    json::object{
                        {"real", values.cffH.real()},
                        {"imaginary", values.cffH.imag()}}},
                {"observable",
                    json::object{
                        {"value", values.observable},
                        {"native_unit_symbol", values.observableUnit},
                        {"quantity", "d4sigma_nb_per_GeV4"}}},
                {"finite", true}}},
        {"theory_configuration",
            json::object{
                {"cff_module", "DVCSCFFStandard"},
                {"process_module", "DVCSProcessGV08"},
                {"observable_module", "DVCSCrossSectionUUMinus"},
                {"xi_converter_module", "DVCSXiConverterXBToXi"},
                {"scales_module", "DVCSScalesQ2Multiplier"},
                {"coefficient_function_order", "NLO"},
                {"gpd_evolution_order", nullptr},
                {"alpha_s",
                    json::object{
                        {"module", "RunningAlphaStrongStandard"},
                        {"implementation_order", "four_loop"},
                        {"scheme", "MSbar"},
                        {"reference_value", 0.1184},
                        {"reference_scale",
                            json::object{
                                {"value", 91.1876},
                                {"unit", "GeV"}}},
                        {"flavor_threshold_source",
                            "installed_quark_masses"},
                        {"threshold_matching_verified", false}}},
                {"factorization_scale_squared", "Q2"},
                {"renormalization_scale_squared", "Q2"},
                {"twist", nullptr},
                {"analysis_order_label", nullptr}}},
        {"physics_omissions",
            json::array{
                "GPD_evolution", "gluon_shadow", "threshold_matching",
                "verified_twist_label", "general_observable_equivalence"}},
        {"analysis_order_label", nullptr},
        {"source_provenance",
            json::object{
                {"shadow_source_sha256", SHADOW_SOURCE_SHA256},
                {"shadow_header_sha256", SHADOW_HEADER_SHA256},
                {"contract",
                    "configs/physics/"
                    "stage09_shadow_identifiability_v1.json"}}}};
}

json::object pseudodataEvaluationJson(
        const PseudodataEvaluation& evaluation,
        const PseudodataValues& values) {
    const auto shapeJson = [](const DVCSInference::DDParameters& shape) {
        return json::object{
            {"normalization", shape.normalization},
            {"a", shape.a},
            {"c", shape.c},
            {"profile_b", shape.profileB},
            {"t_slope",
                json::object{{"value", shape.tSlope}, {"unit", "GeV-2"}}}};
    };
    const auto quarkJson = [](const DDValues& value, double chargeSquared) {
        return json::object{
            {"value", value.up},
            {"plus", value.upPlus},
            {"minus", value.upMinus},
            {"charge_squared_weighted_c_even", chargeSquared * value.upPlus}};
    };
    const auto distributionJson = [&quarkJson](
            const PseudodataGPDValues::Distribution& value) {
        return json::object{
            {"u", quarkJson(value.quarks[0], 4.0 / 9.0)},
            {"d", quarkJson(value.quarks[1], 1.0 / 9.0)},
            {"s", quarkJson(value.quarks[2], 1.0 / 9.0)},
            {"gluon", value.gluon},
            {"charge_squared_weighted_c_even_quark_sum",
                (4.0 / 9.0) * value.quarks[0].upPlus +
                (1.0 / 9.0) * value.quarks[1].upPlus +
                (1.0 / 9.0) * value.quarks[2].upPlus}};
    };
    json::array gpdDiagnostics;
    for (const PseudodataGPDValues& point : values.gpdDiagnostics) {
        gpdDiagnostics.emplace_back(json::object{
            {"x", point.x},
            {"xi", values.cffXi},
            {"t_GeV2", evaluation.observableKinematics.t},
            {"Q2_GeV2", evaluation.observableKinematics.q2},
            {"gpd",
                json::object{
                    {"H", distributionJson(point.byType[0])},
                    {"E", distributionJson(point.byType[1])},
                    {"Htilde", distributionJson(point.byType[2])},
                    {"Etilde", distributionJson(point.byType[3])}}}});
    }
    return json::object{
        {"request_id", evaluation.requestId},
        {"representation", "stage11_lo_multiq2_full_independent_dd_v1"},
        {"representation_schema_version", 1},
        {"composition_module", "PseudodataInputGPD"},
        {"baseline_module", "FixedScaleDDGPD"},
        {"shadow_modules",
            json::object{
                {"H.u", "GPDBDMMS21"},
                {"other_channels",
                    "project_zero_forward_moment_DD_test_direction_v1"},
                {"cff_null_property_verified", false}}},
        {"parameters",
            json::object{
                {"gpd_shapes",
                    [&]() {
                        json::object types;
                        const std::array<std::string, 4> typeNames{
                            "H", "E", "Htilde", "Etilde"};
                        const std::array<std::string, 4> channelNames{
                            "u", "d", "s", "gluon"};
                        for (std::size_t type = 0; type < 4; ++type) {
                            json::object channels;
                            for (std::size_t channel = 0; channel < 4;
                                    ++channel) {
                                channels[channelNames[channel]] = shapeJson(
                                    evaluation.parameters
                                        .ddByTypeAndChannel[type][channel]);
                            }
                            types[typeNames[type]] = std::move(channels);
                        }
                        return types;
                    }()},
                {"shadow_coefficients",
                    json::object{
                        {"H", evaluation.parameters.shadowCoefficientByType[0]},
                        {"E", evaluation.parameters.shadowCoefficientByType[1]},
                        {"Htilde", evaluation.parameters.shadowCoefficientByType[2]},
                        {"Etilde", evaluation.parameters.shadowCoefficientByType[3]}}},
                {"shadow_channel_amplitudes",
                    [&]() {
                        json::object types;
                        const std::array<std::string, 4> typeNames{
                            "H", "E", "Htilde", "Etilde"};
                        const std::array<std::string, 4> channelNames{
                            "u", "d", "s", "gluon"};
                        for (std::size_t type = 0; type < 4; ++type) {
                            json::object channels;
                            for (std::size_t channel = 0; channel < 4;
                                    ++channel) {
                                channels[channelNames[channel]] = evaluation
                                    .parameters.shadowAmplitudeByTypeAndChannel
                                        [type][channel];
                            }
                            types[typeNames[type]] = std::move(channels);
                        }
                        return types;
                    }()}}},
        {"q0_squared",
            json::object{{"value", evaluation.q0Squared}, {"unit", "GeV2"}}},
        {"observable_kinematics",
            json::object{
                {"x_b", evaluation.observableKinematics.xB},
                {"xi_from_native_converter", values.cffXi},
                {"t",
                    json::object{
                        {"value", evaluation.observableKinematics.t},
                        {"unit", "GeV2"}}},
                {"Q2",
                    json::object{
                        {"value", evaluation.observableKinematics.q2},
                        {"unit", "GeV2"}}},
                {"beam_energy",
                    json::object{
                        {"value", evaluation.observableKinematics.beamEnergy},
                        {"unit", "GeV"}}},
                {"phi",
                    json::object{
                        {"value", evaluation.observableKinematics.phi},
                        {"unit", "rad"}}}}},
        {"result",
            json::object{
                {"cffs",
                    json::object{
                        {"H",
                            json::object{
                                {"real", values.cffs[0].real()},
                                {"imaginary", values.cffs[0].imag()}}},
                        {"E",
                            json::object{
                                {"real", values.cffs[1].real()},
                                {"imaginary", values.cffs[1].imag()}}},
                        {"Htilde",
                            json::object{
                                {"real", values.cffs[2].real()},
                                {"imaginary", values.cffs[2].imag()}}},
                        {"Etilde",
                            json::object{
                                {"real", values.cffs[3].real()},
                                {"imaginary", values.cffs[3].imag()}}}}},
                {"observables",
                    json::object{
                        {"DVCSCrossSectionUUMinus",
                            json::object{
                                {"value", values.observables[0]},
                                {"native_unit_symbol",
                                    values.observableUnits[0]},
                                {"quantity", "d4sigma_nb_per_GeV4"}}},
                        {"DVCSCrossSectionDifferenceLUMinus",
                            json::object{
                                {"value", values.observables[1]},
                                {"native_unit_symbol",
                                    values.observableUnits[1]},
                                {"quantity",
                                    "delta_d4sigma_nb_per_GeV4"}}},
                        {"DVCSAc",
                            json::object{
                                {"value", values.observables[2]},
                                {"native_unit_symbol",
                                    values.observableUnits[2]},
                                {"quantity", "beam_charge_asymmetry"}}},
                        {"DVCSAluMinus",
                            json::object{
                                {"value", values.observables[3]},
                                {"native_unit_symbol",
                                    values.observableUnits[3]},
                                {"quantity", "beam_spin_asymmetry"}}},
                        {"DVCSAulMinus",
                            json::object{
                                {"value", values.observables[4]},
                                {"native_unit_symbol",
                                    values.observableUnits[4]},
                                {"quantity", "target_spin_asymmetry"}}},
                        {"DVCSAllMinus",
                            json::object{
                                {"value", values.observables[5]},
                                {"native_unit_symbol",
                                    values.observableUnits[5]},
                                {"quantity", "double_spin_asymmetry"}}}}},
                {"gpd_diagnostics", std::move(gpdDiagnostics)},
                {"finite", true}}},
        {"theory_configuration",
            json::object{
                {"input_scale_squared_GeV2", evaluation.q0Squared},
                {"scale_treatment",
                    evaluation.observableKinematics.q2 == evaluation.q0Squared
                        ? "Q2_equals_Q0_squared_identity_no_data_evolution"
                        : "input_GPD_at_Q0_evolved_to_datum_Q2_no_data_evolution"},
                {"gpd_evolution_module", "GPDEvolutionApfel"},
                {"gpd_evolution_order", "LO"},
                {"factorization_scheme", "MSbar"},
                {"fixed_active_flavors", 3},
                {"alpha_s",
                    json::object{
                        {"module", "RunningAlphaStrongApfel"},
                        {"order", "LO"},
                        {"scheme", "MSbar"},
                        {"reference_value", 0.118},
                        {"reference_scale",
                            json::object{
                                {"value", 91.1876},
                                {"unit", "GeV"}}},
                        {"thresholds", json::array{0, 0, 0}}}},
                {"cff_module", "DVCSCFFStandard"},
                {"coefficient_function_order", "LO"},
                {"process_module", "DVCSProcessGV08"},
                {"observable_modules",
                    json::array{
                        "DVCSCrossSectionUUMinus",
                        "DVCSCrossSectionDifferenceLUMinus",
                        "DVCSAc", "DVCSAluMinus", "DVCSAulMinus",
                        "DVCSAllMinus"}},
                {"xi_converter_module", "DVCSXiConverterXBToXi"},
                {"scales_module", "DVCSScalesQ2Multiplier"},
                {"factorization_scale_squared", "Q2"},
                {"renormalization_scale_squared", "Q2"},
                {"twist", nullptr},
                {"analysis_order_label", nullptr}}},
        {"physics_omissions",
            json::array{
                "D_term", "Etilde_pion_pole",
                "heavy_quark_input", "heavy_flavor_matching",
                "verified_twist_label", "real_data"}},
        {"analysis_order_label", nullptr},
        {"source_provenance",
            json::object{
                {"pseudodata_source_sha256", PSEUDODATA_SOURCE_SHA256},
                {"pseudodata_header_sha256", PSEUDODATA_HEADER_SHA256},
                {"evolution_cache_adapter_source_sha256",
                    TABULATED_PSEUDODATA_SOURCE_SHA256},
                {"evolution_cache_adapter_header_sha256",
                    TABULATED_PSEUDODATA_HEADER_SHA256},
                {"evolution_cache_adapter_contract",
                    "one authoritative GPDEvolutionApfel table per GPD type; "
                    "runtime direct-wrapper anchors require absolute "
                    "agreement within 1e-11"},
                {"dd_source_sha256", DD_SOURCE_SHA256},
                {"dd_header_sha256", DD_HEADER_SHA256},
                {"contract",
                    "configs/physics/"
                    "stage10_pseudodata_systematics_v1.json"}}}};
}

json::object nativeModelHoldoutJson(
        const NativeModelHoldoutEvaluation& evaluation,
        const PseudodataValues& values) {
    const PseudodataEvaluation outputShape{
        evaluation.requestId, {}, 128U, 1.0,
        evaluation.observableKinematics,
        evaluation.diagnosticX};
    json::object result = pseudodataEvaluationJson(outputShape, values);
    result["representation"] = "stage11_native_model_holdout_v1";
    result["representation_schema_version"] = 1;
    result["native_gpd_model"] = evaluation.model;
    result.erase("composition_module");
    result.erase("baseline_module");
    result.erase("shadow_modules");
    result.erase("parameters");
    result.erase("q0_squared");
    json::object& theory = result.at("theory_configuration").as_object();
    theory["scale_treatment"] =
        "native_model_internal_scale_dependence_no_attached_evolution";
    theory["gpd_evolution_module"] = nullptr;
    theory["gpd_evolution_order"] = nullptr;
    theory["fixed_active_flavors"] = nullptr;
    theory.erase("input_scale_squared_GeV2");
    theory.erase("factorization_scheme");
    theory.erase("alpha_s");
    result["physics_omissions"] = json::array{
        "common_Q0_reparameterization", "attached_GPD_evolution",
        "verified_twist_label", "heavy_flavor_matching", "real_data"};
    result["holdout_contract"] = json::object{
        {"training_use", false}, {"validation_use", false},
        {"optuna_use", false}, {"early_stopping_use", false},
        {"post_training_only", true}, {"real_data_used", false},
        {"external_native_validation_use", true},
        {"replaces_dd_outer_test", false}};
    std::string modelHeader;
    std::string modelSource;
    std::string headerHash;
    std::string sourceHash;
    if (evaluation.model == "GPDGK11") {
        modelHeader = "GPDGK11.h";
        modelSource = "GPDGK11.cpp";
        headerHash = GPD_GK11_HEADER_SHA256;
        sourceHash = GPD_GK11_SOURCE_SHA256;
    } else if (evaluation.model == "GPDGK16") {
        modelHeader = "GPDGK16.h";
        modelSource = "GPDGK16.cpp";
        headerHash = GPD_GK16_HEADER_SHA256;
        sourceHash = GPD_GK16_SOURCE_SHA256;
    } else if (evaluation.model == "GPDGK19") {
        modelHeader = "GPDGK19.h";
        modelSource = "GPDGK19.cpp";
        headerHash = GPD_GK19_HEADER_SHA256;
        sourceHash = GPD_GK19_SOURCE_SHA256;
    } else {
        modelHeader = "GPDVGG99.h";
        modelSource = "GPDVGG99.cpp";
        headerHash = GPD_VGG99_HEADER_SHA256;
        sourceHash = GPD_VGG99_SOURCE_SHA256;
    }
    result["source_provenance"] = json::object{
        {"model", evaluation.model},
        {"header_sha256", headerHash},
        {"source_sha256", sourceHash},
        {"header", "dependency://partons@1ad0b7d3/include/partons/"
                   "modules/gpd/" + modelHeader},
        {"source", "dependency://partons@1ad0b7d3/src/partons/"
                   "modules/gpd/" + modelSource}};
    if (evaluation.model == "GPDVGG99") {
        result["forward_pdf"] = json::object{
            {"library", "LHAPDF"},
            {"library_version", "6.5.6"},
            {"set", "MSTW2008nlo68cl"},
            {"member", 0},
            {"archive_sha256",
                "98ec0541e80e223785bb6029ebf81e93ca5111da41d6565b3b5c4aa86d59bb5d"},
            {"info_sha256",
                "07b1ee816adfd9e438bb996d20116dbcb4a0e3624e5b2075aecb7e699a37b0f3"},
            {"member_sha256",
                "26e94045264006998fa0af3dc5d5c0e7e304820ac5ce43eee43413197eae7e78"},
            {"selection_evidence",
                "PARTONS commit ccd142ab parent used mstw2008nlo.00.dat"}};
    }
    return result;
}

json::object ddKinematicsJson(const DDKinematics& input) {
    return json::object{
        {"x", input.x},
        {"xi", input.xi},
        {"canonical_abs_xi", std::abs(input.xi)},
        {"t", json::object{{"value", input.t}, {"unit", "GeV2"}}},
        {"q0_squared",
            json::object{{"value", input.q0Squared}, {"unit", "GeV2"}}},
        {"mu_f_squared",
            json::object{{"value", input.muFSquared}, {"unit", "GeV2"}}}};
}

json::object ddEvaluationJson(const DDEvaluation& evaluation,
        const DDValues& values) {
    json::object result{
        {"h_up", values.up},
        {"h_up_plus", values.upPlus},
        {"h_up_minus", values.upMinus},
        {"finite", true}};
    if (evaluation.evolved) {
        result["h_gluon"] = values.gluon;
    }

    json::object response{
        {"request_id", evaluation.requestId},
        {"representation", evaluation.evolved
            ? "stage06_lo_apfel_evolved_reduced_dd_v1"
            : "stage03_fixed_scale_up_valence_v1"},
        {"representation_schema_version", 1},
        {"module", "FixedScaleDDGPD"},
        {"gpd_type", "H"},
        {"flavor", "up"},
        {"parton_content",
            evaluation.evolved
                ? "reduced_up_valence_input_evolved_fixed_nf3"
                : "reduced_up_valence_forward_input_no_gluon_no_sea"},
        {"parameters",
            json::object{
                {"a", evaluation.parameters.a},
                {"c", evaluation.parameters.c},
                {"profile_b", evaluation.parameters.profileB},
                {"t_slope",
                    json::object{{"value", evaluation.parameters.tSlope},
                                 {"unit", "GeV-2"}}},
                {"up_valence_number", 2.0}}},
        {"numerics",
            json::object{
                {"integrator", "NumA_GaussLegendre"},
                {"quadrature_order", evaluation.quadratureOrder}}},
        {"scale_treatment", evaluation.evolved
            ? "input_at_Q0_then_PARTONS_GPDEvolutionApfel_to_muF"
            : "fixed_Q0_no_evolution"},
        {"kinematics", ddKinematicsJson(evaluation.kinematics)},
        {"result", std::move(result)},
        {"physics_omissions",
            evaluation.evolved
                ? json::array{"D_term", "validated_singlet_gluon_mixing",
                              "heavy_flavor_thresholds", "CFF",
                              "DVCS_observable"}
                : json::array{"D_term", "sea", "gluon", "singlet",
                              "quark_gluon_mixing", "evolution"}},
        {"perturbative_label", nullptr},
        {"source_provenance",
            json::object{{"dd_source_sha256", DD_SOURCE_SHA256},
                         {"dd_header_sha256", DD_HEADER_SHA256},
                         {"convention_registry",
                          "configs/physics/"
                          "stage03_dd_conventions_v1.json"}}}};
    if (evaluation.evolved) {
        response["evolution_configuration"] = json::object{
            {"module", "GPDEvolutionApfel"},
            {"gpd_evolution_order", "LO"},
            {"factorization_scheme", "MSbar"},
            {"alpha_s",
                json::object{
                    {"module", "RunningAlphaStrongApfel"},
                    {"order", "LO"},
                    {"scheme", "MSbar"},
                    {"reference_value", 0.118},
                    {"reference_scale",
                        json::object{{"value", 91.1876},
                                     {"unit", "GeV"}}},
                    {"thresholds", json::array{0, 0, 0}},
                    {"argument", "evolution_factorization_scale"}}},
            {"active_flavors",
                json::object{
                    {"module", "ActiveFlavorsThresholdsConstant"},
                    {"n_flavors", 3}}},
            {"coefficient_function_order", nullptr},
            {"renormalization_scale", nullptr},
            {"twist", nullptr}};
        response["analysis_order_label"] = nullptr;
        response["source_provenance"].as_object()[
            "evolution_contract"] =
            "configs/physics/stage06_evolution_v1.json";
    }
    return response;
}

json::object conformalEvaluationJson(
        const ConformalEvaluation& evaluation,
        const ConformalValues& values) {
    return json::object{
        {"request_id", evaluation.requestId},
        {"representation",
            "stage08_zero_skewness_conformal_moment_v1"},
        {"representation_schema_version", 1},
        {"module", "ConformalMomentGPD"},
        {"gpd_type", "H"},
        {"flavor", "up"},
        {"parton_content",
            "reduced_up_valence_zero_skewness_no_gluon_no_sea"},
        {"parameters",
            json::object{
                {"alpha", evaluation.parameters.alpha},
                {"beta", evaluation.parameters.beta},
                {"trajectory_slope",
                    json::object{
                        {"value", evaluation.parameters.trajectorySlope},
                        {"unit", "GeV-2"}}},
                {"up_valence_number_at_t0", 2.0}}},
        {"moment_ansatz",
            "N*B(j+1-alpha,beta+1)*(j+1-alpha)/"
            "(j+1-alpha-alpha_prime*t)"},
        {"reconstruction",
            "analytic_inverse_Mellin_convolution_at_xi_zero"},
        {"numerics",
            json::object{
                {"integrator", "GSL_qag_GAUSS61"},
                {"integration_limit", evaluation.integrationLimit},
                {"absolute_tolerance", 1e-12},
                {"relative_tolerance", 1e-10}}},
        {"scale_treatment", "fixed_Q0_no_evolution"},
        {"kinematics", ddKinematicsJson(evaluation.kinematics)},
        {"result",
            json::object{
                {"h_up", values.up},
                {"h_up_plus", values.upPlus},
                {"h_up_minus", values.upMinus},
                {"finite", true}}},
        {"physics_omissions",
            json::array{"nonzero_skewness", "D_term", "sea", "gluon",
                        "singlet", "quark_gluon_mixing", "evolution",
                        "CFF", "DVCS_observable"}},
        {"perturbative_label", nullptr},
        {"source_provenance",
            json::object{
                {"conformal_source_sha256", CONFORMAL_SOURCE_SHA256},
                {"conformal_header_sha256", CONFORMAL_HEADER_SHA256},
                {"convention_registry",
                    "configs/physics/"
                    "stage08_conformal_moment_v1.json"}}}};
}

json::object basisEvaluationJson(const BasisEvaluation& evaluation,
        const BasisValues& values) {
    const std::array<const char*, 6> names{
        "up", "down", "strange", "charm", "bottom", "top"};
    json::object physical;
    std::map<int, double> physicalMap{{0, values.gluon}};
    const std::array<int, 6> partonsToPdg{2, 1, 3, 4, 5, 6};
    for (std::size_t index = 0; index < names.size(); ++index) {
        physical[names[index]] = json::object{
            {"h_q", values.quark[index]},
            {"h_q_plus", values.plus[index]},
            {"h_q_minus", values.minus[index]}};
        const double q = evaluation.kinematics.x * values.quark[index];
        const double qbar = evaluation.kinematics.x *
            (values.plus[index] - values.quark[index]);
        physicalMap[partonsToPdg[index]] = q;
        physicalMap[-partonsToPdg[index]] = qbar;
    }
    physical["gluon"] = json::object{
        {"partons_gluon_field", values.gluon},
        {"apfel_x_weighted", true}};

    const std::map<int, double> evolutionBasis =
        apfel::PhysToQCDEv(physicalMap);
    const std::array<const char*, 13> basisNames{
        "g", "sigma", "valence", "t3", "v3", "t8", "v8",
        "t15", "v15", "t24", "v24", "t35", "v35"};
    json::object basis;
    for (std::size_t index = 0; index < basisNames.size(); ++index) {
        basis[basisNames[index]] =
            evolutionBasis.at(static_cast<int>(index));
    }

    return json::object{
        {"request_id", evaluation.requestId},
        {"representation", "stage07_coupled_basis_dd_v1"},
        {"representation_schema_version", 1},
        {"module", "BasisInjectionGPD"},
        {"gpd_type", "H"},
        {"input_basis", "apfel_nf3_sigma_t3_gluon"},
        {"parton_content", "explicit_sigma_t3_gluon_input_fixed_nf3"},
        {"parameters",
            json::object{
                {"shape",
                    json::object{
                        {"a", evaluation.shapeParameters.a},
                        {"c", evaluation.shapeParameters.c},
                        {"profile_b", evaluation.shapeParameters.profileB},
                        {"t_slope",
                            json::object{
                                {"value", evaluation.shapeParameters.tSlope},
                                {"unit", "GeV-2"}}}}},
                {"amplitudes",
                    json::object{
                        {"sigma", evaluation.amplitudes.sigma},
                        {"t3", evaluation.amplitudes.t3},
                        {"gluon", evaluation.amplitudes.gluon}}}}},
        {"numerics",
            json::object{
                {"integrator", "NumA_GaussLegendre"},
                {"quadrature_order", evaluation.quadratureOrder}}},
        {"scale_treatment",
            "input_at_Q0_then_PARTONS_GPDEvolutionApfel_to_muF"},
        {"kinematics", ddKinematicsJson(evaluation.kinematics)},
        {"result",
            json::object{
                {"physical_basis", std::move(physical)},
                {"apfel_evolution_basis_x_weighted", std::move(basis)},
                {"finite", true}}},
        {"evolution_configuration",
            json::object{
                {"module", "GPDEvolutionApfel"},
                {"gpd_evolution_order", "LO"},
                {"factorization_scheme", "MSbar"},
                {"alpha_s",
                    json::object{
                        {"module", "RunningAlphaStrongApfel"},
                        {"order", "LO"},
                        {"scheme", "MSbar"},
                        {"reference_value", 0.118},
                        {"reference_scale",
                            json::object{{"value", 91.1876},
                                         {"unit", "GeV"}}},
                        {"thresholds", json::array{0, 0, 0}},
                        {"argument", "evolution_factorization_scale"}}},
                {"active_flavors",
                    json::object{
                        {"module", "ActiveFlavorsThresholdsConstant"},
                        {"n_flavors", 3}}},
                {"coefficient_function_order", nullptr},
                {"renormalization_scale", nullptr},
                {"twist", nullptr}}},
        {"analysis_order_label", nullptr},
        {"physics_omissions",
            json::array{"heavy_flavor_threshold_crossing", "D_term",
                        "CFF", "DVCS_observable", "shadow_directions"}},
        {"source_provenance",
            json::object{
                {"basis_source_sha256", BASIS_SOURCE_SHA256},
                {"basis_header_sha256", BASIS_HEADER_SHA256},
                {"evolution_contract",
                    "configs/physics/stage07_coupled_basis_v1.json"}}}};
}

json::object capabilitiesResponse() {
    return json::object{
        {"schema_version", BRIDGE_SCHEMA_VERSION},
        {"status", "ok"},
        {"operation", "capabilities"},
        {"backend", backendProvenance()},
        {"capabilities",
         json::object{
             {"commands",
              json::array{"capabilities", "self_test", "evaluate_gpd",
                  "batch_evaluate_gpd", "evaluate_dd_gpd",
                  "batch_evaluate_dd_gpd", "evaluate_evolved_dd_gpd",
                  "batch_evaluate_evolved_dd_gpd",
                  "evaluate_coupled_basis_gpd",
                  "batch_evaluate_coupled_basis_gpd",
                  "evaluate_conformal_moment_gpd",
                  "batch_evaluate_conformal_moment_gpd",
                  "evaluate_shadow_dvcs",
                  "batch_evaluate_shadow_dvcs",
                  "evaluate_pseudodata_dvcs",
                  "batch_evaluate_pseudodata_dvcs",
                  "evaluate_post_training_comparison_dvcs",
                  "batch_evaluate_post_training_comparison_dvcs",
                  "evaluate_native_model_holdout_dvcs",
                  "batch_evaluate_native_model_holdout_dvcs"}},
             {"evaluate_gpd",
              json::object{
                  {"available", true},
                  {"modules", json::array{"GPDGK16"}},
                  {"gpd_types", json::array{"H"}},
                  {"units", json::object{{"t", "GeV2"},
                                        {"mu_f2", "GeV2"},
                                        {"mu_r2", "GeV2"}}},
                  {"scale_treatment",
                   "native_GPDGK16_internal_scale_dependence_no_"
                   "GPDEvolutionModule"},
                  {"model_reference_mu_f2",
                   json::object{{"value", 4.0}, {"unit", "GeV2"}}},
                  {"mu_r2_role",
                      "passed_to_GPDKinematic_not_separately_benchmarked"}}},
             {"evaluate_dd_gpd",
              json::object{
                  {"available", true},
                  {"representation",
                   "stage03_fixed_scale_up_valence_v1"},
                  {"module", "FixedScaleDDGPD"},
                  {"mapping", "PARTONS_GPDModule_factory_GPDService"},
                  {"gpd_types", json::array{"H"}},
                  {"flavors", json::array{"up"}},
                  {"parton_content",
                   "reduced_up_valence_forward_input_no_gluon_no_sea"},
                  {"q0_squared",
                   json::object{{"value", 4.0}, {"unit", "GeV2"}}},
                  {"mu_f_squared",
                   json::object{{"value", 4.0}, {"unit", "GeV2"}}},
                  {"evolution", false},
                  {"integrator", "NumA_GaussLegendre"},
                  {"quadrature_orders", json::array{32, 64, 128, 256}},
                  {"d_term", false},
                  {"perturbative_label", nullptr}}},
             {"gpd_evolution",
              json::object{{"source_runtime_audited", true},
                           {"bridge_available", true},
                           {"operation", "evaluate_evolved_dd_gpd"},
                           {"representation",
                            "stage06_lo_apfel_evolved_reduced_dd_v1"},
                           {"input_scale_squared_GeV2", 4.0},
                           {"factorization_scale_squared_range_GeV2",
                            json::array{4.0, 16.0}},
                           {"positive_x_range",
                            json::array{1e-5, 0.95}},
                           {"verified_order", "LO"},
                           {"factorization_scheme", "MSbar"},
                           {"alpha_s_module", "RunningAlphaStrongApfel"},
                           {"active_flavors_module",
                            "ActiveFlavorsThresholdsConstant"},
                           {"fixed_active_flavors", 3},
                           {"thresholds_GeV", json::array{0, 0, 0}},
                           {"coefficient_function_order", nullptr},
                           {"renormalization_scale", nullptr},
                           {"twist", nullptr},
                           {"analysis_order_label", nullptr},
                           {"validated_singlet_gluon_mixing", false},
                           {"heavy_flavor_thresholds", false},
                           {"fixed_scale_fallback", false}}},
             {"coupled_quark_gluon",
              json::object{
                  {"available", true},
                  {"operation", "evaluate_coupled_basis_gpd"},
                  {"representation", "stage07_coupled_basis_dd_v1"},
                  {"input_basis", "apfel_nf3_sigma_t3_gluon"},
                  {"explicit_directions",
                      json::array{"sigma", "t3", "gluon"}},
                  {"physical_output_flavors",
                      json::array{"up", "down", "strange", "charm",
                                  "bottom", "top", "gluon"}},
                  {"verified_evolution_order", "LO"},
                  {"factorization_scheme", "MSbar"},
                  {"fixed_active_flavors", 3},
                  {"thresholds_GeV", json::array{0, 0, 0}},
                  {"validated_q_to_g_response", true},
                  {"validated_g_to_q_response", true},
                  {"validated_non_singlet_decoupling", true},
                  {"variable_flavor_thresholds", false},
                  {"thread_safe_parallel_execution", false}}},
             {"conformal_moment_family",
              json::object{
                  {"available", true},
                  {"operation", "evaluate_conformal_moment_gpd"},
                  {"representation",
                      "stage08_zero_skewness_conformal_moment_v1"},
                  {"module", "ConformalMomentGPD"},
                  {"mapping", "PARTONS_GPDModule_factory_GPDService"},
                  {"moment_parameterization", "GUMP_k0_single_Regge_term"},
                  {"reconstruction",
                      "analytic_inverse_Mellin_convolution"},
                  {"skewness", 0.0},
                  {"q0_squared_GeV2", 4.0},
                  {"mu_f_squared_GeV2", 4.0},
                  {"t_range_GeV2", json::array{-0.5, 0.0}},
                  {"integration_limits",
                      json::array{256, 512, 1024, 2048}},
                  {"evolution", false},
                  {"perturbative_label", nullptr},
                  {"thread_safe_parallel_execution", false}}},
             {"shadow_identifiability",
              json::object{
                  {"available", true},
                  {"operation", "evaluate_shadow_dvcs"},
                  {"representation",
                      "stage09_gk16_plus_bdmms21_shadow_v1"},
                  {"composition_module", "ShadowInjectionGPD"},
                  {"baseline_module", "GPDGK16"},
                  {"shadow_module", "GPDBDMMS21"},
                  {"shadow_content", "up_sea_quark_only"},
                  {"coefficient_function_module", "DVCSCFFStandard"},
                  {"coefficient_function_order", "NLO"},
                  {"process_module", "DVCSProcessGV08"},
                  {"observable_module", "DVCSCrossSectionUUMinus"},
                  {"gpd_evolution_order", nullptr},
                  {"twist", nullptr},
                  {"analysis_order_label", nullptr},
                  {"thread_safe_parallel_execution", false}}},
             {"pseudodata_dvcs",
              json::object{
                  {"available", true},
                  {"operation", "evaluate_pseudodata_dvcs"},
                  {"representation",
                      "stage11_lo_multiq2_full_independent_dd_v1"},
                  {"composition_module", "PseudodataInputGPD"},
                  {"baseline_module", "FixedScaleDDGPD"},
                  {"shadow_modules",
                      json::object{
                          {"H.u", "GPDBDMMS21"},
                          {"other_channels",
                              "project_zero_forward_moment_DD_test_direction_v1"},
                          {"cff_null_property_verified", false}}},
                  {"gpd_types",
                      json::array{"H", "E", "Htilde", "Etilde"}},
                  {"input_parton_channels",
                      json::array{"u", "d", "s", "gluon"}},
                  {"q0_squared_GeV2", 1.0},
                  {"observable_q2_squared_GeV2_domain", "[1,80]"},
                  {"scale_treatment",
                      "input_GPD_at_Q0_evolved_to_each_datum_Q2_no_data_evolution"},
                  {"configured_gpd_evolution_module",
                      "GPDEvolutionApfel"},
                  {"gpd_evolution_order", "LO"},
                  {"coefficient_function_module", "DVCSCFFStandard"},
                  {"coefficient_function_order", "LO"},
                  {"process_module", "DVCSProcessGV08"},
                  {"observable_modules",
                      json::array{
                          "DVCSCrossSectionUUMinus",
                          "DVCSCrossSectionDifferenceLUMinus",
                          "DVCSAc", "DVCSAluMinus", "DVCSAulMinus",
                          "DVCSAllMinus"}},
                  {"analysis_order_label", nullptr},
                  {"off_scale_dvcs_campaign_available", true},
                  {"thread_safe_parallel_execution", false}}},
             {"post_training_real_comparison_dvcs",
              json::object{
                  {"available", true},
                  {"operations", json::array{
                      "evaluate_post_training_comparison_dvcs",
                      "batch_evaluate_post_training_comparison_dvcs"}},
                  {"role", "post_training_diagnostic_only"},
                  {"mapped_observable", "DVCSAluMinus"},
                  {"real_data_fit", false},
                  {"posterior_updated", false},
                  {"q2_squared_GeV2_domain", "[1,80]"},
                  {"source_Q2_retained", true},
                  {"data_evolved", false},
                  {"gpd_evolved_to_each_datum_Q2", true},
                  {"x_b_domain", "0<xB<1"},
                  {"t_GeV2_domain", "-1<=t<0"},
                  {"beam_energy_GeV_domain", "3<=E<=200"},
                  {"thread_safe_parallel_execution", false}}},
             {"native_model_holdout_dvcs",
              json::object{
                  {"available", true},
                  {"operations", json::array{
                      "evaluate_native_model_holdout_dvcs",
                      "batch_evaluate_native_model_holdout_dvcs"}},
                  {"models", json::array{"GPDGK11", "GPDGK16", "GPDGK19",
                      "GPDVGG99"}},
                  {"role", "post_training_external_native_validation_test"},
                  {"training_use", false},
                  {"external_native_validation_use", true},
                  {"replaces_dd_outer_test", false},
                  {"real_data_fit", false},
                  {"coefficient_function_order", "LO"},
                  {"q2_squared_GeV2_domain", "[1,80]"},
                  {"fresh_kinematics_manifest_required", true},
                  {"native_outputs_generated_during_freeze", false},
                  {"analysis_order_label", nullptr},
                  {"thread_safe_parallel_execution", false}}},
             {"dvcs_cff",
              json::object{{"source_runtime_audited", true},
                           {"bridge_available", true},
                           {"restricted_operation",
                            "evaluate_shadow_dvcs_or_"
                            "evaluate_pseudodata_dvcs"},
                           {"coefficient_orders",
                            json::array{"LO", "NLO"}}}},
             {"dvcs_observable",
              json::object{{"source_runtime_audited", true},
                           {"bridge_available", true},
                           {"restricted_operation",
                            "evaluate_shadow_dvcs_or_"
                            "evaluate_pseudodata_dvcs"},
                           {"modules",
                            json::array{
                                "DVCSCrossSectionUUMinus",
                                "DVCSCrossSectionDifferenceLUMinus",
                                "DVCSAc", "DVCSAluMinus", "DVCSAulMinus",
                                "DVCSAllMinus"}}}},
             {"production_perturbative_configurations",
              json::array{}},
             {"verified_evolution_only_configurations",
              json::array{"stage06_lo_apfel_fixed_nf3_msbar_v1",
                          "stage07_lo_apfel_coupled_fixed_nf3_msbar_v1"}},
             {"thread_safe_parallel_execution", false}}}};
}

json::object executeRequest(const json::object& request,
        const char* executablePath) {
    requireOnlyKeys(request,
        {"schema_version", "operation", "request_id", "module", "model", "gpd_type",
            "kinematics", "requests", "representation", "flavor",
            "parton_content", "parameters", "quadrature_order",
            "integration_limit", "evolution_configuration", "input_basis",
            "gpd_diagnostic_kinematics", "observable_kinematics",
            "theory_configuration", "q0_squared", "gpd_diagnostic_x"},
        "request");
    const json::value schema =
        requireField(request, "schema_version", "request");
    if (!schema.is_int64() ||
            schema.as_int64() != BRIDGE_SCHEMA_VERSION) {
        throw BridgeError("unsupported_schema_version",
            "request.schema_version must be 1", kExitValidation);
    }
    const std::string operation =
        requireString(request, "operation", "request");

    if (operation == "evaluate_gpd") {
        Evaluation evaluation = parseEvaluation(request, "request", true);
        PartonsSession session(executablePath);
        const GPDValues values = evaluateGPD(session.get(),
            evaluation.kinematics);
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluation", evaluationJson(evaluation, values)}};
    }

    if (operation == "batch_evaluate_gpd") {
        requireOnlyKeys(request,
            {"schema_version", "operation", "requests"}, "request");
        const json::value requestsValue =
            requireField(request, "requests", "request");
        if (!requestsValue.is_array()) {
            throw BridgeError("invalid_type",
                "request.requests must be an array", kExitValidation);
        }
        const json::array requests = requestsValue.as_array();
        if (requests.empty()) {
            throw BridgeError("invalid_value",
                "request.requests must not be empty", kExitValidation);
        }

        std::vector<Evaluation> evaluations;
        evaluations.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const json::object item = requireObject(requests[index],
                "request.requests[" + std::to_string(index) + "]");
            evaluations.push_back(parseEvaluation(item,
                "request.requests[" + std::to_string(index) + "]"));
        }

        PartonsSession session(executablePath);
        json::array results;
        results.reserve(evaluations.size());
        for (const Evaluation& evaluation : evaluations) {
            results.emplace_back(evaluationJson(evaluation,
                evaluateGPD(session.get(), evaluation.kinematics)));
        }
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluations", std::move(results)}};
    }

    if (operation == "evaluate_dd_gpd") {
        DDEvaluation evaluation =
            parseDDEvaluation(request, "request", true);
        PartonsSession session(executablePath);
        const DDValues values = evaluateDDGPD(session.get(), evaluation);
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluation", ddEvaluationJson(evaluation, values)}};
    }

    if (operation == "batch_evaluate_dd_gpd") {
        requireOnlyKeys(request,
            {"schema_version", "operation", "requests"}, "request");
        const json::value requestsValue =
            requireField(request, "requests", "request");
        if (!requestsValue.is_array()) {
            throw BridgeError("invalid_type",
                "request.requests must be an array", kExitValidation);
        }
        const json::array requests = requestsValue.as_array();
        if (requests.empty()) {
            throw BridgeError("invalid_value",
                "request.requests must not be empty", kExitValidation);
        }

        std::vector<DDEvaluation> evaluations;
        evaluations.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const json::object item = requireObject(requests[index],
                "request.requests[" + std::to_string(index) + "]");
            evaluations.push_back(parseDDEvaluation(item,
                "request.requests[" + std::to_string(index) + "]"));
        }

        PartonsSession session(executablePath);
        json::array results;
        results.reserve(evaluations.size());
        for (const DDEvaluation& evaluation : evaluations) {
            results.emplace_back(ddEvaluationJson(evaluation,
                evaluateDDGPD(session.get(), evaluation)));
        }
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluations", std::move(results)}};
    }

    if (operation == "evaluate_conformal_moment_gpd") {
        ConformalEvaluation evaluation =
            parseConformalEvaluation(request, "request", true);
        PartonsSession session(executablePath);
        const ConformalValues values =
            evaluateConformalGPD(session.get(), evaluation);
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluation", conformalEvaluationJson(evaluation, values)}};
    }

    if (operation == "batch_evaluate_conformal_moment_gpd") {
        requireOnlyKeys(request,
            {"schema_version", "operation", "requests"}, "request");
        const json::value requestsValue =
            requireField(request, "requests", "request");
        if (!requestsValue.is_array()) {
            throw BridgeError("invalid_type",
                "request.requests must be an array", kExitValidation);
        }
        const json::array requests = requestsValue.as_array();
        if (requests.empty()) {
            throw BridgeError("invalid_value",
                "request.requests must not be empty", kExitValidation);
        }

        std::vector<ConformalEvaluation> evaluations;
        evaluations.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const json::object item = requireObject(requests[index],
                "request.requests[" + std::to_string(index) + "]");
            evaluations.push_back(parseConformalEvaluation(item,
                "request.requests[" + std::to_string(index) + "]"));
        }

        PartonsSession session(executablePath);
        json::array results;
        results.reserve(evaluations.size());
        for (const ConformalEvaluation& evaluation : evaluations) {
            results.emplace_back(conformalEvaluationJson(
                evaluation, evaluateConformalGPD(session.get(), evaluation)));
        }
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluations", std::move(results)}};
    }

    if (operation == "evaluate_evolved_dd_gpd") {
        DDEvaluation evaluation =
            parseDDEvaluation(request, "request", true, true);
        PartonsSession session(executablePath);
        const DDValues values = evaluateDDGPD(session.get(), evaluation);
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluation", ddEvaluationJson(evaluation, values)}};
    }

    if (operation == "batch_evaluate_evolved_dd_gpd") {
        requireOnlyKeys(request,
            {"schema_version", "operation", "requests"}, "request");
        const json::value requestsValue =
            requireField(request, "requests", "request");
        if (!requestsValue.is_array()) {
            throw BridgeError("invalid_type",
                "request.requests must be an array", kExitValidation);
        }
        const json::array requests = requestsValue.as_array();
        if (requests.empty()) {
            throw BridgeError("invalid_value",
                "request.requests must not be empty", kExitValidation);
        }

        std::vector<DDEvaluation> evaluations;
        evaluations.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const json::object item = requireObject(requests[index],
                "request.requests[" + std::to_string(index) + "]");
            evaluations.push_back(parseDDEvaluation(item,
                "request.requests[" + std::to_string(index) + "]",
                false, true));
        }

        PartonsSession session(executablePath);
        json::array results;
        results.reserve(evaluations.size());
        for (const DDEvaluation& evaluation : evaluations) {
            results.emplace_back(ddEvaluationJson(evaluation,
                evaluateDDGPD(session.get(), evaluation)));
        }
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluations", std::move(results)}};
    }

    if (operation == "evaluate_coupled_basis_gpd") {
        BasisEvaluation evaluation =
            parseBasisEvaluation(request, "request", true);
        PartonsSession session(executablePath);
        const BasisValues values =
            evaluateBasisGPD(session.get(), evaluation);
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluation", basisEvaluationJson(evaluation, values)}};
    }

    if (operation == "batch_evaluate_coupled_basis_gpd") {
        requireOnlyKeys(request,
            {"schema_version", "operation", "requests"}, "request");
        const json::value requestsValue =
            requireField(request, "requests", "request");
        if (!requestsValue.is_array()) {
            throw BridgeError("invalid_type",
                "request.requests must be an array", kExitValidation);
        }
        const json::array requests = requestsValue.as_array();
        if (requests.empty()) {
            throw BridgeError("invalid_value",
                "request.requests must not be empty", kExitValidation);
        }

        std::vector<BasisEvaluation> evaluations;
        evaluations.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const json::object item = requireObject(requests[index],
                "request.requests[" + std::to_string(index) + "]");
            evaluations.push_back(parseBasisEvaluation(item,
                "request.requests[" + std::to_string(index) + "]"));
        }

        PartonsSession session(executablePath);
        json::array results;
        results.reserve(evaluations.size());
        for (const BasisEvaluation& evaluation : evaluations) {
            results.emplace_back(basisEvaluationJson(evaluation,
                evaluateBasisGPD(session.get(), evaluation)));
        }
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluations", std::move(results)}};
    }

    if (operation == "evaluate_shadow_dvcs") {
        const ShadowEvaluation evaluation =
            parseShadowEvaluation(request, "request", true);
        PartonsSession session(executablePath);
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluation", shadowEvaluationJson(
                evaluation, evaluateShadowDVCS(session.get(), evaluation))}};
    }

    if (operation == "batch_evaluate_shadow_dvcs") {
        requireOnlyKeys(request,
            {"schema_version", "operation", "requests"}, "request");
        const json::value requestsValue =
            requireField(request, "requests", "request");
        if (!requestsValue.is_array()) {
            throw BridgeError("invalid_type",
                "request.requests must be an array", kExitValidation);
        }
        const json::array requests = requestsValue.as_array();
        if (requests.empty()) {
            throw BridgeError("invalid_value",
                "request.requests must not be empty", kExitValidation);
        }

        std::vector<ShadowEvaluation> evaluations;
        evaluations.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const json::object item = requireObject(requests[index],
                "request.requests[" + std::to_string(index) + "]");
            evaluations.push_back(parseShadowEvaluation(item,
                "request.requests[" + std::to_string(index) + "]"));
        }

        PartonsSession session(executablePath);
        json::array results;
        results.reserve(evaluations.size());
        for (const ShadowEvaluation& evaluation : evaluations) {
            results.emplace_back(shadowEvaluationJson(
                evaluation, evaluateShadowDVCS(session.get(), evaluation)));
        }
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluations", std::move(results)}};
    }

    if (operation == "evaluate_pseudodata_dvcs") {
        const PseudodataEvaluation evaluation =
            parsePseudodataEvaluation(request, "request", true);
        PartonsSession session(executablePath);
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluation", pseudodataEvaluationJson(
                evaluation,
                evaluatePseudodataDVCS(session.get(), evaluation))}};
    }

    if (operation == "batch_evaluate_pseudodata_dvcs") {
        requireOnlyKeys(request,
            {"schema_version", "operation", "requests"}, "request");
        const json::value requestsValue =
            requireField(request, "requests", "request");
        if (!requestsValue.is_array()) {
            throw BridgeError("invalid_type",
                "request.requests must be an array", kExitValidation);
        }
        const json::array requests = requestsValue.as_array();
        if (requests.empty()) {
            throw BridgeError("invalid_value",
                "request.requests must not be empty", kExitValidation);
        }
        std::vector<PseudodataEvaluation> evaluations;
        evaluations.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const json::object item = requireObject(requests[index],
                "request.requests[" + std::to_string(index) + "]");
            evaluations.push_back(parsePseudodataEvaluation(item,
                "request.requests[" + std::to_string(index) + "]"));
        }

        PartonsSession session(executablePath);
        json::array results;
        results.reserve(evaluations.size());
        for (const PseudodataEvaluation& evaluation : evaluations) {
            results.emplace_back(pseudodataEvaluationJson(
                evaluation,
                evaluatePseudodataDVCS(session.get(), evaluation)));
        }
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluations", std::move(results)}};
    }

    if (operation == "evaluate_post_training_comparison_dvcs") {
        const PseudodataEvaluation evaluation =
            parsePseudodataEvaluation(request, "request", true, true);
        PartonsSession session(executablePath);
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"comparison_contract", json::object{
                {"role", "post_training_diagnostic_only"},
                {"real_data_fit", false},
                {"posterior_updated", false},
                {"source_Q2_retained", true},
                {"data_evolved", false},
                {"gpd_evolved_to_each_datum_Q2", true}}},
            {"evaluation", pseudodataEvaluationJson(
                evaluation,
                evaluatePseudodataDVCS(session.get(), evaluation))}};
    }

    if (operation == "batch_evaluate_post_training_comparison_dvcs") {
        requireOnlyKeys(request,
            {"schema_version", "operation", "requests"}, "request");
        const json::value requestsValue =
            requireField(request, "requests", "request");
        if (!requestsValue.is_array()) {
            throw BridgeError("invalid_type",
                "request.requests must be an array", kExitValidation);
        }
        const json::array requests = requestsValue.as_array();
        if (requests.empty()) {
            throw BridgeError("invalid_value",
                "request.requests must not be empty", kExitValidation);
        }
        std::vector<PseudodataEvaluation> evaluations;
        evaluations.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const json::object item = requireObject(requests[index],
                "request.requests[" + std::to_string(index) + "]");
            evaluations.push_back(parsePseudodataEvaluation(item,
                "request.requests[" + std::to_string(index) + "]",
                false, true));
        }
        PartonsSession session(executablePath);
        json::array results;
        results.reserve(evaluations.size());
        for (const PseudodataEvaluation& evaluation : evaluations) {
            results.emplace_back(pseudodataEvaluationJson(
                evaluation,
                evaluatePseudodataDVCS(session.get(), evaluation)));
        }
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"comparison_contract", json::object{
                {"role", "post_training_diagnostic_only"},
                {"real_data_fit", false},
                {"posterior_updated", false},
                {"source_Q2_retained", true},
                {"data_evolved", false},
                {"gpd_evolved_to_each_datum_Q2", true}}},
            {"evaluations", std::move(results)}};
    }

    if (operation == "evaluate_native_model_holdout_dvcs") {
        const auto evaluation = parseNativeModelHoldoutEvaluation(
            request, "request", true);
        PartonsSession session(executablePath);
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluation", nativeModelHoldoutJson(evaluation,
                evaluateNativeModelHoldout(session.get(), evaluation))}};
    }

    if (operation == "batch_evaluate_native_model_holdout_dvcs") {
        requireOnlyKeys(request,
            {"schema_version", "operation", "requests"}, "request");
        const json::value requestsValue =
            requireField(request, "requests", "request");
        if (!requestsValue.is_array() || requestsValue.as_array().empty()) {
            throw BridgeError("invalid_value",
                "request.requests must be a non-empty array", kExitValidation);
        }
        std::vector<NativeModelHoldoutEvaluation> evaluations;
        const json::array requests = requestsValue.as_array();
        evaluations.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const json::object item = requireObject(requests[index],
                "request.requests[" + std::to_string(index) + "]");
            evaluations.push_back(parseNativeModelHoldoutEvaluation(item,
                "request.requests[" + std::to_string(index) + "]"));
        }
        PartonsSession session(executablePath);
        json::array results;
        results.reserve(evaluations.size());
        for (const auto& evaluation : evaluations) {
            results.emplace_back(nativeModelHoldoutJson(evaluation,
                evaluateNativeModelHoldout(session.get(), evaluation)));
        }
        return json::object{
            {"schema_version", BRIDGE_SCHEMA_VERSION},
            {"status", "ok"},
            {"operation", operation},
            {"backend", backendProvenance()},
            {"evaluations", std::move(results)}};
    }

    throw BridgeError("unsupported_operation",
        "request.operation must be 'evaluate_gpd' or "
        "'batch_evaluate_gpd', 'evaluate_dd_gpd', "
        "'batch_evaluate_dd_gpd', 'evaluate_evolved_dd_gpd', or "
        "'batch_evaluate_evolved_dd_gpd', "
        "'evaluate_coupled_basis_gpd', or "
        "'batch_evaluate_coupled_basis_gpd', "
        "'evaluate_conformal_moment_gpd', or "
        "'batch_evaluate_conformal_moment_gpd', "
        "'evaluate_shadow_dvcs', or "
        "'batch_evaluate_shadow_dvcs', "
        "'evaluate_pseudodata_dvcs', or "
        "'batch_evaluate_pseudodata_dvcs', "
        "'evaluate_post_training_comparison_dvcs', or "
        "'batch_evaluate_post_training_comparison_dvcs', "
        "'evaluate_native_model_holdout_dvcs', or "
        "'batch_evaluate_native_model_holdout_dvcs'",
        kExitValidation);
}

std::string readInput(const std::string& path) {
    std::ostringstream buffer;
    if (path == "-") {
        buffer << std::cin.rdbuf();
    } else {
        std::ifstream stream(path);
        if (!stream) {
            throw BridgeError("input_open_failed",
                "unable to open input file '" + path + "'", kExitUsage);
        }
        buffer << stream.rdbuf();
    }
    return buffer.str();
}

json::object selfTest(const char* executablePath) {
    const Evaluation reference{
        "stage01-self-test", Kinematics{0.1, 0.2, -0.1, 2.0, 2.0}};
    constexpr double expectedUp = 5.1000189639572495;
    constexpr double expectedGluon = 1.213564197723455;
    constexpr double tolerance = 5e-13;

    PartonsSession session(executablePath);
    const GPDValues values = evaluateGPD(session.get(),
        reference.kinematics);
    const double upDifference = std::abs(values.up - expectedUp);
    const double gluonDifference = std::abs(values.gluon - expectedGluon);
    const bool passed =
        upDifference <= tolerance && gluonDifference <= tolerance;

    json::object response{
        {"schema_version", BRIDGE_SCHEMA_VERSION},
        {"status", passed ? "ok" : "error"},
        {"operation", "self_test"},
        {"backend", backendProvenance()},
        {"fixture", "stage00-gpd-gk16-reference-v1"},
        {"tolerance_absolute", tolerance},
        {"computed", evaluationJson(reference, values)},
        {"absolute_difference",
         json::object{{"h_up", upDifference},
                      {"h_gluon", gluonDifference}}},
        {"passed", passed}};
    if (!passed) {
        response["error"] = json::object{
            {"code", "self_test_mismatch"},
            {"message", "native result differs from retained fixture"}};
    }
    return response;
}

void printJson(const json::value& value) {
    std::cout << json::serialize(value) << '\n';
}

void printUsage() {
    std::cout
        << "Usage:\n"
        << "  partons_bridge --capabilities\n"
        << "  partons_bridge --self-test\n"
        << "  partons_bridge --input <request.json|->\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--capabilities") {
            printJson(capabilitiesResponse());
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            const json::object response = selfTest(argv[0]);
            const bool passed = response.at("passed").as_bool();
            printJson(response);
            return passed ? 0 : kExitSelfTest;
        }
        if (argc == 2 && (std::string(argv[1]) == "--help" ||
                            std::string(argv[1]) == "-h")) {
            printUsage();
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--input") {
            const std::string input = readInput(argv[2]);
            json::error_code error;
            const json::value requestValue = json::parse(input, error);
            if (error) {
                throw BridgeError("json_parse_error",
                    "invalid JSON: " + error.message(), kExitParse);
            }
            const json::object request =
                requireObject(requestValue, "request");
            printJson(executeRequest(request, argv[0]));
            return 0;
        }

        throw BridgeError("invalid_command_line",
            "use --help for the supported command line", kExitUsage);
    } catch (const BridgeError& error) {
        printJson(errorResponse(error.code(), error.what()));
        return error.exitCode();
    } catch (const ElemUtils::CustomException& error) {
        printJson(errorResponse("backend_error", error.what()));
        return kExitBackend;
    } catch (const std::exception& error) {
        printJson(errorResponse("internal_error", error.what()));
        return kExitBackend;
    }
}
