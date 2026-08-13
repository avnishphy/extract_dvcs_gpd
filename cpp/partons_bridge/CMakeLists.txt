if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost 1.83 REQUIRED COMPONENTS json)
find_package(GSL 2.8 REQUIRED)

find_path(PARTONS_INCLUDE_DIR partons/Partons.h REQUIRED)
find_path(APFELXX_INCLUDE_DIR apfel/version.h REQUIRED)
find_path(ELEMENTARY_INCLUDE_DIR ElementaryUtils/ElementaryUtilsVersion.h REQUIRED)
find_path(NUMA_INCLUDE_DIR NumA/NumAVersion.h REQUIRED)
find_path(LHAPDF_INCLUDE_DIR LHAPDF/Version.h REQUIRED)
find_library(PARTONS_LIBRARY NAMES PARTONS REQUIRED)
find_library(APFELXX_LIBRARY NAMES apfelxx REQUIRED)
find_library(ELEMENTARY_LIBRARY NAMES ElementaryUtils REQUIRED)
find_library(NUMA_LIBRARY NAMES "NumA++" REQUIRED)
find_library(LHAPDF_LIBRARY NAMES LHAPDF REQUIRED)
find_file(PARTONS_SCHEMA_FILE NAMES xmlSchema.xsd
  HINTS ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES share/extract-dvcs-gpd data
  REQUIRED)

set(BRIDGE_SOURCES
  src/main.cpp src/FixedScaleDDGPD.cpp src/BasisInjectionGPD.cpp
  src/ConformalMomentGPD.cpp src/ShadowInjectionGPD.cpp
  src/PseudodataInputGPD.cpp src/TabulatedPseudodataGPD.cpp)
add_executable(partons_bridge ${BRIDGE_SOURCES})
target_compile_features(partons_bridge PRIVATE cxx_std_17)
target_compile_options(partons_bridge PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_include_directories(partons_bridge SYSTEM PRIVATE
  "${PARTONS_INCLUDE_DIR}" "${APFELXX_INCLUDE_DIR}"
  "${ELEMENTARY_INCLUDE_DIR}" "${NUMA_INCLUDE_DIR}" "${LHAPDF_INCLUDE_DIR}")
target_link_libraries(partons_bridge PRIVATE Boost::json GSL::gsl GSL::gslcblas
  "${PARTONS_LIBRARY}" "${APFELXX_LIBRARY}" "${NUMA_LIBRARY}"
  "${ELEMENTARY_LIBRARY}" "${LHAPDF_LIBRARY}")
target_link_options(partons_bridge PRIVATE -Wl,--no-as-needed)
set_target_properties(partons_bridge PROPERTIES
  CXX_EXTENSIONS OFF
  BUILD_RPATH_USE_ORIGIN ON
  INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
  INSTALL_RPATH_USE_LINK_PATH FALSE)

function(source_hash variable file)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${file}" ${variable})
  set(${variable} "${${variable}}" PARENT_SCOPE)
endfunction()
source_hash(BRIDGE_SOURCE_SHA256 src/main.cpp)
source_hash(DD_SOURCE_SHA256 src/FixedScaleDDGPD.cpp)
source_hash(DD_HEADER_SHA256 src/FixedScaleDDGPD.h)
source_hash(BASIS_SOURCE_SHA256 src/BasisInjectionGPD.cpp)
source_hash(BASIS_HEADER_SHA256 src/BasisInjectionGPD.h)
source_hash(CONFORMAL_SOURCE_SHA256 src/ConformalMomentGPD.cpp)
source_hash(CONFORMAL_HEADER_SHA256 src/ConformalMomentGPD.h)
source_hash(SHADOW_SOURCE_SHA256 src/ShadowInjectionGPD.cpp)
source_hash(SHADOW_HEADER_SHA256 src/ShadowInjectionGPD.h)
source_hash(PSEUDODATA_SOURCE_SHA256 src/PseudodataInputGPD.cpp)
source_hash(PSEUDODATA_HEADER_SHA256 src/PseudodataInputGPD.h)
source_hash(TABULATED_PSEUDODATA_SOURCE_SHA256 src/TabulatedPseudodataGPD.cpp)
source_hash(TABULATED_PSEUDODATA_HEADER_SHA256 src/TabulatedPseudodataGPD.h)
foreach(model GK11 GK16 GK19 VGG99)
  file(SHA256 "${PARTONS_INCLUDE_DIR}/partons/modules/gpd/GPD${model}.h"
    GPD_${model}_HEADER_SHA256)
endforeach()
# Source hashes are supplied by the dependency lock and verified at image build.
set(GPD_GK11_SOURCE_SHA256 "6d7823662014526533f0276b687c4d8eaf7d7d032fb465b7ca37a800b6aecafd")
set(GPD_GK16_SOURCE_SHA256 "4512e3d7dcab78bdbf22a6491bc80208eccc49c658d53f2376b6536950d07aee")
set(GPD_GK19_SOURCE_SHA256 "c66062b25d8696bdc75aa4ea580cbb9f84f20f0384bb4201415e59902e06a8fe")
set(GPD_VGG99_SOURCE_SHA256 "06898d77120340abecf2879d08dc03534abec520b982e2df115525f368e44335")
target_compile_definitions(partons_bridge PRIVATE
  BRIDGE_VERSION="0.9.0" BRIDGE_SCHEMA_VERSION=1
  BRIDGE_SOURCE_SHA256="${BRIDGE_SOURCE_SHA256}"
  DD_SOURCE_SHA256="${DD_SOURCE_SHA256}" DD_HEADER_SHA256="${DD_HEADER_SHA256}"
  BASIS_SOURCE_SHA256="${BASIS_SOURCE_SHA256}" BASIS_HEADER_SHA256="${BASIS_HEADER_SHA256}"
  CONFORMAL_SOURCE_SHA256="${CONFORMAL_SOURCE_SHA256}" CONFORMAL_HEADER_SHA256="${CONFORMAL_HEADER_SHA256}"
  SHADOW_SOURCE_SHA256="${SHADOW_SOURCE_SHA256}" SHADOW_HEADER_SHA256="${SHADOW_HEADER_SHA256}"
  PSEUDODATA_SOURCE_SHA256="${PSEUDODATA_SOURCE_SHA256}" PSEUDODATA_HEADER_SHA256="${PSEUDODATA_HEADER_SHA256}"
  TABULATED_PSEUDODATA_SOURCE_SHA256="${TABULATED_PSEUDODATA_SOURCE_SHA256}"
  TABULATED_PSEUDODATA_HEADER_SHA256="${TABULATED_PSEUDODATA_HEADER_SHA256}"
  GPD_GK11_HEADER_SHA256="${GPD_GK11_HEADER_SHA256}" GPD_GK11_SOURCE_SHA256="${GPD_GK11_SOURCE_SHA256}"
  GPD_GK16_HEADER_SHA256="${GPD_GK16_HEADER_SHA256}" GPD_GK16_SOURCE_SHA256="${GPD_GK16_SOURCE_SHA256}"
  GPD_GK19_HEADER_SHA256="${GPD_GK19_HEADER_SHA256}" GPD_GK19_SOURCE_SHA256="${GPD_GK19_SOURCE_SHA256}"
  GPD_VGG99_HEADER_SHA256="${GPD_VGG99_HEADER_SHA256}" GPD_VGG99_SOURCE_SHA256="${GPD_VGG99_SOURCE_SHA256}"
  BRIDGE_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}" BRIDGE_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
  BRIDGE_PARTONS_LIBRARY="libPARTONS.so" BRIDGE_APFELXX_LIBRARY="libapfelxx.so"
  BRIDGE_NUMA_LIBRARY="libNumA++.so" BRIDGE_ELEMENTARY_LIBRARY="libElementaryUtils.so"
  BRIDGE_LHAPDF_LIBRARY="libLHAPDF.so" BRIDGE_GSL_LIBRARY="libgsl.so")

set(BRIDGE_LOGGER_PATH "logger.properties")
set(PARTONS_SCHEMA_PATH "xmlSchema.xsd")
configure_file(config/partons.properties.in partons.properties @ONLY)
configure_file(config/logger.properties.in logger.properties @ONLY)
configure_file("${PARTONS_SCHEMA_FILE}" xmlSchema.xsd COPYONLY)
install(TARGETS partons_bridge RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/partons.properties"
              "${CMAKE_CURRENT_BINARY_DIR}/logger.properties"
              "${PARTONS_SCHEMA_FILE}"
        DESTINATION ${CMAKE_INSTALL_BINDIR})

if(DVCS_BUILD_NATIVE_TESTS)
  foreach(spec
      "stage03_dd_physics_test;tests/stage03_dd_physics_test.cpp;src/FixedScaleDDGPD.cpp"
      "stage06_evolution_physics_test;tests/stage06_evolution_physics_test.cpp;src/FixedScaleDDGPD.cpp"
      "stage07_basis_mixing_test;tests/stage07_basis_mixing_test.cpp;src/FixedScaleDDGPD.cpp;src/BasisInjectionGPD.cpp"
      "stage08_conformal_physics_test;tests/stage08_conformal_physics_test.cpp;src/ConformalMomentGPD.cpp"
      "stage09_shadow_physics_test;tests/stage09_shadow_physics_test.cpp;src/ShadowInjectionGPD.cpp")
    list(GET spec 0 target)
    list(REMOVE_AT spec 0)
    add_executable(${target} ${spec})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_include_directories(${target} SYSTEM PRIVATE "${PARTONS_INCLUDE_DIR}"
      "${APFELXX_INCLUDE_DIR}" "${ELEMENTARY_INCLUDE_DIR}" "${NUMA_INCLUDE_DIR}" "${LHAPDF_INCLUDE_DIR}")
    target_link_libraries(${target} PRIVATE GSL::gsl GSL::gslcblas "${PARTONS_LIBRARY}"
      "${APFELXX_LIBRARY}" "${NUMA_LIBRARY}" "${ELEMENTARY_LIBRARY}" "${LHAPDF_LIBRARY}")
    set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/../../../${CMAKE_INSTALL_LIBDIR}")
    install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/extract-dvcs-gpd/tests)
  endforeach()
endif()
