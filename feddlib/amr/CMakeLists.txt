# This roughly follows the structure of packages in the simple tribits example project
# Probably need to restructure project and core to also use this structure 
# TODO: finish restructuring this like tribits example
#
# A) Define the package
#

tribits_package(amr)
set(${PACKAGE_NAME}_VERSION "1.0.0")

#
# B) Set up package-specific options
#

tribits_add_show_deprecated_warnings_option()

foreach(TRILINOS_PACKAGE_NAME in ${Trilinos_PACKAGE_LIST})
    if(${TRILINOS_PACKAGE_NAME} STREQUAL "NOX")
        set(FEDD_HAVE_NOX TRUE)
    endif()
    if(${TRILINOS_PACKAGE_NAME} STREQUAL "Teko")
        set(FEDD_HAVE_TEKO TRUE)
    endif()
endforeach()


tribits_configure_file(${PACKAGE_NAME}_config.h)

set(HEADERS "")
set(SOURCES "")

tribits_include_directories(${CMAKE_CURRENT_BINARY_DIR})
append_set(HEADERS
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.h
  )

tribits_include_directories(${CMAKE_CURRENT_SOURCE_DIR})

append_set(HEADERS
  AdaptiveMeshRefinement.hpp
	AdaptiveMeshRefinement_decl.hpp
	AdaptiveMeshRefinement_def.hpp
	RefinementFactory.hpp
	RefinementFactory_decl.hpp
	RefinementFactory_def.hpp
	ErrorEstimation.hpp
	ErrorEstimation_decl.hpp
	ErrorEstimation_def.hpp
	ExporterParaViewAMR.hpp
	ExporterParaViewAMR_decl.hpp
	ExporterParaViewAMR_def.hpp
  )

append_set(SOURCES
  AdaptiveMeshRefinement.cpp
	RefinementFactory.cpp
	ErrorEstimation.cpp
	ExporterParaViewAMR.cpp
  )
	
# Add library target
tribits_add_library(
amr
HEADERS ${HEADERS}
SOURCES ${SOURCES}
)

tribits_add_test_directories(tests)

#
# Do standard postprocessing
#

tribits_package_postprocess()

