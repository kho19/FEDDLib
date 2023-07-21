include(TribitsPackageMacros)
include(TribitsAddOptionAndDefine)
include(TribitsLibraryMacros)

#
# A) Define the package
#

tribits_package(core)
set(${PACKAGE_NAME}_VERSION "1.0.0")

#
# B) Set up package-specific options
#

tribits_add_show_deprecated_warnings_option()

foreach(TRILINOS_PACKAGE_NAME in ${Trilinos_PACKAGE_LIST})
    if(${TRILINOS_PACKAGE_NAME} STREQUAL "NOX")
        set(FEDD_HAVE_NOX TRUE)
    endif()
endforeach()

foreach(TRILINOS_PACKAGE_NAME in ${Trilinos_PACKAGE_LIST})
    if(${TRILINOS_PACKAGE_NAME} STREQUAL "Teko")
        set(FEDD_HAVE_TEKO TRUE)
    endif()
endforeach()


if(HAVE_ACEGENINTERFACE EQUAL TRUE)
	message (" Have ACEGEN Interface ")
	set(FEDD_HAVE_ACEGENINTERFACE TRUE)
endif()

tribits_configure_file(${PACKAGE_NAME}_config.h)

set(HEADERS "")
set(SOURCES "")

set(HEADERS ${HEADERS}
    ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.h
)

set(HEADERS ${HEADERS} FEDDCore.hpp)

add_subdirectory(FE)
set(HEADERS ${HEADERS} ${FE_HEADERS})
set(SOURCES ${SOURCES} ${FE_SOURCES})

add_subdirectory(General)
set(HEADERS ${HEADERS} ${General_HEADERS})
set(SOURCES ${SOURCES} ${General_SOURCES})

add_subdirectory(LinearAlgebra)
set(HEADERS ${HEADERS} ${LinearAlgebra_HEADERS})
set(SOURCES ${SOURCES} ${LinearAlgebra_SOURCES})

add_subdirectory(Mesh)
set(HEADERS ${HEADERS} ${Mesh_HEADERS})
set(SOURCES ${SOURCES} ${Mesh_SOURCES})

add_subdirectory(AceFemAssembly)
set(HEADERS ${HEADERS} ${AceFemAssembly_HEADERS})
set(SOURCES ${SOURCES} ${AceFemAssembly_SOURCES})

add_subdirectory(AceFemAssembly/specific)
set(HEADERS ${HEADERS} ${AceFemAssemblySpecific_HEADERS})
set(SOURCES ${SOURCES} ${AceFemAssemblySpecific_SOURCES})

add_subdirectory(AceFemAssembly/TestFE)
set(HEADERS ${HEADERS} ${AceFemAssemblyTestFE_HEADERS})
set(SOURCES ${SOURCES} ${AceFemAssemblyTestFE_SOURCES})

add_subdirectory(AceFemAssembly/AceInterface)
set(HEADERS ${HEADERS} ${AceFemAssemblyAceInterface_HEADERS})
set(SOURCES ${SOURCES} ${AceFemAssemblyAceInterface_SOURCES})

#ADD_SUBDIRECTORY(Utils)
#SET(HEADERS ${HEADERS} ${Utils_HEADERS})
#SET(SOURCES ${SOURCES} ${Utils_SOURCES})

tribits_exclude_autotools_files()
# This is the list of files excluded from the tarball
tribits_exclude_files()

# Add library target
tribits_add_library(
core
HEADERS ${HEADERS}
SOURCES ${SOURCES}
)

tribits_add_test_directories(FE/tests)
tribits_add_test_directories(General/tests)
tribits_add_test_directories(Mesh/tests)
tribits_add_test_directories(LinearAlgebra/tests)
tribits_add_test_directories(AceFemAssembly/tests)

#TRIBITS_ADD_EXAMPLE_DIRECTORIES(examples)

#
# Do standard postprocessing
#

tribits_package_postprocess()

