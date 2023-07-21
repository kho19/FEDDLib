################################################################################
#                                                                              #
#                        FEDDLib                                               #
#                                                                              #
################################################################################

# Define your minimum CMake version
cmake_minimum_required(VERSION 3.23.0 FATAL_ERROR)

# For debugging
include(CMakePrintHelpers)
#
# A) Define project name and set up major project options
#
# NOTE: Don't set options that would impact what packages get defined or
# enabled/disabled in this file as that would not impact other tools that
# don't process this file.
#

# Get PROJECT_NAME (must be in a file for other parts of system to read)
include("${CMAKE_CURRENT_SOURCE_DIR}/ProjectName.cmake")

# CMake requires that you declare the CMake project in the top-level file and
# not in an include file :-(
project(${PROJECT_NAME})

# Set version number and project mode
include("${CMAKE_CURRENT_SOURCE_DIR}/Version.cmake")

# TODO: Should compiler and linker options be placed somewhere else? Check the tribits docs
set(${PROJECT_NAME}_ENABLE_Fortran ON CACHE BOOL "Enable Fortran compiler")

# Set default C++ standard to C++17
set(FEDDLib_ENABLE_CXX17_DEFAULT ON)

# set up a Profiling build type
set(CMAKE_CXX_FLAGS_PROFILING "-O3 -pg -g" CACHE STRING "CXX profiling flags")
set(CMAKE_C_FLAGS_PROFILING   "-O3 -pg -g" CACHE STRING "C profiling flags")

#set(GCC_COVERAGE_LINK_FLAGS    "-lm")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}" )

#
# B) Pull in the TriBITS system and execute
#
# TODO: tribits source tree should be git fetched automatically by cmake if not already present 
set(${PROJECT_NAME}_TRIBITS_DIR
  "/home/kho/Workspace/feddlibtemp/cmake/tribits/"  CACHE  STRING
  "TriBITS base directory (default assumes in TriBITS source tree)")
include("${${PROJECT_NAME}_TRIBITS_DIR}/TriBITS.cmake")

include_directories(${CMAKE_SOURCE_DIR})

# Do all of the processing for this Tribits project
tribits_project()

# TODO: this needs to be changed. The new tribits doesn't have FEDDLib_ENABLED_PACKAGES
# Not critical since FEDD_HAVE_... are not used for now
foreach(ENABLED_PACKAGE ${FEDDLib_ENABLED_PACKAGES})
	string(TOUPPER ${ENABLED_PACKAGE} UPPER_PACKAGE)
	set(FEDD_HAVE_${UPPER_PACKAGE} ON)
endforeach()

configure_file(	${CMAKE_SOURCE_DIR}/${PROJECT_NAME}_Config.h.in
				${CMAKE_BINARY_DIR}/${PROJECT_NAME}_Config.h)
# We need the next line to install feddlib_config.h.in too
install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}_config.h DESTINATION include)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/feddlib/debug.areas
DESTINATION share/feddlib)
