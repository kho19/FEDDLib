TRIBITS_ADD_EXECUTABLE(
  steadyLDC
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
  steadyLDC
  NAME steadyLDC
	ARGS
#--precfile=parametersPrecMumps.xml
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyLDC
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml parametersProblem.xml parametersSolver.xml
  EXEDEPS steadyLDC
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyLDC
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
	SOURCE_FILES square.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES square.mesh
  EXEDEPS steadyLDC
)
