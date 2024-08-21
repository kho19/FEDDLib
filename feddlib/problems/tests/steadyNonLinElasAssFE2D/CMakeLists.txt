TRIBITS_ADD_EXECUTABLE(
  steadyNonLinElasAssFE2D
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
  steadyNonLinElasAssFE2D
  NAME steadyNonLinElasAssFE2D
	ARGS
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyNonLinElasAssFE2D
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml  parametersProblem.xml parametersSolver.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml  parametersProblem.xml parametersSolver.xml
  EXEDEPS steadyNonLinElasAssFE2D
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyNonLinElasAssFE2D
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
	SOURCE_FILES square.mesh
  SOURCE_FILES beam2D59k.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES square.mesh
  DEST_FILES beam2D59k.mesh
  EXEDEPS steadyNonLinElasAssFE2D
)
