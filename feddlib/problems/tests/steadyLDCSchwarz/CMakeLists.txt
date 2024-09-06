TRIBITS_ADD_EXECUTABLE(
  steadyLDCSchwarz
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
  steadyLDCSchwarz
  NAME steadyLDCSchwarz
	ARGS
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyLDCSchwarz
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCE_FILES parametersProblem.xml parametersSolverNonLinSchwarz.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
  DEST_FILES parametersProblem.xml parametersSolverNonLinSchwarz.xml
  EXEDEPS steadyLDCSchwarz
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyLDCSchwarz
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
	SOURCE_FILES square.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES square.mesh
  EXEDEPS steadyLDCSchwarz
)
