TRIBITS_ADD_EXECUTABLE(
	nonLinElasWithNonLinSchwarz
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
	nonLinElasWithNonLinSchwarz
    NAME nonLinElasWithNonLinSchwarz
    ARGS
    NUM_MPI_PROCS 1
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_nonLinElasWithNonLinSchwarz
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCE_FILES parametersProblem.xml parametersSolverNonLinSchwarz.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
  DEST_FILES parametersProblem.xml parametersSolverNonLinSchwarz.xml
	EXEDEPS nonLinElasWithNonLinSchwarz
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_nonLinElasWithNonLinSchwarz
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
	SOURCE_FILES square.mesh
  SOURCE_FILES beam2D26k.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES square.mesh
  DEST_FILES beam2D26k.mesh
  EXEDEPS nonLinElasWithNonLinSchwarz
)

