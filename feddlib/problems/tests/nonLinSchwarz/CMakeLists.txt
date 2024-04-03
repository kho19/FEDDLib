TRIBITS_ADD_EXECUTABLE(
  nonLinSchwarz
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
  nonLinSchwarz
  NAME nonLinSchwarz 
	ARGS
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_nonLinSchwarz
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml  parametersProblem.xml parametersSolver.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml  parametersProblem.xml parametersSolver.xml
  EXEDEPS nonLinSchwarz
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_nonLinSchwarz
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
	SOURCE_FILES simple_square.mesh
	SOURCE_FILES square.mesh
	SOURCE_FILES big_square.mesh
  SOURCE_FILES rectangle.mesh
  SOURCE_FILES large_square.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES simple_square.mesh
	DEST_FILES square.mesh
	DEST_FILES big_square.mesh
  DEST_FILES rectangle.mesh
  DEST_FILES large_square.mesh
  EXEDEPS nonLinSchwarz
)
