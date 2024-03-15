set(nonlinear_schwarz_HEADERS
    Solver/NonLinearSchwarzSolver/SimpleOverlappingOperator.hpp
    Solver/NonLinearSchwarzSolver/SimpleOverlappingOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/SimpleOverlappingOperator_def.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSchwarzOperator.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSchwarzOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSchwarzOperator_def.hpp
    Solver/NonLinearSchwarzSolver/CoarseNonLinearSchwarzOperator.hpp
    Solver/NonLinearSchwarzSolver/CoarseNonLinearSchwarzOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/CoarseNonLinearSchwarzOperator_def.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSumOperator.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSumOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSumOperator_def.hpp
    CACHE INTERNAL "")

set(nonlinear_schwarz_SOURCES
    Solver/NonLinearSchwarzSolver/SimpleOverlappingOperator.cpp
    Solver/NonLinearSchwarzSolver/NonLinearSchwarzOperator.cpp
    Solver/NonLinearSchwarzSolver/CoarseNonLinearSchwarzOperator.cpp
    Solver/NonLinearSchwarzSolver/NonLinearSumOperator.cpp
    CACHE INTERNAL "")

set(LOCAL_HEADERS)
foreach(INC ${nonlinear_schwarz_HEADERS})
  string(REPLACE "NonLinearSchwarzSolver/" "" LOCAL_INC ${INC})
  set(LOCAL_HEADERS ${LOCAL_HEADERS} ${LOCAL_INC})
endforeach()

install(
  FILES ${LOCAL_HEADERS}
  DESTINATION
    "${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}/feddlib/problems/Solver/NonLinearSchwarzSolver"
  COMPONENT ${PACKAGE_NAME})
