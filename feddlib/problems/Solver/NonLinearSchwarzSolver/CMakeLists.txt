set(nonlinear_schwarz_HEADERS
    Solver/NonLinearSchwarzSolver/CoarseNonLinearSchwarzOperator.hpp
    Solver/NonLinearSchwarzSolver/CoarseNonLinearSchwarzOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/CoarseNonLinearSchwarzOperator_def.hpp
    Solver/NonLinearSchwarzSolver/CombineOperator.hpp
    Solver/NonLinearSchwarzSolver/CombineOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/CombineOperator_def.hpp
    Solver/NonLinearSchwarzSolver/H1Operator.hpp
    Solver/NonLinearSchwarzSolver/H1Operator_decl.hpp
    Solver/NonLinearSchwarzSolver/H1Operator_def.hpp
    Solver/NonLinearSchwarzSolver/NonLinearCombineOperator.hpp
    Solver/NonLinearSchwarzSolver/NonLinearCombineOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/NonLinearCombineOperator_def.hpp
    Solver/NonLinearSchwarzSolver/NonLinearH1Operator.hpp
    Solver/NonLinearSchwarzSolver/NonLinearH1Operator_decl.hpp
    Solver/NonLinearSchwarzSolver/NonLinearH1Operator_def.hpp
    Solver/NonLinearSchwarzSolver/NonLinearOperator.hpp
    Solver/NonLinearSchwarzSolver/NonLinearOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/NonLinearOperator_def.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSchwarzOperator.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSchwarzOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSchwarzOperator_def.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSumOperator.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSumOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/NonLinearSumOperator_def.hpp
    Solver/NonLinearSchwarzSolver/SimpleCoarseOperator.hpp
    Solver/NonLinearSchwarzSolver/SimpleCoarseOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/SimpleCoarseOperator_def.hpp
    Solver/NonLinearSchwarzSolver/SimpleOverlappingOperator.hpp
    Solver/NonLinearSchwarzSolver/SimpleOverlappingOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/SimpleOverlappingOperator_def.hpp
    Solver/NonLinearSchwarzSolver/SumOperator.hpp
    Solver/NonLinearSchwarzSolver/SumOperator_decl.hpp
    Solver/NonLinearSchwarzSolver/SumOperator_def.hpp
    CACHE INTERNAL "")

set(nonlinear_schwarz_SOURCES
    Solver/NonLinearSchwarzSolver/CoarseNonLinearSchwarzOperator.cpp
    Solver/NonLinearSchwarzSolver/CombineOperator.cpp
    Solver/NonLinearSchwarzSolver/H1Operator.cpp
    Solver/NonLinearSchwarzSolver/NonLinearCombineOperator.cpp
    Solver/NonLinearSchwarzSolver/NonLinearH1Operator.cpp
    Solver/NonLinearSchwarzSolver/NonLinearOperator.cpp
    Solver/NonLinearSchwarzSolver/NonLinearSchwarzOperator.cpp
    Solver/NonLinearSchwarzSolver/NonLinearSumOperator.cpp
    Solver/NonLinearSchwarzSolver/SimpleCoarseOperator.cpp
    Solver/NonLinearSchwarzSolver/SimpleOverlappingOperator.cpp
    Solver/NonLinearSchwarzSolver/SumOperator.cpp
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
