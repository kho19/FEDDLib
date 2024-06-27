#ifndef NONLINEARCOMBINEOPERATOR_DEF_HPP
#define NONLINEARCOMBINEOPERATOR_DEF_HPP

#include "CombineOperator_decl.hpp"
#include "NonLinearCombineOperator_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearOperator_decl.hpp"
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_RCPDecl.hpp>
/*!
 @brief Implementation of NonlinearCombineOperator which extends the FROSch Combine operator to allow non-const apply()
 methods
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC, class LO, class GO, class NO>
NonLinearCombineOperator<SC, LO, GO, NO>::NonLinearCombineOperator(CommPtr comm)
    : CombineOperator<SC, LO, GO, NO>(comm) {}

template <class SC, class LO, class GO, class NO>
void NonLinearCombineOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly,
                                                     Teuchos::ETransp mode, SC alpha, SC beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "This apply() overload should not be used with nonlinear operators");
}

template <class SC, class LO, class GO, class NO>
int NonLinearCombineOperator<SC, LO, GO, NO>::addOperator(NonLinearOperatorPtr op) {
    NonLinearOperatorVector_.push_back(op);
    EnableNonLinearOperators_.push_back(true);
    return 0;
}

template <class SC, class LO, class GO, class NO>
int NonLinearCombineOperator<SC, LO, GO, NO>::addOperators(NonLinearOperatorPtrVecPtr operators) {
    int ret = 0;
    for (UN i = 1; i < operators.size(); i++) {
        if (0 > addOperator(operators[i]))
            ret -= pow(10, i);
    }
    return ret;
}

}; // namespace FROSch

#endif
