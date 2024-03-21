#ifndef NONLINEARSUMOPERATOR_DEF_HPP
#define NONLINEARSUMOPERATOR_DEF_HPP

#include "NonLinearSumOperator_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearOperator_decl.hpp"
#include <FROSch_SumOperator_decl.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_RCPDecl.hpp>
/*!
 Implementation of NonlinearSumOperator which extends the FROSch sum operator to allow non-const apply() methods

 @brief Implements the coarse correction T_0 from the nonlinear Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC, class LO, class GO, class NO>
NonLinearSumOperator<SC, LO, GO, NO>::NonLinearSumOperator(CommPtr comm) : SumOperator<SC, LO, GO, NO>(comm) {}

// Y = alpha * A^mode * X + beta * Y
template <class SC, class LO, class GO, class NO>
void NonLinearSumOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, SC alpha, SC beta) {
    if (this->NonLinearOperatorVector_.size() > 0) {
        if (this->XTmp_.is_null())
            this->XTmp_ = MultiVectorFactory<SC, LO, GO, NO>::Build(x.getMap(), x.getNumVectors());
        *this->XTmp_ = x; // Incase x=y
        bool firstOp = true;
        for (UN i = 0; i < this->NonLinearOperatorVector_.size(); i++) {
            if (this->EnableNonLinearOperators_[i]) {
                // We can be sure that all operators are of type NonLinearOperator since this is checked when adding
                // We need to dynamic_cast here anyway because NonLinearOperator and SchwarzOperator are not related
                // This could be changed by modifying FROSch to allow virtual inheritance
                rcp_dynamic_cast<NonLinearOperator<SC, LO, GO, NO>>(this->NonLinearOperatorVector_[i])
                    ->apply(*this->XTmp_, y, alpha, beta);
                if (firstOp) {
                    beta = ScalarTraits<SC>::one();
                    firstOp = false;
                }
            }
        }
    } else {
        y.update(alpha, x, beta);
    }
}

template <class SC, class LO, class GO, class NO>
void NonLinearSumOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly,
                                                 Teuchos::ETransp mode, SC alpha, SC beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "This apply() overload should not be used with nonlinear operators");
}

template <class SC, class LO, class GO, class NO>
int NonLinearSumOperator<SC, LO, GO, NO>::addOperator(NonLinearOperatorPtr op) {
    NonLinearOperatorVector_.push_back(op);
    EnableNonLinearOperators_.push_back(true);
    return 0;
}

template <class SC, class LO, class GO, class NO>
int NonLinearSumOperator<SC, LO, GO, NO>::addOperators(NonLinearOperatorPtrVecPtr operators) {
    int ret = 0;
    for (UN i = 1; i < operators.size(); i++) {
        if (0 > addOperator(operators[i]))
            ret -= pow(10, i);
    }
    return ret;
}

}; // namespace FROSch

#endif
