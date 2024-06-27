#ifndef NONLINEARSUMOPERATOR_DEF_HPP
#define NONLINEARSUMOPERATOR_DEF_HPP

#include "NonLinearSumOperator_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearOperator_decl.hpp"
#include "NonLinearCombineOperator_decl.hpp"
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_RCPDecl.hpp>
/*!
 @brief Implementation of NonlinearSumOperator which extends the FROSch sum operator to allow non-const apply() methods
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC, class LO, class GO, class NO>
NonLinearSumOperator<SC, LO, GO, NO>::NonLinearSumOperator(CommPtr comm) : NonLinearCombineOperator<SC, LO, GO, NO>(comm) {}

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

}; // namespace FROSch

#endif
