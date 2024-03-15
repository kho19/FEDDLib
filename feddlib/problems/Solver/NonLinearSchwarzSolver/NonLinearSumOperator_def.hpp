#ifndef NONLINEARSUMOPERATOR_DEF_HPP
#define NONLINEARSUMOPERATOR_DEF_HPP

#include "NonLinearSumOperator_decl.hpp"
#include <FROSch_SumOperator_decl.hpp>
#include <Teuchos_BLAS_types.hpp>
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
    // Dummy variables for compatibility
    bool usePreconditionerOnly = true;
    ETransp mode = Teuchos::NO_TRANS;
    if (this->OperatorVector_.size() > 0) {
        if (this->XTmp_.is_null())
            this->XTmp_ = MultiVectorFactory<SC, LO, GO, NO>::Build(x.getMap(), x.getNumVectors());
        *this->XTmp_ = x; // Incase x=y
        UN itmp = 0;
        for (UN i = 0; i < this->OperatorVector_.size(); i++) {
            if (this->EnableOperators_[i]) {
                this->OperatorVector_[i]->apply(*this->XTmp_, y, usePreconditionerOnly, mode, alpha, beta);
                if (itmp == 0)
                    beta = ScalarTraits<SC>::one();
                itmp++;
            }
        }
    } else {
        y.update(alpha, x, beta);
    }
}
}; // namespace FROSch

#endif
