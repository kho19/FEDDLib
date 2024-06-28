#ifndef NONLINEARH1OPERATOR_DEF_HPP
#define NONLINEARH1OPERATOR_DEF_HPP

#include "NonLinearCombineOperator_decl.hpp"
#include "NonLinearH1Operator_decl.hpp"
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_RCPDecl.hpp>
/*!
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC, class LO, class GO, class NO>
NonLinearH1Operator<SC, LO, GO, NO>::NonLinearH1Operator(CommPtr comm)
    : NonLinearCombineOperator<SC, LO, GO, NO>(comm) {}

// Y = alpha * A^mode * X + beta * Y
template <class SC, class LO, class GO, class NO>
void NonLinearH1Operator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, SC alpha, SC beta) {
    FROSCH_TIMER_START_LEVELID(applyTime, "H1Operator::apply");
    FROSCH_ASSERT(this->NonLinearOperatorVector_.size() == 2, "H1 operator can only be applied with two levels")

    // We do not explicitly check if the operators have been activated here as is done e.g. in the SumOperator
    // We can be sure that all operators are of type NonLinearOperator since this is checked when adding
    // We need to dynamic_cast here anyway because NonLinearOperator and SchwarzOperator are not related
    // This could be changed by modifying FROSch to allow virtual inheritance
    auto one = ScalarTraits<SC>::one();
    auto zero = ScalarTraits<SC>::zero();
    if (this->XTmp_.is_null())
        this->XTmp_ = MultiVectorFactory<SC, LO, GO, NO>::Build(x.getMap(), x.getNumVectors());
    if (this->YTmp_.is_null())
        this->YTmp_ = MultiVectorFactory<SC, LO, GO, NO>::Build(y.getMap(), y.getNumVectors());
    if (this->gTmp_.is_null())
        this->gTmp_ = MultiVectorFactory<SC, LO, GO, NO>::Build(y.getMap(), y.getNumVectors());

    *this->XTmp_ = x;

    rcp_dynamic_cast<NonLinearOperator<SC, LO, GO, NO>>(this->NonLinearOperatorVector_[1])
        ->apply(*this->XTmp_, *this->gTmp_, one, zero);
    // Set YTmp_ = u - g0
    this->YTmp_->update(one, *this->XTmp_, zero);
    this->YTmp_->update(-one, *this->gTmp_, one);
    // Get overlapping correction g evaluated at u - g0
    // This will build the local Jacobians at v_i = u_0 - P_iT_i(u_0)
    rcp_dynamic_cast<NonLinearOperator<SC, LO, GO, NO>>(this->NonLinearOperatorVector_[0])
        ->apply(*this->YTmp_, *this->XTmp_, one, zero);
    // Add the coarse correction to the overlapping correction g(u-g0) + g0
    y.update(alpha, *this->gTmp_, beta);
    y.update(one, *this->XTmp_, one);
}

}; // namespace FROSch

#endif
