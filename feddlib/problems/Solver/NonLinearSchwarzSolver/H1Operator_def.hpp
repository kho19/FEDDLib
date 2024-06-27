#ifndef H1OPERATOR_DEF_HPP
#define H1OPERATOR_DEF_HPP

#include "H1Operator_decl.hpp"
#include <FROSch_Output.h>

namespace FROSch {

template <class SC, class LO, class GO, class NO>
H1Operator<SC, LO, GO, NO>::H1Operator(CommPtr comm) : CombineOperator<SC, LO, GO, NO>(comm) {}

template <class SC, class LO, class GO, class NO>
H1Operator<SC, LO, GO, NO>::H1Operator(SchwarzOperatorPtrVecPtr operators)
    : CombineOperator<SC, LO, GO, NO>(operators[0]->getRangeMap()->getComm()) {}

//  This apply method with the option usePreconditionerOnly is due to the inheritance from SchwarzOperator. The option
//  may be used in alternative hybrid type methods but is not needed here.
template <class SC, class LO, class GO, class NO>
void H1Operator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly, ETransp mode,
                                       SC alpha, SC beta) const {
    FROSCH_TIMER_START_LEVELID(applyTime, "H1Operator::apply");
    FROSCH_ASSERT(this->OperatorVector_.size() == 2, "H1 operator can only be applied with two levels")

    auto one = ScalarTraits<SC>::one();
    auto zero = ScalarTraits<SC>::zero();
    if (this->XTmp_.is_null())
        this->XTmp_ = MultiVectorFactory<SC, LO, GO, NO>::Build(x.getMap(), x.getNumVectors());
    if (z0_.is_null())
        z0_ = MultiVectorFactory<SC, LO, GO, NO>::Build(y.getMap(), y.getNumVectors());
    if (z1_.is_null())
        z1_ = MultiVectorFactory<SC, LO, GO, NO>::Build(y.getMap(), y.getNumVectors());

    // In case x and y reference the same variable
    *this->XTmp_ = x;
    // This operator applies Q_0 + sum Q_i(I-Q_0)
    // Build z0 = Q_0x
    // y = alpha*Op(x) + beta*y
    this->OperatorVector_[1]->apply(*this->XTmp_, *z0_, mode, one, zero);
    // Build z1 = (I-Q_0)x
    this->z1_->update(one, *this->XTmp_, zero);
    this->z1_->update(-one, *z0_, one);
    // Build sum Q_i(I-Q_0)
    this->OperatorVector_[0]->apply(*z1_, *this->XTmp_, mode, one, zero);
    // Form Q_0 + sum Q_i(I-Q_0)
    y.update(one, *this->XTmp_, zero);
    y.update(one, *z0_, one);
}
} // namespace FROSch

#endif
