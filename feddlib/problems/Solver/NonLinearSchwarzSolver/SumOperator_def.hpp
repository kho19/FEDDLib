#ifndef SUMOPERATOR_DEF_HPP
#define SUMOPERATOR_DEF_HPP

#include "SumOperator_decl.hpp"

namespace FROSch {

template <class SC, class LO, class GO, class NO>
NewSumOperator<SC, LO, GO, NO>::NewSumOperator(CommPtr comm) : CombineOperator<SC, LO, GO, NO>(comm) {}

template <class SC, class LO, class GO, class NO>
NewSumOperator<SC, LO, GO, NO>::NewSumOperator(SchwarzOperatorPtrVecPtr operators)
    : CombineOperator<SC, LO, GO, NO>(operators[0]->getRangeMap()->getComm()) {}

// Y = alpha * A^mode * X + beta * Y
template <class SC, class LO, class GO, class NO>
void NewSumOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly,
                                           ETransp mode, SC alpha, SC beta) const {
    FROSCH_TIMER_START_LEVELID(applyTime, "SumOperator::apply");
    if (this->OperatorVector_.size() > 0) {
        if (this->XTmp_.is_null())
            this->XTmp_ = MultiVectorFactory<SC, LO, GO, NO>::Build(x.getMap(), x.getNumVectors());
        *this->XTmp_ = x;
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
} // namespace FROSch

#endif
