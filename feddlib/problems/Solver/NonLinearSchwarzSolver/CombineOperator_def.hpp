#ifndef COMBINEOPERATOR_DEF_HPP
#define COMBINEOPERATOR_DEF_HPP

#include "CombineOperator_decl.hpp"

namespace FROSch {

template <class SC, class LO, class GO, class NO>
CombineOperator<SC, LO, GO, NO>::CombineOperator(CommPtr comm) : SchwarzOperator<SC, LO, GO, NO>(comm) {
    FROSCH_DETAILTIMER_START_LEVELID(sumOperatorTime, "CombineOperator::CombineOperator");
}

template <class SC, class LO, class GO, class NO>
CombineOperator<SC, LO, GO, NO>::CombineOperator(SchwarzOperatorPtrVecPtr operators)
    : SchwarzOperator<SC, LO, GO, NO>(operators[0]->getRangeMap()->getComm()) {
    FROSCH_DETAILTIMER_START_LEVELID(sumOperatorTime, "CombineOperator::CombineOperator");
    FROSCH_ASSERT(operators.size() > 0, "operators.size()<=0");
    OperatorVector_.push_back(operators[0]);
    for (unsigned i = 1; i < operators.size(); i++) {
        FROSCH_ASSERT(operators[i]->getDomainMap()->isSameAs(*OperatorVector_[0]->getDomainMap()),
                      "The DomainMaps of the operators are not identical.");
        FROSCH_ASSERT(operators[i]->getRangeMap()->isSameAs(*OperatorVector_[0]->getRangeMap()),
                      "The RangeMaps of the operators are not identical.");

        OperatorVector_.push_back(operators[i]);
        EnableOperators_.push_back(true);
    }
}

template <class SC, class LO, class GO, class NO> int CombineOperator<SC, LO, GO, NO>::initialize() {
    if (this->Verbose_) {
        FROSCH_ASSERT(false, "ERROR: Each of the Operators has to be initialized manually.");
    }
    return 0;
}

template <class SC, class LO, class GO, class NO>
int CombineOperator<SC, LO, GO, NO>::initialize(ConstXMapPtr repeatedMap) {
    if (this->Verbose_) {
        FROSCH_ASSERT(false, "ERROR: Each of the Operators has to be initialized manually.");
    }
    return 0;
}

template <class SC, class LO, class GO, class NO> int CombineOperator<SC, LO, GO, NO>::compute() {
    if (this->Verbose_) {
        FROSCH_ASSERT(false, "ERROR: Each of the Operators has to be computed manually.");
    }
    return 0;
}

template <class SC, class LO, class GO, class NO>
const typename CombineOperator<SC, LO, GO, NO>::ConstXMapPtr CombineOperator<SC, LO, GO, NO>::getDomainMap() const {
    return OperatorVector_[0]->getDomainMap();
}

template <class SC, class LO, class GO, class NO>
const typename CombineOperator<SC, LO, GO, NO>::ConstXMapPtr CombineOperator<SC, LO, GO, NO>::getRangeMap() const {
    return OperatorVector_[0]->getRangeMap();
}

template <class SC, class LO, class GO, class NO>
void CombineOperator<SC, LO, GO, NO>::describe(FancyOStream &out, const EVerbosityLevel verbLevel) const {
    FROSCH_ASSERT(false, "describe() has to be implemented properly...");
}

template <class SC, class LO, class GO, class NO> string CombineOperator<SC, LO, GO, NO>::description() const {
    string labelString = "Combine operator: ";

    for (UN i = 0; i < OperatorVector_.size(); i++) {
        labelString += OperatorVector_[i]->description();
        if (i < OperatorVector_.size() - 1) {
            labelString += ",";
        }
    }
    return labelString;
}

template <class SC, class LO, class GO, class NO>
int CombineOperator<SC, LO, GO, NO>::addOperator(SchwarzOperatorPtr op) {
    FROSCH_DETAILTIMER_START_LEVELID(addOperatorTime, "CombineOperator::addOperator");
    int ret = 0;
    if (OperatorVector_.size() > 0) {
        if (!op->getDomainMap()->isSameAs(*OperatorVector_[0]->getDomainMap())) {
            if (this->Verbose_)
                cerr << "CombineOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr "
                        "op)\t\t!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())\n";
            ret -= 1;
        }
        if (!op->getRangeMap()->isSameAs(*OperatorVector_[0]->getRangeMap())) {
            if (this->Verbose_)
                cerr << "CombineOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr "
                        "op)\t\t!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())\n";
            ret -= 10;
        }
    }
    OperatorVector_.push_back(op);
    EnableOperators_.push_back(true);
    return ret;
}

template <class SC, class LO, class GO, class NO>
int CombineOperator<SC, LO, GO, NO>::addOperators(SchwarzOperatorPtrVecPtr operators) {
    FROSCH_DETAILTIMER_START_LEVELID(addOperatorsTime, "CombineOperator::addOperators");
    int ret = 0;
    for (UN i = 1; i < operators.size(); i++) {
        if (0 > addOperator(operators[i]))
            ret -= pow(10, i);
    }
    return ret;
}

template <class SC, class LO, class GO, class NO>
int CombineOperator<SC, LO, GO, NO>::resetOperator(UN iD, SchwarzOperatorPtr op) {
    FROSCH_DETAILTIMER_START_LEVELID(resetOperatorTime, "CombineOperator::resetOperator");
    FROSCH_ASSERT(iD < OperatorVector_.size(), "iD exceeds the length of the OperatorVector_");
    int ret = 0;
    if (!op->getDomainMap()->isSameAs(*OperatorVector_[0]->getDomainMap())) {
        if (this->Verbose_)
            cerr << "CombineOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr "
                    "op)\t\t!op->getDomainMap().isSameAs(OperatorVector_[0]->getDomainMap())\n";
        ret -= 1;
    }
    if (!op->getRangeMap()->isSameAs(*OperatorVector_[0]->getRangeMap())) {
        if (this->Verbose_)
            cerr << "CombineOperator<SC,LO,GO,NO>::addOperator(SchwarzOperatorPtr "
                    "op)\t\t!op->getRangeMap().isSameAs(OperatorVector_[0]->getRangeMap())\n";
        ret -= 10;
    }
    OperatorVector_[iD] = op;
    return ret;
}

template <class SC, class LO, class GO, class NO>
int CombineOperator<SC, LO, GO, NO>::enableOperator(UN iD, bool enable) {
    FROSCH_DETAILTIMER_START_LEVELID(enableOperatorTime, "CombineOperator::enableOperatorOperator");
    EnableOperators_[iD] = enable;
    return 0;
}

template <class SC, class LO, class GO, class NO>
typename CombineOperator<SC, LO, GO, NO>::UN CombineOperator<SC, LO, GO, NO>::getNumOperators() {
    return OperatorVector_.size();
}
} // namespace FROSch

#endif
