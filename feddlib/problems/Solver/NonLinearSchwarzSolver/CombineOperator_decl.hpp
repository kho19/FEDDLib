#ifndef COMBINEOPERATOR_DECL_HPP
#define COMBINEOPERATOR_DECL_HPP

#include "CombineOperator_decl.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include <FROSch_SchwarzOperator_def.hpp>

/*!
 Declaration of CombineOperator. This is an interface operator which is specialized into e.g. SumOperator,
 MultiplicativeOperator etc.

 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */
namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class CombineOperator : public SchwarzOperator<SC, LO, GO, NO> {

  protected:
    using CommPtr = typename SchwarzOperator<SC, LO, GO, NO>::CommPtr;

    using XMapPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMapPtr;
    using ConstXMapPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMapPtr;

    using XMultiVector = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVector;
    using XMultiVectorPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVectorPtr;

    using SchwarzOperatorPtr = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtr;
    using SchwarzOperatorPtrVec = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtrVec;
    using SchwarzOperatorPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtrVecPtr;

    using UN = typename SchwarzOperator<SC, LO, GO, NO>::UN;

    using BoolVec = typename SchwarzOperator<SC, LO, GO, NO>::BoolVec;

  public:
    CombineOperator(CommPtr comm);

    CombineOperator(SchwarzOperatorPtrVecPtr operators);

    ~CombineOperator() = default;

    virtual int initialize();

    virtual int initialize(ConstXMapPtr repeatedMap);

    virtual int compute();

    virtual void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly, ETransp mode = NO_TRANS,
                       SC alpha = ScalarTraits<SC>::one(), SC beta = ScalarTraits<SC>::zero()) const = 0;

    virtual ConstXMapPtr getDomainMap() const;

    virtual ConstXMapPtr getRangeMap() const;

    virtual void describe(FancyOStream &out, const EVerbosityLevel verbLevel = Describable::verbLevel_default) const;

    virtual string description() const;

    int addOperator(SchwarzOperatorPtr op);

    int addOperators(SchwarzOperatorPtrVecPtr operators);

    int resetOperator(UN iD, SchwarzOperatorPtr op);

    int enableOperator(UN iD, bool enable);

    UN getNumOperators();

  protected:
    SchwarzOperatorPtrVec OperatorVector_ = SchwarzOperatorPtrVec(0);

    // Temp Vectors for apply()
    mutable XMultiVectorPtr XTmp_;

    BoolVec EnableOperators_ = BoolVec(0);
};

} // namespace FROSch

#endif
