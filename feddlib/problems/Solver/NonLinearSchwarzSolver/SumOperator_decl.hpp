#ifndef SUMOPERATOR_DECL_HPP
#define SUMOPERATOR_DECL_HPP

#include "SumOperator_decl.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/CombineOperator_decl.hpp"
#include <FROSch_SchwarzOperator_def.hpp>

/*!
 Declaration of SumOperator. This is a reimplementation of the FROSch::SumOperator that inherits from the
 CombineOperator interface. This enables using a single CombineOperator object which can use apply() in an additive,
 multiplicative etc. way.

 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */
namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class NewSumOperator : public CombineOperator<SC, LO, GO, NO> {

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
    NewSumOperator(CommPtr comm);

    NewSumOperator(SchwarzOperatorPtrVecPtr operators);

    ~NewSumOperator() = default;

    virtual void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly, ETransp mode = NO_TRANS,
                       SC alpha = ScalarTraits<SC>::one(), SC beta = ScalarTraits<SC>::zero()) const override;
};

} // namespace FROSch

#endif
