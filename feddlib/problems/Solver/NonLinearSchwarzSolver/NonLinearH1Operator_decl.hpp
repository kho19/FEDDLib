#ifndef NONLINEARH1OPERATOR_DECL_HPP
#define NONLINEARH1OPERATOR_DECL_HPP
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearCombineOperator_decl.hpp"
#include <FROSch_SchwarzOperator_def.hpp>

/*!
 Declaration of NonLinearH1Operator. This implements the nonlinear Schwarz operator in the hybrid-1 fashion as
introduced in "Additive and hybrid nonlinear two-level Schwarz methods and energy minimizing coarse spaces for
unstructured grids"

 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class NonLinearH1Operator : public NonLinearCombineOperator<SC, LO, GO, NO> {

  protected:
    using SchwarzOperatorPtr = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtr;
    using SchwarzOperatorPtrVec = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtrVec;
    using SchwarzOperatorPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtrVecPtr;
    using NonLinearOperatorPtr = Teuchos::RCP<NonLinearOperator<SC, LO, GO, NO>>;
    using NonLinearOperatorPtrVec = Array<NonLinearOperatorPtr>;
    using NonLinearOperatorPtrVecPtr = ArrayRCP<NonLinearOperatorPtr>;

    using CommPtr = typename NonLinearCombineOperator<SC, LO, GO, NO>::CommPtr;
    using BoolVec = typename NonLinearCombineOperator<SC, LO, GO, NO>::BoolVec;
    using XMultiVector = typename NonLinearCombineOperator<SC, LO, GO, NO>::XMultiVector;
    using XMultiVectorPtr = typename NonLinearCombineOperator<SC, LO, GO, NO>::XMultiVectorPtr;
    using UN = typename NonLinearCombineOperator<SC, LO, GO, NO>::UN;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    NonLinearH1Operator(CommPtr comm);

    ~NonLinearH1Operator() = default;

    void apply(const XMultiVector &x, XMultiVector &y, SC alpha = ScalarTraits<SC>::one(),
               SC beta = ScalarTraits<SC>::zero()) override;

  protected:
    mutable XMultiVectorPtr YTmp_;
    mutable XMultiVectorPtr gTmp_;
};
} // namespace FROSch

#endif
