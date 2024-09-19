#ifndef NONLINEARCOMBINEOPERATOR_DECL_HPP
#define NONLINEARCOMBINEOPERATOR_DECL_HPP
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearOperator_decl.hpp"
#include <FROSch_SchwarzOperator_def.hpp>
#include <Teuchos_ArrayRCPDecl.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Xpetra_Matrix.hpp>

#include "CombineOperator_decl.hpp"
/*!
 Declaration of NonlinearCombineOperator which extends the FROSch Combine operator to allow non-const apply() methods. This is
 necessary since the apply() methods of nonlinear operators perform nonlinear solves which requires changing member
 variables. This class does not implement the apply() method itself. This is left up to e.g. sumOperator or h1Operator

 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */
// NOTE: kho this class basically reimplements FROSch_CombineOperator. If FROSch used virtual inheritance this operator
// could use most of the existing code since casting between NonlinearOperator and SchwarzOperator would be possible
// (without using dynamic_cast)

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class NonLinearCombineOperator : public CombineOperator<SC, LO, GO, NO>, public NonLinearOperator<SC, LO, GO, NO> {

  protected:
    using SchwarzOperatorPtr = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtr;
    using SchwarzOperatorPtrVec = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtrVec;
    using SchwarzOperatorPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtrVecPtr;
    using NonLinearOperatorPtr = Teuchos::RCP<NonLinearOperator<SC, LO, GO, NO>>;
    using NonLinearOperatorPtrVec = Array<NonLinearOperatorPtr>;
    using NonLinearOperatorPtrVecPtr = ArrayRCP<NonLinearOperatorPtr>;

    using CommPtr = typename CombineOperator<SC, LO, GO, NO>::CommPtr;
    using BoolVec = typename CombineOperator<SC, LO, GO, NO>::BoolVec;
    using XMultiVector = typename CombineOperator<SC, LO, GO, NO>::XMultiVector;
    using UN = typename CombineOperator<SC, LO, GO, NO>::UN;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    NonLinearCombineOperator(CommPtr comm);

    ~NonLinearCombineOperator() = default;

    void apply(const XMultiVector &x, XMultiVector &y, SC alpha = ScalarTraits<SC>::one(),
               SC beta = ScalarTraits<SC>::zero()) override = 0;

    void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly,
               Teuchos::ETransp mode = Teuchos::NO_TRANS, SC alpha = ST::one(), SC beta = ST::zero()) const override;

    int addOperator(NonLinearOperatorPtr op);

    int addOperators(NonLinearOperatorPtrVecPtr operators);

  protected:
    NonLinearOperatorPtrVec NonLinearOperatorVector_ = NonLinearOperatorPtrVec(0);

    BoolVec EnableNonLinearOperators_ = BoolVec(0);
};
} // namespace FROSch

#endif
