#ifndef NONLINEARSUMOPERATOR_DECL_HPP
#define NONLINEARSUMOPERATOR_DECL_HPP
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearOperator_decl.hpp"
#include <FROSch_SchwarzOperator_def.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Xpetra_Matrix.hpp>

#include <FROSch_SumOperator_decl.hpp>
/*!
 Declaration of NonlinearSumOperator which extends the FROSch sum operator to allow non-const apply() methods

 @brief Implements the coarse correction T_0 from the nonlinear Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class NonLinearSumOperator : public SumOperator<SC, LO, GO, NO>, public NonLinearOperator<SC, LO, GO, NO> {

  protected:
    using SchwarzOperatorPtr = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtr;
    using SchwarzOperatorPtrVec = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtrVec;
    using SchwarzOperatorPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::SchwarzOperatorPtrVecPtr;

    using CommPtr = typename SumOperator<SC, LO, GO, NO>::CommPtr;
    using XMultiVector = typename SumOperator<SC, LO, GO, NO>::XMultiVector;
    using UN = typename SumOperator<SC, LO, GO, NO>::UN;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    NonLinearSumOperator(CommPtr comm);

    ~NonLinearSumOperator() = default;

    void apply(const XMultiVector &x, XMultiVector &y, SC alpha = ScalarTraits<SC>::one(),
               SC beta = ScalarTraits<SC>::zero()) override;

    void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly,
               Teuchos::ETransp mode = Teuchos::NO_TRANS, SC alpha = ST::one(), SC beta = ST::zero()) const override;

    int addOperator(SchwarzOperatorPtr op);

    int addOperators(SchwarzOperatorPtrVecPtr operators);
};
} // namespace FROSch

#endif
