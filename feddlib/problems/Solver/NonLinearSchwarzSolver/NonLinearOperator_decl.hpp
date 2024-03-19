#ifndef NONLINEAROPERATOR_DECL_hpp
#define NONLINEAROPERATOR_DECL_hpp

#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include <FROSch_SchwarzOperator_def.hpp>

/*!
 Declration of a nonlinear operator

 @brief  Nonlinear Operator
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {
template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class NonLinearOperator {

  private:
    using XMultiVector = Xpetra::MultiVector<SC, LO, GO, NO>;
    using BlockMultiVectorPtrFEDD = typename Teuchos::RCP<FEDD::BlockMultiVector<SC, LO, GO, NO>>;
    using MapConstPtrFEDD = typename Teuchos::RCP<const FEDD::Map<LO, GO, NO>>;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    NonLinearOperator() = default;
    ~NonLinearOperator() = default;

    // non-const apply() interface which nonlinear operators should implement since they cannot be applied without
    // performing internal calculations
    virtual void apply(const XMultiVector &x, XMultiVector &y, SC alpha = ST::one(), SC beta = ST::zero()) = 0;
};
} // namespace FROSch
#endif
