#ifndef NONLINEARSCHWARZOPERATOR_DEF_HPP
#define NONLINEARSCHWARZOPERATOR_DEF_HPP

#include "NonLinearSchwarzOperator_decl.hpp"
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MultiVectorFactory_decl.hpp>
#include <stdexcept>

/*!
 Implementation of NonLinearSchwarzOperator

 @brief Implements the surrogate problem $\mathcal{F}(u)$ from the nonlinear Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC, class LO, class GO, class NO>
NonLinearSchwarzOperator<SC, LO, GO, NO>::NonLinearSchwarzOperator(ConstXMatrixPtr k, ParameterListPtr parameterList)
    : SchwarzOperator<SC, LO, GO, NO>(k, parameterList) {}

template <class SC, class LO, class GO, class NO>
NonLinearSchwarzOperator<SC, LO, GO, NO>::~NonLinearSchwarzOperator() {}

template <class SC, class LO, class GO, class NO>
int NonLinearSchwarzOperator<SC, LO, GO, NO>::initialize(int overlap) {

    // TODO build all necessary maps and data structures for storing results of and carrying out the local Newtons
    // method
    // TODO increase overlap of the dual graph by the specified number of layers
    return 0;
}

template <class SC, class LO, class GO, class NO> int NonLinearSchwarzOperator<SC, LO, GO, NO>::compute() {

    TEUCHOS_TEST_FOR_EXCEPTION(this->x_.is_null(), std::runtime_error,
                               "The the current value of the nonlinear operator requires an input to be computed.");

    int numIters = 0;
    // TODO build the residual from the current solution
    int residual = 1;
    // TODO build localJacobian and localRHS on a serial communicator
    // this is how it is done in FROSch. Same here?
    RCP<const Comm<LO>> SerialComm = rcp(new MpiComm<LO>(MPI_COMM_SELF));
    auto localJacobian = Xpetra::MatrixFactory<SC, LO, GO, NO>::Build();
    auto localRHS = Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build();
    // TODO initialize the solution. Should this be distributed the same as the Jacobian and RHS only not locally? Or
    // actually locally while computing and then replaceMap to the distributed communicator?
    while (residual > this->newtonTol_ && numIters < this->maxNumIts_) {
        numIters++;

        // TODO iterate over all local elements (via dual graph) to assemble local rhs and Jacobian.
    }
    return 0;
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly,
                                                     ETransp mode, SC alpha, SC beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(mode != NO_TRANS, std::runtime_error,
                               "The transpose of a nonlinear operator does not exist.");
    TEUCHOS_TEST_FOR_EXCEPTION(usePreconditionerOnly != false, std::runtime_error,
                               "Nonlinear Schwarz operator cannot be used as a preconditioner.");

    // Save the current input
    this->x_.reset();
    this->x_ = x;

    // Compute the value of the nonlinear operator
    this->compute();

    // y = alpha*f(x) + beta*y
    y.update(alpha, *x_, beta);
}

} // namespace FROSch

#endif
