#ifndef ASPENOVERLAPPINGOPERATPR_DEF_HPP
#define ASPENOVERLAPPINGOPERATPR_DEF_HPP

#include "ASPENOverlappingOperator_decl.hpp"
#include <Teuchos_RCPDecl.hpp>
#include <Xpetra_Map_decl.hpp>
#include <stdexcept>

namespace FROSch {

template <class SC, class LO, class GO, class NO>
ASPENOverlappingOperator<SC, LO, GO, NO>::ASPENOverlappingOperator(ConstXMatrixPtr k, ParameterListPtr parameterList)
    : OverlappingOperator<SC, LO, GO, NO>(k, parameterList) {}

template <class SC, class LO, class GO, class NO>
int ASPENOverlappingOperator<SC, LO, GO, NO>::initialize(CommPtr serialComm, ConstXMatrixPtr localJacobian,
                                                         ConstXMapPtr overlappingMap) {
    // AlgebraicOverlappingOperator does: calculates overlap multiplicity if needed and does symbolic extraction
    // of local subdomain matrix and initialization of solver (symbolic factorization)
    // Here we just read in the localSubdomainMatrix since it already exists
    // TODO: Need to set the following as AlgebraicOverlappingOperator initialize()
    // x OverlappingMap_ (global)
    // x OverlappingMatrix_ (global and local)
    // x Multiplicity_
    // x SubdomainSolver_
    // x IsInitialzed_
    // x IsComputed_
    // - subdomainMatrix_ (NO: only used as an intermediate overlapping matrix from which values are taken for
    // localSubdomainMatrix_)
    // - localSubdomainMatrix_ (NO: this is the non-const predecessor of OverlappingMatrix_ used for extraction and
    // factorization in AlgebraicOverlappingOperator)
    // - ExtractLocalSubdomainMatrix_Symbolic_Done flag (NO: should not need this since not doing any symbolic
    // extraction)

    // Need to pass the serial communicator on which localJacobian lives
    this->SerialComm_ = serialComm;
    this->OverlappingMatrix_ = localJacobian;
    this->OverlappingMap_ = overlappingMap;

    //  Calculate overlap multiplicity if needed
    this->initializeOverlappingOperator();
    // Compute symbolic factorization
    this->initializeSubdomainSolver(this->OverlappingMatrix_);
    this->IsInitialized_ = true;
    this->IsComputed_ = false;
    return 0;
}

template <class SC, class LO, class GO, class NO> int ASPENOverlappingOperator<SC, LO, GO, NO>::compute() {
    // AlgebraicOverlappingOperator does: gets values of the local subdomain matrices and computes the numerical
    // factorization
    // Here we do not need to fill values into the sparsity pattern since they are already there
    // ==> updateLocalOverlappingMatrices() is a no-op
    TEUCHOS_TEST_FOR_EXCEPTION(!this->IsInitialized_, std::runtime_error,
                               "ASPENOverlappingOperator must be initialized before calling compute()");
    this->computeOverlappingOperator();
    return 0;
}

template <class SC, class LO, class GO, class NO>
void ASPENOverlappingOperator<SC, LO, GO, NO>::describe(FancyOStream &out, const EVerbosityLevel verbLevel) const {
    TEUCHOS_TEST_FOR_EXCEPTION(false, std::runtime_error, "describe() has to be implemented properly...");
}

template <class SC, class LO, class GO, class NO>
std::string ASPENOverlappingOperator<SC, LO, GO, NO>::description() const {
    return "ASPEN Overlapping Operator";
}

} // namespace FROSch

#endif // ASPENOVERLAPPINGOPERATPR_DEF_HPP
