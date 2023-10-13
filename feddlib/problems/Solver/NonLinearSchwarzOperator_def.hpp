#ifndef NONLINEARSCHWARZOPERATOR_DEF_HPP
#define NONLINEARSCHWARZOPERATOR_DEF_HPP

#include "NonLinearSchwarzOperator_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector_decl.hpp"
#include <Tacho_Driver.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MapFactory_decl.hpp>
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

// NOTE: KHo Working with a FEDDLib mesh object for now. This has a partitioned dual graph and corresponding maps. When
// making this solver more portable a mesh format will have to be decided on and METIS partitioning of the mesh and dual
// graph construction integrated. Could also make FROSch constructor more fitting.
// NOTE: KHo Passing in a FEDD::Problem for now. This could be made more general by accepting a NOX::Thyra::Group
// TODO: KHo Implementing for problems with a single domain to get a version running. This is going to have to be
// changed. Need to think about where to deal with multiple domains. Depends on how the nonlinear method deals with
// them.
// TODO: KHo all todos that are not for immediate development are lowercase.
namespace FROSch {
template <class SC, class LO, class GO, class NO>
NonLinearSchwarzOperator<SC, LO, GO, NO>::NonLinearSchwarzOperator(CommPtr mpiComm, ParameterListPtr parameterList,
                                                                   NonLinearProblemPtr problem)
    : SchwarzOperator<SC, LO, GO, NO>(mpiComm), problem_(problem), dualGraph_(), localJacobian_(), localRHS_(), fX_(),
      xTmp_(), yTmp_(), newtonTol_(), maxNumIts_(), mapRepeatedMPI_(), mapUniqueMPI_(), elementMapMPI_(),
      elementMapOverlappingMPI_(), mapOverlappingMPI_() {

    // Ensure that the mesh object has been initialized and a dual graph generated
    auto domainPtr_vec = this->problem_->getDomainVector();
    for (auto &domainPtr : domainPtr_vec) {
        TEUCHOS_ASSERT(!domainPtr->getElementMap().is_null());
        TEUCHOS_ASSERT(!domainPtr->getMapRepeated().is_null());
        TEUCHOS_ASSERT(!domainPtr->getDualGraph().is_null());
    }
    // todo KHo to avoid making a new constructor for SchwarzOperator, which would require editing Trilinos source code,
    // SchwarzOperator protected variables are set here. Should be avoided!
    this->ParameterList_ = parameterList;

    // Initialize members that cannot be null after construction
    this->x_ =
        Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build(domainPtr_vec.at(0)->getMapRepeated()->getXpetraMap(), 1);
}

template <class SC, class LO, class GO, class NO>
NonLinearSchwarzOperator<SC, LO, GO, NO>::~NonLinearSchwarzOperator() {}

// NOTE KHo increase overlap of the dual graph by the specified number of layers. Not doing this here for now since
// MeshPartitioner already does.
template <class SC, class LO, class GO, class NO>
int NonLinearSchwarzOperator<SC, LO, GO, NO>::initialize(int overlap) {

    // Need to replace the following:
    // - comm_ to SerialComm_ in problem, domainVec, Mesh ==> DONE
    // - u_rep_ to u_overlapping_ in problemSpecific ==> DONE
    // - mapRepeated_ and mapUnique_ to mapOverlappingLocal ==> DONE
    // - pointsRep to pointsOverlapping ==> DONE

    auto domainVec = this->problem_->getDomainVector();
    auto mesh = domainVec.at(0)->getMesh();

    // Store all distributed properties that need replacing for local computations
    this->mapRepeatedMPI_ = mesh->getMapRepeated()->getXpetraMap();
    this->mapUniqueMPI_ = mesh->getMapUnique()->getXpetraMap();
    /* this->elementMapMPI_ = mesh->getElementMap()->getXpetraMap(); */
    this->elementMapOverlappingMPI_ = mesh->getElementMapOverlapping()->getXpetraMap();
    this->mapOverlappingMPI_ = mesh->getMapOverlapping()->getXpetraMap();

    // Replace communicators
    this->problem_->comm_ = this->SerialComm_;
    domainVec.at(0)->setComm(this->SerialComm_);
    mesh->comm_ = this->SerialComm_;

    // Build serial overlapping element and node maps
    // todo KHo Are repeated and unique maps needed?
    auto elementMapOverlappingLocal = Xpetra::MapFactory<LO, GO, NO>::Build(
        this->elementMapOverlappingMPI_->lib(), this->elementMapOverlappingMPI_->getLocalNumElements(), 0,
        this->SerialComm_);
    auto mapOverlappingLocal = Xpetra::MapFactory<LO, GO, NO>::Build(
        this->mapOverlappingMPI_->lib(), this->mapOverlappingMPI_->getLocalNumElements(), 0, this->SerialComm_);

    // Replace repeated and unique members to "trick" FEDD::Problem assembly routines to assemble locally on the
    // overlapping subdomain. No destinction between unique and repeated needs to be made since assembly is only local
    // here.
    mesh->replaceRepeatedMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingLocal)),
                                 mesh->pointsOverlapping_);
    mesh->replaceUniqueMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingLocal)), mesh->pointsOverlapping_);

    // Problems block vectors and matrices need to be reinitialized
    this->problem_->initializeProblem();

    // rebuild problem->u_rep_ to use overlapping map
    this->problem_->reInitSpecificProblemVectors(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingLocal)));
    /* this->localJacobian_ = Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(); */
    /* this->localRHS_ = Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build(); */

    return 0;
}

template <class SC, class LO, class GO, class NO> int NonLinearSchwarzOperator<SC, LO, GO, NO>::compute() {

    TEUCHOS_TEST_FOR_EXCEPTION(this->x_.is_null(), std::runtime_error,
                               "The the current value of the nonlinear operator requires an input to be computed.");

    int numIters = 0;
    this->problem_->calculateNonLinResidualVec("reverse", 0.);
    auto residual = this->problem_->calculateResidualNorm();
    std::cout << "==> Current residual is " << residual << std::endl;
    // TODO KHo initialize the solution. Should this be distributed the same as the Jacobian and RHS only not locally?
    // Or actually locally while computing and then replaceMap to the distributed communicator?
    // TODO maybe I can use the nonlinear solver built into the FEDDLib here?
    /* while (residual > this->newtonTol_ && numIters < this->maxNumIts_) { */
    /*     numIters++; */
    /**/
    /*     // TODO KHo iterate over all local elements (via dual graph) to assemble local rhs and Jacobian. */
    /* } */
    return 0;
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, SC alpha, SC beta) {

    // Save the current input
    auto importer = Xpetra::ImportFactory<LO, GO>::Build(x.getMap(), this->x_->getMap());
    this->x_->doImport(x, *importer, Xpetra::ADD);
    // Compute the value of the nonlinear operator
    this->compute();
    /* // y = alpha*f(x) + beta*y */
    /* y.update(alpha, *x_, beta); */
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly,
                                                     ETransp mode, SC alpha, SC beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(false, std::runtime_error,
                               "This version of apply does not make sense in the context of nonlinear operator.");
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::describe(FancyOStream &out, const EVerbosityLevel verbLevel) const {
    TEUCHOS_TEST_FOR_EXCEPTION(false, std::runtime_error, "describe() has to be implemented properly...");
}

template <class SC, class LO, class GO, class NO> string NonLinearSchwarzOperator<SC, LO, GO, NO>::description() const {
    return "Nonlinear Schwarz Operator";
}

} // namespace FROSch

#endif
