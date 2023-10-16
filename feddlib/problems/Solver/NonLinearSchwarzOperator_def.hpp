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
      xTmp_(), yTmp_(), newtonTol_(), maxNumIts_(), mapRepeatedMpiTmp_(), mapUniqueMpiTmp_(), elementMapMpiTmp_(),
      elementMapOverlappingMpiTmp_(), mapOverlappingMpiTmp_(), pointsRepTmp_(), pointsUniTmp_() {

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

    auto domainVec = this->problem_->getDomainVector();
    auto mesh = domainVec.at(0)->getMesh();

    // Build serial overlapping element and node maps
    auto elementMapOverlappingMPI = mesh->getElementMapOverlapping()->getXpetraMap();
    auto mapOverlappingMPI = mesh->getMapOverlapping()->getXpetraMap();
    auto elementMapOverlappingLocal = Xpetra::MapFactory<LO, GO, NO>::Build(
        elementMapOverlappingMPI->lib(), elementMapOverlappingMPI->getLocalNumElements(), 0, this->SerialComm_);
    auto mapOverlappingLocal = Xpetra::MapFactory<LO, GO, NO>::Build(
        mapOverlappingMPI->lib(), mapOverlappingMPI->getLocalNumElements(), 0, this->SerialComm_);

    // Store all distributed properties that need replacing for local computations
    this->mapRepeatedMpiTmp_ = mesh->getMapRepeated()->getXpetraMap();
    this->mapUniqueMpiTmp_ = mesh->getMapUnique()->getXpetraMap();
    this->pointsRepTmp_ = mesh->getPointsRepeated();
    this->pointsUniTmp_ = mesh->getPointsUnique();

    // Replace the following shared objects within this->problem_ to "trick" FEDD::Problem assembly routines to assemble
    // locally on the overlapping subdomain
    //   1. comm_ to SerialComm_ in problem, domainVec, Mesh
    //   2. mapRepeated_ and mapUnique_ to mapOverlappingLocal
    //   3. pointsRep to pointsOverlapping
    //   4. u_rep_ to u_overlapping_ in problemSpecific
    // No destinction between unique and repeated needs to be made since assembly is only local here.

    // 1. replace communicators
    this->problem_->comm_ = this->SerialComm_;
    domainVec.at(0)->setComm(this->SerialComm_);
    mesh->comm_ = this->SerialComm_;

    // 2. and 3. replace repeated and unique members
    mesh->replaceRepeatedMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingLocal)),
                                 mesh->pointsOverlapping_);
    mesh->replaceUniqueMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingLocal)), mesh->pointsOverlapping_);
    // Problems block vectors and matrices need to be reinitialized
    this->problem_->initializeProblem();

    // 4. rebuild problem->u_rep_ to use overlapping map
    this->problem_->reInitSpecificProblemVectors(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingLocal)));
    /* this->localJacobian_ = Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(); */
    /* this->localRHS_ = Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build(); */

    return 0;
}

template <class SC, class LO, class GO, class NO> int NonLinearSchwarzOperator<SC, LO, GO, NO>::compute() {

    TEUCHOS_TEST_FOR_EXCEPTION(this->x_.is_null(), std::runtime_error,
                               "The the current value of the nonlinear operator requires "
                               "an input to be computed.");

    bool verbose = this->problem_->getVerbose();
    double	gmresIts = 0.;
    double residual0 = 1.;
    double residual = 1.;
    double tol = this->problem_->getParameterList()->sublist("Parameter").get("relNonLinTol",1.0e-6);
    int nlIts=0;
    int maxNonLinIts = this->problem_->getParameterList()->sublist("Parameter").get("MaxNonLinIts",10);
    double criterionValue = 1.;
    std::string criterion = this->problem_->getParameterList()->sublist("Parameter").get("Criterion","Residual");

    while ( nlIts < maxNonLinIts ) {
        //this makes only sense for Navier-Stokes/Stokes, for other problems, e.g., non linear elasticity, it should do nothing.

        this->problem_->calculateNonLinResidualVec("reverse");

        if (criterion=="Residual")
            residual = this->problem_->calculateResidualNorm();

        this->problem_->assemble("Newton");

        this->problem_->setBoundariesSystem();

        if (nlIts==0)
            residual0 = residual;
        
        if (criterion=="Residual"){
            criterionValue = residual/residual0;
            if (verbose)
                cout << "### Newton iteration : " << nlIts << "  relative nonlinear residual : " << criterionValue << endl;
            if ( criterionValue < tol )
                break;
        }

        gmresIts += this->problem_->solveAndUpdate( criterion, criterionValue );
        nlIts++;
        if(criterion=="Update"){
            if (verbose)
                cout << "### Newton iteration : " << nlIts << "  residual of update : " << criterionValue << endl;
            if ( criterionValue < tol )
                break;
        }

        // ####### end FPI #######
    }

    gmresIts/=nlIts;
    if (verbose)
        cout << "### Total Newton iterations : " << nlIts << "  with average gmres its : " << gmresIts << endl;
    if ( this->problem_->getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts",false) ) {
        TEUCHOS_TEST_FOR_EXCEPTION(nlIts == maxNonLinIts ,std::runtime_error,"Maximum nonlinear Iterations reached. Problem might have converged in the last step. Still we cancel here.");
    }

    std::cout << "==> Current residual: " << residual << std::endl;

    // TODO KHo reset u_rep_ to repeated map
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
