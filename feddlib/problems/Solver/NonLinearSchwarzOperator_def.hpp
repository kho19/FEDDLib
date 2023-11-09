#ifndef NONLINEARSCHWARZOPERATOR_DEF_HPP
#define NONLINEARSCHWARZOPERATOR_DEF_HPP

#include "NonLinearSchwarzOperator_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector_decl.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
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
    // Store all distributed properties that need replacing for local computations
    this->mapRepeatedMpiTmp_ = mesh->getMapRepeated()->getXpetraMap();
    this->mapUniqueMpiTmp_ = mesh->getMapUnique()->getXpetraMap();
    this->pointsRepTmp_ = mesh->getPointsRepeated();
    this->pointsUniTmp_ = mesh->getPointsUnique();
    this->bcFlagRepTmp_ = mesh->getBCFlagRepeated();
    this->bcFlagUniTmp_ = mesh->getBCFlagUnique();
    this->elementsCTmp_ = mesh->getElementsC();

    return 0;
}

template <class SC, class LO, class GO, class NO> int NonLinearSchwarzOperator<SC, LO, GO, NO>::compute() {

    TEUCHOS_TEST_FOR_EXCEPTION(this->x_.is_null(), std::runtime_error,
                               "The the current value of the nonlinear operator requires "
                               "an input to be computed.");

    auto domainVec = this->problem_->getDomainVector();
    auto mesh = domainVec.at(0)->getMesh();

    // ================= Replace shared objects ===============================
    // Done to "trick" FEDD::Problem assembly routines to assemble locally on the overlapping subdomain
    //    1. comm_ to SerialComm_ in problem, domainVec, Mesh
    //    2. mapRepeated_ and mapUnique_ to mapOverlappingLocal
    //    3. pointsRep to pointsOverlapping
    //    4. bcFlagRep_ and bcFlagUni_ to bcFlagOverlapping_
    //    5. elementsC_ to elementsOverlapping_
    //    6. u_rep_ to u_overlapping_ in problemSpecific
    //  No destinction between unique and repeated needs to be made since assembly is only local here.

    // 1. replace communicators
    this->problem_->comm_ = this->SerialComm_;
    domainVec.at(0)->setComm(this->SerialComm_);
    mesh->comm_ = this->SerialComm_;

    // 2., 3., 4. and 5. replace repeated and unique members
    auto mapOverlappingMPI = mesh->getMapOverlapping()->getXpetraMap();
    auto mapOverlappingLocal = Xpetra::MapFactory<LO, GO, NO>::Build(
        mapOverlappingMPI->lib(), mapOverlappingMPI->getLocalNumElements(), 0, this->SerialComm_);
    mesh->replaceRepeatedMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingLocal)), mesh->pointsOverlapping_,
                                 mesh->bcFlagOverlapping_);
    mesh->replaceUniqueMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingLocal)), mesh->pointsOverlapping_,
                               mesh->bcFlagOverlapping_);
    mesh->setElementsC(mesh->elementsOverlapping_);

    // Problems block vectors and matrices need to be reinitialized
    this->problem_->initializeProblem();

    // 6. rebuild problem->u_rep_ to use overlapping map
    this->problem_->reInitSpecificProblemVectors(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingLocal)));

    // Solve local nonlinear problems
    // TODO KHo need to set the initial value here to reflect the current linearisation point
    // TODO KHo change this to be a nox solver? Don't want FEDDLib nonlinear solver to remove depedency on the FEDDLib.
    bool verbose = this->problem_->getVerbose();
    double gmresIts = 0.;
    double residual0 = 1.;
    double residual = 1.;
    int nlIts = 0;
    double criterionValue = 1.;
    auto tol = this->problem_->getParameterList()
                   ->sublist("Nonlinear Schwarz Solver")
                   .sublist("Inner Newton")
                   .get("relNonLinTol", 1.0e-6);
    auto maxNonLinIts = this->problem_->getParameterList()
                            ->sublist("Nonlinear Schwarz Solver")
                            .sublist("Inner Newton")
                            .get("MaxNonLinIts", 10);
    auto criterion = this->problem_->getParameterList()
                         ->sublist("Nonlinear Schwarz Solver")
                         .sublist("Inner Newton")
                         .get("Criterion", "Residual");

    while (nlIts < maxNonLinIts) {

        this->problem_->calculateNonLinResidualVec("reverse");

        if (criterion == "Residual")
            residual = this->problem_->calculateResidualNorm();

        this->problem_->assemble("Newton");

        this->problem_->setBoundariesSystem();

        if (nlIts == 0)
            residual0 = residual;

        if (criterion == "Residual") {
            criterionValue = residual / residual0;
            if (verbose)
                cout << "### Newton iteration : " << nlIts << "  relative nonlinear residual : " << criterionValue
                     << endl;
            if (criterionValue < tol)
                break;
        }

        gmresIts += this->problem_->solveAndUpdate(criterion, criterionValue);
        nlIts++;
        if (criterion == "Update") {
            if (verbose)
                cout << "### Newton iteration : " << nlIts << "  residual of update : " << criterionValue << endl;
            if (criterionValue < tol)
                break;
        }
    }

    gmresIts /= nlIts;
    if (verbose)
        cout << "### Total Newton iterations : " << nlIts << "  with average gmres its : " << gmresIts << endl;
    if (this->problem_->getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts", false)) {
        TEUCHOS_TEST_FOR_EXCEPTION(nlIts == maxNonLinIts, std::runtime_error,
                                   "Maximum nonlinear Iterations reached. Problem might have converged in the last "
                                   "step. Still we cancel here.");
    }

    /* this->MpiComm_->barrier(); */
    /* std::cout << "==> Current residual: " << residual << std::endl; */

    // ================= Restore shared objects ===============================
    //    1. comm_ to MpiComm_ in problem, domainVec, Mesh
    //    2. mapRepeated_ and mapUnique_
    //    3. pointsRep
    //    4. bcFlagRep_ and bcFlagUni_
    //    5. elementsC_
    //    6. u_rep_

    // 1. replace communicators
    this->problem_->comm_ = this->MpiComm_;
    domainVec.at(0)->setComm(this->MpiComm_);
    mesh->comm_ = this->MpiComm_;

    // 2., 3. and 4. replace repeated and unique members
    mesh->replaceRepeatedMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(this->mapRepeatedMpiTmp_)), this->pointsRepTmp_,
                                 this->bcFlagRepTmp_);
    mesh->replaceUniqueMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(this->mapUniqueMpiTmp_)), this->pointsUniTmp_,
                               this->bcFlagUniTmp_);

    // Problems block vectors and matrices need to be reinitialized
    mesh->setElementsC(this->elementsCTmp_);
    // TODO KHo Could consider putting surrounding code into this function and do the same with the code above the solve
    this->replaceMapAndExportProblem();

    // 5. rebuild problem->u_rep_ to use overlapping map
    this->problem_->reInitSpecificProblemVectors(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(this->mapRepeatedMpiTmp_)));
    /* FEDD::logGreen("Map repeated", this->MpiComm_); */
    /* mesh->mapRepeated_->print(); */
    /* FEDD::logGreen("Map unique", this->MpiComm_); */
    /* mesh->mapUnique_->print(); */
    /* FEDD::logGreen("Map overlapping", this->MpiComm_); */
    /* mesh->mapOverlapping_->print(); */
    this->MpiComm_->barrier();

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

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::replaceMapAndExportProblem() {
    problem_->system_.reset(new FEDD::BlockMatrix<SC, LO, GO, NO>(1));

    // TODO KHo make this work for block and vector-valued systems
    // TODO KHo can't import like this for now because one of the maps is serial. How is this done in FROSch?
    auto domainPtr_vec = problem_->getDomainVector();
    auto mapUnique = domainPtr_vec[0]->getMapUnique();
    auto mapOverlapping = domainPtr_vec[0]->getMapOverlapping();
    auto oldSolution = problem_->solution_->getBlockNonConst(0);

    /* logGreen("old solution serial map", this->MpiComm_); */
    /* oldSolution->print(); */
    /* comm_->barrier(); */
    /* logGreen("old second rank", comm_); */
    /* if (comm_->getRank() == 1) { */
    /*     auto solData = oldSolution->getData(0); */
    /*     std::cout << '\n'; */
    /*     for (auto &temp : solData) { */
    /*         std::cout << temp << ", "; */
    /*     } */
    /*     std::cout << std::endl; */
    /* } */
    /* comm_->barrier(); */

    oldSolution->replaceMap(mapOverlapping);

    FEDD::logGreen("old solution mpi map", this->MpiComm_);
    oldSolution->print();
    /* logGreen("overlapping map", comm_); */
    /* mapOverlapping->print(); */

    auto newSolution = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(mapUnique));
    // Use export operation here since oldSolution is on overlapping map and newSolution on the unique map
    newSolution->exportFromVector(oldSolution, true, "Add");
    FEDD::logGreen("new solution", this->MpiComm_);
    newSolution->print();
    /* logGreen("new map", comm_); */
    /* newSolution->getMap()->print(); */
    problem_->solution_->addBlock(newSolution, 0);
}
} // namespace FROSch

#endif
