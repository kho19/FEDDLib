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
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MapFactory_decl.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MultiVectorFactory_decl.hpp>
#include <stdexcept>
#include <vector>

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
                                                                   NonLinearProblemPtrFEDD problem)
    : SchwarzOperator<SC, LO, GO, NO>(mpiComm), problem_{problem}, newtonTol_{}, maxNumIts_{}, criterion_{""},
      recombinationMode_{RecombinationMode::Add}, mapRepeatedMpiTmp_{}, mapUniqueMpiTmp_{}, elementMapMpiTmp_{},
      elementMapOverlappingMpiTmp_{}, mapOverlappingMpiTmp_{}, pointsRepTmp_{}, pointsUniTmp_{}, bcFlagRepTmp_{},
      bcFlagUniTmp_{}, elementsCTmp_{} {

    // Ensure that the mesh object has been initialized and a dual graph generated
    auto domainPtr_vec = problem_->getDomainVector();
    for (auto &domainPtr : domainPtr_vec) {
        TEUCHOS_ASSERT(!domainPtr->getElementMap().is_null());
        TEUCHOS_ASSERT(!domainPtr->getMapRepeated().is_null());
        TEUCHOS_ASSERT(!domainPtr->getDualGraph().is_null());
    }
    // todo KHo to avoid making a new constructor for SchwarzOperator, which would require editing Trilinos source code,
    // SchwarzOperator protected variables are set here. Should be avoided!
    this->ParameterList_ = parameterList;

    // Initialize members that cannot be null after construction
    x_.reset(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1));
    x_->addBlock(Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(domainPtr_vec.at(0)->getMapOverlapping(), 1)), 0);
    y_.reset(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1));
    multiplicity_.reset(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1));
}

template <class SC, class LO, class GO, class NO>
NonLinearSchwarzOperator<SC, LO, GO, NO>::~NonLinearSchwarzOperator() {}

// NOTE KHo increase overlap of the dual graph by the specified number of layers. Not doing this here for now since
// MeshPartitioner already does.
template <class SC, class LO, class GO, class NO>
int NonLinearSchwarzOperator<SC, LO, GO, NO>::initialize(int overlap) {

    auto domainVec = problem_->getDomainVector();
    auto mesh = domainVec.at(0)->getMesh();

    // Extract info from parameterList
    newtonTol_ = problem_->getParameterList()
                     ->sublist("Nonlinear Schwarz Solver")
                     .sublist("Inner Newton")
                     .get("Relative Tolerance", 1.0e-6);
    maxNumIts_ = problem_->getParameterList()
                     ->sublist("Nonlinear Schwarz Solver")
                     .sublist("Inner Newton")
                     .get("Max Iterations", 10);
    criterion_ = problem_->getParameterList()
                     ->sublist("Nonlinear Schwarz Solver")
                     .sublist("Inner Newton")
                     .get("Criterion", "Residual");
    auto recombinationModeTemp = problem_->getParameterList()
                                     ->sublist("Nonlinear Schwarz Solver")
                                     .sublist("Inner Newton")
                                     .get("Recombination Mode", "Addition");

    if (recombinationModeTemp == "Add") {
        recombinationMode_ = RecombinationMode::Add;
    } else if (recombinationModeTemp == "Average") {
        recombinationMode_ = RecombinationMode::Average;
    } else if (recombinationModeTemp == "Restricted") {
        recombinationMode_ = RecombinationMode::Restricted;
    } else {
        if (this->MpiComm_->getRank() == 0) {
            std::cout << "\nInvalid Recombination Mode. Defaulting to Addition" << std::endl;
        }
        recombinationMode_ = RecombinationMode::Add;
    }

    // Build serial overlapping element and node maps
    auto elementMapOverlappingMPI = mesh->getElementMapOverlapping()->getXpetraMap();
    // Store all distributed properties that need replacing for local computations
    mapRepeatedMpiTmp_ = mesh->getMapRepeated()->getXpetraMap();
    mapUniqueMpiTmp_ = mesh->getMapUnique()->getXpetraMap();
    pointsRepTmp_ = mesh->getPointsRepeated();
    pointsUniTmp_ = mesh->getPointsUnique();
    bcFlagRepTmp_ = mesh->getBCFlagRepeated();
    bcFlagUniTmp_ = mesh->getBCFlagUnique();
    elementsCTmp_ = mesh->getElementsC();
    solutionTmp_ = problem_->getSolution();

    // Compute overlap multiplicity
    if (recombinationMode_ == RecombinationMode::Average) {
        auto multiplicityUnique = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(mesh->getMapUnique(), 1));

        auto multiplicityRepeated = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(mesh->getMapOverlappingInterior(), 1));
        multiplicityRepeated->putScalar(ScalarTraits<SC>::one());
        multiplicityUnique->exportFromVector(multiplicityRepeated, false, "Add", "Forward");
        multiplicity_->addBlock(multiplicityUnique, 0);
    //TODO: want to add boundary nodes on ghost layer
    }
    return 0;
}

template <class SC, class LO, class GO, class NO> int NonLinearSchwarzOperator<SC, LO, GO, NO>::compute() {

    TEUCHOS_TEST_FOR_EXCEPTION(x_.is_null(), std::runtime_error,
                               "The the current value of the nonlinear operator requires "
                               "an input to be computed.");

    auto domainVec = problem_->getDomainVector();
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
    problem_->comm_ = this->SerialComm_;
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
    problem_->initializeProblem();
    // Set solution to the current input
    x_->getBlockNonConst(0)->replaceMap(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingLocal)));
    problem_->solution_ = Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>{x_});

    // 6. rebuild problem->u_rep_ to use overlapping map
    problem_->reInitSpecificProblemVectors(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingLocal)));

    // Solve local nonlinear problems
    // TODO KHo need to set the initial value here to reflect the current linearisation point
    // TODO KHo change this to be a nox solver? Don't want FEDDLib nonlinear solver to remove depedency on the FEDDLib.
    bool verbose = problem_->getVerbose();
    double gmresIts = 0.;
    double residual0 = 1.;
    double residual = 1.;
    int nlIts = 0;
    double criterionValue = 1.;

    // Not using the constructor that accepts a block map since it uses const_cast rather than constructing new maps
    std::vector<MapConstPtrFEDD> mapVector;
    for (int i = 0; i < x_->getMap()->size(); i++) {
        mapVector.push_back(x_->getMap()->getBlock(i));
    }
    // Current solution on the subdomain interior c.f. R_i*F(u-P_i*gLocal) = 0
    auto gLocal =
        Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>{mapVector, static_cast<int>(x_->getNumVectors())});
    gLocal->putScalar(ST::zero());

    // Need to update solution_ within each iteration to assemble at u+P_i*g_i but update only g_i
    // This is necessary since u is nonzero on the artificial (interface) zero Dirichlet boundary
    // It would be more efficient to only store u on the boundary and update this value in each iteration
    while (nlIts < maxNumIts_) {

        // Set solution_ to be u+P_i*g_i. g_i is zero on the boundary, simulating P_i locally
        // this = alpha*xTmp + beta*this
        /* FEDD::logGreen("Before update:", this->MpiComm_); */
        /* problem_->solution_->getBlock(0)->print(); */

        problem_->solution_->update(ST::one(), *x_, ST::one());
        problem_->calculateNonLinResidualVec("reverse");
        /* FEDD::logGreen("After update:", this->MpiComm_); */
        /* problem_->solution_->getBlock(0)->print(); */

        if (criterion_ == "Residual") {
            residual = problem_->calculateResidualNorm();
        }

        if (nlIts == 0) {
            residual0 = residual;
        }

        if (criterion_ == "Residual") {
            criterionValue = residual / residual0;
            if (verbose) {
                cout << "### Newton iteration : " << nlIts << "  relative nonlinear residual : " << criterionValue
                     << endl;
            }
            if (criterionValue < newtonTol_) {
                // Set solution_ to g_i
                problem_->solution_->update(-ST::one(), *x_, ST::one());
                break;
            }
        }

        problem_->assemble("Newton");
        /* FEDD::logGreen("RHS:", this->MpiComm_); */
        /* problem_->residualVec_->getBlock(0)->print(); */
        // After this rows corresponding to Dirichlet nodes are unity and residualVec_ = 0
        problem_->setBoundariesSystem();
        /* FEDD::logGreen("System:", this->MpiComm_); */
        /* problem_->system_->getBlock(0, 0)->print(); */

        // Changing the solution here changes the result after solveAndUpdate() since the Newton update \delta is added
        // to the current solution
        // Set solution_ to g_i
        problem_->solution_->update(-ST::one(), *x_, ST::one());

        gmresIts += problem_->solveAndUpdate(criterion_, criterionValue);
        nlIts++;
        if (criterion_ == "Update") {
            if (verbose) {
                cout << "### Newton iteration : " << nlIts << "  residual of update : " << criterionValue << endl;
            }
            if (criterionValue < newtonTol_) {
                break;
            }
        }
    }
    // Set solution on ghost points to zero
    for (int i = 0; i < mesh->bcFlagOverlapping_->size(); i++) {
        if (mesh->bcFlagOverlapping_->at(i) == -99) {
            problem_->solution_->getBlockNonConst(0)->replaceLocalValue(static_cast<GO>(i), 0, ST::zero());
        }
    }

    // Set all solutions to zero except for rank 0 for testing
    /* if (this->MpiComm_->getRank() != 0) */
    /*     problem_->solution_->putScalar(0.); */

    gmresIts /= nlIts;
    if (verbose)
        cout << "### Total Newton iterations : " << nlIts << "  with average gmres its : " << gmresIts << endl;
    if (problem_->getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts", false)) {
        TEUCHOS_TEST_FOR_EXCEPTION(nlIts == maxNumIts_, std::runtime_error,
                                   "Maximum nonlinear Iterations reached. Problem might have converged in the last "
                                   "step. Still we cancel here.");
    }

    // ================= Restore shared objects ===============================
    //    1. comm_ to MpiComm_ in problem, domainVec, Mesh
    //    2. mapRepeated_ and mapUnique_
    //    3. pointsRep
    //    4. bcFlagRep_ and bcFlagUni_
    //    5. elementsC_
    //    6. u_rep_

    // 1. replace communicators
    problem_->comm_ = this->MpiComm_;
    domainVec.at(0)->setComm(this->MpiComm_);
    mesh->comm_ = this->MpiComm_;

    // 2., 3. and 4. replace repeated and unique members
    mesh->replaceRepeatedMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapRepeatedMpiTmp_)), pointsRepTmp_,
                                 bcFlagRepTmp_);
    mesh->replaceUniqueMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapUniqueMpiTmp_)), pointsUniTmp_, bcFlagUniTmp_);

    // 5. replace elementsC_
    mesh->setElementsC(elementsCTmp_);
    this->replaceMapAndExportProblem();
    this->MpiComm_->barrier();

    // 6. rebuild problem->u_rep_ to use overlapping map
    problem_->reInitSpecificProblemVectors(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapRepeatedMpiTmp_)));
    // Restore current global solution
    problem_->solution_ = solutionTmp_;
    return 0;
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const BlockMultiVectorPtrFEDD x, BlockMultiVectorPtrFEDD y,
                                                     SC alpha, SC beta) {

    // Save the current input
    x_->getBlockNonConst(0)->importFromVector(x->getBlock(0), true, "Insert");
    // Compute the value of the nonlinear operator
    this->compute();
    // y = alpha*f(x) + beta*y
    y->getBlockNonConst(0)->update(alpha, y_->getBlock(0), beta);
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly,
                                                     ETransp mode, SC alpha, SC beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(false, std::runtime_error,
                               "This version of apply does not make sense in the context of nonlinear operators.");
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::describe(FancyOStream &out, const EVerbosityLevel verbLevel) const {
    TEUCHOS_TEST_FOR_EXCEPTION(false, std::runtime_error, "describe() has to be implemented properly...");
}

template <class SC, class LO, class GO, class NO> string NonLinearSchwarzOperator<SC, LO, GO, NO>::description() const {
    return "Nonlinear Schwarz Operator";
}

// NOTE: KHo if FROSch_OverlappingOperator is modified this functionality could be shared
template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::replaceMapAndExportProblem() {
    problem_->system_.reset(new FEDD::BlockMatrix<SC, LO, GO, NO>(1));

    // TODO: KHo make this work for block and vector-valued systems
    auto domainPtr_vec = problem_->getDomainVector();
    auto mapUnique = domainPtr_vec.at(0)->getMapUnique();
    auto mapOverlapping = domainPtr_vec.at(0)->getMapOverlapping();
    auto overlappingY = problem_->solution_->getBlockNonConst(0);

    // For testing difference between add and insert
    /* if (this->MpiComm_->getRank() == 0) { */
    /*     overlappingY->putScalar(1.); */
    /* } else if (this->MpiComm_->getRank() == 1) { */
    /*     overlappingY->putScalar(2.); */
    /* } else if (this->MpiComm_->getRank() == 2) { */
    /*     overlappingY->putScalar(4.); */
    /* } else if (this->MpiComm_->getRank() == 3) { */
    /*     overlappingY->putScalar(7.); */
    /* } */

    overlappingY->replaceMap(mapOverlapping);

    auto uniqueY = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(mapUnique));
    // Use export operation here since oldSolution is on overlapping map and newSolution on the unique map
    // Use Insert since newSolution does not contain any values yet

    if (recombinationMode_ == RecombinationMode::Restricted) {
        GO globID = 0;
        LO localID = 0;
        for (auto i = 0; i < uniqueY->getNumVectors(); i++) {
            auto overlappingYData = overlappingY->getData(i);
            for (auto j = 0; j < mapUnique->getNodeNumElements(); j++) {
                globID = mapUnique->getGlobalElement(j);
                localID = mapOverlapping->getLocalElement(globID);
                uniqueY->getDataNonConst(i)[j] = overlappingYData[localID];
            }
        }
        // MultiVector insert mode does not add any entries i.e. only one entry from a rank is taken
        // Export in Forward mode: the rank which is chosen to give value seems random
        // Export in reverse mode: the rank which is chosen is the owning rank
        // Import in Forward mode: like Export in reverse mode
        // Import in Reverse mode: like Export in forward mode
        // Conclusion: using an Importer results in a correct distribution. Probably because order in which mapping is
        // done happens to be correct. Probably cannot be relied on.
        /* uniqueY->importFromVector(overlappingY, true, "Insert", "Forward"); */
    } else {
        uniqueY->exportFromVector(overlappingY, true, "Add", "Forward");
    }
    if (recombinationMode_ == RecombinationMode::Average) {

        auto scaling = multiplicity_->getBlock(0)->getData(0);
        for (auto j = 0; j < uniqueY->getNumVectors(); j++) {
            auto values = uniqueY->getDataNonConst(j);
            for (auto i = 0; i < values.size(); i++) {
                values[i] = values[i] / scaling[i];
            }
        }
    }
    y_->addBlock(uniqueY, 0);
}
} // namespace FROSch

#endif
