#ifndef NONLINEARSCHWARZOPERATOR_DEF_HPP
#define NONLINEARSCHWARZOPERATOR_DEF_HPP

#include "NonLinearSchwarzOperator_decl.hpp"
#include "feddlib/core/FE/FE_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/LinearAlgebra/Matrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector_decl.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
#include <Tacho_Driver.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MapFactory_decl.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MultiVectorFactory_decl.hpp>
#include <stdexcept>
#include <string>
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

namespace FROSch {
template <class SC, class LO, class GO, class NO>
NonLinearSchwarzOperator<SC, LO, GO, NO>::NonLinearSchwarzOperator(CommPtr serialComm, NonLinearProblemPtrFEDD problem,
                                                                   ParameterListPtr parameterList)
    : SchwarzOperator<SC, LO, GO, NO>(problem->system_->getBlock(0, 0)->getXpetraMatrix(), parameterList),
      problem_{problem}, x_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      y_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      localJacobianGhosts_{Teuchos::rcp(new FEDD::BlockMatrix<SC, LO, GO, NO>(1))}, mapOverlappingGhostsLocal_{},
      newtonTol_{}, maxNumIts_{}, criterion_{""}, combinationMode_{CombinationMode::Full},
      multiplicity_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))}, mapRepeatedMpiTmp_{},
      mapUniqueMpiTmp_{}, pointsRepTmp_{}, pointsUniTmp_{}, bcFlagRepTmp_{}, bcFlagUniTmp_{}, elementsCTmp_{},
      solutionTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      systemTmp_{Teuchos::rcp(new FEDD::BlockMatrix<SC, LO, GO, NO>(1))},
      feFactoryTmp_{Teuchos::rcp(new FEDD::FE<SC, LO, GO, NO>())},
      feFactoryGhostsLocal_{Teuchos::rcp(new FEDD::FE<SC, LO, GO, NO>())} {

    // Ensure that the mesh object has been initialized and a dual graph generated
    auto domainPtr_vec = problem_->getDomainVector();
    for (auto &domainPtr : domainPtr_vec) {
        TEUCHOS_ASSERT(!domainPtr->getElementMap().is_null());
        TEUCHOS_ASSERT(!domainPtr->getMapRepeated().is_null());
        TEUCHOS_ASSERT(!domainPtr->getDualGraph().is_null());
    }

    // Assigning parent class protected members is not good practice, but is done here to avoid modifying FROSch code
    this->SerialComm_ = serialComm;

    // Initialize members that cannot be null after construction
    x_->addBlock(Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(domainPtr_vec.at(0)->getMapOverlappingGhosts(), 1)),
                 0);
    // Max. num. entries per row unkown at this stage
    localJacobianGhosts_->addBlock(
        Teuchos::rcp(new FEDD::Matrix<SC, LO, GO, NO>(domainPtr_vec.at(0)->getMapOverlappingGhosts(),
                                                      domainPtr_vec.at(0)->getApproxEntriesPerRow())),
        0, 0);
    feFactoryGhostsLocal_->addFE(domainPtr_vec.at(0));
}

template <class SC, class LO, class GO, class NO> int NonLinearSchwarzOperator<SC, LO, GO, NO>::initialize() {

    auto domainVec = problem_->getDomainVector();
    auto mesh = domainVec.at(0)->getMesh();

    // Extract info from parameterList
    newtonTol_ =
        problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Relative Tolerance", 1.0e-6);
    maxNumIts_ = problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Max Iterations", 10);
    criterion_ = problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Criterion", "Residual");
    auto combineModeTemp = problem_->getParameterList()->get("Combine Mode", "Addition");

    if (combineModeTemp == "Averaging") {
        combinationMode_ = CombinationMode::Averaging;
    } else if (combineModeTemp == "Full") {
        combinationMode_ = CombinationMode::Full;
    } else if (combineModeTemp == "Restricted") {
        combinationMode_ = CombinationMode::Restricted;
    } else {
        if (this->MpiComm_->getRank() == 0) {
            std::cout << "\nInvalid Recombination Mode. Defaulting to Addition" << std::endl;
        }
        combinationMode_ = CombinationMode::Full;
    }

    // Store all distributed properties that need replacing for local computations
    mapRepeatedMpiTmp_ = mesh->getMapRepeated()->getXpetraMap();
    mapUniqueMpiTmp_ = mesh->getMapUnique()->getXpetraMap();
    pointsRepTmp_ = mesh->getPointsRepeated();
    pointsUniTmp_ = mesh->getPointsUnique();
    bcFlagRepTmp_ = mesh->getBCFlagRepeated();
    bcFlagUniTmp_ = mesh->getBCFlagUnique();
    elementsCTmp_ = mesh->getElementsC();
    solutionTmp_ = problem_->getSolution();
    systemTmp_ = problem_->system_;
    feFactoryTmp_ = problem_->feFactory_;

    mapOverlappingGhostsLocal_ = Xpetra::MapFactory<LO, GO, NO>::Build(
        mesh->getMapOverlappingGhosts()->getXpetraMap()->lib(),
        mesh->getMapOverlappingGhosts()->getXpetraMap()->getLocalNumElements(), 0, this->SerialComm_);

    // Compute overlap multiplicity
    if (combinationMode_ == CombinationMode::Averaging) {
        auto multiplicityUnique = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(mesh->getMapUnique(), 1));

        auto multiplicityRepeated = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(mesh->getMapOverlapping(), 1));
        multiplicityRepeated->putScalar(ST::one());
        multiplicityUnique->exportFromVector(multiplicityRepeated, false, "Add", "Forward");
        multiplicity_->addBlock(multiplicityUnique, 0);
    }
    return 0;
}

template <class SC, class LO, class GO, class NO> int NonLinearSchwarzOperator<SC, LO, GO, NO>::compute() {

    TEUCHOS_TEST_FOR_EXCEPTION(x_.is_null(), std::runtime_error,
                               "The current value of the nonlinear operator requires "
                               "an input to be computed.");

    auto domainVec = problem_->getDomainVector();
    auto mesh = domainVec.at(0)->getMesh();

    // ================= Replace shared objects ===============================
    // Done to "trick" FEDD::Problem assembly routines to assemble locally on the overlapping subdomain
    //    1. comm_ to SerialComm_ in problem, domainVec, Mesh
    //    2. mapRepeated_ and mapUnique_ to mapOverlappingLocal_
    //    3. pointsRep to pointsOverlapping
    //    4. bcFlagRep_ and bcFlagUni_ to bcFlagOverlappingGhosts_
    //    5. elementsC_ to elementsOverlappingGhosts_
    //    6. u_rep_ to u_overlapping_ in problemSpecific
    //    7. feFactory_ to feFactoryLocal_ in problem
    //  No destinction between unique and repeated needs to be made since assembly is only local here.

    // 1. replace communicators
    problem_->comm_ = this->SerialComm_;
    domainVec.at(0)->setComm(this->SerialComm_);
    mesh->comm_ = this->SerialComm_;

    // 2., 3., 4. and 5. replace repeated and unique members
    auto FEDDMapOverlappingGhostsLocal = Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapOverlappingGhostsLocal_));
    mesh->replaceRepeatedMembers(FEDDMapOverlappingGhostsLocal, mesh->pointsOverlappingGhosts_,
                                 mesh->bcFlagOverlappingGhosts_);
    mesh->replaceUniqueMembers(FEDDMapOverlappingGhostsLocal, mesh->pointsOverlappingGhosts_,
                               mesh->bcFlagOverlappingGhosts_);
    mesh->setElementsC(mesh->elementsOverlappingGhosts_);

    // Problems block vectors and matrices need to be reinitialized
    problem_->initializeProblem();
    // Map of current input
    x_->getBlockNonConst(0)->replaceMap(FEDDMapOverlappingGhostsLocal);
    // 6. rebuild problem->u_rep_ to use overlapping map
    problem_->reInitSpecificProblemVectors(FEDDMapOverlappingGhostsLocal);
    // 7. feFactory. Needs replacing because stores an AssembleFEFactoryObject
    problem_->feFactory_ = feFactoryGhostsLocal_;

    // Set Dirichlet BC on ghost points to current global solution. This ensures:
    // 1. The linear system solved in each Newton iteration will have a solution of zero at the ghost points
    // 2. This implies that residual is zero at the ghost points i.e. original solution is maintained
    auto tempMV = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(FEDDMapOverlappingGhostsLocal, 1));
    for (int i = 0; i < mesh->bcFlagOverlappingGhosts_->size(); i++) {
        if (mesh->bcFlagOverlappingGhosts_->at(i) == -99) {
            tempMV->replaceLocalValue(static_cast<GO>(i), 0, x_->getBlock(0)->getData(0)[i]);
        }
    }

    auto tempVecFlag = problem_->bcFactory_->getVecFlag();
    for (int i = 0; i < tempVecFlag.size(); i++) {
        if (tempVecFlag.at(i) == -99) {
            problem_->bcFactory_->setVecExternalSolAtIndex(i, tempMV);
        }
    }

    // Solve local nonlinear problems
    bool verbose = problem_->getVerbose();
    double residual0 = 1.;
    double residual = 1.;
    int nlIts = 0;
    double criterionValue = 1.;

    // Set solution_ to be u+P_i*g_i. g_i is zero on the boundary, simulating P_i locally
    // this = alpha*xTmp + beta*this
    problem_->solution_->update(ST::one(), *x_, ST::one());

    // Need to update solution_ within each iteration to assemble at u+P_i*g_i but update only g_i
    // This is necessary since u is nonzero on the artificial (interface) zero Dirichlet boundary
    // It would be more efficient to only store u on the boundary and update this value in each iteration
    while (nlIts < maxNumIts_) {

        problem_->calculateNonLinResidualVec("reverse");

        if (criterion_ == "Residual") {
            residual = problem_->calculateResidualNorm();
        }

        if (nlIts == 0) {
            residual0 = residual;
        }

        if (criterion_ == "Residual") {
            criterionValue = residual / residual0;
            FEDD::logGreen("Inner Newton iteration: " + std::to_string(nlIts), this->MpiComm_);
            FEDD::logGreen("Relative residual: " + std::to_string(criterionValue), this->MpiComm_);
            if (criterionValue < newtonTol_) {
                // Set solution_ to g_i
                break;
            }
        }

        problem_->assemble("Newton");

        // After this rows corresponding to Dirichlet nodes are unity and residualVec_ = 0
        problem_->setBoundariesSystem();

        problem_->solveAndUpdate(criterion_, criterionValue);
        nlIts++;
        if (criterion_ == "Update") {
            FEDD::logGreen("Inner Newton iteration: " + std::to_string(nlIts), this->MpiComm_);
            FEDD::logGreen("Residual of update: " + std::to_string(criterionValue), this->MpiComm_);
            if (criterionValue < newtonTol_) {
                break;
            }
        }
    }

    // Set solution_ to g_i
    problem_->solution_->update(-ST::one(), *x_, ST::one());
    FEDD::logGreen("Total inner Newton iters: " + std::to_string(nlIts), this->MpiComm_);

    if (problem_->getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts", false)) {
        TEUCHOS_TEST_FOR_EXCEPTION(nlIts == maxNumIts_, std::runtime_error,
                                   "Maximum nonlinear Iterations reached. Problem might have converged in the last "
                                   "step. Still we cancel here.");
    }
    // Set solution on ghost points to zero to build \sum P_ig_i
    for (int i = 0; i < mesh->bcFlagOverlappingGhosts_->size(); i++) {
        if (mesh->bcFlagOverlappingGhosts_->at(i) == -99) {
            problem_->solution_->getBlockNonConst(0)->replaceLocalValue(static_cast<LO>(i), 0, ST::zero());
        }
    }

    // The currently assembled Jacobian is from the previous Newton iteration. The error this causes is negligable.
    // Worth it since a reassemble is avoided.

    localJacobianGhosts_->addBlock(problem_->system_->getBlock(0, 0), 0, 0);
    // Assigning like this does not copy across the map pointer and results in a non-fillComplete matrix
    /* localJacobian_->getBlock(0, 0) = problem_->system_->getBlock(0, 0); */

    // Set all solutions to zero except for rank 0 for testing
    /* if (this->MpiComm_->getRank() != 0) */
    /*     problem_->solution_->putScalar(0.); */

    // ================= Restore shared objects ===============================
    //    1. comm_ to MpiComm_ in problem, domainVec, Mesh
    //    2. mapRepeated_ and mapUnique_
    //    3. pointsRep
    //    4. bcFlagRep_ and bcFlagUni_
    //    5. elementsC_
    //    6. u_rep_
    //    7. feFactory_

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

    // 6. rebuild problem->u_rep_ to use repeated map
    problem_->reInitSpecificProblemVectors(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapRepeatedMpiTmp_)));

    // Restore system state
    // TODO: kho maybe these reinits are not needed?
    problem_->initializeProblem();
    problem_->solution_ = solutionTmp_;
    problem_->system_ = systemTmp_;

    // Restore feFactory_
    problem_->feFactory_ = feFactoryTmp_;

    x_->getBlockNonConst(0)->replaceMap(
        Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mesh->getMapOverlappingGhosts()->getXpetraMap())));

    return 0;
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const BlockMultiVectorPtrFEDD x, BlockMultiVectorPtrFEDD y,
                                                     SC alpha, SC beta) {

    // Save the current input
    x_->getBlockNonConst(0)->importFromVector(x->getBlock(0), true, "Insert", "Forward");
    // Compute the value of the nonlinear operator
    this->compute();
    // y = alpha*f(x) + beta*y
    y->getBlockNonConst(0)->update(alpha, y_->getBlock(0), beta);
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly,
                                                     ETransp mode, SC alpha, SC beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "This version of apply does not make sense in the context of nonlinear operators.");
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::describe(FancyOStream &out, const EVerbosityLevel verbLevel) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "describe() has to be implemented properly...");
}

template <class SC, class LO, class GO, class NO> string NonLinearSchwarzOperator<SC, LO, GO, NO>::description() const {
    return "Nonlinear Schwarz Operator";
}

template <class SC, class LO, class GO, class NO>
Teuchos::RCP<FEDD::BlockMatrix<SC, LO, GO, NO>>
NonLinearSchwarzOperator<SC, LO, GO, NO>::getLocalJacobianGhosts() const {
    return localJacobianGhosts_;
}

// NOTE: KHo if FROSch_OverlappingOperator is modified this functionality could be shared
template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::replaceMapAndExportProblem() {

    // TODO: KHo make this work for block and vector-valued systems
    auto domainPtr_vec = problem_->getDomainVector();
    auto mapUnique = domainPtr_vec.at(0)->getMapUnique();
    auto mapOverlappingGhosts = domainPtr_vec.at(0)->getMapOverlappingGhosts();
    auto y_overlapping = problem_->solution_->getBlockNonConst(0);

    // For testing difference between add and insert
    /* if (this->MpiComm_->getRank() == 0) { */
    /*     y_overlapping->putScalar(1.); */
    /* } else if (this->MpiComm_->getRank() == 1) { */
    /*     y_overlapping->putScalar(2.); */
    /* } else if (this->MpiComm_->getRank() == 2) { */
    /*     y_overlapping->putScalar(4.); */
    /* } else if (this->MpiComm_->getRank() == 3) { */
    /*     y_overlapping->putScalar(7.); */
    /* } */

    y_overlapping->replaceMap(mapOverlappingGhosts);

    auto y_unique = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(mapUnique));
    if (combinationMode_ == CombinationMode::Restricted) {
        GO globID = 0;
        LO localID = 0;
        //  MultiVector insert mode does not add any entries i.e. only one entry from a rank is taken
        //  Export in Forward mode: the rank which is chosen to give value seems random
        //  Export in reverse mode: the rank which is chosen is the owning rank
        //  Import in Forward mode: like Export in reverse mode
        //  Import in Reverse mode: like Export in forward mode
        //  Conclusion: using an Importer results in a correct distribution. Probably because order in which mapping is
        //  done happens to be correct. Probably cannot be relied on.
        /* y_unique_->importFromVector(y_overlapping, true, "Insert", "Forward"); */
        for (auto i = 0; i < y_unique->getNumVectors(); i++) {
            auto y_overlappingData = y_overlapping->getData(i);
            for (auto j = 0; j < mapUnique->getNodeNumElements(); j++) {
                globID = mapUnique->getGlobalElement(j);
                localID = mapOverlappingGhosts->getLocalElement(globID);
                y_unique->getDataNonConst(i)[j] = y_overlappingData[localID];
            }
        }
    } else {
        // Use export operation here since oldSolution is on overlapping map and newSolution on the unique map
        // Use Insert since newSolution does not contain any values yet
        y_unique->exportFromVector(y_overlapping, true, "Add", "Forward");
    }
    if (combinationMode_ == CombinationMode::Averaging) {

        auto scaling = multiplicity_->getBlock(0)->getData(0);
        for (auto j = 0; j < y_unique->getNumVectors(); j++) {
            auto values = y_unique->getDataNonConst(j);
            for (auto i = 0; i < values.size(); i++) {
                values[i] = values[i] / scaling[i];
            }
        }
    }
    y_->addBlock(y_unique, 0);
}
} // namespace FROSch

#endif
