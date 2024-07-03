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
#include <Teuchos_Array.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
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
#include <algorithm>
#include <iterator>
#include <numeric>
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
      mapVecFieldOverlappingGhostsLocal_{}, relNewtonTol_{}, absNewtonTol_{}, maxNumIts_{},
      combinationMode_{},
      multiplicity_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))}, mapRepeatedMpiTmp_{},
      mapUniqueMpiTmp_{}, mapVecFieldRepeatedMpiTmp_{}, mapVecFieldUniqueMpiTmp_{}, pointsRepTmp_{}, pointsUniTmp_{},
      bcFlagRepTmp_{}, bcFlagUniTmp_{}, elementsCTmp_{},
      systemTmp_{Teuchos::rcp(new FEDD::BlockMatrix<SC, LO, GO, NO>(1))},
      solutionTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      rhsTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      sourceTermTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      previousSolutionTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      residualVecTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      feFactoryTmp_{Teuchos::rcp(new FEDD::FE<SC, LO, GO, NO>())},
      feFactoryGhostsLocal_{Teuchos::rcp(new FEDD::FE<SC, LO, GO, NO>())}, totalIters_{0} {

    // Ensure that the mesh object has been initialized and a dual graph generated
    auto domainPtr_vec = problem_->getDomainVector();
    for (auto &domainPtr : domainPtr_vec) {
        TEUCHOS_ASSERT(!domainPtr->getElementMap().is_null());
        TEUCHOS_ASSERT(!domainPtr->getMapRepeated().is_null());
        TEUCHOS_ASSERT(!domainPtr->getDualGraph().is_null());
    }

    // Assigning parent class protected members is not good practice, but is done here to avoid modifying FROSch code
    this->SerialComm_ = serialComm;

    // If we have more than one dof per node we need a special map to store these since e.g. mapOverlappingGhosts maps
    // nodes and not dofs
    MapConstPtrFEDD map;
    if (problem->getDofsPerNode(0) > 1) {
        map = domainPtr_vec.at(0)->getMapVecFieldOverlappingGhosts();
    } else {
        map = domainPtr_vec.at(0)->getMapOverlappingGhosts();
    }
    // Initialize members that cannot be null after construction
    x_->addBlock(Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(map, 1)), 0);
    // Max. num. entries per row unkown at this stage
    localJacobianGhosts_->addBlock(
        Teuchos::rcp(new FEDD::Matrix<SC, LO, GO, NO>(map, domainPtr_vec.at(0)->getApproxEntriesPerRow())), 0, 0);
    feFactoryGhostsLocal_->addFE(domainPtr_vec.at(0));
}

template <class SC, class LO, class GO, class NO> int NonLinearSchwarzOperator<SC, LO, GO, NO>::initialize() {

    auto domainVec = problem_->getDomainVector();
    auto mesh = domainVec.at(0)->getMesh();

    // Extract info from parameterList
    relNewtonTol_ =
        problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Relative Tolerance", 1.0e-6);
    absNewtonTol_ =
        problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Absolute Tolerance", 1.0e-6);
    maxNumIts_ = problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Max Iterations", 10);
    totalIters_ = 0;
    auto combineModeTemp = problem_->getParameterList()->get("Combine Mode", "Restricted");

    if (combineModeTemp == "Averaging") {
        combinationMode_ = CombinationMode::Averaging;
    } else if (combineModeTemp == "Full") {
        combinationMode_ = CombinationMode::Full;
    } else if (combineModeTemp == "Restricted") {
        combinationMode_ = CombinationMode::Restricted;
    } else {
        if (this->MpiComm_->getRank() == 0) {
            std::cerr << "\nInvalid Recombination Mode in NonLinearSchwarzOperator: \"" << combineModeTemp << "\". Defaulting to \"Restricted\"" << std::endl;
        }
        combinationMode_ = CombinationMode::Restricted;
    }

    mapOverlappingGhostsLocal_ = Xpetra::MapFactory<LO, GO, NO>::Build(
        mesh->getMapOverlappingGhosts()->getXpetraMap()->lib(),
        mesh->getMapOverlappingGhosts()->getXpetraMap()->getLocalNumElements(), 0, this->SerialComm_);
    mapVecFieldOverlappingGhostsLocal_ =
        FEDD::Map<LO, GO, NO>(mapOverlappingGhostsLocal_).buildVecFieldMap(problem_->getDofsPerNode(0))->getXpetraMap();

    // Compute overlap multiplicity
    if (combinationMode_ == CombinationMode::Averaging) {
        MapConstPtrFEDD mapUnique;
        MapConstPtrFEDD mapOverlapping;
        // Get the maps appropriate for the dofs per node
        if (problem_->getDofsPerNode(0) > 1) {
            mapUnique = domainVec.at(0)->getMapVecFieldUnique();
            mapOverlapping = domainVec.at(0)->getMapVecFieldOverlapping();
        } else {
            mapUnique = domainVec.at(0)->getMapUnique();
            mapOverlapping = domainVec.at(0)->getMapOverlapping();
        }
        auto multiplicityUnique = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(mapUnique, 1));

        auto multiplicityRepeated = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(mapOverlapping, 1));
        multiplicityRepeated->putScalar(ST::one());
        multiplicityUnique->exportFromVector(multiplicityRepeated, false, "Add", "Forward");
        multiplicity_->addBlock(multiplicityUnique, 0);
    }
    return 0;
}

template <class SC, class LO, class GO, class NO> int NonLinearSchwarzOperator<SC, LO, GO, NO>::compute() {

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "The nonlinear operator does not implement compute(). Simply call apply() directly.");
    return 0;
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const BlockMultiVectorPtrFEDD x, BlockMultiVectorPtrFEDD y,
                                                     SC alpha, SC beta) {

    FEDD_TIMER_START(InnerTimer, " - Schwarz - inner solve");
    // Save the current input
    x_->getBlockNonConst(0)->importFromVector(x->getBlock(0), true, "Insert", "Forward");

    auto domainVec = problem_->getDomainVector();
    auto mesh = domainVec.at(0)->getMesh();

    // Store all distributed properties that need replacing for local computations
    mapRepeatedMpiTmp_ = mesh->getMapRepeated()->getXpetraMap();
    mapUniqueMpiTmp_ = mesh->getMapUnique()->getXpetraMap();
    mapVecFieldRepeatedMpiTmp_ = domainVec.at(0)->getMapVecFieldRepeated()->getXpetraMap();
    mapVecFieldUniqueMpiTmp_ = domainVec.at(0)->getMapVecFieldUnique()->getXpetraMap();
    pointsRepTmp_ = mesh->getPointsRepeated();
    pointsUniTmp_ = mesh->getPointsUnique();
    bcFlagRepTmp_ = mesh->getBCFlagRepeated();
    bcFlagUniTmp_ = mesh->getBCFlagUnique();
    elementsCTmp_ = mesh->getElementsC();
    systemTmp_ = problem_->system_;
    solutionTmp_ = problem_->getSolution();
    rhsTmp_ = problem_->rhs_;
    sourceTermTmp_ = problem_->sourceTerm_;
    feFactoryTmp_ = problem_->feFactory_;
    previousSolutionTmp_ = problem_->previousSolution_;
    residualVecTmp_ = problem_->residualVec_;
    // Do not need to store rhsFuncVec_ because reseting problem does not erase it

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
    auto FEDDMapVecFieldOverlappingGhostsLocal =
        Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapVecFieldOverlappingGhostsLocal_));
    domainVec.at(0)->replaceRepeatedMembers(FEDDMapVecFieldOverlappingGhostsLocal, FEDDMapOverlappingGhostsLocal,
                                            mesh->pointsOverlappingGhosts_, mesh->bcFlagOverlappingGhosts_);
    domainVec.at(0)->replaceUniqueMembers(FEDDMapVecFieldOverlappingGhostsLocal, FEDDMapOverlappingGhostsLocal,
                                          mesh->pointsOverlappingGhosts_, mesh->bcFlagOverlappingGhosts_);
    mesh->setElementsC(mesh->elementsOverlappingGhosts_);

    // Problems block vectors and matrices need to be reinitialized
    problem_->initializeProblem();

    if (problem_->getDofsPerNode(0) > 1) {
        x_->getBlockNonConst(0)->replaceMap(FEDDMapVecFieldOverlappingGhostsLocal);
        problem_->reInitSpecificProblemVectors(FEDDMapVecFieldOverlappingGhostsLocal);
    } else {
        // Map of current input
        x_->getBlockNonConst(0)->replaceMap(FEDDMapOverlappingGhostsLocal);
        // 6. rebuild problem->u_rep_ to use overlapping map
        problem_->reInitSpecificProblemVectors(FEDDMapOverlappingGhostsLocal);
    }
    // 7. feFactory. Needs replacing because stores an AssembleFEFactoryObject
    problem_->feFactory_ = feFactoryGhostsLocal_;

    // Set Dirichlet BC on ghost points to current global solution. This ensures:
    // 1. The linear system solved in each Newton iteration will have a solution of zero at the ghost points
    // 2. This implies that residual is zero at the ghost points i.e. original solution is maintained
    Teuchos::RCP<FEDD::MultiVector<SC, LO, GO, NO>> tempMV;
    if (problem_->getDofsPerNode(0) > 1) {
        tempMV = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(FEDDMapVecFieldOverlappingGhostsLocal, 1));
    } else {
        tempMV = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(FEDDMapOverlappingGhostsLocal, 1));
    }
    for (int i = 0; i < mesh->bcFlagOverlappingGhosts_->size(); i++) {
        if (mesh->bcFlagOverlappingGhosts_->at(i) == -99) {
            for (int j = 0; j < problem_->getDofsPerNode(0); j++) {
                tempMV->replaceLocalValue(static_cast<GO>(problem_->getDofsPerNode(0) * i + j), 0,
                                          x_->getBlock(0)->getData(0)[problem_->getDofsPerNode(0) * i + j]);
            }
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
    double relResidual = 1.;
    double absResidual = 1.;
    int nlIts = 0;

    // Need to initialize the rhs_ and set boundary values in rhs_
    problem_->assemble();
    problem_->setBoundaries();

    // Set solution_ to be u+P_i*g_i. g_i is zero on the boundary, simulating P_i locally
    // this = alpha*xTmp + beta*this
    problem_->solution_->update(ST::one(), *x_, -ST::one());

    // Need to update solution_ within each iteration to assemble at u+P_i*g_i but update only g_i
    // This is necessary since u is nonzero on the artificial (interface) zero Dirichlet boundary
    // It would be more efficient to only store u on the boundary and update this value in each iteration
    while (nlIts < maxNumIts_) {
        problem_->calculateNonLinResidualVec("reverse");

        absResidual = problem_->calculateResidualNorm();

        if (nlIts == 0) {
            residual0 = absResidual;
        }

        relResidual = absResidual / residual0;
        FEDD::logGreen("Inner Newton iteration: " + std::to_string(nlIts), this->MpiComm_);
        FEDD::print("Absolute residual: ", this->MpiComm_);
        FEDD::print(absResidual, this->MpiComm_, 0, 10);
        FEDD::print("\nRelative residual: ", this->MpiComm_);
        FEDD::print(relResidual, this->MpiComm_, 0, 10);
        FEDD::print("\n", this->MpiComm_);

        if (relResidual < relNewtonTol_ || absResidual < absNewtonTol_) {
            break;
        }

        problem_->assemble("Newton");

        // After this rows corresponding to Dirichlet nodes are unity and residualVec_ = 0
        problem_->setBoundariesSystem();

        // Passing relative residual here is not correct since the residual of the update is returned
        // Not using this value here though so it does not matter
        problem_->solveAndUpdate("", absResidual);

        nlIts++;
    }

    // Set solution_ to g_i
    problem_->solution_->update(ST::one(), *x_, -ST::one());
    FEDD::logGreen("Terminated inner Newton", this->MpiComm_);
    totalIters_ += nlIts;

    if (problem_->getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts", false)) {
        TEUCHOS_TEST_FOR_EXCEPTION(nlIts == maxNumIts_, std::runtime_error,
                                   "Maximum nonlinear Iterations reached. Problem might have converged in the last "
                                   "step. Still we cancel here.");
    }
    // Set solution on ghost points to zero to build \sum P_ig_i
    for (int i = 0; i < mesh->bcFlagOverlappingGhosts_->size(); i++) {
        if (mesh->bcFlagOverlappingGhosts_->at(i) == -99) {
            for (int j = 0; j < problem_->getDofsPerNode(0); j++) {
                problem_->solution_->getBlockNonConst(0)->replaceLocalValue(
                    static_cast<LO>(problem_->getDofsPerNode(0) * i + j), 0, ST::zero());
            }
        }
    }

    // The currently assembled Jacobian is from the previous Newton iteration. The error this causes is negligable.
    // Worth it since a reassemble is avoided. Set the rows corresponding to Dirichlet nodes to unity since some problem
    // classes reassemble the tangent when calculating the residual
    problem_->setBoundariesSystem();
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
    domainVec.at(0)->replaceRepeatedMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapVecFieldRepeatedMpiTmp_)),
                                            Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapRepeatedMpiTmp_)), pointsRepTmp_,
                                            bcFlagRepTmp_);
    domainVec.at(0)->replaceUniqueMembers(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapVecFieldUniqueMpiTmp_)),
                                          Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapUniqueMpiTmp_)), pointsUniTmp_,
                                          bcFlagUniTmp_);
    // 5. replace elementsC_
    mesh->setElementsC(elementsCTmp_);
    this->replaceMapAndExportProblem();
    this->MpiComm_->barrier();
    // Restore system state
    problem_->initializeProblem();
    problem_->system_ = systemTmp_;
    problem_->solution_ = solutionTmp_;
    problem_->rhs_ = rhsTmp_;
    problem_->sourceTerm_ = sourceTermTmp_;
    problem_->previousSolution_ = previousSolutionTmp_;
    problem_->residualVec_ = residualVecTmp_;

    // Restore feFactory_
    problem_->feFactory_ = feFactoryTmp_;

    if (problem_->getDofsPerNode(0) > 1) {
        x_->getBlockNonConst(0)->replaceMap(domainVec.at(0)->getMapVecFieldOverlappingGhosts());
        // 6. rebuild problem->u_rep_ to use repeated map
        problem_->reInitSpecificProblemVectors(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapVecFieldRepeatedMpiTmp_)));
    } else {
        x_->getBlockNonConst(0)->replaceMap(domainVec.at(0)->getMapOverlappingGhosts());
        problem_->reInitSpecificProblemVectors(Teuchos::rcp(new FEDD::Map<LO, GO, NO>(mapRepeatedMpiTmp_)));
    }

    // y = alpha*f(x) + beta*y
    y->getBlockNonConst(0)->update(alpha, y_->getBlock(0), beta);
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, SC alpha, SC beta) {
    // Wraps another apply() method for compatibility
    auto feddX = Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1));
    auto feddY = Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1));
    // non owning rcp objects since they should not destroy x, y when going out of scope
    auto rcpX = Teuchos::rcp(&x, false);
    auto rcpY = Teuchos::rcp(&y, false);

    // TODO: kho this needs to be changed since the input vector is copied here (const)
    feddX->addBlock(Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(rcpX)), 0);
    feddY->addBlock(Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(rcpY)), 0);
    apply(feddX, feddY, alpha, beta);
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly,
                                                     ETransp mode, SC alpha, SC beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "This apply() overload should not be used with nonlinear operators");
}

template <class SC, class LO, class GO, class NO>
Teuchos::RCP<FEDD::BlockMatrix<SC, LO, GO, NO>>
NonLinearSchwarzOperator<SC, LO, GO, NO>::getLocalJacobianGhosts() const {
    return localJacobianGhosts_;
}

template <class SC, class LO, class GO, class NO>
std::vector<SC> NonLinearSchwarzOperator<SC, LO, GO, NO>::getRunStats() const {

    // Import all iteration counts to rank 0
    std::vector<LO> totalItersVec({0});
    if (this->MpiComm_->getRank() == 0) {
        totalItersVec = std::vector<LO>(this->MpiComm_->getSize());
    }
    Teuchos::gather(&totalIters_, 1, totalItersVec.data(), 1, 0, *this->MpiComm_);
    auto maxIters = std::max_element(totalItersVec.begin(), totalItersVec.end());

    auto minIters = std::min_element(totalItersVec.begin(), totalItersVec.end());
    SC avgIters = 0;
    for (auto val : totalItersVec) {
        avgIters += val;
    }

    avgIters = avgIters / this->MpiComm_->getSize();
    return std::vector<SC>{static_cast<SC>(*minIters), avgIters, static_cast<SC>(*maxIters)};
}

template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::describe(FancyOStream &out, const EVerbosityLevel verbLevel) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "describe() has to be implemented properly...");
}

template <class SC, class LO, class GO, class NO> string NonLinearSchwarzOperator<SC, LO, GO, NO>::description() const {
    return "Nonlinear Schwarz Operator";
}

// NOTE: KHo if FROSch_OverlappingOperator is modified this functionality could be shared
template <class SC, class LO, class GO, class NO>
void NonLinearSchwarzOperator<SC, LO, GO, NO>::replaceMapAndExportProblem() {

    // TODO: kho make this work for block and vector-valued systems
    auto domainPtr_vec = problem_->getDomainVector();
    MapConstPtrFEDD mapUnique;
    MapConstPtrFEDD mapOverlappingGhosts;
    if (problem_->getDofsPerNode(0) > 1) {
        mapUnique = domainPtr_vec.at(0)->getMapVecFieldUnique();
        mapOverlappingGhosts = domainPtr_vec.at(0)->getMapVecFieldOverlappingGhosts();
    } else {
        mapUnique = domainPtr_vec.at(0)->getMapUnique();
        mapOverlappingGhosts = domainPtr_vec.at(0)->getMapOverlappingGhosts();
    }
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
        GO globalID = 0;
        LO localID = 0;
        //  MultiVector insert mode does not add any entries i.e. only one entry from a rank is taken
        //  Export in Forward mode: the rank which is chosen to give value seems random
        //  Export in reverse mode: the rank which is chosen is the owning rank
        //  Import in Forward mode: like Export in reverse mode
        //  Import in Reverse mode: like Export in forward mode
        //  Conclusion: using an Importer results in a correct distribution. Probably because order in which mapping
        //  is done happens to be correct. Probably cannot be relied on.
        /* y_unique_->importFromVector(y_overlapping, true, "Insert", "Forward"); */
        for (auto i = 0; i < y_unique->getNumVectors(); i++) {
            auto y_overlappingData = y_overlapping->getData(i);
            for (auto j = 0; j < mapUnique->getNodeNumElements(); j++) {
                globalID = mapUnique->getGlobalElement(j);
                localID = mapOverlappingGhosts->getLocalElement(globalID);
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
