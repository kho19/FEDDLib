#ifndef COARSENONLINEARSCHWARZOPERATOR_DEF_HPP
#define COARSENONLINEARSCHWARZOPERATOR_DEF_HPP

#include "CoarseNonLinearSchwarzOperator_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector_decl.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
#include <FROSch_IPOUHarmonicCoarseOperator_decl.hpp>
#include <FROSch_Tools_decl.hpp>
#include <FROSch_Types.h>
#include <Tacho_Driver.hpp>
#include <Teuchos_ArrayRCPDecl.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_implicit_cast.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MapFactory_decl.hpp>
#include <Xpetra_Map_decl.hpp>
#include <Xpetra_Matrix.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MultiVectorFactory_decl.hpp>
#include <Xpetra_MultiVector_decl.hpp>
#include <stdexcept>
#include <string>
#include <vector>

/*!
 Implementation of CoarseNonLinearSchwarzOperator

 @brief Implements the coarse correction T_0 from the nonlinear Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

// The communicator for this object is taken from the matrix passed to IPOUHarmonicCoarseOperator
template <class SC, class LO, class GO, class NO>
CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::CoarseNonLinearSchwarzOperator(NonLinearProblemPtrFEDD problem,
                                                                               ParameterListPtr parameterList)
    : IPOUHarmonicCoarseOperator<SC, LO, GO, NO>(problem->system_->getMergedMatrix()->getXpetraMatrix(), parameterList),
      problem_{problem},
      x_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(problem->getDomainVector().size()))},
      y_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(problem->getDomainVector().size()))}, relNewtonTol_{},
      absNewtonTol_{}, maxNumIts_{}, coarseResidualVec_{}, coarseDeltaG0_{}, deltaG0Merged_{},
      deltaG0_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(problem->getDomainVector().size()))},
      solutionTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(problem->getDomainVector().size()))},
      systemTmp_{Teuchos::rcp(new FEDD::BlockMatrix<SC, LO, GO, NO>(problem->getDomainVector().size()))},
      rhsTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(problem->getDomainVector().size()))},
      sourceTermTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(problem->getDomainVector().size()))},
      previousSolutionTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(problem->getDomainVector().size()))},
      residualVecTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(problem->getDomainVector().size()))},
      totalIters_{0} {

    // Ensure that the mesh object has been initialized and a dual graph generated
    auto domainPtr_vec = problem_->getDomainVector();
    for (auto &domainPtr : domainPtr_vec) {
        TEUCHOS_ASSERT(!domainPtr->getElementMap().is_null());
        TEUCHOS_ASSERT(!domainPtr->getMapRepeated().is_null());
        TEUCHOS_ASSERT(!domainPtr->getDualGraph().is_null());
    }

    // Initialize members that cannot be null after construction
    /* x_->addBlock(Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(domainPtr_vec.at(0)->getMapOverlappingGhosts(),
     * 1)), */
    /*              0); */
}

template <class SC, class LO, class GO, class NO> int CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::initialize() {
    // Extract info from parameterList
    relNewtonTol_ =
        problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Relative Tolerance", 1.0e-6);
    absNewtonTol_ =
        problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Absolute Tolerance", 1.0e-6);
    maxNumIts_ = problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Max Iterations", 10);
    auto dimension = problem_->getParameterList()->sublist("Parameter").get("Dimension", 2);
    totalIters_ = 0;

    auto domainVec = problem_->getDomainVector();

    Teuchos::RCP<const FEDD::Map<LO, GO, NO>> uniqueDofsMap;
    if (domainVec.size() == 1) {

        auto mesh = domainVec.at(0)->getMesh();
        ConstXMapPtr repeatedNodesMap;
        ConstXMapPtr repeatedDofsMap;
        if (problem_->getDofsPerNode(0) > 1) {
            repeatedNodesMap = domainVec.at(0)->getMapRepeated()->getXpetraMap();
            repeatedDofsMap = domainVec.at(0)->getMapVecFieldRepeated()->getXpetraMap();
            uniqueDofsMap = domainVec.at(0)->getMapVecFieldUnique();
        } else {
            repeatedNodesMap = domainVec.at(0)->getMapRepeated()->getXpetraMap();
            repeatedDofsMap = repeatedNodesMap;
            uniqueDofsMap = domainVec.at(0)->getMapUnique();
        }

        auto tempMV = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(uniqueDofsMap, 1));
        deltaG0_->addBlock(tempMV, 0);

        auto dofsPerNode = problem_->getDofsPerNode(0);

        // Initialize the underlying IPOUHarmonicCoarseOperator object
        // dofs of a node are lumped together
        int dofOrdering = 0;
        auto dofsMaps = Teuchos::ArrayRCP<ConstXMapPtr>(1);
        BuildDofMaps(repeatedDofsMap, dofsPerNode, dofOrdering, repeatedNodesMap, dofsMaps);

        // Build nodeList
        auto nodeListFEDD = mesh->getPointsRepeated();
        auto nodeList = Xpetra::MultiVectorFactory<SC, LO, GO>::Build(repeatedNodesMap, nodeListFEDD->at(0).size());
        for (auto i = 0; i < nodeListFEDD->size(); i++) {
            // pointsRepeated are distributed
            for (auto j = 0; j < nodeListFEDD->at(0).size(); j++) {
                nodeList->getVectorNonConst(j)->replaceLocalValue(i, nodeListFEDD->at(i).at(j));
            }
        }

        // Build nullspace as in TwoLevelPreconditioner line 195
        NullSpaceType nullSpaceType;
        if (!this->ParameterList_->get("Null Space Type", "Laplace").compare("Laplace")) {
            nullSpaceType = NullSpaceType::Laplace;
        } else if (!this->ParameterList_->get("Null Space Type", "Laplace").compare("Linear Elasticity")) {
            nullSpaceType = NullSpaceType::Elasticity;
        } else {
            FROSCH_ASSERT(false, "Null Space Type unknown.");
        }
        // nullSpaceBasis is a multiVector with one vector for each function in the nullspace e.g. translations and
        // rotations for elasticity. Each vector contains the nullspace function on the repeated map.
        auto nullSpaceBasis = BuildNullSpace(dimension, nullSpaceType, repeatedDofsMap, dofsPerNode, dofsMaps,
                                             implicit_cast<ConstXMultiVectorPtr>(nodeList));
        // Build the vector of Dirichlet node indices
        // See FROSch::FindOneEntryOnlyRowsGlobal() for reference
        auto dirichletBoundaryDofsVec = Teuchos::rcp(new std::vector<GO>(0));
        int block = 0;
        int loc = 0;
        auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
        /* FEDD::logGreen("repeatedNodesmap", this->MpiComm_); */
        /* repeatedNodesMap->describe(*out, VERB_EXTREME); */
        /* FEDD::logGreen("bcFlagRep_", this->MpiComm_); */
        /* FEDD::logVec(*mesh->bcFlagRep_, this->MpiComm_); */
        /* std::cout << std::flush; */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* FEDD::logVec(*mesh->bcFlagRep_, this->MpiComm_, 1); */
        /* std::cout << std::flush; */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* FEDD::logVec(*mesh->bcFlagRep_, this->MpiComm_, 2); */
        /* std::cout << std::flush; */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* FEDD::logVec(*mesh->bcFlagRep_, this->MpiComm_, 3); */
        /* std::cout << std::flush; */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        for (auto i = 0; i < repeatedNodesMap->getLocalNumElements(); i++) {
            auto flag = mesh->bcFlagRep_->at(i);
            // The vector vecFlag_ contains the flags that have been set with addBC().
            // loc: local index in vecFlag_ of the flag if found. Is set by the function.
            // block: block id in which to search. Needs to be provided to the function.
            if (problem_->getBCFactory()->findFlag(flag, block, loc)) {
                for (auto j = 0; j < dofsPerNode; j++) {
                    dirichletBoundaryDofsVec->push_back(repeatedDofsMap->getGlobalElement(dofsPerNode * i + j));
                }
            }
        }
        /* FEDD::logGreen("dirichletBoundaryDofsVec", this->MpiComm_); */
        /* FEDD::logVec(*dirichletBoundaryDofsVec, this->MpiComm_); */
        /* std::cout << std::flush; */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* FEDD::logVec(*dirichletBoundaryDofsVec, this->MpiComm_, 1); */
        /* std::cout << std::flush; */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* FEDD::logVec(*dirichletBoundaryDofsVec, this->MpiComm_, 2); */
        /* std::cout << std::flush; */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* FEDD::logVec(*dirichletBoundaryDofsVec, this->MpiComm_, 3); */
        /* std::cout << std::flush; */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */
        /* this->MpiComm_->barrier(); */

        // Convert std::vector to ArrayRCP
        auto dirichletBoundaryDofs = arcp<GO>(dirichletBoundaryDofsVec);
        // This builds the coarse spaces, assembles the coarse solve map and does symbolic factorization of the coarse
        // problem
        IPOUHarmonicCoarseOperator<SC, LO, GO, NO>::initialize(
            dimension, dofsPerNode, repeatedNodesMap, dofsMaps, nullSpaceBasis,
            implicit_cast<ConstXMultiVectorPtr>(nodeList), dirichletBoundaryDofs);
    } else { // else we are dealing with a block system

        // dofsPerNode, repeatedNodesMap, dofsMaps, nullSpaceBasis, nodeList, dirichletBoundaryDofs
        // iPOUHarmonicCoarseOperator->initialize(dimension,dofsPerNode,   repeatedNodesMap,   dofsMaps, nullSpaceBasis,
        // nodeList,   dirichletBoundaryDofs);
        // iPOUHarmonicCoarseOperator->initialize(dimension,dofsPerNodeVec,repeatedNodesMapVec,dofsMapsVec,nullSpaceBasisVec,nodeListVec,dirichletBoundaryDofsVec);
        // Dof count in each block
        auto dofsPerNodeVec = Teuchos::ArrayRCP<unsigned>(domainVec.size());
        // The distribution of dofs. One map for each block. This gets built heuristically in
        // TwoLevelBlockPreconditioner using the unique distribution if not provided but only for one block.
        auto repeatedDofsMapVec = Teuchos::ArrayRCP<Teuchos::RCP<const Xpetra::Map<LO, GO, NO>>>(domainVec.size());
        // The distribution of nodes. This gets built from the distribution of dofs in TwoLevelBlockPreconditioner. Here
        // we already know this distribution, so we just ensure the built one is the same as the existing one.
        auto repeatedNodesMapVec = Teuchos::ArrayRCP<Teuchos::RCP<const Xpetra::Map<LO, GO, NO>>>(domainVec.size());
        // The distribution of dofs, but this is a 2D vector. For each block there is a vector that contains a map entry
        // for each dof, unlike repeatedDofsMapVec.
        auto dofsMapsVec =
            Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<const Xpetra::Map<LO, GO, NO>>>>(domainVec.size());
        // A multivector for each block. Columns in the multivector for each basis function
        auto nullSpaceBasisVec =
            Teuchos::ArrayRCP<Teuchos::RCP<const Xpetra::MultiVector<SC, LO, GO, NO>>>(domainVec.size());
        // Coords of nodes. Can be different for each block
        auto nodeListVec = Teuchos::ArrayRCP<Teuchos::RCP<const Xpetra::MultiVector<SC, LO, GO, NO>>>(domainVec.size());
        // List of indices of the nodes that are on the Dirichlet boundary
        auto dirichletBoundaryDofsVec = Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>(domainVec.size());

        deltaG0_ = Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(domainVec.size()));
        GO offset = 0;
        for (int i = 0; i < domainVec.size(); i++) {
            auto mesh = domainVec.at(i)->getMesh();
            dofsPerNodeVec[i] = problem_->getDofsPerNode(i);
            if (dofsPerNodeVec[i] > 1) {
                repeatedNodesMapVec[i] = domainVec.at(i)->getMapRepeated()->getXpetraMap();
                repeatedDofsMapVec[i] = domainVec.at(i)->getMapVecFieldRepeated()->getXpetraMap();
                uniqueDofsMap = domainVec.at(i)->getMapVecFieldUnique();
            } else {
                repeatedNodesMapVec[i] = domainVec.at(i)->getMapRepeated()->getXpetraMap();
                repeatedDofsMapVec[i] = repeatedNodesMapVec[i];
                uniqueDofsMap = domainVec.at(i)->getMapUnique();
            }

            auto tempMV = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(uniqueDofsMap, 1));
            deltaG0_->addBlock(tempMV, i);

            // auto dofsPerNode = problem_->getDofsPerNode(0);

            // Initialize the underlying IPOUHarmonicCoarseOperator object
            // dofs of a node are lumped together
            int dofOrdering = 0;
            Teuchos::RCP<const Xpetra::Map<LO, GO, NO>> dummyRepeatedNodesMap;
            BuildDofMaps(repeatedDofsMapVec[i], dofsPerNodeVec[i], dofOrdering, dummyRepeatedNodesMap, dofsMapsVec[i],
                         offset);
            FROSCH_ASSERT(repeatedNodesMapVec[i]->isSameAs(*dummyRepeatedNodesMap), "Error with the repeatedNodesMap.");
            offset += repeatedDofsMapVec[i]->getMaxAllGlobalIndex() + 1;

            // FEDD::logGreen("dofsMapsVec in block " + std::to_string(i), this->MpiComm_);
            // auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
            // for (int j = 0; j < dofsPerNodeVec[i]; j++) {
            //     FEDD::logGreen("dofsMaps for dof " + std::to_string(j), this->MpiComm_);
            //     dofsMapsVec[i][j]->describe(*out, VERB_EXTREME);
            // }
            // Build nodeList
            auto nodeListFEDD = mesh->getPointsRepeated();
            // TODO: kho do I need to build the nodeList in a different way?
            nodeListVec[i] =
                Xpetra::MultiVectorFactory<SC, LO, GO>::Build(repeatedNodesMapVec[i], nodeListFEDD->at(i).size());
            for (auto j = 0; j < nodeListFEDD->size(); j++) {
                // pointsRepeated are distributed
                for (auto k = 0; k < nodeListFEDD->at(j).size(); k++) {
                    Teuchos::rcp_const_cast<Xpetra::MultiVector<SC, LO, GO, NO>>(nodeListVec[i])
                        ->getVectorNonConst(k)
                        ->replaceLocalValue(j, nodeListFEDD->at(j).at(k));
                }
            }

            // Build nullspace as in TwoLevelPreconditioner line 195
            NullSpaceType nullSpaceType;
            if (!this->ParameterList_->get("Null Space Type", "Laplace").compare("Laplace")) {
                nullSpaceType = NullSpaceType::Laplace;
            } else if (!this->ParameterList_->get("Null Space Type", "Laplace").compare("Linear Elasticity")) {
                nullSpaceType = NullSpaceType::Elasticity;
            } else {
                FROSCH_ASSERT(false, "Null Space Type unknown.");
            }
            // nullSpaceBasis is a multiVector with one vector for each function in the nullspace e.g. translations
            // and rotations for elasticity. Each vector contains the nullspace function on the repeated map. At
            // this point different nullspaces could be built for each block
            nullSpaceBasisVec[i] = BuildNullSpace(dimension, nullSpaceType, repeatedDofsMapVec[i], dofsPerNodeVec[i],
                                                  dofsMapsVec[i], implicit_cast<ConstXMultiVectorPtr>(nodeListVec[i]));
            // Build the vector of Dirichlet node indices
            // See FROSch::FindOneEntryOnlyRowsGlobal() for reference
            int loc = 0;
            auto tempDirichletBoundaryDofs = Teuchos::rcp(new std::vector<GO>(0));
            for (auto j = 0; j < repeatedNodesMapVec[i]->getLocalNumElements(); j++) {
                auto flag = mesh->bcFlagRep_->at(j);
                // The vector vecFlag_ contains the flags that have been set with addBC().
                // loc: local index in vecFlag_ of the flag if found. Is set by the function.
                // block: block id in which to search. Needs to be provided to the function.
                if (problem_->getBCFactory()->findFlag(flag, i, loc)) {
                    for (auto k = 0; k < dofsPerNodeVec[i]; k++) {
                        tempDirichletBoundaryDofs->push_back(
                            repeatedDofsMapVec[i]->getGlobalElement(dofsPerNodeVec[i] * j + k));
                    }
                }
            }
            dirichletBoundaryDofsVec[i] = Teuchos::arcp<GO>(tempDirichletBoundaryDofs);
        }
        // This builds the coarse spaces, assembles the coarse solve map and does symbolic factorization of the
        // coarse problem
        IPOUHarmonicCoarseOperator<SC, LO, GO, NO>::initialize(dimension, dofsPerNodeVec, repeatedNodesMapVec,
                                                               dofsMapsVec, nullSpaceBasisVec, nodeListVec,
                                                               dirichletBoundaryDofsVec);
        // iPOUHarmonicCoarseOperator<SC, LO, GO, NO>::initialize(dimension, dofsPerNodeVec, repeatedNodesMapVec,
        //                                                     dofsMapsVec, nullSpaceBasisVec, nodeListVec,
        //                                                     dirichletBoundaryDofsVec);
    }
    coarseResidualVec_ =
        Xpetra::MultiVectorFactory<SC, LO, GO>::Build(this->GatheringMaps_[this->GatheringMaps_.size() - 1], 1);
    coarseDeltaG0_ =
        Xpetra::MultiVectorFactory<SC, LO, GO>::Build(this->GatheringMaps_[this->GatheringMaps_.size() - 1], 1);
    deltaG0Merged_ = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(deltaG0_->getMergedVector()->getMap()));

    return 0;
}

template <class SC, class LO, class GO, class NO>
void CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const BlockMultiVectorPtrFEDD x, BlockMultiVectorPtrFEDD y,
                                                           SC alpha, SC beta) {

    FEDD_TIMER_START(CoarseTimer, " - Schwarz - coarse solve");
    TEUCHOS_TEST_FOR_EXCEPTION(!x->getMergedVector()->getMap()->getXpetraMap()->isSameAs(*this->getDomainMap()),
                               std::runtime_error, "input map does not correspond to domain map of nonlinear operator");
    x_ = x;
    // Save problem state
    // Not replacing maps here --> much less needs replacing then in nonlinearSchwarzOperator
    systemTmp_ = problem_->system_;
    solutionTmp_ = problem_->getSolution();
    rhsTmp_ = problem_->rhs_;
    sourceTermTmp_ = problem_->sourceTerm_;
    previousSolutionTmp_ = problem_->previousSolution_;
    residualVecTmp_ = problem_->residualVec_;

    // Reset problem
    problem_->initializeProblem();

    // Solve coarse nonlinear problem
    bool verbose = problem_->getVerbose();
    double residual0 = 1.;
    double relResidual = 1.;
    double absResidual = 1.;
    int nlIts = 0;
    coarseResidualVec_->putScalar(0.);
    coarseDeltaG0_->putScalar(0.);
    deltaG0_->putScalar(0.);
    // Need to initialize the rhs_ and set boundary values in rhs_
    problem_->assemble();
    problem_->setBoundaries();

    // Set solution_ to be u+P_0*g_0. g_0 is zero on the boundary, simulating P_0 locally
    // x_ corresponds to u and problem_->solution_ corresponds to g_0
    // this = alpha*xTmp + beta*this
    problem_->solution_->update(ST::one(), *x_, -ST::one());

    // Need to update solution_ within each iteration to assemble at u+P_0*g_0 but update only g_0
    // This is necessary since u is nonzero on the artificial (interface) zero Dirichlet boundary
    // It would be more efficient to only store u on the boundary and update this value in each iteration
    while (nlIts < maxNumIts_) {

        problem_->calculateNonLinResidualVec("reverse");

        // Restrict the residual to the coarse space
        this->applyPhiT(*problem_->getResidualVector()->getMergedVector()->getXpetraMultiVector(), *coarseResidualVec_);

        Teuchos::Array<SC> residualArray(1);
        coarseResidualVec_->norm2(residualArray());
        absResidual = residualArray[0];

        if (nlIts == 0) {
            if (absResidual < absNewtonTol_) {
                std::cout
                    << "==> Exiting coarse Newton solver immediately: absolute residual is already below the tolerance."
                    << std::endl;
                break; // We are already done
            } else {
                residual0 = absResidual;
            }
        }

        relResidual = absResidual / residual0;
        FEDD::logGreen("Coarse Newton iteration: " + std::to_string(nlIts), this->MpiComm_);
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

        // Update the coarse matrix and the coarse solver (coarse factorization)
        this->K_ = problem_->system_->getMergedMatrix()->getXpetraMatrix();
        this->setUpCoarseOperator();

        // Apply the coarse solution
        this->applyCoarseSolve(*coarseResidualVec_, *coarseDeltaG0_, ETransp::NO_TRANS);
        // TODO: kho fix this in FROSch
        //  Required because applyCoarseSolve switches out the map without restoring initial map. BAD!!
        coarseResidualVec_->replaceMap(this->GatheringMaps_[this->GatheringMaps_.size() - 1]);

        // Project the coarse nonlinear correction update into the global space
        this->applyPhi(*coarseDeltaG0_, *deltaG0Merged_->getXpetraMultiVectorNonConst());
        deltaG0_->setMergedVector(deltaG0Merged_);
        deltaG0_->split();

        // Update the current coarse nonlinear correction g_0
        problem_->solution_->update(ST::one(), *deltaG0_, ST::one());

        nlIts++;
    }

    // Set solution_ to g_i
    problem_->solution_->update(ST::one(), *x_, -ST::one());
    totalIters_ += nlIts;
    FEDD::logGreen("Terminated coarse Newton", this->MpiComm_);

    if (problem_->getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts", false)) {
        TEUCHOS_TEST_FOR_EXCEPTION(nlIts == maxNumIts_, std::runtime_error,
                                   "Maximum nonlinear Iterations reached. Problem might have converged in the last "
                                   "step. Still we cancel here.");
    }

    // y->getBlockNonConst(0)->update(alpha, problem_->getSolution()->getBlock(0), beta);
    y->update(alpha, problem_->getSolution(), beta);
    // Restore problem state
    problem_->initializeProblem();
    problem_->system_ = systemTmp_;
    problem_->solution_ = solutionTmp_;
    problem_->rhs_ = rhsTmp_;
    problem_->sourceTerm_ = sourceTermTmp_;
    problem_->previousSolution_ = previousSolutionTmp_;
    problem_->residualVec_ = residualVecTmp_;
}

template <class SC, class LO, class GO, class NO>
void CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, SC alpha, SC beta) {
    // This version of apply does not make sense for nonlinear operators
    // Wraps another apply() method for compatibility
    // non owning rcp objects since they should not destroy x, y when going out of scope
    auto rcpX = Teuchos::rcp(&x, false);
    auto rcpY = Teuchos::rcp(&y, false);
    auto rcpFEDDX = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(rcpX));
    auto rcpFEDDY = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(rcpY));

    // Using the solution to setup the block vectors correctly. Values are set with setMergedVector()
    auto feddX = Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(problem_->solution_));
    auto feddY = Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(problem_->solution_));
    // This copies pointers (not values) since the pointers are non-const
    feddX->setMergedVector(rcpFEDDX);
    feddY->setMergedVector(rcpFEDDY);
    feddX->split();
    feddY->split();
    apply(feddX, feddY, alpha, beta);
    feddY->merge();
    y.update(alpha, *feddY->getMergedVector()->getXpetraMultiVector(), beta);
}

template <class SC, class LO, class GO, class NO>
void CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y,
                                                           bool usePreconditionerOnly, ETransp mode, SC alpha,
                                                           SC beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "This apply() overload should not be used with nonlinear operators");
}

template <class SC, class LO, class GO, class NO>
void CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::exportCoarseBasis() {

    auto comm = problem_->getComm();
    FEDD::logGreen("Exporting coarse basis functions", comm);

    FEDD::ParameterListPtr_Type pList = problem_->getParameterList();
    FEDD::ParameterListPtr_Type pLCoarse = sublist(pList, "Coarse Nonlinear Schwarz");

    TEUCHOS_TEST_FOR_EXCEPTION(!pLCoarse->isParameter("RCP(Phi)"), std::runtime_error,
                               "No parameter to extract Phi pointer.");

    // This will have numNodes*dofs rows and numCoarseBasisFunctions columns. Each column is a coarse basis function
    // that has a value for each dof at each node.
    Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> phiXpetra = this->Phi_;

    // Convert to a FEDD matrix object
    auto phiMatrix = Teuchos::rcp(new FEDD::Matrix<SC, LO, GO, NO>(phiXpetra));
    // Convert to a FEDD multivector
    Teuchos::RCP<FEDD::MultiVector<SC, LO, GO, NO>> phiMV;
    phiMatrix->toMV(phiMV);

    int numberOfBlocks = pLCoarse->get("Number of blocks", 1);
    std::vector<Teuchos::RCP<const FEDD::Map<LO, GO, NO>>> blockMaps(numberOfBlocks);
    // Build corresponding map object
    for (UN i = 0; i < numberOfBlocks; i++) {
        int dofsPerNode = pLCoarse->get("DofsPerNode" + std::to_string(i + 1), 1);
        Teuchos::RCP<const FEDD::Map<LO, GO, NO>> map;
        if (dofsPerNode > 1) {
            if (!problem_.is_null()) {
                TEUCHOS_TEST_FOR_EXCEPTION(problem_->getDomain(i)->getFEType() == "P0", std::logic_error,
                                           "Vector field map not implemented for P0 elements.");
                map = problem_->getDomain(i)->getMapVecFieldUnique();
            }
        } else {
            if (!problem_.is_null()) {
                TEUCHOS_TEST_FOR_EXCEPTION(problem_->getDomain(i)->getFEType() == "P0", std::logic_error,
                                           "Vector field map not implemented for P0 elements.");
                map = problem_->getDomain(i)->getMapUnique();
            }
        }
        blockMaps[i] = map;
    }

    // Convert into multivector object
    Teuchos::RCP<FEDD::BlockMultiVector<SC, LO, GO, NO>> phiBlockMV =
        Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(blockMaps, phiMV->getNumVectors()));
    phiBlockMV->setMergedVector(phiMV);
    phiBlockMV->split();

    Teuchos::RCP<FEDD::BlockMultiVector<SC, LO, GO, NO>> phiSumBlockMV = phiBlockMV->sumColumns();

    // Export the basis block by block
    for (int i = 0; i < phiBlockMV->size(); i++) {
        bool exportThisBlock =
            !pList->sublist("Exporter").get("Exclude coarse functions block" + std::to_string(i + 1), false);

        if (exportThisBlock) {
            Teuchos::RCP<FEDD::ExporterParaView<SC, LO, GO, NO>> exporter(new FEDD::ExporterParaView<SC, LO, GO, NO>());

            Teuchos::RCP<const FEDD::Domain<SC, LO, GO, NO>> domain;
            if (!problem_.is_null()) {
                domain = problem_->getDomain(i);
            }
            int dofsPerNode = domain->getDimension();
            std::string fileName =
                pList->sublist("Exporter").get("Name coarse functions block" + std::to_string(i + 1), "phi");

            Teuchos::RCP<FEDD::Mesh<SC, LO, GO, NO>> meshNonConst =
                Teuchos::rcp_const_cast<FEDD::Mesh<SC, LO, GO, NO>>(domain->getMesh());
            exporter->setup(fileName, meshNonConst, domain->getFEType());

            for (int j = 0; j < phiBlockMV->getNumVectors(); j++) {

                Teuchos::RCP<const FEDD::MultiVector<SC, LO, GO, NO>> exportPhi = phiBlockMV->getBlock(i)->getVector(j);

                std::string varName = fileName + std::to_string(j);
                if (pLCoarse->get("DofsPerNode" + std::to_string(i + 1), 1) > 1) {
                    exporter->addVariable(exportPhi, varName, "Vector", dofsPerNode, domain->getMapUnique());
                } else {
                    exporter->addVariable(exportPhi, varName, "Scalar", 1, domain->getMapUnique());
                }
            }
            Teuchos::RCP<const FEDD::MultiVector<SC, LO, GO, NO>> exportSumPhi = phiSumBlockMV->getBlock(i);
            std::string varName = fileName + "Sum";
            if (pLCoarse->get("DofsPerNode" + std::to_string(i + 1), 1) > 1) {
                exporter->addVariable(exportSumPhi, varName, "Vector", dofsPerNode, domain->getMapUnique());
            } else {
                exporter->addVariable(exportSumPhi, varName, "Scalar", 1, domain->getMapUnique());
            }
            exporter->save(0.0);
        }
    }
}

template <class SC, class LO, class GO, class NO>
std::vector<SC> CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::getRunStats() const {
    return std::vector<SC>{static_cast<SC>(totalIters_)};
}

template <class SC, class LO, class GO, class NO>
void CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::describe(FancyOStream &out,
                                                              const EVerbosityLevel verbLevel) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "describe() has to be implemented properly...");
}

template <class SC, class LO, class GO, class NO>
string CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::description() const {
    return "Coarse Nonlinear Schwarz Operator";
}

} // namespace FROSch

#endif
