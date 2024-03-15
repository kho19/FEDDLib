#ifndef COARSENONLINEARSCHWARZOPERATOR_DEF_HPP
#define COARSENONLINEARSCHWARZOPERATOR_DEF_HPP

#include "CoarseNonLinearSchwarzOperator_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector_decl.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
#include <FROSch_IPOUHarmonicCoarseOperator_decl.hpp>
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
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MultiVectorFactory_decl.hpp>
#include <stdexcept>
#include <string>

/*!
 Implementation of CoarseNonLinearSchwarzOperator

 @brief Implements the coarse correction T_0 from the nonlinear Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

// TODO: kho how to pass in a block system? might need to construct a monolithic matrix out of the block for this? The
// initially passed in matrix is used to build the coarse spaces

// The communicator for this object is taken from the matrix passed to IPOUHarmonicCoarseOperator
template <class SC, class LO, class GO, class NO>
CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::CoarseNonLinearSchwarzOperator(NonLinearProblemPtrFEDD problem,
                                                                               ParameterListPtr parameterList)
    : IPOUHarmonicCoarseOperator<SC, LO, GO, NO>(problem->system_->getBlock(0, 0)->getXpetraMatrix(), parameterList),
      problem_{problem}, x_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      y_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      coarseJacobian_{Teuchos::rcp(new FEDD::BlockMatrix<SC, LO, GO, NO>(1))}, newtonTol_{}, maxNumIts_{},
      criterion_{""}, solutionTmp_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      systemTmp_{Teuchos::rcp(new FEDD::BlockMatrix<SC, LO, GO, NO>(1))} {

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

    auto domainVec = problem_->getDomainVector();
    auto mesh = domainVec.at(0)->getMesh();
    auto repeatedMap = domainVec.at(0)->getMapRepeated()->getXpetraMap();

    // Extract info from parameterList
    newtonTol_ =
        problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Relative Tolerance", 1.0e-6);
    maxNumIts_ = problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Max Iterations", 10);
    criterion_ = problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Criterion", "Residual");
    auto coarseParameterList = problem_->getParameterList()->sublist("Coarse Nonlinear Schwarz");
    auto dimension = problem_->getParameterList()->sublist("Parameter").get("Dimension", 2);
    auto dofsPerNode = problem_->getDofsPerNode(0);
    // Initialize the underlying IPOUHarmonicCoarseOperator object
    // TODO: kho dofsMaps need to be built properly for problems with more than one dof per node
    auto dofsMaps = Teuchos::ArrayRCP<ConstXMapPtr>(1);
    dofsMaps[0] = domainVec.at(0)->getMapRepeated()->getXpetraMap();

    // Build nodeList
    // TODO: kho get these from the problem object
    auto nodeListFEDD = mesh->getPointsRepeated();
    auto nodeList = Xpetra::MultiVectorFactory<SC, LO, GO>::Build(repeatedMap, nodeListFEDD->at(0).size());
    for (auto i = 0; i < nodeListFEDD->size(); i++) {
        // pointsRepeated are distributed
        for (auto j = 0; j < nodeListFEDD->at(0).size(); j++) {
            nodeList->getVectorNonConst(j)->replaceLocalValue(i, nodeListFEDD->at(i).at(j));
        }
    }

    // Build nullspace as in TwoLevelPreconditioner line 195
    NullSpaceType nullSpaceType;
    if (!coarseParameterList.get("Null Space Type", "Laplace").compare("Laplace")) {
        nullSpaceType = NullSpaceType::Laplace;
    } else if (!coarseParameterList.get("Null Space Type", "Laplace").compare("Linear Elasticity")) {
        nullSpaceType = NullSpaceType::Elasticity;
    } else {
        FROSCH_ASSERT(false, "Null Space Type unknown.");
    }
    auto nullSpaceBasis = BuildNullSpace(dimension, nullSpaceType, repeatedMap, dofsPerNode, dofsMaps,
                                         implicit_cast<ConstXMultiVectorPtr>(nodeList));

    // TODO: kho figure out when these are needed
    GOVecPtr dirichletBoundaryDofs = null;
    // This builds the coars spaces, assembles the coarse solve map and does symbolic factorization of the coarse
    // problem
    IPOUHarmonicCoarseOperator<SC, LO, GO, NO>::initialize(
        dimension, dofsPerNode, repeatedMap, dofsMaps, nullSpaceBasis, implicit_cast<ConstXMultiVectorPtr>(nodeList),
        dirichletBoundaryDofs);
    return 0;
}

template <class SC, class LO, class GO, class NO>
void CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const BlockMultiVectorPtrFEDD x, BlockMultiVectorPtrFEDD y,
                                                           SC alpha, SC beta) {

    // TODO: kho how do we have to deal with boundary conditions here? What are the dirichletBoundaryDofs used for?
    TEUCHOS_TEST_FOR_EXCEPTION(x->getBlock(0)->getMapXpetra()->isSameAs(*this->getDomainMap()), std::runtime_error,
                               "input map does not correspond to domain map of nonlinear operator");
    x_ = x;

    // Save problem state
    solutionTmp_ = problem_->getSolution();
    systemTmp_ = problem_->system_;

    // Reset problem
    problem_->initializeProblem();

    // Solve coarse nonlinear problem
    bool verbose = problem_->getVerbose();
    double residual0 = 1.;
    double residual = 1.;
    int nlIts = 0;
    double criterionValue = 1.;
    auto coarseResidualVec =
        Xpetra::MultiVectorFactory<SC, LO, GO>::Build(this->GatheringMaps_[this->GatheringMaps_.size() - 1], 1);
    auto coarseDeltaG0 =
        Xpetra::MultiVectorFactory<SC, LO, GO>::Build(this->GatheringMaps_[this->GatheringMaps_.size() - 1], 1);
    auto tempMV = Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(problem_->getDomain(0)->getMapUnique(), 1));
    auto deltaG0 = Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1));
    deltaG0->addBlock(tempMV, 0);

    // Set solution_ to be u+P_0*g_0. g_0 is zero on the boundary, simulating P_0 locally
    // this = alpha*xTmp + beta*this
    problem_->solution_->update(ST::one(), *x_, ST::one());

    // Need to update solution_ within each iteration to assemble at u+P_0*g_0 but update only g_0
    // This is necessary since u is nonzero on the artificial (interface) zero Dirichlet boundary
    // It would be more efficient to only store u on the boundary and update this value in each iteration
    while (nlIts < maxNumIts_) {

        problem_->calculateNonLinResidualVec("reverse");
        auto out = Teuchos::VerboseObjectBase::getDefaultOStream();

        // Restrict the residual to the coarse space
        this->applyPhiT(*problem_->getResidualVector()->getBlock(0)->getXpetraMultiVector(), *coarseResidualVec);

        if (criterion_ == "Residual") {
            Teuchos::Array<SC> residualArray(1);
            coarseResidualVec->norm2(residualArray());
            residual = residualArray[0];
        }

        if (nlIts == 0) {
            residual0 = residual;
        }

        if (criterion_ == "Residual") {
            criterionValue = residual / residual0;
            FEDD::logGreen("Coarse Newton iteration: " + std::to_string(nlIts), this->MpiComm_);
            FEDD::logGreen("Relative residual: " + std::to_string(criterionValue), this->MpiComm_);

            if (criterionValue < newtonTol_) {
                break;
            }
        }
        problem_->assemble("Newton");

        // After this rows corresponding to Dirichlet nodes are unity and residualVec_ = 0
        // TODO: kho is this the right way to deal with boundary conditions?
        problem_->setBoundariesSystem();

        // Update the coarse matrix and the coarse solver (coarse factorization)
        this->K_ = problem_->system_->getBlock(0, 0)->getXpetraMatrix();
        this->setUpCoarseOperator();

        // Apply the coarse solution
        this->applyCoarseSolve(*coarseResidualVec, *coarseDeltaG0, ETransp::NO_TRANS);
        // Required because applyCoarseSolve switches out the map without restoring initial map. BAD!!
        coarseResidualVec->replaceMap(this->GatheringMaps_[this->GatheringMaps_.size() - 1]);

        // Project the coarse nonlinear correction update into the global space
        this->applyPhi(*coarseDeltaG0, *deltaG0->getBlockNonConst(0)->getXpetraMultiVectorNonConst());

        // Update the current coarse nonlinear correction g_0
        // TODO: kho the signs might need adjusting
        problem_->solution_->update(ST::one(), *deltaG0, ST::one());

        nlIts++;
        if (criterion_ == "Update") {
            FEDD::logGreen("Coarse Newton iteration: " + std::to_string(nlIts), this->MpiComm_);
            FEDD::logGreen("Residual of update: " + std::to_string(criterionValue), this->MpiComm_);
            if (criterionValue < newtonTol_) {
                break;
            }
        }
    }

    // Set solution_ to g_i
    problem_->solution_->update(-ST::one(), *x_, ST::one());

    FEDD::logGreen("Total coarse Newton iters: " + std::to_string(nlIts), this->MpiComm_);

    if (problem_->getParameterList()->sublist("Parameter").get("Cancel MaxNonLinIts", false)) {
        TEUCHOS_TEST_FOR_EXCEPTION(nlIts == maxNumIts_, std::runtime_error,
                                   "Maximum nonlinear Iterations reached. Problem might have converged in the last "
                                   "step. Still we cancel here.");
    }

    // The currently assembled Jacobian is from the previous Newton iteration. The error this causes is negligable.
    // Worth it since a reassemble is avoided.
    coarseJacobian_->addBlock(problem_->system_->getBlock(0, 0), 0, 0);
    y->getBlockNonConst(0)->update(alpha, problem_->getSolution()->getBlock(0), beta);
    // Restore problem state
    problem_->initializeProblem();
    problem_->solution_ = solutionTmp_;
    problem_->system_ = systemTmp_;
}

template <class SC, class LO, class GO, class NO>
void CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, SC alpha, SC beta) {
    // This version of apply does not make sense for nonlinear operators
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
void CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y,
                                                           bool usePreconditionerOnly, ETransp mode, SC alpha,
                                                           SC beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "This apply() overload should not be used with nonlinear operators");
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

template <class SC, class LO, class GO, class NO>
Teuchos::RCP<FEDD::BlockMatrix<SC, LO, GO, NO>>
CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::getCoarseJacobian() const {
    return coarseJacobian_;
}
} // namespace FROSch

#endif
