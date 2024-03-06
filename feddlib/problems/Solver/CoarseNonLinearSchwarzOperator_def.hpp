#ifndef COARSENONLINEARSCHWARZOPERATOR_DEF_HPP
#define COARSENONLINEARSCHWARZOPERATOR_DEF_HPP

#include "CoarseNonLinearSchwarzOperator_decl.hpp"
#include "feddlib/core/FE/FE_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/LinearAlgebra/Matrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector_decl.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
#include <FROSch_IPOUHarmonicCoarseOperator_decl.hpp>
#include <Tacho_Driver.hpp>
#include <Teuchos_ArrayRCPDecl.hpp>
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
 Implementation of CoarseNonLinearSchwarzOperator

 @brief Implements the coarse correction T_0 from the nonlinear Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {
template <class SC, class LO, class GO, class NO>
CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::CoarseNonLinearSchwarzOperator(CommPtr mpiComm,
                                                                               ParameterListPtr parameterList,
                                                                               NonLinearProblemPtrFEDD problem)
    : SchwarzOperator<SC, LO, GO, NO>(mpiComm), problem_{problem},
      x_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))},
      y_{Teuchos::rcp(new FEDD::BlockMultiVector<SC, LO, GO, NO>(1))}, newtonTol_{}, maxNumIts_{}, criterion_{""} {

    // Ensure that the mesh object has been initialized and a dual graph generated
    auto domainPtr_vec = problem_->getDomainVector();
    for (auto &domainPtr : domainPtr_vec) {
        TEUCHOS_ASSERT(!domainPtr->getElementMap().is_null());
        TEUCHOS_ASSERT(!domainPtr->getMapRepeated().is_null());
        TEUCHOS_ASSERT(!domainPtr->getDualGraph().is_null());
    }

    // Assigning parent class protected members is not good practice, but is done here to avoid modifying FROSch code
    this->ParameterList_ = parameterList;

    // Initialize members that cannot be null after construction
    x_->addBlock(Teuchos::rcp(new FEDD::MultiVector<SC, LO, GO, NO>(domainPtr_vec.at(0)->getMapOverlappingGhosts(), 1)),
                 0);
}

template <class SC, class LO, class GO, class NO> int CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::initialize() {

    auto domainVec = problem_->getDomainVector();
    auto mesh = domainVec.at(0)->getMesh();
    auto repeatedMap = domainVec.at(0)->getMapRepeated();

    // Extract info from parameterList
    newtonTol_ =
        problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Relative Tolerance", 1.0e-6);
    maxNumIts_ = problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Max Iterations", 10);
    criterion_ = problem_->getParameterList()->sublist("Inner Newton Nonlinear Schwarz").get("Criterion", "Residual");

    // Initialize the underlying IPOUHarmonicCoarseOperator object
    auto coarseParameterList = problem_->getParameterList()->sublist("Coarse Nonlinear Schwarz");
    // TODO: kho dofsMaps need to be built properly for problems with more than one dof per node
    auto dofsMaps = Teuchos::ArrayRCP<ConstXMapPtr>(domainVec.at(0)->getMapRepeated());
    ConstXMultiVectorPtr nullSpaceBasis = null;
    if (coarseParameterList->isParameter("Null Space")) {
        nullSpaceBasis = ExtractPtrFromParameterList<XMultiVector>(*coarseParameterList, "Null Space").getConst();
        if (nullSpaceBasis.is_null()) {
            RCP<Tpetra::MultiVector<SC, LO, GO, NO>> nullSpaceBasisTmp =
                ExtractPtrFromParameterList<Tpetra::MultiVector<SC, LO, GO, NO>>(*coarseParameterList, "Null Space");

            RCP<const Xpetra::TpetraMultiVector<SC, LO, GO, NO>> xTpetraNullSpaceBasis(
                new const Xpetra::TpetraMultiVector<SC, LO, GO, NO>(nullSpaceBasisTmp));
            nullSpaceBasis = rcp_dynamic_cast<ConstXMultiVector>(xTpetraNullSpaceBasis);
        }
    }
    // Build nodeList

    ConstXMultiVectorPtr nodeList = null;
    if (coarseParameterList->isParameter("Coordinates List")) {
        nodeList = ExtractPtrFromParameterList<XMultiVector>(*coarseParameterList, "Coordinates List").getConst();
        if (nodeList.is_null()) {
            RCP<Tpetra::MultiVector<SC, LO, GO, NO>> coordinatesListTmp =
                ExtractPtrFromParameterList<Tpetra::MultiVector<SC, LO, GO, NO>>(*coarseParameterList,
                                                                                 "Coordinates List");

            RCP<const Xpetra::TpetraMultiVector<SC, LO, GO, NO>> xTpetraCoordinatesList(
                new const Xpetra::TpetraMultiVector<SC, LO, GO, NO>(coordinatesListTmp));
            nodeList = rcp_dynamic_cast<ConstXMultiVector>(xTpetraCoordinatesList);
        }
    }
    // Communicate nodeList
    if (!nodeList.is_null()) {
        FROSCH_DETAILTIMER_START_LEVELID(communicateNodeListTime, "Communicate Node List");
        ConstXMapPtr nodeListMap = nodeList->getMap();
        if (!nodeListMap->isSameAs(*repeatedMap)) {
            RCP<MultiVector<SC, LO, GO, NO>> tmpNodeList =
                MultiVectorFactory<SC, LO, GO, NO>::Build(repeatedMap, nodeList->getNumVectors());
            RCP<Import<LO, GO, NO>> scatter = ImportFactory<LO, GO, NO>::Build(nodeListMap, repeatedMap);
            tmpNodeList->doImport(*nodeList, *scatter, INSERT);
            nodeList = tmpNodeList.getConst();
        }
    }
    // TODO: kho does this need to be changed?
    GOVecPtr dirichletBoundaryDofs = null;
    // This builds the coars spaces, assembles the coarse solve map and does symbolic factorization of the coarse
    // problem
    this->initialize(coarseParameterList->get("Dimension", 2), coarseParameterList->get("DofsPerNode", 1), repeatedMap,
                     dofsMaps, nullSpaceBasis, nodeList, dirichletBoundaryDofs);
    return 0;
}

template <class SC, class LO, class GO, class NO>
void CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const BlockMultiVectorPtrFEDD x, BlockMultiVectorPtrFEDD y,
                                                           SC alpha, SC beta) {

    // TODO: kho how do we have to deal with boundary conditions here? What are the dirichletBoundaryDofs used for?

    /* x_->getBlockNonConst(0)->importFromVector(x->getBlock(0), true, "Insert", "Forward"); */
    // TODO: kho is it okay to assign like this? importing should not be necessary here
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
    XMultiVector coarseResidualVec;
    XMultiVector coarseDeltaG0;
    XMultiVector deltaG0;

    // Need to update solution_ within each iteration to assemble at u+P_0*g_0 but update only g_0
    // This is necessary since u is nonzero on the artificial (interface) zero Dirichlet boundary
    // It would be more efficient to only store u on the boundary and update this value in each iteration
    while (nlIts < maxNumIts_) {

        // Set solution_ to be u+P_0*g_0. g_0 is zero on the boundary, simulating P_0 locally
        // this = alpha*xTmp + beta*this
        problem_->solution_->update(ST::one(), *x_, ST::one());
        problem_->calculateNonLinResidualVec("reverse");

        // Restrict the residual to the coarse space
        this->applyPhiT(problem_->getResidualVector(), coarseResidualVec);

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
                // Set solution_ to g_0
                problem_->solution_->update(-ST::one(), *x_, ST::one());
                break;
            }
        }

        problem_->assemble("Newton");

        // After this rows corresponding to Dirichlet nodes are unity and residualVec_ = 0
        // TODO: kho is this the right way to deal with boundary conditions?
        problem_->setBoundariesSystem();

        // Update the coarse matrix and the coarse solver (coarse factorization)
        this->K_ = problem_->system_->getBlock(0, 0);
        this->setUpCoarseOperator();

        // Apply the coarse solution
        this->applyCoarseSolve(coarseResidualVec, coarseDeltaG0, ETransp::NO_TRANS);

        // Project the coarse nonlinear correction update into the global space
        this->applyPhi(coarseDeltaG0, deltaG0);

        // Update the current coarse nonlinear correction g_0
        // TODO: kho the signs might need adjusting
        problem_->solution_->update(ST::one(), deltaG0, ST::one());

        // Changing the solution here changes the result after solveAndUpdate() since the Newton update \delta is added
        // to the current solution
        // Set solution_ to g_i
        problem_->solution_->update(-ST::one(), *x_, ST::one());

    // TODO: kho why is it necessary to update solution to g_i here?
        problem_->solveAndUpdate(criterion_, criterionValue);
        nlIts++;
        if (criterion_ == "Update") {
            FEDD::logGreen("Coarse Newton iteration: " + std::to_string(nlIts), this->MpiComm_);
            FEDD::logGreen("Residual of update: " + std::to_string(criterionValue), this->MpiComm_);
            if (criterionValue < newtonTol_) {
                break;
            }
        }
    }

    FEDD::logGreen("Total inner Newton iters: " + std::to_string(nlIts), this->MpiComm_);

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
void CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y,
                                                           bool usePreconditionerOnly, ETransp mode, SC alpha,
                                                           SC beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "This version of apply does not make sense in the context of nonlinear operators.");
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
