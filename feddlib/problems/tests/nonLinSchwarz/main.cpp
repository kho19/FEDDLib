#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/General/ExporterParaView.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include "feddlib/core/Mesh/MeshPartitioner.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzOperator.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzOperator_decl.hpp"
#include "feddlib/problems/specific/NonLinLaplace.hpp"
#include <FROSch_AlgebraicOverlappingOperator_decl.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Thyra_LinearOpBase_decl.hpp>
#include <Thyra_TpetraThyraWrappers_decl.hpp>
#include <Tpetra_Operator.hpp>
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MultiVectorFactory_decl.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "Epetra_MpiComm.h"

void zeroDirichlet(double *x, double *res, double t, const double *parameters) { res[0] = 0.; }

void oneFunc2D(double *x, double *res, double *parameters) {
    res[0] = 1.;
    res[1] = 1.;
}

void oneFunc3D(double *x, double *res, double *parameters) {
    res[0] = 1.;
    res[1] = 1.;
    res[2] = 1.;
}

typedef unsigned UN;
typedef default_sc SC;
typedef default_lo LO;
typedef default_go GO;
typedef default_no NO;

using namespace FEDD;
using namespace Teuchos;
using namespace std;

namespace NonLinearSchwarz {
typedef MeshUnstructured<SC, LO, GO, NO> MeshUnstr_Type;
typedef RCP<MeshUnstr_Type> MeshUnstrPtr_Type;
typedef Mesh<SC, LO, GO, NO> Mesh_Type;
typedef Teuchos::RCP<Mesh_Type> MeshPtr_Type;
typedef Xpetra::CrsGraph<LO, GO, NO> Graph_Type;
typedef RCP<Graph_Type> GraphPtr_Type;

typedef Domain<SC, LO, GO, NO> Domain_Type;
typedef RCP<Domain_Type> DomainPtr_Type;
typedef ExporterParaView<SC, LO, GO, NO> ExporterPV_Type;
typedef RCP<ExporterPV_Type> ExporterPVPtr_Type;
typedef MeshPartitioner<SC, LO, GO, NO> MeshPartitioner_Type;

typedef Map<LO, GO, NO> Map_Type;
typedef RCP<Map_Type> MapPtr_Type;
typedef MultiVector<SC, LO, GO, NO> MultiVector_Type;
typedef RCP<MultiVector_Type> MultiVectorPtr_Type;
typedef RCP<const MultiVector_Type> MultiVectorConstPtr_Type;
typedef BlockMultiVector<SC, LO, GO, NO> BlockMultiVector_Type;
typedef RCP<BlockMultiVector_Type> BlockMultiVectorPtr_Type;

typedef NonLinearProblem<SC, LO, GO, NO> NonLinearProblem_Type;
typedef RCP<NonLinearProblem_Type> NonLinearProblemPtr_Type;
typedef Teuchos::ScalarTraits<SC> ST;

} // namespace NonLinearSchwarz

using namespace NonLinearSchwarz;

void solve(NonLinearProblemPtr_Type problem);
int solveThyraLinOp(Teuchos::RCP<const Thyra::LinearOpBase<SC>> thyraLinOp, MultiVectorPtr_Type x,
                    MultiVectorConstPtr_Type b, ParameterListPtr_Type parameterList, bool verbose = true);

int main(int argc, char *argv[]) {

    Teuchos::oblackholestream blackhole;
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

    Teuchos::RCP<const Teuchos::Comm<int>> comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();

    // ########################
    // Set default values for command line parameters
    // ########################

    const bool debug = false;
    if (comm->getRank() == 0 && debug) {
        waitForGdbAttach<LO>();
    }
    comm->barrier();

    bool verbose(comm->getRank() == 0);

    if (verbose) {
        cout << "###############################################################" << endl;
        cout << "########## Starting nonlinear graph partitioning ... ##########" << endl;
        cout << "###############################################################" << endl;
    }

    string xmlProblemFile = "parametersProblem.xml";
    string xmlPrecFile = "parametersPrec.xml";
    string xmlSolverFile = "parametersSolver.xml";

    ParameterListPtr_Type parameterListProblem = Teuchos::getParametersFromXmlFile(xmlProblemFile);
    ParameterListPtr_Type parameterListPrec = Teuchos::getParametersFromXmlFile(xmlPrecFile);
    ParameterListPtr_Type parameterListSolver = Teuchos::getParametersFromXmlFile(xmlSolverFile);

    ParameterListPtr_Type parameterListAll(new Teuchos::ParameterList(*parameterListProblem));
    parameterListAll->setParameters(*parameterListPrec);
    parameterListAll->setParameters(*parameterListSolver);

    int dim = parameterListProblem->sublist("Parameter").get("Dimension", 2);
    string meshType = parameterListProblem->sublist("Parameter").get("Mesh Type", "unstructured");
    // m = dimension of square on each processor in num elements
    int m = parameterListProblem->sublist("Parameter").get("H/h", 5);
    string FEType = parameterListProblem->sublist("Parameter").get("Discretization", "P1");

    int numProcsCoarseSolve = parameterListProblem->sublist("General").get("Mpi Ranks Coarse", 0);
    int size = comm->getSize() - numProcsCoarseSolve;

    double length = 4;
    int minNumberSubdomains = 1;
    // n = number of procs per side
    int n;
    m = 1;

    // ########################
    // Build mesh
    // ########################
    DomainPtr_Type domain;
    if (!meshType.compare("unstructured")) {
        Teuchos::RCP<Domain<SC, LO, GO, NO>> domainP1;
        domainP1.reset(new Domain<SC, LO, GO, NO>(comm, dim));

        MeshPartitioner_Type::DomainPtrArray_Type domainP1Array(1);
        domainP1Array[0] = domainP1;

        ParameterListPtr_Type pListPartitioner = sublist(parameterListAll, "Mesh Partitioner");
        MeshPartitioner<SC, LO, GO, NO> partitionerP1(domainP1Array, pListPartitioner, "P1", dim);

        partitionerP1.readMesh();
        partitionerP1.buildDualGraph(0);
        partitionerP1.partitionDualGraphWithOverlap(0, 2);
        partitionerP1.buildSubdomainFEsAndNodeLists(0);

        domain = domainP1;
    } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                                   "Can only use unstructured mesh with nonlinear Schwarz for now.");
    }
    // ########################
    // Set flags for the boundary conditions
    // ########################

    Teuchos::RCP<BCBuilder<SC, LO, GO, NO>> bcFactory(new BCBuilder<SC, LO, GO, NO>());
    bcFactory->addBC(zeroDirichlet, 1, 0, domain, "Dirichlet", 1);
    bcFactory->addBC(zeroDirichlet, 2, 0, domain, "Dirichlet", 1);
    bcFactory->addBC(zeroDirichlet, 3, 0, domain, "Dirichlet", 1);
    bcFactory->addBC(zeroDirichlet, 4, 0, domain, "Dirichlet", 1);
    // This boundary condition must be set for nonlinear Schwarz solver to correctly solve on the subdomains
    bcFactory->addBC(zeroDirichlet, -99, 0, domain, "Dirichlet", 1);

    auto nonLinLaplace = Teuchos::rcp(new NonLinLaplace<SC, LO, GO, NO>(domain, FEType, parameterListAll));

    nonLinLaplace->addBoundaries(bcFactory);

    if (dim == 2) {
        nonLinLaplace->addRhsFunction(oneFunc2D);
    } else if (dim == 3) {
        nonLinLaplace->addRhsFunction(oneFunc3D);
    }
    // Initializes the system matrix (no values) and initializes the solution, rhs, residual vectors and splits them
    // between the subdomains
    nonLinLaplace->initializeProblem();
    // Set initial guess
    nonLinLaplace->solution_->putScalar(0.);

    // ########################
    // Solve the problem using nonlinear Schwarz
    // ########################

    solve(nonLinLaplace);

    comm->barrier();

    // Export Solution
    bool boolExportSolution = true;
    if (boolExportSolution) {
        Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exPara(new ExporterParaView<SC, LO, GO, NO>());

        Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolution = nonLinLaplace->getSolution()->getBlock(0);

        /* logGreen("Solution map for paraview", comm); */
        /* exportSolution->getMap()->print(); */
        /* logGreen("Solution data", comm); */
        /* exportSolution->print(); */

        exPara->setup("solutionNonLinSchwarz", domain->getMesh(), FEType);
        exPara->addVariable(exportSolution, "u", "Scalar", 1, domain->getMapUnique(), domain->getMapUniqueP2());
        exPara->save(0.0);
    }

    return (EXIT_SUCCESS);
}

// Solve the problem
void solve(NonLinearProblemPtr_Type problem) {

    logGreen("Commencing nonlinear Schwarz solve", problem->getComm());
    // NOTE: KHo turn on/off printing
    /* problem->verbose_ = false; */
    // Define nonlinear Schwarz operator
    auto domainVec = problem->getDomainVector();
    auto mpiComm = domainVec.at(0)->getComm();
    auto NonLinearSchwarzOp = Teuchos::rcp(
        new FROSch::NonLinearSchwarzOperator<SC, LO, GO, NO>(mpiComm, problem->getParameterList(), problem));

    // FROSch overlapping operator
    // TODO KHo fill with dummy matrix and replace below
    auto dummyMat = Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(domainVec.at(0)->getMapUnique()->getXpetraMap());
    dummyMat->fillComplete();
    auto FROSchOverlappingOperator =
        Teuchos::rcp(new FROSch::AlgebraicOverlappingOperator<SC, LO, GO, NO>(dummyMat, problem->getParameterList()));

    // Init block vectors for nonlinear residual and outer Newton update
    auto g = Teuchos::rcp(new BlockMultiVector_Type(1));
    auto gBlock = Teuchos::rcp(new MultiVector_Type(domainVec.at(0)->getMesh()->getMapUnique(), 1));
    gBlock->putScalar(0.);
    g->addBlock(gBlock, 0);
    auto deltaSolution = Teuchos::rcp(new BlockMultiVector_Type(1));
    auto deltaSolutionBlock = Teuchos::rcp(new MultiVector_Type(domainVec.at(0)->getMesh()->getMapUnique(), 1));
    deltaSolutionBlock->putScalar(0.);
    deltaSolution->addBlock(deltaSolutionBlock, 0);

    // Define convergence requirements
    double gmresIts = 0.;
    double residual0 = 1.;
    double residual = 1.;
    double outerTol = problem->getParameterList()->sublist("Parameter").get("relNonLinTol", 1.0e-6);
    int outerNonLinIts = 0;
    int maxOuterNonLinIts = problem->getParameterList()->sublist("Parameter").get("MaxNonLinIts", 10);
    auto relativeResidual = residual / residual0;
    // TODO: KHo fetch from paremeter list
    auto overlap = 2;

    writeParameterListToXmlFile(*problem->getParameterList(), "debug_xml_file.xml");

    // Fill solution with values corresponding to global index
    /* for (int i = 0; i < problem->solution_->getBlock(0)->getLocalLength(); i++) { */
    /*     problem->solution_->getBlockNonConst(0)->replaceLocalValue( */
    /*         i, 0, problem->solution_->getBlock(0)->getMap()->getGlobalElement(i)); */
    /* } */

    /* logGreen("Increasing solution", mpiComm); */
    /* problem->solution_->getBlock(0)->print(); */

    // Compute the residual
    problem->calculateNonLinResidualVec("reverse");
    residual0 = problem->calculateResidualNorm();
    residual = residual0;
    relativeResidual = residual / residual0;
    // Outer Newton iterations
    NonLinearSchwarzOp->initialize();
    while (relativeResidual > outerTol && outerNonLinIts < 1) {
        logGreen("Outer Newton iteration: " + std::to_string(outerNonLinIts), mpiComm);
        logSimple("Residual: " + std::to_string(relativeResidual), mpiComm);
        // Compute current g
        // Must call initialize here again to avoid overwriting maps etc of problem
        // TODO: KHo for ASPEN the Schwarz Op should store the DF(u_i) and stitch them together and return them upon
        // request as a Thyra linear op
        // TODO: KHo probably can call initialize() only once since pointers are stored so value updates will also be
        // stored.
        logGreen("Computing nonlinear Schwarz operator", mpiComm);
        /* problem->solution_->putScalar(0.); */
        NonLinearSchwarzOp->apply(problem->getSolution(), g);

        // TODO: KHo need to incorporate the boundary conditions here in some further manner? Possibly also in the
        // Schwarz operator below?
        // TODO: KHo assembleFEElements_ needs to be different for global assembly and local assembly.
        // Problem->feFactory_->assembleFEElements_

        // Compute DF(u) (ASPIN)
        logGreen("Building ASPIN tangent with FROSch", mpiComm);
        problem->assemble("Newton");
        problem->setBoundariesSystem();

        // Compute D\mathcal{F}(u) using FROSch and DF(u)
        auto jacobian = problem->getSystem()->getBlock(0, 0)->getXpetraMatrix();
        /* logGreen("Jacobian", mpiComm); */
        /* problem->getSystem()->getBlock(0, 0)->print(); */
        FROSchOverlappingOperator->resetMatrix(jacobian);
        FROSchOverlappingOperator->initialize(overlap);
        FROSchOverlappingOperator->compute();

        /* const auto out = Teuchos::VerboseObjectBase::getDefaultOStream(); */
        /* auto testVec = Xpetra::MultiVectorFactory<SC, LO, GO, NO>::Build(g->getBlock(0)->getMapXpetra(), 1); */
        /* FROSchOverlappingOperator->apply(*g->getBlock(0)->getXpetraMultiVector(), *testVec, true); */
        /* logGreen("g", mpiComm); */
        /* g->print(); */
        /* logGreen("A*g", mpiComm); */
        /* testVec->describe(*out, Teuchos::VERB_EXTREME); */
        /* return; */

        // Convert SchwarzOperator to Thyra::LinearOpBase
        auto XpetraFROSchOverlappingOperator =
            rcp_dynamic_cast<Xpetra::Operator<SC, LO, GO, NO>>(FROSchOverlappingOperator);

        // TODO: KHo is the overlap being used by the FROSch operator the same?
        RCP<FROSch::TpetraPreconditioner<SC, LO, GO, NO>> TpetraFROSchOverlappingOperator(
            new FROSch::TpetraPreconditioner<SC, LO, GO, NO>(XpetraFROSchOverlappingOperator));

        auto TpetraOverlappingOperator =
            rcp_dynamic_cast<Tpetra::Operator<SC, LO, GO, NO>>(TpetraFROSchOverlappingOperator);

        auto ThyraFROSchOverlappingOperator = Thyra::createLinearOp(TpetraOverlappingOperator);

        // Solve linear system with GMRES
        logGreen("Solving outer linear system", mpiComm);
        solveThyraLinOp(ThyraFROSchOverlappingOperator, deltaSolution->getBlockNonConst(0), g->getBlock(0),
                        problem->getParameterList());
        // Update the current solution
        // solution = alpha * deltaSolution + beta * solution
        problem->solution_->update(ST::one(), deltaSolution, ST::one());
        logGreen("Update @ iteration " + std::to_string(outerNonLinIts), mpiComm);
        g->getBlock(0)->print();

        /* problem->solution_->update(ST::one(), g, ST::zero()); */
        // Compute the residual
        problem->calculateNonLinResidualVec("reverse");
        residual = problem->calculateResidualNorm();
        relativeResidual = residual / residual0;

        outerNonLinIts += 1;
    }
    logGreen("Outer Newton iteration terminated with residual: " + std::to_string(relativeResidual), mpiComm);
}

int solveThyraLinOp(Teuchos::RCP<const Thyra::LinearOpBase<SC>> thyraLinOp, MultiVectorPtr_Type x,
                    MultiVectorConstPtr_Type b, ParameterListPtr_Type parameterList, bool verbose) {
    int its = 0;
    x->putScalar(0.);

    auto thyraX = x->getThyraMultiVector();

    auto thyraB = b->getThyraMultiVectorConst();

    auto pListThyraSolver =
        sublist(sublist(parameterList, "Outer Newton Nonlinear Schwarz"), "Thyra Solver Outer Newton");

    writeParameterListToXmlFile(*pListThyraSolver, "thyra_solver_xml_file.xml");
    auto linearSolverBuilder = Teuchos::rcp(new Stratimikos::DefaultLinearSolverBuilder());
    linearSolverBuilder->setParameterList(pListThyraSolver);
    auto lowsFactory = linearSolverBuilder->createLinearSolveStrategy("");

    auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
    lowsFactory->setOStream(out);
    lowsFactory->setVerbLevel(Teuchos::VERB_LOW);

    auto solver = lowsFactory->createOp();
    Thyra::initializeOp<SC>(*lowsFactory, thyraLinOp, solver.ptr());

    Thyra::SolveStatus<SC> status = Thyra::solve<SC>(*solver, Thyra::NOTRANS, *thyraB, thyraX.ptr());

    if (verbose) {
        std::cout << status << std::endl;
    }
    if (!pListThyraSolver->get("Linear Solver Type", "Belos").compare("Belos")) {
        its = status.extraParameters->get("Belos/Iteration Count", 0);
    } else {
        its = 0;
    }
    return its;
}
