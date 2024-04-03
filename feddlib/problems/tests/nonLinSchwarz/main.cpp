#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include "feddlib/core/LinearAlgebra/Matrix_decl.hpp"
#include "feddlib/core/Mesh/MeshPartitioner_decl.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/CoarseNonLinearSchwarzOperator_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearOperator_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearSchwarzOperator_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearSumOperator_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/SimpleCoarseOperator_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/SimpleOverlappingOperator_decl.hpp"
#include "feddlib/problems/specific/NonLinLaplace_decl.hpp"
#include <FROSch_AlgebraicOverlappingOperator_decl.hpp>
#include <FROSch_ExtractSubmatrices_decl.hpp>
#include <FROSch_OverlappingOperator_decl.hpp>
#include <FROSch_SchwarzOperator_decl.hpp>
#include <FROSch_SumOperator_decl.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_BLAS_types.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Thyra_LinearOpBase_decl.hpp>
#include <Thyra_TpetraThyraWrappers_decl.hpp>
#include <Tpetra_Operator.hpp>
#include <Xpetra_ConfigDefs.hpp>
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#include <Xpetra_Import.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MultiVectorFactory_decl.hpp>
#include <stdexcept>
#include <string>
#include <vector>

void zeroDirichlet(double *x, double *res, double t, const double *parameters) { res[0] = 0.; }

// Required for setting the Dirichlet BC on the ghost points to the current global solution in nonlinear Schwarz
void currentSolutionDirichlet(double *x, double *res, double t, const double *parameters) { res[0] = x[0]; }

void oneFunc2D(double *x, double *res, double *parameters) {
    res[0] = 1.;
    res[1] = 1.;
}

void oneFunc3D(double *x, double *res, double *parameters) {
    res[0] = 1.;
    res[1] = 1.;
    res[2] = 1.;
}

void initialValue2D(double *x, double *res, double *parameters) { res[0] = x[0] * x[1] * (1 - x[0]) * (1 - x[1]); }

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

enum class NonlinSchwarzVariant { ASPIN, ASPEN };
void solve(NonLinearProblemPtr_Type problem, int overlap = 2);
int solveThyraLinOp(Teuchos::RCP<const Thyra::LinearOpBase<SC>> thyraLinOp, MultiVectorPtr_Type x,
                    MultiVectorConstPtr_Type b, ParameterListPtr_Type parameterList, bool verbose = true);

int main(int argc, char *argv[]) {

    Teuchos::oblackholestream blackhole;
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

    Teuchos::RCP<const Teuchos::Comm<int>> comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();

    // ########################
    // Set default values for command line parameters
    // ########################

    bool debug = false;
    if (argc > 1) {
        debug = true;
    }
    if (comm->getRank() == 1 && debug) {
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
    string FEType = parameterListProblem->sublist("Parameter").get("Discretization", "P1");
    auto overlap = parameterListSolver->get("Overlap", 1);

    // ########################
    // Build mesh
    // ########################
    DomainPtr_Type domain;
    Teuchos::RCP<Domain<SC, LO, GO, NO>> domainP1;
    domainP1.reset(new Domain<SC, LO, GO, NO>(comm, dim));

    MeshPartitioner_Type::DomainPtrArray_Type domainP1Array(1);
    domainP1Array[0] = domainP1;

    ParameterListPtr_Type pListPartitioner = sublist(parameterListAll, "Mesh Partitioner");
    MeshPartitioner<SC, LO, GO, NO> partitionerP1(domainP1Array, pListPartitioner, "P1", dim);

    partitionerP1.readMesh();
    partitionerP1.buildDualGraph(0);
    partitionerP1.partitionDualGraphWithOverlap(0, overlap);
    partitionerP1.buildSubdomainFEsAndNodeLists(0);

    domain = domainP1;

    // ########################
    // Set flags for the boundary conditions
    // ########################

    Teuchos::RCP<BCBuilder<SC, LO, GO, NO>> bcFactory(new BCBuilder<SC, LO, GO, NO>());
    bcFactory->addBC(zeroDirichlet, 1, 0, domain, "Dirichlet", 1);
    bcFactory->addBC(zeroDirichlet, 2, 0, domain, "Dirichlet", 1);
    bcFactory->addBC(zeroDirichlet, 3, 0, domain, "Dirichlet", 1);
    bcFactory->addBC(zeroDirichlet, 4, 0, domain, "Dirichlet", 1);
    // The current global solution must be set as the Dirichlet BC on the ghost nodes for nonlinear Schwarz solver to
    // correctly solve on the subdomains
    bcFactory->addBC(currentSolutionDirichlet, -99, 0, domain, "Dirichlet", 1);

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

    // ########################
    // Solve the problem using nonlinear Schwarz
    // ########################

    solve(nonLinLaplace, overlap);

    comm->barrier();

    // Export Solution
    bool boolExportSolution = true;
    if (boolExportSolution) {
        Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exPara(new ExporterParaView<SC, LO, GO, NO>());

        Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolution = nonLinLaplace->getSolution()->getBlock(0);

        exPara->setup("solutionNonLinSchwarz", domain->getMesh(), FEType);
        exPara->addVariable(exportSolution, "u", "Scalar", 1, domain->getMapUnique(), domain->getMapUniqueP2());
        exPara->save(0.0);
    }

    return (EXIT_SUCCESS);
}

// Solve the problem
void solve(NonLinearProblemPtr_Type problem, int overlap) {

    auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
    logGreen("Commencing nonlinear Schwarz solve", problem->getComm());

    // Define nonlinear Schwarz operator
    auto domainVec = problem->getDomainVector();
    auto mpiComm = domainVec.at(0)->getComm();
    auto serialComm = Teuchos::createSerialComm<LO>();

    auto variantString = problem->getParameterList()->get("Nonlin Schwarz Variant", "ASPEN");
    NonlinSchwarzVariant variant;
    if (variantString == "ASPEN") {
        variant = NonlinSchwarzVariant::ASPEN;
    } else if (variantString == "ASPIN") {
        variant = NonlinSchwarzVariant::ASPIN;
    } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Invalid nonlinear Schwarz solver type");
    }
    problem->initSolutionWithFunction(initialValue2D, 0, std::vector<double>{0});

    // The coarse space is built using this Jacobian
    problem->assemble("Newton");
    problem->setBoundariesSystem();

    // The operators
    auto nonLinearSchwarzOp = Teuchos::rcp(
        new FROSch::NonLinearSchwarzOperator<SC, LO, GO, NO>(serialComm, problem, problem->getParameterList()));
    nonLinearSchwarzOp->initialize();
    auto coarseOperator = Teuchos::rcp(new FROSch::CoarseNonLinearSchwarzOperator<SC, LO, GO, NO>(
        problem, sublist(problem->getParameterList(), "Coarse Nonlinear Schwarz")));
    coarseOperator->initialize();
    coarseOperator->compute();
    /* coarseOperator->exportCoarseBasis(); */

    // When using auto, this is an rcp to a const NonLinearSumOperator. We need non-const here to ensure that the
    // non-const (nonlinear) apply() overload is called
    Teuchos::RCP<FROSch::NonLinearSumOperator<SC, LO, GO, NO>> rhsSumOperator =
        Teuchos::rcp(new FROSch::NonLinearSumOperator<SC, LO, GO, NO>(mpiComm));
    rhsSumOperator->addOperator(rcp_implicit_cast<FROSch::NonLinearOperator<SC, LO, GO, NO>>(nonLinearSchwarzOp));
    rhsSumOperator->addOperator(rcp_implicit_cast<FROSch::NonLinearOperator<SC, LO, GO, NO>>(coarseOperator));

    auto simpleOverlappingOperator = Teuchos::rcp(new FROSch::SimpleOverlappingOperator<SC, LO, GO, NO>(
        problem->system_->getBlock(0, 0)->getXpetraMatrix(), problem->getParameterList()));
    auto simpleCoarseOperator = Teuchos::rcp(new FROSch::SimpleCoarseOperator<SC, LO, GO, NO>(
        problem->system_->getBlock(0, 0)->getXpetraMatrix(), problem->getParameterList()));
    simpleCoarseOperator->initialize(coarseOperator);
    auto simpleSumOperator = Teuchos::rcp(new FROSch::SumOperator<SC, LO, GO, NO>(mpiComm));
    simpleSumOperator->addOperator(simpleOverlappingOperator);
    simpleSumOperator->addOperator(simpleCoarseOperator);

    // Init block vectors for nonlinear residual and outer Newton update
    auto gBlock = Teuchos::rcp(new MultiVector_Type(domainVec.at(0)->getMesh()->getMapUnique(), 1));
    gBlock->putScalar(0.);
    auto g = Teuchos::rcp(new BlockMultiVector_Type(1));
    g->addBlock(gBlock, 0);
    auto deltaSolutionBlock = Teuchos::rcp(new MultiVector_Type(domainVec.at(0)->getMesh()->getMapUnique(), 1));
    deltaSolutionBlock->putScalar(0.);
    auto deltaSolution = Teuchos::rcp(new BlockMultiVector_Type(1));
    deltaSolution->addBlock(deltaSolutionBlock, 0);

    Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> jacobianGhosts;
    Teuchos::RCP<Xpetra::Import<LO, GO>> uniqueToOverlappingGhostsImporter;
    RCP<ParameterList> params = parameterList();
    if (variant == NonlinSchwarzVariant::ASPIN) {
        jacobianGhosts = Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(
            domainVec.at(0)->getMapOverlappingGhosts()->getXpetraMap(), domainVec.at(0)->getApproxEntriesPerRow());
        uniqueToOverlappingGhostsImporter =
            Xpetra::ImportFactory<LO, GO>::Build(domainVec.at(0)->getMapUnique()->getXpetraMap(),
                                                 domainVec.at(0)->getMapOverlappingGhosts()->getXpetraMap());
        // So we can call doImport after calling resumeFill on a fillComplete matrix
        params->set("Optimize Storage", false);
    }

    // Define convergence requirements
    double gmresIts = 0.;
    double outerTol = problem->getParameterList()->sublist("Parameter").get("relNonLinTol", 1.0e-6);
    int outerNonLinIts = 0;
    int maxOuterNonLinIts = problem->getParameterList()->sublist("Parameter").get("MaxNonLinIts", 10);

    // Compute the residual
    problem->calculateNonLinResidualVec("reverse");
    auto residual0 = problem->calculateResidualNorm();
    auto residual = residual0;
    auto relativeResidual = residual / residual0;

    // Outer Newton iterations
    while (relativeResidual > outerTol && outerNonLinIts < maxOuterNonLinIts) {
        logGreen("Starting outer Newton iteration: " + std::to_string(outerNonLinIts), mpiComm);
        logSimple("Residual: " + std::to_string(relativeResidual), mpiComm);

        // Compute the residual of the alternative problem \mathcal{F} = g
        // g fulfills the boundary conditions
        // TODO: kho apply the sum operator here resulting in g
        logGreen("Computing nonlinear Schwarz operator", mpiComm);
        rhsSumOperator->apply(*problem->getSolution()->getBlock(0)->getXpetraMultiVector(),
                              *g->getBlockNonConst(0)->getXpetraMultiVectorNonConst());
        // Update SimpleCoarseOperator to ensure it wraps the current CoarseNonLinearSchwarzOperator
        simpleCoarseOperator->initialize(coarseOperator);

        if (variant == NonlinSchwarzVariant::ASPEN) {
            logGreen("Building ASPEN tangent", mpiComm);

            auto localJacobian = nonLinearSchwarzOp->getLocalJacobianGhosts()->getBlock(0, 0)->getXpetraMatrix();
            simpleOverlappingOperator->initialize(serialComm, localJacobian,
                                                  domainVec.at(0)->getMesh()->getMapOverlapping()->getXpetraMap(),
                                                  domainVec.at(0)->getMesh()->getMapOverlappingGhosts()->getXpetraMap(),
                                                  domainVec.at(0)->getMesh()->getMapUnique()->getXpetraMap(),
                                                  domainVec.at(0)->getMesh()->getBCFlagOverlappingGhosts());
            simpleOverlappingOperator->compute();
        } else if (variant == NonlinSchwarzVariant::ASPIN) {

            logGreen("Building ASPIN tangent with FROSch", mpiComm);
            problem->assemble("Newton");
            problem->setBoundariesSystem();
            // Compute D\mathcal{F}(u) using FROSch and DF(u)
            auto jacobian = problem->getSystem()->getBlock(0, 0)->getXpetraMatrix();
            logGreen("Global num entries: " +
                         std::to_string(problem->getSystem()->getBlock(0, 0)->getXpetraMatrix()->getGlobalNumEntries()),
                     mpiComm);
            jacobianGhosts->setAllToScalar(ST::zero());
            jacobianGhosts->resumeFill();
            jacobianGhosts->doImport(*jacobian, *uniqueToOverlappingGhostsImporter, Xpetra::ADD);
            jacobianGhosts->fillComplete(params);
            auto localJacobian = FROSch::ExtractLocalSubdomainMatrix(
                jacobianGhosts.getConst(), domainVec.at(0)->getMapOverlappingGhosts()->getXpetraMap());
            simpleOverlappingOperator->initialize(serialComm, localJacobian,
                                                  domainVec.at(0)->getMesh()->getMapOverlapping()->getXpetraMap(),
                                                  domainVec.at(0)->getMesh()->getMapOverlappingGhosts()->getXpetraMap(),
                                                  domainVec.at(0)->getMesh()->getMapUnique()->getXpetraMap(),
                                                  domainVec.at(0)->getMesh()->getBCFlagOverlappingGhosts());
            simpleOverlappingOperator->compute();
        }

        // Convert SchwarzOperator to Thyra::LinearOpBase
        auto xpetraOverlappingOperator = rcp_static_cast<Xpetra::Operator<SC, LO, GO, NO>>(simpleSumOperator);

        RCP<FROSch::TpetraPreconditioner<SC, LO, GO, NO>> tpetraFROSchOverlappingOperator(
            new FROSch::TpetraPreconditioner<SC, LO, GO, NO>(xpetraOverlappingOperator));

        auto tpetraOverlappingOperator =
            rcp_static_cast<Tpetra::Operator<SC, LO, GO, NO>>(tpetraFROSchOverlappingOperator);

        auto thyraOverlappingOperator = Thyra::createLinearOp(tpetraOverlappingOperator);

        // Solve linear system with GMRES
        logGreen("Solving outer linear system", mpiComm);
        solveThyraLinOp(thyraOverlappingOperator, deltaSolution->getBlockNonConst(0), g->getBlock(0),
                        problem->getParameterList());
        // Update the current solution
        // solution = alpha * deltaSolution + beta * solution
        problem->solution_->update(ST::one(), deltaSolution, ST::one());

        // Compute the residual
        problem->calculateNonLinResidualVec("reverse");
        residual = problem->calculateResidualNorm();
        relativeResidual = residual / residual0;
        /* logGreen("update: ", mpiComm); */
        /* deltaSolution->print(); */

        outerNonLinIts += 1;
    }
    logGreen("Outer Newton terminated", mpiComm);
    logSimple("Iterations: " + std::to_string(outerNonLinIts), mpiComm);
    logSimple("Residual: " + std::to_string(relativeResidual), mpiComm);
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
