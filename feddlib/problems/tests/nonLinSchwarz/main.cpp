#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include "feddlib/core/Mesh/MeshPartitioner_decl.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
#include "feddlib/problems/Solver/NonLinearSolver.hpp"
#include "feddlib/problems/specific/NonLinLaplace_decl.hpp"
#include <Teuchos_StackedTimer.hpp>
#include <Xpetra_DefaultPlatform.hpp>

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

int main(int argc, char *argv[]) {

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

    Teuchos::oblackholestream blackhole;
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

    Teuchos::RCP<const Teuchos::Comm<int>> comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();

    Teuchos::RCP<StackedTimer> stackedTimer = rcp(new StackedTimer("Nonlinear Schwarz solver", true));
    TimeMonitor::setStackedTimer(stackedTimer);
    // ########################
    // Set default values for command line parameters
    // ########################
    // Command Line Parameters
    Teuchos::CommandLineProcessor myCLP;
    string ulib_str = "Tpetra";
    myCLP.setOption("ulib", &ulib_str, "Underlying lib");

    string xmlProblemFile = "parametersProblem.xml";
    myCLP.setOption("problemfile", &xmlProblemFile, ".xml file with Inputparameters.");
    string xmlPrecFile = "parametersPrec.xml";
    myCLP.setOption("precfile", &xmlPrecFile, ".xml file with Inputparameters.");
    string xmlSolverFile = "parametersSolver.xml";
    myCLP.setOption("solverfile", &xmlSolverFile, ".xml file with Inputparameters.");

    myCLP.recogniseAllOptions(true);
    myCLP.throwExceptions(false);
    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = myCLP.parse(argc, argv);
    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
        mpiSession.~GlobalMPISession();
        return 0;
    }

    bool debug = false;
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

    ParameterListPtr_Type parameterListProblem = Teuchos::getParametersFromXmlFile(xmlProblemFile);
    ParameterListPtr_Type parameterListPrec = Teuchos::getParametersFromXmlFile(xmlPrecFile);
    ParameterListPtr_Type parameterListSolver = Teuchos::getParametersFromXmlFile(xmlSolverFile);

    ParameterListPtr_Type parameterListAll(new Teuchos::ParameterList(*parameterListProblem));
    parameterListAll->setParameters(*parameterListPrec);
    parameterListAll->setParameters(*parameterListSolver);

    int dim = parameterListProblem->sublist("Parameter").get("Dimension", 2);
    int minNumberSubdomains = 1;
    int n = -1;
    int m = parameterListProblem->sublist("Parameter").get("H/h", 5);
    string meshType = parameterListProblem->sublist("Parameter").get("Mesh Type", "structured");
    string FEType = parameterListProblem->sublist("Parameter").get("Discretization", "P1");
    auto overlap = parameterListSolver->get("Overlap", 1);
    int numProcsCoarseSolve = 0;
    int size = comm->getSize() - numProcsCoarseSolve;

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

    if (!meshType.compare("structured")) {
        TEUCHOS_TEST_FOR_EXCEPTION(size % minNumberSubdomains != 0, std::logic_error,
                                   "Wrong number of processors for structured mesh.");
        if (dim == 2) {
            n = (int)(std::pow(size, 1 / 2.) + 100. * Teuchos::ScalarTraits<double>::eps()); // 1/H
            std::vector<double> x(2);
            x[0] = 0.0;
            x[1] = 0.0;
            domainP1.reset(new Domain<SC, LO, GO, NO>(x, 1., 1., comm));
            domainP1Array[0] = domainP1;
            domainP1->buildMesh(1, "Square", dim, FEType, n, m, numProcsCoarseSolve);
        } else if (dim == 3) {
            n = (int)(std::pow(size, 1 / 3.) + 100. * Teuchos::ScalarTraits<SC>::eps()); // 1/H
            std::vector<double> x(3);
            x[0] = 0.0;
            x[1] = 0.0;
            x[2] = 0.0;
            domainP1.reset(new Domain<SC, LO, GO, NO>(x, 1., 1., 1., comm));
            domainP1Array[0] = domainP1;
            domainP1->buildMesh(1, "Square", dim, FEType, n, m, numProcsCoarseSolve);
        }
        partitionerP1 = MeshPartitioner<SC, LO, GO, NO>(domainP1Array, pListPartitioner, "P1", dim);
        partitionerP1.buildOverlappingDualGraphFromDistributedParMETIS(0, 1);
        partitionerP1.buildSubdomainFromDualGraphStructured(0);
    } else if (!meshType.compare("unstructured")) {
        partitionerP1.readMesh();
        partitionerP1.buildDualGraph(0);
        partitionerP1.partitionDualGraphWithOverlap(0, overlap);
    partitionerP1.buildSubdomainFromDualGraphUnstructured(0);
    }
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
    // Set initial value that is used to build the coarse basis functions
    nonLinLaplace->initSolutionWithFunction(initialValue2D, 0, std::vector<double>{0});

    // ########################
    // Solve the problem using nonlinear Schwarz
    // ########################
    NonLinearSolver<SC, LO, GO, NO> nlSolverAssFE("NonLinearSchwarz");
    FEDD_TIMER_START(SolveTimer, " - Schwarz - global solve");
    nlSolverAssFE.solve(*nonLinLaplace);
    FEDD_TIMER_STOP(SolveTimer);

    comm->barrier();

    Teuchos::TimeMonitor::report(cout, "FEDD");
    stackedTimer->stop("Nonlinear Schwarz solver");
    StackedTimer::OutputOptions options;
    options.output_fraction = options.output_histogram = options.output_minmax = true;
    stackedTimer->report((std::cout), comm, options);
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
