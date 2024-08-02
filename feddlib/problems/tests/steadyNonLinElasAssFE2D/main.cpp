#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"

#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/General/ExporterParaView.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include "feddlib/core/Mesh/MeshPartitioner.hpp"
#include "feddlib/problems/Solver/NonLinearSolver.hpp"
#include "feddlib/problems/specific/NonLinElasAssFE.hpp"
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Xpetra_DefaultPlatform.hpp>

void zeroDirichlet2D(double *x, double *res, double t, const double *parameters) {
    res[0] = 0.;
    res[1] = 0.;

    return;
}

void dummyFunc(double *x, double *res, double t, const double *parameters) { return; }

void rhs2D(double *x, double *res, double *parameters) {
    // parameters[0] is the time, not needed here
    res[0] = 0.;
    res[1] = parameters[1];

    return;
}

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

    Teuchos::oblackholestream blackhole;
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

    Teuchos::RCP<const Teuchos::Comm<int>> comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();

    Teuchos::RCP<StackedTimer> stackedTimer = rcp(new StackedTimer("Nonlinear Schwarz solver", true));
    TimeMonitor::setStackedTimer(stackedTimer);
 
    // Command Line Parameters
    Teuchos::CommandLineProcessor myCLP;
    string ulib_str = "Tpetra";
    myCLP.setOption("ulib", &ulib_str, "Underlying lib");
    // int dim = 2;
    // myCLP.setOption("dim",&dim,"dim");
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

    bool verbose(comm->getRank() == 0); // Print-Ausgaben nur auf rank = 0
    if (verbose) {
        cout << "###############################################################" << endl;
        cout << "############ Starting 2D Steady Nonlinear Elasticity ... ############" << endl;
        cout << "###############################################################" << endl;
    }

    ParameterListPtr_Type parameterListProblem = Teuchos::getParametersFromXmlFile(xmlProblemFile);
    ParameterListPtr_Type parameterListPrec = Teuchos::getParametersFromXmlFile(xmlPrecFile);
    ParameterListPtr_Type parameterListSolver = Teuchos::getParametersFromXmlFile(xmlSolverFile);

    ParameterListPtr_Type parameterListAll(new Teuchos::ParameterList(*parameterListProblem));
    parameterListAll->setParameters(*parameterListPrec);
    parameterListAll->setParameters(*parameterListSolver);

    // Only 2D for this example
    int dim = 2;
    string meshType = parameterListProblem->sublist("Parameter").get("Mesh Type", "structured");
    string meshName = parameterListProblem->sublist("Parameter").get("Mesh Name", "cube_0_1.mesh");
    string meshDelimiter = parameterListProblem->sublist("Parameter").get("Mesh Delimiter", " ");
    int n;
    int m = parameterListProblem->sublist("Parameter").get("H/h", 5);
    // Only P1 for this problem
    string FEType = "P1";

    int numProcsCoarseSolve = parameterListProblem->sublist("General").get("Mpi Ranks Coarse", 0);
    int size = comm->getSize() - numProcsCoarseSolve;
    double length = 4.;
    int minNumberSubdomains = 1;

    Teuchos::RCP<Teuchos::Time> totalTime(Teuchos::TimeMonitor::getNewCounter("main: Total Time"));
    Teuchos::RCP<Teuchos::Time> buildMesh(Teuchos::TimeMonitor::getNewCounter("main: Build Mesh"));
    Teuchos::RCP<Teuchos::Time> solveTime(Teuchos::TimeMonitor::getNewCounter("main: Solve problem time"));

    DomainPtr_Type domain;
    if (!meshType.compare("structured")) {
        TEUCHOS_TEST_FOR_EXCEPTION(size % minNumberSubdomains != 0, std::logic_error,
                                   "Wrong number of processors for structured mesh.");
        n = (int)(std::pow(size, 1 / 2.) + 100. * Teuchos::ScalarTraits<double>::eps()); // 1/H
        std::vector<double> x(2);
        x[0] = 0.0;
        x[1] = 0.0;
        domain = Teuchos::rcp(new Domain<SC, LO, GO, NO>(x, 1., 1., comm));
        domain->buildMesh(1, "Square", dim, FEType, n, m, numProcsCoarseSolve);
    } else if (!meshType.compare("unstructured")) {
        domain.reset(new Domain<SC, LO, GO, NO>(comm, dim));

        MeshPartitioner_Type::DomainPtrArray_Type domainArray(1);
        domainArray[0] = domain;

        ParameterListPtr_Type pListPartitioner = sublist(parameterListAll, "Mesh Partitioner");
        MeshPartitioner<SC, LO, GO, NO> partitioner(domainArray, pListPartitioner, "P1", dim);

        partitioner.readAndPartition();
    }

    Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaF(new ExporterParaView<SC, LO, GO, NO>());

    Teuchos::RCP<MultiVector<SC, LO, GO, NO>> exportSolution(new MultiVector<SC, LO, GO, NO>(domain->getMapUnique()));
    vec_int_ptr_Type BCFlags = domain->getBCFlagUnique();

    Teuchos::ArrayRCP<SC> entries = exportSolution->getDataNonConst(0);
    for (int i = 0; i < entries.size(); i++) {
        entries[i] = BCFlags->at(i);
    }

    Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionConst = exportSolution;

    exParaF->setup("Flags", domain->getMesh(), FEType);

    exParaF->addVariable(exportSolutionConst, "Flags", "Scalar", 1, domain->getMapUnique(), domain->getMapUniqueP2());

    exParaF->save(0.0);

    Teuchos::RCP<BCBuilder<SC, LO, GO, NO>> bcFactory(new BCBuilder<SC, LO, GO, NO>());

    bcFactory->addBC(zeroDirichlet2D, 2, 0, domain, "Dirichlet", dim);
    bcFactory->addBC(zeroDirichlet2D, 4, 0, domain, "Dirichlet", dim);

    NonLinElasAssFE<SC, LO, GO, NO> NonLinElasAssFE(domain, FEType, parameterListAll);

    NonLinElasAssFE.addBoundaries(bcFactory);

    NonLinElasAssFE.addRhsFunction(rhs2D);

    double force = parameterListAll->sublist("Parameter").get("Volume force", 0.);
    double degree = 0;

    NonLinElasAssFE.addParemeterRhs(force);
    NonLinElasAssFE.addParemeterRhs(degree);

    NonLinElasAssFE.initializeProblem();
    NonLinElasAssFE.assemble();
    NonLinElasAssFE.setBoundaries();
    NonLinElasAssFE.setBoundariesRHS();

    std::string nlSolverType = "Newton";
    NonLinearSolver<SC, LO, GO, NO> nlSolverAssFE(nlSolverType);
    FEDD_TIMER_START(SolveTimer, " - NKS - global solve");
    nlSolverAssFE.solve(NonLinElasAssFE);
    FEDD_TIMER_STOP(SolveTimer);

    comm->barrier();

    Teuchos::TimeMonitor::report(cout, "FEDD");
    stackedTimer->stop("Nonlinear Schwarz solver");
    StackedTimer::OutputOptions options;
    options.output_fraction = options.output_histogram = options.output_minmax = true;
    stackedTimer->report((std::cout), comm, options);

    Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exPara(new ExporterParaView<SC, LO, GO, NO>());

    exPara->setup("displacements", domain->getMesh(), FEType);

    MultiVectorConstPtr_Type solution = NonLinElasAssFE.getSolution()->getBlock(0);
    exPara->addVariable(solution, "valuesNonLinElasAssFE", "Vector", dim, domain->getMapUnique());

    exPara->save(0.0);

    return (EXIT_SUCCESS);
}
