#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/General/ExporterParaView.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include "feddlib/core/Mesh/MeshPartitioner.hpp"

#include "feddlib/problems/Solver/NonLinearSolver.hpp"
#include "feddlib/problems/specific/NavierStokes.hpp"

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#include <stdexcept>

/*!
 main of steady-state Navier-Stokes problem

 @brief steady-state Navier-Stokes main
 @author Christian Hochmuth
 @version 1.0
 @copyright CH
 */

using namespace std;

void initialValue2D(double *x, double *res, double *parameters) { res[0] = 1;res[1] = 1; }
// Required for setting the Dirichlet BC on the ghost points to the current global solution in nonlinear Schwarz
void currentSolutionDirichlet(double *x, double *res, double t, const double *parameters) { res[0] = x[0]; }

void zeroDirichlet(double *x, double *res, double t, const double *parameters) {

    res[0] = 0.;

    return;
}

void zeroDirichlet2D(double *x, double *res, double t, const double *parameters) {

    res[0] = 0.;
    res[1] = 0.;

    return;
}

// For Lid Driven Cavity Test
void ldcFunc2D(double *x, double *res, double t, const double *parameters) {

    res[0] = 1.;//* parameters[0];
    res[1] = 0.;

    return;
}

void dummyFunc(double *x, double *res, double *parameters) {
    if (parameters[0] == 2)
        res[0] = 1;
    else
        res[0] = 0.;

    return;
}

typedef unsigned UN;
typedef default_sc SC;
typedef default_lo LO;
typedef default_go GO;
typedef default_no NO;
using namespace Teuchos;

using namespace FEDD;

int main(int argc, char *argv[]) {

    typedef MeshPartitioner<SC, LO, GO, NO> MeshPartitioner_Type;
    typedef Teuchos::RCP<Domain<SC, LO, GO, NO>> DomainPtr_Type;

    Teuchos::oblackholestream blackhole;
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

    Teuchos::RCP<const Teuchos::Comm<int>> comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();
    bool verbose(comm->getRank() == 0);

    //    Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();

    if (verbose) {
        cout << "###############################################################" << endl;
        cout << "##################### Steady Navier-Stokes ####################" << endl;
        cout << "###############################################################" << endl;
    }

    // Command Line Parameters
    Teuchos::CommandLineProcessor myCLP;

    string xmlProblemFile = "parametersProblem.xml";
    myCLP.setOption("problemfile", &xmlProblemFile, ".xml file with Inputparameters.");
    string xmlSchwarzSolverFile = "parametersSolverNonLinSchwarz.xml";
    myCLP.setOption("schwarzsolverfile", &xmlSchwarzSolverFile, ".xml file with Inputparameters.");
    bool debug = false;
    myCLP.setOption("debug", "", &debug, "bool option for debugging");

    double length = 4.;
    myCLP.setOption("length", &length, "length of domain.");

    myCLP.recogniseAllOptions(true);
    myCLP.throwExceptions(false);
    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = myCLP.parse(argc, argv);
    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
        MPI_Finalize();
        return 0;
    }

    if (comm->getRank() == 1 && debug) {
        waitForGdbAttach<LO>();
    }
    comm->barrier();
    comm->barrier();

    // Teuchos::RCP<StackedTimer> stackedTimer = rcp(new StackedTimer("Steady Navier-Stokes", true));
    // TimeMonitor::setStackedTimer(stackedTimer);
    ParameterListPtr_Type parameterListProblem = Teuchos::getParametersFromXmlFile(xmlProblemFile);

    ParameterListPtr_Type parameterListSolver = Teuchos::getParametersFromXmlFile(xmlSchwarzSolverFile);

    int dim = parameterListProblem->sublist("Parameter").get("Dimension", 2);

    std::string discVelocity = parameterListProblem->sublist("Parameter").get("Discretization Velocity", "P2");
    std::string discPressure = parameterListProblem->sublist("Parameter").get("Discretization Pressure", "P1");

    string meshType = parameterListProblem->sublist("Parameter").get("Mesh Type", "structured");
    string meshName = parameterListProblem->sublist("Parameter").get("Mesh Name", "circle2D_1800.mesh");
    string meshDelimiter = parameterListProblem->sublist("Parameter").get("Mesh Delimiter", " ");
    int m = parameterListProblem->sublist("Parameter").get("H/h", 5);
    string linearization = parameterListProblem->sublist("General").get("Linearization", "FixedPoint");
    string precMethod = parameterListProblem->sublist("General").get("Preconditioner Method", "Monolithic");
    int mixedFPIts = parameterListProblem->sublist("General").get("MixedFPIts", 1);
    auto overlap = parameterListSolver->get("Overlap", 1);
    int n;

    TEUCHOS_TEST_FOR_EXCEPTION(dim != 2, std::runtime_error, "Only 2D implemented for now");
    ParameterListPtr_Type parameterListAll(new Teuchos::ParameterList(*parameterListProblem));
    parameterListAll->setParameters(*parameterListSolver);

    std::string bcType = parameterListProblem->sublist("Parameter").get("BC Type", "parabolic");

    int minNumberSubdomains = 1;
    if (!meshType.compare("structured")) {
        minNumberSubdomains = 1;
    }

    int numProcsCoarseSolve = parameterListProblem->sublist("General").get("Mpi Ranks Coarse", 0);
    int size = comm->getSize() - numProcsCoarseSolve;

    Teuchos::RCP<Teuchos::Time> totalTime(Teuchos::TimeMonitor::getNewCounter("main: Total Time"));
    Teuchos::RCP<Teuchos::Time> buildMesh(Teuchos::TimeMonitor::getNewCounter("main: Build Mesh"));
    Teuchos::RCP<Teuchos::Time> solveTime(Teuchos::TimeMonitor::getNewCounter("main: Solve problem time"));
    DomainPtr_Type domainPressure;
    DomainPtr_Type domainVelocity;
    Teuchos::TimeMonitor totalTimeMonitor(*totalTime);
    Teuchos::TimeMonitor buildMeshMonitor(*buildMesh);
    if (verbose) {
        cout << "-- Building Mesh ..." << flush;
    }

    MeshPartitioner_Type::DomainPtrArray_Type domainP1Array(2);
    domainP1Array[0] = domainPressure;
    domainP1Array[1] = domainVelocity;

    ParameterListPtr_Type pListPartitioner = sublist(parameterListAll, "Mesh Partitioner");
    MeshPartitioner<SC, LO, GO, NO> partitionerP1;

    if (!meshType.compare("structured_ldc")) {
        // Structured Mesh for Lid-Driven Cavity Test
        TEUCHOS_TEST_FOR_EXCEPTION(size % minNumberSubdomains != 0, std::logic_error,
                                   "Wrong number of processors for structured mesh.");
        n = (int)(std::pow(size / minNumberSubdomains, 1 / 2.) + 100 * Teuchos::ScalarTraits<double>::eps()); // 1/H
        std::vector<double> x(2);
        x[0] = 0.0;
        x[1] = 0.0;
        domainPressure.reset(new Domain<SC, LO, GO, NO>(x, 1., 1., comm));
        domainVelocity.reset(new Domain<SC, LO, GO, NO>(x, 1., 1., comm));
        domainP1Array[0] = domainPressure;
        domainP1Array[1] = domainVelocity;
        domainPressure->buildMesh(5, "Square", dim, discPressure, n, m, numProcsCoarseSolve);
        domainVelocity->buildMesh(5, "Square", dim, discVelocity, n, m, numProcsCoarseSolve);

        partitionerP1 = MeshPartitioner<SC, LO, GO, NO>(domainP1Array, pListPartitioner, "P1", dim);
        partitionerP1.buildOverlappingDualGraphFromDistributedParMETIS(0, overlap);
        partitionerP1.buildSubdomainFromDualGraphStructured(0);
        partitionerP1.buildOverlappingDualGraphFromDistributedParMETIS(1, overlap);
        partitionerP1.buildSubdomainFromDualGraphStructured(1);
    }

    std::vector<double> parameter_vec(1, parameterListProblem->sublist("Parameter").get("MaxVelocity", 1.));

    Teuchos::RCP<BCBuilder<SC, LO, GO, NO>> bcFactory(new BCBuilder<SC, LO, GO, NO>());

    if (!bcType.compare("LDC")) {
        parameter_vec.push_back(0.); // Dummy
        bcFactory->addBC(zeroDirichlet2D, 1, 0, domainVelocity, "Dirichlet", dim);
        bcFactory->addBC(zeroDirichlet2D, 3, 0, domainVelocity, "Dirichlet", dim);
        bcFactory->addBC(ldcFunc2D, 2, 0, domainVelocity, "Dirichlet", dim, parameter_vec);
        bcFactory->addBC(zeroDirichlet, 3, 1, domainPressure, "Dirichlet", 1);
    } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Select a valid boundary condition.");
    }
    // The current global solution must be set as the Dirichlet BC on the ghost nodes for nonlinear Schwarz solver to
    // correctly solve on the subdomains
    bcFactory->addBC(currentSolutionDirichlet, -99, 0, domainVelocity, "Dirichlet", dim);
    bcFactory->addBC(currentSolutionDirichlet, -99, 1, domainPressure, "Dirichlet", 1);
    NavierStokes<SC, LO, GO, NO> navierStokes(domainVelocity, discVelocity, domainPressure, discPressure,
                                              parameterListAll);

    domainVelocity->info();
    domainPressure->info();
    navierStokes.info();

    Teuchos::TimeMonitor solveTimeMonitor(*solveTime);

    navierStokes.addBoundaries(bcFactory);
    navierStokes.addRhsFunction(dummyFunc);

    navierStokes.initializeProblem();
    navierStokes.assemble();

    navierStokes.setBoundariesRHS();

    // navierStokes.getSystem()->getBlock(1,1)->print();
    std::string nlSolverType = parameterListProblem->sublist("General").get("Linearization", "NonlinearSchwarz");
    NonLinearSolver<SC, LO, GO, NO> nlSolver(nlSolverType);
    nlSolver.solve(navierStokes);
    comm->barrier();

    Teuchos::TimeMonitor::report(cout);
    // stackedTimer->stop("Steady Navier-Stokes");
    // StackedTimer::OutputOptions options;
    // options.output_fraction = options.output_histogram = options.output_minmax = true;
    // stackedTimer->report((std::cout), comm, options);

    if (parameterListAll->sublist("General").get("ParaViewExport", false)) {
        Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaVelocity(new ExporterParaView<SC, LO, GO, NO>());
        Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaPressure(new ExporterParaView<SC, LO, GO, NO>());

        Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionV = navierStokes.getSolution()->getBlock(0);
        Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionP = navierStokes.getSolution()->getBlock(1);

        DomainPtr_Type dom = domainVelocity;

        exParaVelocity->setup("velocity", dom->getMesh(), dom->getFEType());

        UN dofsPerNode = dim;
        exParaVelocity->addVariable(exportSolutionV, "u", "Vector", dofsPerNode, dom->getMapUnique());

        dom = domainPressure;
        exParaPressure->setup("pressure", dom->getMesh(), dom->getFEType());

        exParaPressure->addVariable(exportSolutionP, "p", "Scalar", 1, dom->getMapUnique());

        exParaVelocity->save(0.0);
        exParaPressure->save(0.0);
    }

    return (EXIT_SUCCESS);
}
