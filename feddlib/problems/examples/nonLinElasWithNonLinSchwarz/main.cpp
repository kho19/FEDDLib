#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/ExporterParaView.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector_decl.hpp"
#include "feddlib/core/Mesh/MeshPartitioner.hpp"
#include "feddlib/problems/Solver/NonLinearSolver.hpp"
#include "feddlib/problems/specific/NonLinElasticity.hpp"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_RCPBoostSharedPtrConversions.hpp"
#include "Teuchos_RCPDecl.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include <Teuchos_GlobalMPISession.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#include <vector>

using namespace std;

// TODO: kho:
//  - Implement construction of surfaces in MeshPartitioner on the overlapping subdomains. Without this surface forces
//  cannot be used

// Apply a force in y direction if the boundary element flag is three
// parameters contains [time, force, degree (unused), flag]
void rhsY(double *x, double *res, double *parameters) {
    res[0] = 0.;
    double force = parameters[1];

    if (parameters[3] == 3)
        res[1] = force;
    else
        res[1] = 0.;

    return;
}
void rhs2D(double *x, double *res, double *parameters) {
    // parameters[0] is the time, not needed here
    res[0] = 0.;
    res[1] = parameters[1];

    return;
}

void rhs(double *x, double *res, double *parameters) {
    // parameters[0] is the time, not needed here
    res[0] = 0.;
    res[1] = parameters[1];
    res[2] = 0.;
    return;
}

void rhsX(double *x, double *res, double *parameters) {
    // parameters[0] is the time, not needed here
    res[0] = parameters[1];
    res[1] = 0.;
    res[2] = 0.;
    return;
}

void zeroBC(double *x, double *res, double t, const double *parameters) {

    res[0] = 0.;

    return;
}

void zeroDirichlet2D(double *x, double *res, double t, const double *parameters) {

    res[0] = 0.;
    res[1] = 0.;

    return;
}

void zeroDirichlet3D(double *x, double *res, double t, const double *parameters) {

    res[0] = 0.;
    res[1] = 0.;
    res[2] = 0.;

    return;
}

typedef unsigned UN;
typedef default_sc SC;
typedef default_lo LO;
typedef default_go GO;
typedef default_no NO;

using namespace FEDD;

int main(int argc, char *argv[]) {

    typedef MeshPartitioner<SC, LO, GO, NO> MeshPartitioner_Type;

    Teuchos::oblackholestream blackhole;
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

    Teuchos::RCP<const Teuchos::Comm<int>> comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();

    // Command Line Parameters
    Teuchos::CommandLineProcessor myCLP;
    string ulib_str = "Tpetra";
    myCLP.setOption("ulib", &ulib_str, "Underlying lib");
    double length = 4.;
    myCLP.setOption("length", &length, "length of domain.");
    string xmlProblemFile = "parametersProblem.xml";
    myCLP.setOption("problemfile", &xmlProblemFile, ".xml file with Inputparameters.");
    string xmlSchwarzSolverFile = "parametersSolverNonLinSchwarz.xml";
    myCLP.setOption("schwarzsolverfile", &xmlSchwarzSolverFile, ".xml file with Inputparameters.");

    myCLP.recogniseAllOptions(true);
    myCLP.throwExceptions(false);
    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = myCLP.parse(argc, argv);
    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
        mpiSession.~GlobalMPISession();
        return 0;
    }

    ParameterListPtr_Type parameterListProblem = Teuchos::getParametersFromXmlFile(xmlProblemFile);
    ParameterListPtr_Type parameterListSchwarzSolver = Teuchos::getParametersFromXmlFile(xmlSchwarzSolverFile);

    int dim = parameterListProblem->sublist("Parameter").get("Dimension", 3);
    std::string elasticityType = parameterListProblem->sublist("Parameter").get("Elasticity Type", "linear");
    string meshType = parameterListProblem->sublist("Parameter").get("Mesh Type", "structured");
    string meshName = parameterListProblem->sublist("Parameter").get("Mesh Name", "tetrahedron.mesh");
    string meshDelimiter = parameterListProblem->sublist("Parameter").get("Mesh Delimiter", " ");
    int m = parameterListProblem->sublist("Parameter").get("H/h", 5);

    int n;

    ParameterListPtr_Type parameterListAll(new Teuchos::ParameterList(*parameterListProblem));
    parameterListAll->setParameters(*parameterListSchwarzSolver);
    int minNumberSubdomains = (int)length;

    Teuchos::RCP<Teuchos::Time> totalTime(Teuchos::TimeMonitor::getNewCounter("main: Total Time"));
    Teuchos::RCP<Teuchos::Time> buildMesh(Teuchos::TimeMonitor::getNewCounter("main: Build Mesh"));
    Teuchos::RCP<Teuchos::Time> solveTime(Teuchos::TimeMonitor::getNewCounter("main: Solve problem time"));

    int numProcsCoarseSolve = parameterListProblem->sublist("General").get("Mpi Ranks Coarse", 0);

    int size = comm->getSize() - numProcsCoarseSolve;

    std::string FEType = parameterListProblem->sublist("Parameter").get("Discretization", "P1");
    // Nonlinear Schwarz related params
    auto overlap = parameterListSchwarzSolver->get("Overlap", 1);

    int partition = 0;
    bool verbose(comm->getRank() == 0);
    if (verbose) {
        cout << "######################################" << endl;
        cout << "########## Nonlinear Elasticity ######" << endl;
        cout << "######################################" << endl;
    }

    Teuchos::TimeMonitor totalTimeMonitor(*totalTime);
    Teuchos::RCP<Domain<SC, LO, GO, NO>> domain;
    Teuchos::RCP<Domain<SC, LO, GO, NO>> domainP1;
    domainP1.reset(new Domain<SC, LO, GO, NO>(comm, dim));

    MeshPartitioner_Type::DomainPtrArray_Type domainP1Array(1);
    domainP1Array[0] = domainP1;

    ParameterListPtr_Type pListPartitioner = sublist(parameterListAll, "Mesh Partitioner");
    MeshPartitioner<SC, LO, GO, NO> partitionerP1(domainP1Array, pListPartitioner, "P1", dim);

    if (!meshType.compare("structured")) {
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
        partitionerP1.buildOverlappingDualGraphFromDistributedParMETIS(0, overlap);
        partitionerP1.buildSubdomainFromDualGraphStructured(0);
    } else if (!meshType.compare("unstructured")) {
        partitionerP1.readMesh();
        partitionerP1.buildDualGraph(0);
        partitionerP1.partitionDualGraphWithOverlap(0, overlap);
        partitionerP1.buildSubdomainFromDualGraphUnstructured(0);
    }
    domain = domainP1;

    Teuchos::RCP<BCBuilder<SC, LO, GO, NO>> bcFactory(new BCBuilder<SC, LO, GO, NO>());
    if (meshType == "structured") {
        if (dim == 2) {
            /* bcFactory->addBC(zeroDirichlet2D, 1, 0, domain, "Dirichlet", dim); */
            bcFactory->addBC(zeroDirichlet2D, 2, 0, domain, "Dirichlet", dim);
            /* bcFactory->addBC(zeroDirichlet2D, 3, 0, domain, "Dirichlet", dim); */
            bcFactory->addBC(zeroDirichlet2D, 4, 0, domain, "Dirichlet", dim);
        } else if (dim == 3) {
            bcFactory->addBC(zeroDirichlet3D, 2, 0, domain, "Dirichlet", dim);
        }
    } else if (meshType == "unstructured") {
        if (dim == 2) {
            // For unstructured meshes and surface force set zeroBC at flag 1 since the surface force is applied at flag
            // 3. For volume forces any boundary flag can be taken. This will affect which side of the geometry is
            // stationary
            /* bcFactory->addBC(zeroDirichlet2D, 1, 0, domain, "Dirichlet", dim); */
            bcFactory->addBC(zeroDirichlet2D, 2, 0, domain, "Dirichlet", dim);
            /* bcFactory->addBC(zeroDirichlet2D, 3, 0, domain, "Dirichlet", dim); */
            bcFactory->addBC(zeroDirichlet2D, 4, 0, domain, "Dirichlet", dim);

        } else if (dim == 3) {
            bcFactory->addBC(zeroDirichlet3D, 1, 0, domain, "Dirichlet", dim);
        }
    }
    // The current global solution must be set as the Dirichlet BC on the ghost nodes for nonlinear Schwarz solver to
    // correctly solve on the subdomains
    std::vector<double> funcParams{static_cast<double>(dim)};
    bcFactory->addBC(Helper::currentSolutionDirichlet, -99, 0, domain, "Dirichlet", dim, funcParams);

    NonLinElasticity<SC, LO, GO, NO> elasticity(domain, FEType, parameterListAll);

    domain->info();
    elasticity.info();

    Teuchos::TimeMonitor solveTimeMonitor(*solveTime);

    elasticity.addBoundaries(bcFactory);

    if (dim == 2)
        // Use this in conjuction with Source Type = Surface. It applies a force only on the surface elements
        // corresponding to the specified boundary
        /* elasticity.addRhsFunction(rhsY); */
        // Use this in conjuction with Source Type = Volume. It applies a constant force at every node
        // Applying an elongating force results in a stable solution for much larger deformations than a compressing
        // force
        elasticity.addRhsFunction(rhs2D);
    else if (dim == 3)
        elasticity.addRhsFunction(rhsX);

    double force = parameterListAll->sublist("Parameter").get("Volume force", 0.);
    double degree = 0;

    elasticity.addParemeterRhs(force);
    elasticity.addParemeterRhs(degree);
    elasticity.initializeProblem();
    elasticity.reInitSpecificProblemVectors(domain->getMapVecFieldOverlappingGhosts());
    elasticity.assemble();

    elasticity.setBoundaries();

    std::string nlSolverType = parameterListProblem->sublist("General").get("Linearization", "Newton");
    NonLinearSolver<SC, LO, GO, NO> elasticitySolver(nlSolverType);
    FEDD_TIMER_START(SolveTimer, " - Schwarz - global solve");
    elasticitySolver.solve(elasticity);
    FEDD_TIMER_STOP(SolveTimer);

    auto rankVec = Teuchos::RCP<MultiVector<SC, LO, GO, NO>>(new MultiVector<SC, LO, GO, NO>(domain->getMapUnique()));
    rankVec->putScalar(comm->getRank());

    if (parameterListAll->sublist("General").get("ParaViewExport", false)) {
        Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaDisp(new ExporterParaView<SC, LO, GO, NO>());

        Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionU = elasticity.getSolution()->getBlock(0);
        Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> rank = rankVec;

        exParaDisp->setup("displacement", domain->getMesh(), domain->getFEType());

        UN dofsPerNode = dim;

        exParaDisp->addVariable(exportSolutionU, "u", "Vector", dofsPerNode, domain->getMapUnique());
        exParaDisp->addVariable(rank, "rank", "Scalar", 1, domain->getMapUnique());
        exParaDisp->save(0.0);
        Teuchos::TimeMonitor::report(std::cout);
    }

    return (EXIT_SUCCESS);
}
