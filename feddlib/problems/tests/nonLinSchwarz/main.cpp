#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/General/ExporterParaView.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include "feddlib/core/Mesh/MeshPartitioner.hpp"
#include "feddlib/problems/Solver/NonLinearSolver.hpp"
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_DefaultPlatform.hpp>

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
} // namespace NonLinearSchwarz

using namespace NonLinearSchwarz;

int main(int argc, char *argv[]) {

    Teuchos::oblackholestream blackhole;
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

    Teuchos::RCP<const Teuchos::Comm<int>> comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();

    // ########################
    // Set default values for command line parameters
    // ########################

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
    string meshName = parameterListProblem->sublist("Parameter").get("Mesh Name", "simple_square.mesh");
    string meshDelimiter = parameterListProblem->sublist("Parameter").get("Mesh Delimiter", " ");
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
    GraphPtr_Type dualGraph;
    if (!meshType.compare("structured")) {
        TEUCHOS_TEST_FOR_EXCEPTION(size % minNumberSubdomains != 0, std::logic_error,
                                   "Wrong number of processors for structured mesh.");
        if (dim == 2) {
            n = (int)(std::pow(size, 1 / 2.) + 100. * Teuchos::ScalarTraits<double>::eps()); // 1/H
            std::vector<double> x(2);
            // Starting coordinates for the rectangle (lower left corner?)
            x[0] = 0.0;
            x[1] = 0.0;
            domain = Teuchos::rcp(new Domain<SC, LO, GO, NO>(x, 1., 1., comm));
            domain->buildMesh(1, "Square", dim, FEType, n, m, numProcsCoarseSolve);
        } else if (dim == 3) {
            n = (int)(std::pow(size, 1 / 3.) + 100. * Teuchos::ScalarTraits<SC>::eps()); // 1/H
            std::vector<double> x(3);
            x[0] = 0.0;
            x[1] = 0.0;
            x[2] = 0.0;
            domain = Teuchos::rcp(new Domain<SC, LO, GO, NO>(x, 1., 1., 1., comm));
            domain->buildMesh(1, "Square", dim, FEType, n, m, numProcsCoarseSolve);
        }
    } else if (!meshType.compare("unstructured")) {
        Teuchos::RCP<Domain<SC, LO, GO, NO>> domainP1;
        domainP1.reset(new Domain<SC, LO, GO, NO>(comm, dim));

        MeshPartitioner_Type::DomainPtrArray_Type domainP1Array(1);
        domainP1Array[0] = domainP1;

        ParameterListPtr_Type pListPartitioner = sublist(parameterListAll, "Mesh Partitioner");
        MeshPartitioner<SC, LO, GO, NO> partitionerP1(domainP1Array, pListPartitioner, "P1", dim);

        partitionerP1.readMesh();
        partitionerP1.buildDualGraph(0);
        partitionerP1.partitionDualGraphWithOverlap(0, 1);

        domain = domainP1;
    }
    /* cout << "########################### Final print of maps " */
    /*         "####################################" */
    /*      << endl; */
    /* Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream(); */
    /* domain->getMesh()->getDualGraph()->getMap()->describe(*out, Teuchos::VERB_EXTREME); */
    /* domain->getMapUnique()->print(); */
    // ########################
    // Solve the problem using nonlinear Schwarz
    // ########################

    return (EXIT_SUCCESS);
}

// Solve the problem
void solve() {

    /* RCP<LHelementList; */
    /* buildAndPartitionDualGraph(elementList, dualGraphPartition); */

    // Outer Newton iterations
    /* while (notConverged && !maxIters) { */
    /*     // Form the global problem */
    /* } */
}

// Extend partition of a dual graph
// Essentialy a wrapper of the FROSch function ExtendOverlapByOneLayer()
void extendDualGraphPartition() {}

// Assemble the Schwarz operator by combining local operators
void globalAssembly() {}

// Solve the linear problem on the local overlapping subdomain
void localAssembly() {}
