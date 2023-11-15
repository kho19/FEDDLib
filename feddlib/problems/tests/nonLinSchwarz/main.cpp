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
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#include <stdexcept>

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
} // namespace NonLinearSchwarz

using namespace NonLinearSchwarz;

void solve(NonLinearProblemPtr_Type problem);
void localAssembly(NonLinearProblemPtr_Type problem);

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
    GraphPtr_Type dualGraph;
    if (!meshType.compare("unstructured")) {
        Teuchos::RCP<Domain<SC, LO, GO, NO>> domainP1;
        domainP1.reset(new Domain<SC, LO, GO, NO>(comm, dim));

        MeshPartitioner_Type::DomainPtrArray_Type domainP1Array(1);
        domainP1Array[0] = domainP1;

        ParameterListPtr_Type pListPartitioner = sublist(parameterListAll, "Mesh Partitioner");
        MeshPartitioner<SC, LO, GO, NO> partitionerP1(domainP1Array, pListPartitioner, "P1", dim);

        partitionerP1.readMesh();
        partitionerP1.buildDualGraph(0);
        partitionerP1.partitionDualGraphWithOverlap(0, 0);
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
    // Set initial guess. This has no effect here since vectors are reinitiliazed on serial comm.
    /* nonLinLaplace->solution_->putScalar(1.); */

    // ########################
    // Solve the problem using nonlinear Schwarz
    // ########################

    solve(nonLinLaplace);

    comm->barrier();

    // Export Solution
    bool boolExportSolution = false;
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

    // Define nonlinear Schwarz operator
    auto domainVec = problem->getDomainVector();
    auto mpiComm = domainVec.at(0)->getComm();
    auto NonLinearSchwarzOp = Teuchos::rcp(
        new FROSch::NonLinearSchwarzOperator<SC, LO, GO, NO>(mpiComm, problem->getParameterList(), problem));

    // Set initial guess
    auto initialU = Teuchos::rcp(new BlockMultiVector_Type(1));
    auto tempBlockU = Teuchos::rcp(new MultiVector_Type(domainVec.at(0)->getMesh()->getMapUnique(), 1));
    tempBlockU->putScalar(1.);
    initialU->addBlock(tempBlockU, 0);

    auto tempOut = Teuchos::rcp(new BlockMultiVector_Type(1));
    auto tempBlockOut = Teuchos::rcp(new MultiVector_Type(domainVec.at(0)->getMesh()->getMapUnique(), 1));
    tempBlockOut->putScalar(0.);
    tempOut->addBlock(tempBlockOut, 0);

    NonLinearSchwarzOp->initialize();
    NonLinearSchwarzOp->apply(initialU, tempOut);

    logGreen("solution_", mpiComm);
    problem->solution_->print();
    logGreen("tempOut", mpiComm);
    tempOut->print();

    // Define convergence requirements
    /* double gmresIts = 0.; */
    /* double residual0 = 1.; */
    /* double residual = 1.; */
    /* double outerTol = problem->getParameterList()->sublist("Parameter").get("relNonLinTol", 1.0e-6); */
    /* int outerNonLinIts = 0; */
    /* int maxOuterNonLinIts = problem->getParameterList()->sublist("Parameter").get("MaxNonLinIts", 10); */
    /* double relativeResidual = residual / residual0; */

    // Outer Newton iterations
    /* while (relativeResidual > outerTol && outerNonLinIts < maxOuterNonLinIts) { */
    // TODO solve the local problems. These are divided over the ranks, one problem per rank.
    /* localAssembly(problem); */
    // TODO build the global Jacobian and rhs
    //  Points to consider:
    //   - The local problems are distributed, so building the global problem will require communicating these
    //   - The global (linear) problem should be solved with gmres, the problem matrix should also be distributed.
    //   Have to think of how to do this assembly plus redistribution.
    //   - Do I actually assemble a matrix or build an operator that does the same thing as a matrix? Check how
    //   FROSch achieves this with operators.
    //   - Set the Problems system matrix and rhs to be the constructed global Jacobian and rhs. Then we can solve
    //   the system as bellow.
    //
    // TODO solve using the linear solve method built into Problem. This can be specified in the xml file to be
    // gmres together with a type of preconditioner (none in this case)
    //
    // TODO update the current solution
    /* } */
}

// Solve the linear problem on the local overlapping subdomain
void localAssembly(NonLinearProblemPtr_Type problem) {

    // TODO iterate over each element here to assemble the local Jacobian and rhs (like in FE_def)
    // maybe even consider using the existing assemble function in FE_def? The elements over which are assembled are
    // those stored in elementsC. The only thing that might not fit is that the whole system is assembled together.
    // - use a FROSch subdomain solver in each Newton iteration! The apply method inverts matrix A
    //
    // Notes FROSch:
    //  - submatrices are stored in a Matrix object (how is the map constructed?)
    //  - local solutions are computed with direct solvers (Amesos2) in a three-step process: 1. the symbolic
    //  factorisation is computed giving the sparsity pattern of the matrix, 2. the numerical factorisation is computed
    //  resulting in L and U, 3. the system is solved.
    //  - SchwarzOperator has systemMatrix K_, subdomainMatrix_, localSubdomainMatrix_
    //     - K_ holds the system nonoverlapping
    //     - subdomainMatrix_ holds the systemMatrix in an overlapping fashion
    //     - localSubdomainMatrix_ (I THINK) holds the local subdomain on each rank. I am not sure how this is inverted
    //       though since I thought inversion would act as if the matrix is distributed across all ranks.
    //     - What about OverlappingMatrix_ seems to serve the same purpose as localSubdomainMatrix_
    //        - initializeSubdomainSolver is done with overlappingMatrix_ in overlappingOperator in method
    //          computeOverlappingOperator which is only called in compute() in algebraicOverlappingOperator.
    //        - with localSubdomainMatrix_ in algebraicOverlappingOperator in method initialize()
    //
    //  - FROSch_OverlappingOperator::computeOverlappingOperator() calls the subdomainSolver_ compute() function. In
    //    this function either the LU factorisation of the local matrix is computed or the preconditioner is generated
    //    if solving iteratively. The subdomainSolver_->apply() then solves the system, but using either one of the
    //    precomputed LU factorisation or preconditioner.
}
