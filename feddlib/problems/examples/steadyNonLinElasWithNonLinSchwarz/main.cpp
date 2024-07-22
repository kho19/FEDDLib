#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"

#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/General/ExporterParaView.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include "feddlib/core/Mesh/MeshPartitioner.hpp"
#include "feddlib/problems/Solver/NonLinearSolver.hpp"
#include "feddlib/problems/specific/NonLinElasAssFE.hpp"
#include <Teuchos_GlobalMPISession.hpp>
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

    // Command Line Parameters
    Teuchos::CommandLineProcessor myCLP;
    string ulib_str = "Tpetra";
    myCLP.setOption("ulib", &ulib_str, "Underlying lib");
    // int dim = 2;
    // myCLP.setOption("dim",&dim,"dim");
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

    ParameterListPtr_Type parameterListAll(new Teuchos::ParameterList(*parameterListProblem));
    parameterListAll->setParameters(*parameterListSchwarzSolver);

    // Only 2D for this problem
    int dim = 2;
    string meshType = parameterListProblem->sublist("Parameter").get("Mesh Type", "structured");
    int n;
    int m = parameterListProblem->sublist("Parameter").get("H/h", 5);
    string FEType = parameterListProblem->sublist("Parameter").get("Discretization", "P2");
    auto overlap = parameterListSchwarzSolver->get("Overlap", 1);

    int numProcsCoarseSolve = parameterListProblem->sublist("General").get("Mpi Ranks Coarse", 0);
    int size = comm->getSize() - numProcsCoarseSolve;

    bool verbose(comm->getRank() == 0);
    if (verbose) {
        cout << "###############################################################" << endl;
        cout << "############ Starting Steady Nonlinear Elasticity ... ############" << endl;
        cout << "###############################################################" << endl;
    }

    Teuchos::RCP<Domain<SC, LO, GO, NO>> domain;
    Teuchos::RCP<Domain<SC, LO, GO, NO>> domainP1;
    domainP1.reset(new Domain<SC, LO, GO, NO>(comm, dim));

    MeshPartitioner_Type::DomainPtrArray_Type domainP1Array(1);
    domainP1Array[0] = domainP1;

    ParameterListPtr_Type pListPartitioner = sublist(parameterListAll, "Mesh Partitioner");
    MeshPartitioner<SC, LO, GO, NO> partitionerP1(domainP1Array, pListPartitioner, "P1", dim);

    if (!meshType.compare("structured")) {
        n = (int)(std::pow(size, 1 / 2.) + 100. * Teuchos::ScalarTraits<double>::eps()); // 1/H
        std::vector<double> x(2);
        x[0] = 0.0;
        x[1] = 0.0;
        domainP1.reset(new Domain<SC, LO, GO, NO>(x, 1., 1., comm));
        domainP1Array[0] = domainP1;
        domainP1->buildMesh(1, "Square", dim, FEType, n, m, numProcsCoarseSolve);
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
        /* bcFactory->addBC(zeroDirichlet2D, 1, 0, domain, "Dirichlet", dim); */
        bcFactory->addBC(zeroDirichlet2D, 2, 0, domain, "Dirichlet", dim);
        /* bcFactory->addBC(zeroDirichlet2D, 3, 0, domain, "Dirichlet", dim); */
        bcFactory->addBC(zeroDirichlet2D, 4, 0, domain, "Dirichlet", dim);
    } else if (meshType == "unstructured") {
        // For unstructured meshes and surface force set zeroBC at flag 1 since the surface force is applied at flag
        // 3. For volume forces any boundary flag can be taken. This will affect which side of the geometry is
        // stationary
        /* bcFactory->addBC(zeroDirichlet2D, 1, 0, domain, "Dirichlet", dim); */
        bcFactory->addBC(zeroDirichlet2D, 2, 0, domain, "Dirichlet", dim);
        /* bcFactory->addBC(zeroDirichlet2D, 3, 0, domain, "Dirichlet", dim); */
        bcFactory->addBC(zeroDirichlet2D, 4, 0, domain, "Dirichlet", dim);
    }
    // The current global solution must be set as the Dirichlet BC on the ghost nodes for nonlinear Schwarz solver to
    // correctly solve on the subdomains
    std::vector<double> funcParams{static_cast<double>(dim)};
    bcFactory->addBC(Helper::currentSolutionDirichlet, -99, 0, domain, "Dirichlet", dim, funcParams);

    // Export boundary condition flags for ParaView
    // Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exParaF(new ExporterParaView<SC, LO, GO, NO>());
    // Teuchos::RCP<MultiVector<SC, LO, GO, NO>> exportSolution(new MultiVector<SC, LO, GO,
    // NO>(domain->getMapUnique())); vec_int_ptr_Type BCFlags = domain->getBCFlagUnique(); Teuchos::ArrayRCP<SC> entries
    // = exportSolution->getDataNonConst(0);
    //
    // for (int i = 0; i < entries.size(); i++) {
    //     entries[i] = BCFlags->at(i);
    // }
    //
    // Teuchos::RCP<const MultiVector<SC, LO, GO, NO>> exportSolutionConst = exportSolution;
    // exParaF->setup("Flags", domain->getMesh(), FEType);
    // exParaF->addVariable(exportSolutionConst, "Flags", "Scalar", 1, domain->getMapUnique(),
    // domain->getMapUniqueP2()); exParaF->save(0.0);

    NonLinElasAssFE<SC, LO, GO, NO> nonLinearElasticity(domain, FEType, parameterListAll);

    nonLinearElasticity.addBoundaries(bcFactory);
    nonLinearElasticity.addRhsFunction(rhs2D);

    double force = parameterListAll->sublist("Parameter").get("Volume force", 0.);
    double degree = 0;
    nonLinearElasticity.addParemeterRhs(force);
    nonLinearElasticity.addParemeterRhs(degree);

    nonLinearElasticity.initializeProblem();
    nonLinearElasticity.reInitSpecificProblemVectors(domain->getMapVecFieldOverlappingGhosts());
    nonLinearElasticity.assemble();
    nonLinearElasticity.setBoundaries();
    nonLinearElasticity.setBoundariesRHS();

    std::string nlSolverType = parameterListProblem->sublist("General").get("Linearization", "Newton");
    NonLinearSolver<SC, LO, GO, NO> nlSolver(nlSolverType);
    nlSolver.solve(nonLinearElasticity);
    comm->barrier();

    if (parameterListAll->sublist("General").get("ParaViewExport", false)) {
        Teuchos::RCP<ExporterParaView<SC, LO, GO, NO>> exPara(new ExporterParaView<SC, LO, GO, NO>());

        exPara->setup("displacements", domain->getMesh(), FEType);

        MultiVectorConstPtr_Type valuesSolidConst = nonLinearElasticity.getSolution()->getBlock(0);
        exPara->addVariable(valuesSolidConst, "valuesNonLinElasAssFE", "Vector", dim, domain->getMapUnique());
        exPara->save(0.0);
    }
    return (EXIT_SUCCESS);
}
