#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/Mesh/MeshPartitioner.hpp"
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_Assert.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#include <stdexcept>

typedef unsigned UN;
typedef default_sc SC;
typedef default_lo LO;
typedef default_go GO;
typedef default_no NO;

using namespace FEDD;
using namespace Teuchos;

int main(int argc, char *argv[]) {

    Teuchos::oblackholestream blackhole;
    Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);

    Teuchos::RCP<const Teuchos::Comm<int>> comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();
    const auto myRank = comm->getRank();
    int dim = 2;

    ParameterListPtr_Type parameterListTest = rcp(new ParameterList("problem"));
    ParameterList parameterListParameter("Parameter");
    parameterListParameter.set("Dimension", dim);
    parameterListParameter.set("Discretization", "P1");
    parameterListParameter.set("Mesh Name", "simple_square.mesh");
    parameterListParameter.set("Source Type", "surface");
    parameterListParameter.set("Mesh Delimiter", " ");
    ParameterList parameterListPartitioner("Mesh Partitioner");
    parameterListPartitioner.set("Contiguous", true);
    parameterListPartitioner.set("Mesh 1 Name", "simple_square.mesh");
    parameterListTest->set("Parameter", parameterListParameter);
    parameterListTest->set("Mesh Partitioner", parameterListPartitioner);

    Teuchos::RCP<Domain<SC, LO, GO, NO>> domain;
    domain.reset(new Domain<SC, LO, GO, NO>(comm, dim));

    MeshPartitioner<SC, LO, GO, NO>::DomainPtrArray_Type domainArray(1);
    domainArray[0] = domain;

    ParameterListPtr_Type pListPartitioner = sublist(parameterListTest, "Mesh Partitioner");
    MeshPartitioner<SC, LO, GO, NO> partitioner(domainArray, pListPartitioner, "P1", dim);

    partitioner.readMesh();
    partitioner.buildDualGraph(0);
    partitioner.partitionDualGraphWithOverlap(0, 0);
    partitioner.buildSubdomainFEsAndNodeLists(0);

    auto elementMap = domain->getElementMap();
    auto elementMapOverlapping = domain->getElementMapOverlapping();
    auto mapRepeated = domain->getMapRepeated();
    auto mapOverlapping = domain->getMapOverlapping();
    auto elementsC = domain->getElementsC();

    auto elementMapVecIs = createVector(elementMap->getNodeElementList());
    std::vector<GO> elementMapVec{4, 5};
    if (myRank == 0) {
        TEUCHOS_ASSERT(elementMapVecIs == elementMapVec);
    }

    auto elementMapOverlappingVecIs = createVector(elementMapOverlapping->getNodeElementList());
    std::vector<GO> elementMapOverlappingVec{1, 4, 5, 7};
    if (myRank == 0) {
        TEUCHOS_ASSERT(elementMapOverlappingVecIs == elementMapOverlappingVec);
    }

    auto mapRepeatedVecIs = createVector(mapRepeated->getNodeElementList());
    std::vector<GO> mapRepeatedVec{3, 4, 6, 7};
    if (myRank == 0) {
        TEUCHOS_ASSERT(mapRepeatedVecIs == mapRepeatedVec);
    }
    auto mapOverlappingVecIs = createVector(mapOverlapping->getNodeElementList());
    std::vector<GO> mapOverlappingVec{0, 3, 4, 6, 7, 8};
    if (myRank == 0) {
        TEUCHOS_ASSERT(mapOverlappingVecIs == mapOverlappingVec);
    }

    mapOverlapping->print();
    auto elementsNodeListIs = elementsC->getElementsNodeList();
    vec2D_LO_Type elementsNodeList{{0, 2, 1}, {1, 2, 4}, {1, 3, 4}, {2, 4, 5}};
    if (myRank == 0) {
        TEUCHOS_ASSERT(elementsNodeListIs == elementsNodeList);
    }
    return (EXIT_SUCCESS);
}
