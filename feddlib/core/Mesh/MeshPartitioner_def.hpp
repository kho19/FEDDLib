#ifndef MeshPartitioner_def_hpp
#define MeshPartitioner_def_hpp

#include "MeshPartitioner_decl.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
#include <FROSch_Tools_def.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_Assert.hpp>
#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_RCPBoostSharedPtrConversionsDecl.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_dyn_cast.hpp>
#include <Teuchos_implicit_cast.hpp>
#include <Tpetra_CombineMode.hpp>
#include <Tpetra_Export_decl.hpp>
#include <Xpetra_ConfigDefs.hpp>
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_CrsGraphFactory.hpp>
#include <Xpetra_ExportFactory.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MapFactory_decl.hpp>
#include <Xpetra_Map_def.hpp>
#include <Xpetra_TpetraCrsGraph_decl.hpp>
#include <Zoltan2_MeshAdapter.hpp>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

/*!
 Definition of MeshPartitioner

 @brief  MeshPartitioner
 @author Christian Hochmuth
 @version 1.0
 @copyright CH
 */

using namespace std;
namespace FEDD {
template <class SC, class LO, class GO, class NO> MeshPartitioner<SC, LO, GO, NO>::MeshPartitioner() {}

template <class SC, class LO, class GO, class NO>
MeshPartitioner<SC, LO, GO, NO>::MeshPartitioner(DomainPtrArray_Type domains, ParameterListPtr_Type pL,
                                                 std::string feType, int dimension) {
    domains_ = domains;
    pList_ = pL;
    feType_ = feType;
    comm_ = domains_[0]->getComm();
    rankRanges_.resize(domains_.size());
    dim_ = dimension;
}

template <class SC, class LO, class GO, class NO> MeshPartitioner<SC, LO, GO, NO>::~MeshPartitioner() {}

template <class SC, class LO, class GO, class NO> void MeshPartitioner<SC, LO, GO, NO>::readAndPartition(int volumeID) {
    if (volumeID != 10) {
        if (this->comm_->getRank() == 0) {
            cout << " #### WARNING: The volumeID was set manually and is no longer 10. Please make sure your volumeID "
                    "corresponds to the volumeID in your mesh file. #### "
                 << endl;
        }
    }
    // Read
    string delimiter = pList_->get("Delimiter", " ");
    for (int i = 0; i < domains_.size(); i++) {
        std::string meshName = pList_->get("Mesh " + std::to_string(i + 1) + " Name", "noName");
        TEUCHOS_TEST_FOR_EXCEPTION(meshName == "noName", std::runtime_error, "No mesh name given.");
        domains_[i]->initializeUnstructuredMesh(domains_[i]->getDimension(), "P1",
                                                volumeID); // we only allow to read P1 meshes.
        domains_[i]->readMeshSize(meshName, delimiter);
    }

    this->determineRanks();

    for (int i = 0; i < domains_.size(); i++) {
        this->readAndPartitionMesh(i);
        domains_[i]->getMesh()->rankRange_ = rankRanges_[i];
    }
}

template <class SC, class LO, class GO, class NO> void MeshPartitioner<SC, LO, GO, NO>::determineRanks() {
    bool verbose(comm_->getRank() == 0);
    vec_int_Type fractions(domains_.size(), 0);
    bool autoPartition = pList_->get("Automatic partition", false);
    if (autoPartition) {
        // determine sum of elements and fractions based of domain contributions
        GO sumElements = 0;
        for (int i = 0; i < domains_.size(); i++)
            sumElements += domains_[i]->getNumElementsGlobal();

        for (int i = 0; i < fractions.size(); i++)
            fractions[i] = (domains_[i]->getNumElementsGlobal() * 100) / sumElements;

        int diff = std::accumulate(fractions.begin(), fractions.end(), 0) - 100;
        auto iterator = fractions.begin();
        while (diff > 0) {
            (*iterator)--;
            iterator++;
            diff--;
        }
        iterator = fractions.begin();
        while (diff < 0) {
            (*iterator)++;
            iterator++;
            diff++;
        }

        this->determineRanksFromFractions(fractions);

        if (verbose) {
            std::cout << "\t --- ---------------- ---" << std::endl;
            std::cout << "\t --- Mesh Partitioner ---" << std::endl;
            std::cout << "\t --- ---------------- ---" << std::endl;
            std::cout << "\t --- Automatic partition for " << comm_->getSize() << " ranks" << std::endl;
            for (int i = 0; i < domains_.size(); i++) {
                std::cout << "\t --- Fraction mesh " << to_string(i + 1) << " : " << fractions[i]
                          << " of 100; rank range: " << get<0>(rankRanges_[i]) << " to " << get<1>(rankRanges_[i])
                          << std::endl;
            }
        }

    } else if (autoPartition == false && pList_->get("Mesh 1 fraction ranks", -1) >= 0) {
        for (int i = 0; fractions.size(); i++)
            fractions[i] = pList_->get("Mesh " + std::to_string(i + 1) + " fraction ranks", -1);

        TEUCHOS_TEST_FOR_EXCEPTION(std::accumulate(fractions.begin(), fractions.end(), 0) != 100, std::runtime_error,
                                   "Fractions do not sum up to 100!");
        this->determineRanksFromFractions(fractions);

        if (verbose) {
            std::cout << "\t --- ---------------- ---" << std::endl;
            std::cout << "\t --- Mesh Partitioner ---" << std::endl;
            std::cout << "\t --- ---------------- ---" << std::endl;
            std::cout << "\t --- Fraction partition for " << comm_->getSize() << " ranks" << std::endl;
            for (int i = 0; i < domains_.size(); i++) {
                std::cout << "\t --- Fraction mesh " << to_string(i + 1) << " : " << fractions[i]
                          << " of 100; rank range: " << get<0>(rankRanges_[i]) << " to " << get<1>(rankRanges_[i])
                          << std::endl;
            }
        }
    } else if (autoPartition == false && pList_->get("Mesh 1 fraction ranks", -1) < 0 &&
               pList_->get("Mesh 1 number ranks", -1) > 0) {
        int size = comm_->getSize();
        vec_int_Type numberRanks(domains_.size());
        for (int i = 0; i < numberRanks.size(); i++)
            numberRanks[i] = pList_->get("Mesh " + std::to_string(i + 1) + " number ranks", 0);

        TEUCHOS_TEST_FOR_EXCEPTION(std::accumulate(numberRanks.begin(), numberRanks.end(), 0) > size,
                                   std::runtime_error, "Too many ranks requested for mesh partition!");
        this->determineRanksFromNumberRanks(numberRanks);
        if (verbose) {
            std::cout << "\t --- ---------------- ---" << std::endl;
            std::cout << "\t --- Mesh Partitioner ---" << std::endl;
            std::cout << "\t --- ---------------- ---" << std::endl;
            std::cout << "\t --- Rank number partition for " << comm_->getSize() << " ranks" << std::endl;
            for (int i = 0; i < domains_.size(); i++) {
                std::cout << "\t --- Rank range mesh " << to_string(i + 1) << " :" << get<0>(rankRanges_[i]) << " to "
                          << get<1>(rankRanges_[i]) << std::endl;
            }
        }

    } else {
        for (int i = 0; i < domains_.size(); i++)
            rankRanges_[i] = std::make_tuple(0, comm_->getSize() - 1);
        if (verbose) {
            std::cout << "\t --- ---------------- ---" << std::endl;
            std::cout << "\t --- Mesh Partitioner ---" << std::endl;
            std::cout << "\t --- ---------------- ---" << std::endl;
            std::cout << "\t --- Every mesh on every rank" << std::endl;
        }
    }
}

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::determineRanksFromFractions(vec_int_Type &fractions) {

    int lowerRank = 0;
    int size = comm_->getSize();
    int upperRank = 0;
    for (int i = 0; i < fractions.size(); i++) {
        upperRank = lowerRank + fractions[i] / 100. * size - 1;
        if (upperRank < lowerRank)
            upperRank++;
        rankRanges_[i] = std::make_tuple(lowerRank, upperRank);
        if (size > 1)
            lowerRank = upperRank + 1;
        else
            lowerRank = upperRank;
    }
    int startLoc = 0;
    while (upperRank > size - 1) {
        for (int i = startLoc; i < rankRanges_.size(); i++) {
            if (i > 0)
                std::get<0>(rankRanges_[i])--;
            std::get<1>(rankRanges_[i])--;
        }
        startLoc++;
        upperRank--;
    }
    startLoc = 0;
    while (upperRank < size - 1) {
        for (int i = startLoc; i < rankRanges_.size(); i++) {
            if (i > 0)
                std::get<0>(rankRanges_[i])++;
            std::get<1>(rankRanges_[i])++;
        }
        startLoc++;
        upperRank++;
    }
}

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::determineRanksFromNumberRanks(vec_int_Type &numberRanks) {

    int lowerRank = 0;
    int size = comm_->getSize();
    int upperRank = 0;
    for (int i = 0; i < numberRanks.size(); i++) {
        upperRank = lowerRank + numberRanks[i] - 1;
        rankRanges_[i] = std::make_tuple(lowerRank, upperRank);
        lowerRank = upperRank + 1;
    }

    int startLoc = 0;
    while (upperRank > size - 1) {
        for (int i = startLoc; i < rankRanges_.size(); i++) {
            if (i > 0)
                std::get<0>(rankRanges_[i])--;
            std::get<1>(rankRanges_[i])--;
        }
        startLoc++;
        upperRank--;
    }
    startLoc = 0;
    while (upperRank < size - 1) {
        for (int i = startLoc; i < rankRanges_.size(); i++) {
            if (i > 0)
                std::get<0>(rankRanges_[i])++;
            std::get<1>(rankRanges_[i])++;
        }
        startLoc++;
        upperRank++;
    }
}

/// Reading and partioning of the mesh. Input File is .mesh. Reading is serial and at some point the mesh entities are
/// distributed along the processors.

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::readAndPartitionMesh(int meshNumber) {

    typedef Teuchos::OrdinalTraits<GO> OTGO;

#ifdef UNDERLYING_LIB_TPETRA
    string underlyingLib = "Tpetra";
#endif

    MeshUnstrPtr_Type meshUnstr = Teuchos::rcp_dynamic_cast<MeshUnstr_Type>(domains_[meshNumber]->getMesh());

    // Reading nodes
    meshUnstr->readMeshEntity("node");
    // We delete the point at this point. We only need the flags to determine surface elements. We will load them again
    // later.
    meshUnstr->pointsRep_.reset();
    // Reading elements
    meshUnstr->readMeshEntity("element");
    // Reading surfaces
    meshUnstr->readMeshEntity("surface");
    // Reading line segments
    meshUnstr->readMeshEntity("line");

    bool verbose(comm_->getRank() == 0);
    bool buildEdges = pList_->get("Build Edge List", true);
    bool buildSurfaces = pList_->get("Build Surface List", true);

    // Adding surface as subelement to elements
    if (buildSurfaces)
        this->setSurfacesToElements(meshNumber);
    else
        meshUnstr->deleteSurfaceElements();

    // Serially distributed elements
    ElementsPtr_Type elementsMesh = meshUnstr->getElementsC();

    // Setup Metis
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_SEED] = 666;
    options[METIS_OPTION_CONTIG] =
        pList_->get("Contiguous", false); // 0: Does not force contiguous partitions; 1: Forces contiguous partitions.
    options[METIS_OPTION_MINCONN] = 0;    // 1: Explicitly minimize the maximum connectivity.
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT; // or METIS_OBJTYPE_VOL
    //    options[METIS_OPTION_RTYPE] = METIS_RTYPE_GREEDY;
    options[METIS_OPTION_NITER] = 50; // default is 10
    options[METIS_OPTION_CCORDER] = 1;
    idx_t ne = meshUnstr->getNumElementsGlobal();               // Global number of elements
    idx_t nn = meshUnstr->getNumGlobalNodes();                  // Global number of nodes
    idx_t ned = meshUnstr->getEdgeElements()->numberElements(); // Global number of edges

    int dim = meshUnstr->getDimension();
    std::string FEType = domains_[meshNumber]->getFEType();

    // Setup for paritioning with metis
    vec_idx_Type eptr_vec(0); // Vector for local elements ptr (local is still global at this point)
    vec_idx_Type eind_vec(0); // Vector for local elements ids

    makeContinuousElements(elementsMesh, eind_vec, eptr_vec);

    idx_t *eptr = &eptr_vec.at(0);
    idx_t *eind = &eind_vec.at(0);

    idx_t ncommon;
    int orderSurface;
    if (dim == 2) {
        if (FEType == "P1") {
            ncommon = 2;
        } else if (FEType == "P2") {
            ncommon = 3;
        }
    } else if (dim == 3) {
        if (FEType == "P1") {
            ncommon = 3;
        } else if (FEType == "P2") {
            ncommon = 6;
        }
    } else
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Wrong Dimension.");

    idx_t objval = 0;
    // Partition vectors for elements and nodes
    // For each element/node the vector stores the partition index it has been assigned
    vec_idx_Type epart(ne, -1);
    vec_idx_Type npart(nn, -1);

    if (verbose)
        cout << "-- Start partitioning with Metis ... " << flush;

    {
        FEDD_TIMER_START(partitionTimer, " : MeshPartitioner : Partition Elements");
        // Number of available ranks = number of parts to partition into
        idx_t nparts = std::get<1>(rankRanges_[meshNumber]) - std::get<0>(rankRanges_[meshNumber]) + 1;
        if (nparts > 1) {
            // upperRange - lowerRange +1
            idx_t returnCode = METIS_PartMeshDual(&ne, &nn, eptr, eind, NULL, NULL, &ncommon, &nparts, NULL, options,
                                                  &objval, &epart[0], &npart[0]);
            if (verbose)
                cout << "\n--\t Metis return code: " << returnCode;
        } else {
            for (int i = 0; i < ne; i++)
                epart[i] = 0;
        }
    }

    if (verbose) {
        cout << "\n--\t objval: " << objval << endl;
        cout << "-- done!" << endl;
    }

    if (verbose)
        cout << "-- Set Elements ... " << flush;

    vec_GO_Type locepart(0);
    vec_GO_Type pointsRepIndices(0);

    // Get global IDs of element's nodes
    for (int i = 0; i < ne; i++) {
        // If element has been assigned to current rank, store global index and nodes
        if (epart[i] == comm_->getRank() - std::get<0>(rankRanges_[meshNumber])) {
            locepart.push_back(i);
            for (int j = eptr[i]; j < eptr[i + 1]; j++)
                pointsRepIndices.push_back(eind[j]); // Ids of element nodes, globalIDs
        }
    }
    // TODO KHo why erase the vectors here? eind points to the underlying array and is used later.
    eind_vec.erase(eind_vec.begin(), eind_vec.end());
    eptr_vec.erase(eptr_vec.begin(), eptr_vec.end());

    // Sorting ids with global and corresponding local values to create repeated map
    make_unique(pointsRepIndices);
    if (verbose)
        cout << "done!" << endl;

    // Building repeated node map
    Teuchos::ArrayView<GO> pointsRepGlobMapping = Teuchos::arrayViewFromVector(pointsRepIndices);
    meshUnstr->mapRepeated_.reset(
        new Map<LO, GO, NO>(underlyingLib, OTGO::invalid(), pointsRepGlobMapping, 0, this->comm_));
    MapConstPtr_Type mapRepeated = meshUnstr->mapRepeated_;

    // TODO KHo why create another rcp to the elements? elementsMesh will still point to the old elements after
    // elementsC_ has been reset.
    //
    // We keep the global elements if we want to build edges later. Otherwise they will be
    // deleted
    ElementsPtr_Type elementsGlobal = Teuchos::rcp(new Elements_Type(*elementsMesh));

    // Resetting elements to add the corresponding local IDs instead of global ones since global elements were stored
    // when reading in the mesh
    meshUnstr->elementsC_.reset(new Elements(FEType, dim));
    {
        // elementsGlobalMapping -> elements per Processor i.e. global indices of the elements owned by the current rank
        // Pass invalid since the global number of elements is unknown (includes repetition across the ranks)
        Teuchos::ArrayView<GO> elementsGlobalMapping = Teuchos::arrayViewFromVector(locepart);
        meshUnstr->elementMap_.reset(
            new Map<LO, GO, NO>(underlyingLib, OTGO::invalid(), elementsGlobalMapping, 0, this->comm_));

        {
            int localSurfaceCounter = 0;
            for (int i = 0; i < locepart.size(); i++) {
                // Build local element and save
                std::vector<int> tmpElement;
                // Iterate over the nodes in an element
                for (int j = eptr[locepart.at(i)]; j < eptr[locepart.at(i) + 1]; j++) {
                    // Map the node index from global to local and save
                    int index = mapRepeated->getLocalElement(Teuchos::implicit_cast<long long>(eind[j]));
                    tmpElement.push_back(index);
                }
                FiniteElement fe(tmpElement, elementsGlobal->getElement(locepart.at(i)).getFlag());
                // Convert global IDs of (old) globally owned subelements to local IDs
                if (buildSurfaces) {
                    FiniteElement feGlobalIDs = elementsGlobal->getElement(locepart.at(i));
                    if (feGlobalIDs.subElementsInitialized()) {
                        ElementsPtr_Type subEl = feGlobalIDs.getSubElements();
                        subEl->globalToLocalIDs(mapRepeated);
                        fe.setSubElements(subEl);
                    }
                }
                meshUnstr->elementsC_->addElement(fe);
            }
        }
    }

    // Now distribute the coordinates and flags correctly
    // Reread the nodes, as they were deleted earlier
    meshUnstr->readMeshEntity("node");

    if (verbose)
        cout << "-- Build Repeated Points Volume ... " << flush;

    // Build the unique map
    meshUnstr->mapUnique_ = meshUnstr->mapRepeated_->buildUniqueMap(rankRanges_[meshNumber]);

    if (verbose)
        cout << "-- Building unique & repeated points ... " << flush;
    {
        vec2D_dbl_Type points = *meshUnstr->getPointsRepeated();
        vec_int_Type flags = *meshUnstr->getBCFlagRepeated();
        meshUnstr->pointsRep_.reset(new std::vector<std::vector<double>>(meshUnstr->mapRepeated_->getNodeNumElements(),
                                                                         std::vector<double>(dim, -1.)));
        meshUnstr->bcFlagRep_.reset(new std::vector<int>(meshUnstr->mapRepeated_->getNodeNumElements(), 0));

        int pointIDcont;
        for (int i = 0; i < pointsRepIndices.size(); i++) {
            pointIDcont = pointsRepIndices.at(i);
            for (int j = 0; j < dim; j++)
                meshUnstr->pointsRep_->at(i).at(j) = points[pointIDcont][j];
            meshUnstr->bcFlagRep_->at(i) = flags[pointIDcont];
        }
    }

    // Setting unique points and flags
    meshUnstr->pointsUni_.reset(new std::vector<std::vector<double>>(meshUnstr->mapUnique_->getNodeNumElements(),
                                                                     std::vector<double>(dim, -1.)));
    meshUnstr->bcFlagUni_.reset(new std::vector<int>(meshUnstr->mapUnique_->getNodeNumElements(), 0));
    GO indexGlobal;
    MapConstPtr_Type map = meshUnstr->getMapRepeated();
    vec2D_dbl_ptr_Type pointsRep = meshUnstr->pointsRep_;
    for (int i = 0; i < meshUnstr->mapUnique_->getNodeNumElements(); i++) {
        indexGlobal = meshUnstr->mapUnique_->getGlobalElement(i);
        for (int j = 0; j < dim; j++) {
            meshUnstr->pointsUni_->at(i).at(j) = pointsRep->at(map->getLocalElement(indexGlobal)).at(j);
        }
        meshUnstr->bcFlagUni_->at(i) = meshUnstr->bcFlagRep_->at(map->getLocalElement(indexGlobal));
    }

    // Finally build the edges. Since they rely on nodes and elements, edges are built last to avoid any local and
    // global IDs mix ups
    if (!buildEdges)
        elementsGlobal.reset();

    locepart.erase(locepart.begin(), locepart.end());
    if (verbose)
        cout << "done!" << endl;

    if (buildSurfaces) {
        this->setEdgesToSurfaces(
            meshNumber); // Adding edges as subelements in the 3D case. All dim-1-Subelements were already set
    } else
        meshUnstr->deleteSurfaceElements();

    if (buildEdges) {
        if (verbose)
            cout << "-- Build edge element list ..." << flush;

        // Generates a list of objects of class edge. These are a specialisation of Elements. They are defined by
        // starting and ending node and contain information on surrounding elements.
        buildEdgeListParallel(meshUnstr, elementsGlobal);

        if (verbose)
            cout << "\n done!" << endl;

        MapConstPtr_Type elementMap = meshUnstr->getElementMap();

        FEDD_TIMER_START(partitionEdgesTimer, " : MeshPartitioner : Partition Edges");
        meshUnstr->getEdgeElements()->partitionEdges(elementMap, mapRepeated);
        FEDD_TIMER_STOP(partitionEdgesTimer);

        // Global Edge IDs of local elements
        vec_GO_Type locedpart(0);
        // edge global indices on different processors
        for (int i = 0; i < meshUnstr->getEdgeElements()->numberElements(); i++) {
            locedpart.push_back(meshUnstr->getEdgeElements()->getGlobalID((LO)i));
        }

        // Setup for the EdgeMap
        Teuchos::ArrayView<GO> edgesGlobalMapping = Teuchos::arrayViewFromVector(locedpart);
        meshUnstr->edgeMap_.reset(new Map<LO, GO, NO>(underlyingLib, (GO)-1, edgesGlobalMapping, 0, this->comm_));
    }

    if (verbose)
        cout << "-- Partition interface ... " << flush;
    meshUnstr->partitionInterface();

    if (verbose)
        cout << "done!" << endl;
}

/// Function that (addionally) adds edges as subelements in the 3D case. This is relevant when the .mesh file also
/// contains edge information in 3D. Otherwise edges are not set as subelements in 3D.
template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::setEdgesToSurfaces(int meshNumber) {
    bool verbose(comm_->getRank() == 0);
    MeshUnstrPtr_Type meshUnstr = Teuchos::rcp_dynamic_cast<MeshUnstr_Type>(domains_[meshNumber]->getMesh());
    ElementsPtr_Type elementsMesh = meshUnstr->getElementsC();
    MapConstPtr_Type mapRepeated = meshUnstr->mapRepeated_;
    if (verbose)
        cout << "-- Set edges of surfaces of elements ... " << flush;

    FEDD_TIMER_START(surfacesTimer, " : MeshPartitioner : Set Surfaces of Edge Elements");
    vec2D_int_Type localEdgeIDPermutation;
    setLocalSurfaceEdgeIndices(localEdgeIDPermutation, meshUnstr->getEdgeElementOrder());

    int volumeID = meshUnstr->volumeID_;

    ElementsPtr_Type elements = meshUnstr->getElementsC();
    ElementsPtr_Type edgeElements = meshUnstr->getSurfaceEdgeElements();

    /* First, we convert the surface edge elements to a 2D array so we can use std::find.*/
    // Can we use/implement find for Elements_Type?
    vec2D_int_Type edgeElements_vec(edgeElements->numberElements());
    vec_int_Type edgeElementsFlag_vec(edgeElements->numberElements());
    for (int i = 0; i < edgeElements_vec.size(); i++) {
        vec_int_Type edge = edgeElements->getElement(i).getVectorNodeListNonConst();
        std::sort(edge.begin(), edge.end());
        edgeElements_vec.at(i) = edge;
        edgeElementsFlag_vec.at(i) = edgeElements->getElement(i).getFlag();
    }

    vec_int_ptr_Type flags = meshUnstr->bcFlagRep_;
    int elementEdgeSurfaceCounter;
    for (int i = 0; i < elements->numberElements(); i++) {
        elementEdgeSurfaceCounter = 0;
        bool mark = false;
        for (int j = 0; j < elements->getElement(i).size(); j++) {
            if (flags->at(elements->getElement(i).getNode(j)) < volumeID)
                elementEdgeSurfaceCounter++;
        }
        if (elementEdgeSurfaceCounter >= meshUnstr->getEdgeElementOrder()) {
            // We want to find all surfaces of element i and set the surfaces to the element
            findAndSetSurfaceEdges(edgeElements_vec, edgeElementsFlag_vec, elements->getElement(i),
                                   localEdgeIDPermutation, mapRepeated);
        }
    }
    if (verbose)
        cout << "done!" << endl;
}

/// Adding surfaces as subelements to the corresponding elements. This adresses surfaces in 3D or edges in 2D.
/// If in the 3D case edges are also part of the .mesh file, they will be added as subelements later.
template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::setSurfacesToElements(int meshNumber) {

    bool verbose(comm_->getRank() == 0);
    MeshUnstrPtr_Type meshUnstr = Teuchos::rcp_dynamic_cast<MeshUnstr_Type>(domains_[meshNumber]->getMesh());
    ElementsPtr_Type elementsMesh = meshUnstr->getElementsC(); // Previously read Elements

    if (verbose)
        cout << "-- Set surfaces of elements ... " << flush;

    FEDD_TIMER_START(surfacesTimer, " : MeshPartitioner : Set Surfaces of Elements");

    vec2D_int_Type localSurfaceIDPermutation;
    // get permutations
    setLocalSurfaceIndices(localSurfaceIDPermutation, meshUnstr->getSurfaceElementOrder());

    int volumeID = meshUnstr->volumeID_;
    ElementsPtr_Type surfaceElements = meshUnstr->getSurfaceElements();

    /* First, we convert the surface Elements to a 2D array so we can use std::find.
     and partition it at the same time. We use a simple linear partition. This is done to reduce
     times spend for std::find. The result is then communicated. We therefore use the unpartitoned elements*/
    // Can we use/implement find for Elements_Type?

    int size = this->comm_->getSize();

    LO numSurfaceEl = surfaceElements->numberElements(); // / size;
    /*LO rest = surfaceElements->numberElements() % size;

    vec_GO_Type offsetVec(size);
    for (int i=0; i<size; i++) {
        offsetVec[i] = numSurfaceEl * i;
        if ( i<rest && i == this->comm_->getRank() ) {
            numSurfaceEl++;
            offsetVec[i]+=i;
        }
        else
            offsetVec[i]+=rest;
    }*/

    vec2D_int_Type surfElements_vec(numSurfaceEl);
    vec2D_int_Type surfElements_vec_sorted(numSurfaceEl);

    vec_int_Type surfElementsFlag_vec(numSurfaceEl);
    vec_GO_Type offsetVec(size);
    int offset = offsetVec[this->comm_->getRank()];

    for (int i=0; i<surfElements_vec.size(); i++){
        vec_int_Type surface = surfaceElements->getElement(i ).getVectorNodeListNonConst(); // surfaceElements->getElement(i + offset).getVectorNodeListNonConst();
        surfElements_vec.at(i)  = surface;
        std::sort( surface.begin(), surface.end() ); // We need to maintain a consistent numbering in the surface elements, so we use a sorted and unsorted vector
        surfElements_vec_sorted.at(i) = surface;
        surfElementsFlag_vec.at(i) =
            surfaceElements->getElement(i).getFlag(); // surfaceElements->getElement(i + offset).getFlag();
    }

    // Delete the surface elements. They will be added to the elements in the following loop.
    surfaceElements.reset();
    vec_int_ptr_Type flags = meshUnstr->bcFlagRep_;

    int elementSurfaceCounter;
    int surfaceElOrder = meshUnstr->getSurfaceElementOrder();
    for (int i = 0; i < elementsMesh->numberElements(); i++) {
        elementSurfaceCounter = 0;
        for (int j = 0; j < elementsMesh->getElement(i).size(); j++) {
            if (flags->at(elementsMesh->getElement(i).getNode(j)) < volumeID)
                elementSurfaceCounter++;
        }

        if (elementSurfaceCounter >= surfaceElOrder) {
            FEDD_TIMER_START(findSurfacesTimer, " : MeshPartitioner : Find and Set Surfaces");
            // We want to find all surfaces of element i and set the surfaces to the element
            this->findAndSetSurfacesPartitioned(surfElements_vec_sorted, surfElements_vec, surfElementsFlag_vec,
                                                elementsMesh->getElement(i), localSurfaceIDPermutation, offsetVec, i);
        }
    }

    if (verbose)
        cout << "done!" << endl;
}

/// Function that sets edges (2D) or surfaces(3D) to the corresponding element. It determines for the specific element
/// 'element' if it has a surface with a non volumeflag that can be set as subelement
template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::findAndSetSurfacesPartitioned(
    vec2D_int_Type &surfElements_vec, vec2D_int_Type &surfElements_vec_unsorted, vec_int_Type &surfElementsFlag_vec,
    FiniteElement &element, vec2D_int_Type &permutation, vec_GO_Type &linearSurfacePartitionOffset, int globalElID) {

    // In general we look through the different permutations the input element 'element' can have and if they correspond
    // to a surface. The mesh's surface elements 'surfElements_vec' are then used to determine the corresponding surface
    // If found, the nodes are then used to build the subelement and the corresponding surfElementFlag is set.
    // The Ids are global at this point, as the values are not distributed yet.

    int loc, id1Glob, id2Glob, id3Glob;
    int size = this->comm_->getSize();
    vec_int_Type locAll(size);
    if (dim_ == 2) {
        for (int j = 0; j < permutation.size(); j++) {
            id1Glob = element.getNode(permutation.at(j).at(0));
            id2Glob = element.getNode(permutation.at(j).at(1));

            vec_int_Type tmpSurface(2);
            if (id1Glob > id2Glob) {
                tmpSurface[0] = id2Glob;
                tmpSurface[1] = id1Glob;
            } else {
                tmpSurface[0] = id1Glob;
                tmpSurface[1] = id2Glob;
            }

            loc = searchInSurfaces(surfElements_vec, tmpSurface);

            Teuchos::gatherAll<int, int>(*this->comm_, 1, &loc, locAll.size(), &locAll[0]);

            int surfaceRank = -1;
            int counter = 0;
            while (surfaceRank < 0 && counter < size) {
                if (locAll[counter] > -1)
                    surfaceRank = counter;
                counter++;
            }
            int surfFlag = -1;
            if (loc > -1)
                surfFlag = surfElementsFlag_vec[loc];

            if (surfaceRank > -1) {
                Teuchos::broadcast<int, int>(*this->comm_, surfaceRank, 1, &loc);
                Teuchos::broadcast<int, int>(*this->comm_, surfaceRank, 1, &surfFlag);

                FiniteElement feSurface(tmpSurface, surfFlag);
                if (!element.subElementsInitialized())
                    element.initializeSubElements("P1", 1); // only P1 for now

                element.addSubElement(feSurface);
            }
        }
    } else if (dim_ == 3) {
        for (int j = 0; j < permutation.size(); j++) {

            id1Glob = element.getNode(permutation.at(j).at(0));
            id2Glob = element.getNode(permutation.at(j).at(1));
            id3Glob = element.getNode(permutation.at(j).at(2));

            vec_int_Type tmpSurface = {id1Glob, id2Glob, id3Glob};
            sort(tmpSurface.begin(), tmpSurface.end());
            loc = searchInSurfaces(surfElements_vec, tmpSurface);
            /*Teuchos::gatherAll<int,int>( *this->comm_, 1, &loc, locAll.size(), &locAll[0] );

            int surfaceRank = -1;
            int counter = 0;
            while (surfaceRank<0 && counter<size) {
                if (locAll[counter] > -1)
                    surfaceRank = counter;
                counter++;
            }
            int surfFlag = -1;
            if (loc>-1)
                surfFlag = surfElementsFlag_vec[loc];

            if (surfaceRank>-1) {*/
            if (loc > -1) {
                // Teuchos::broadcast<int,int>(*this->comm_,surfaceRank,1,&loc);
                // Teuchos::broadcast<int,int>(*this->comm_,surfaceRank,1,&surfFlag);

                int surfFlag = surfElementsFlag_vec[loc];
                // cout << " Surfaces set to elements on Proc  " << this->comm_->getRank() << " "  <<
                // surfElements_vec_unsorted[loc][0] << " " << surfElements_vec_unsorted[loc][1] << " " <<
                // surfElements_vec_unsorted[loc][2] << endl;
                FiniteElement feSurface(surfElements_vec_unsorted[loc], surfFlag);
                if (!element.subElementsInitialized())
                    element.initializeSubElements("P1", 2); // only P1 for now

                element.addSubElement(feSurface);
            }
        }
    }
}

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::buildEdgeListParallel(MeshUnstrPtr_Type mesh, ElementsPtr_Type elementsGlobal) {
    FEDD_TIMER_START(edgeListTimer, " : MeshReader : Build Edge List");
    ElementsPtr_Type elements = mesh->getElementsC();

    TEUCHOS_TEST_FOR_EXCEPTION(elements->getFiniteElementType() != "P1", std::runtime_error,
                               "Unknown discretization for method buildEdgeList(...).");
    CommConstPtr_Type comm = mesh->getComm();
    bool verbose(comm->getRank() == 0);

    MapConstPtr_Type repeatedMap = mesh->getMapRepeated();
    // Building local edges with repeated node list
    vec2D_int_Type localEdgeIndices(0);
    setLocalEdgeIndices(localEdgeIndices);
    EdgeElementsPtr_Type edges = Teuchos::rcp(new EdgeElements_Type());
    for (int i = 0; i < elementsGlobal->numberElements(); i++) {
        for (int j = 0; j < localEdgeIndices.size(); j++) {

            int id1 = elementsGlobal->getElement(i).getNode(localEdgeIndices[j][0]);
            int id2 = elementsGlobal->getElement(i).getNode(localEdgeIndices[j][1]);
            vec_int_Type edgeVec(2);
            if (id1 < id2) {
                edgeVec[0] = id1;
                edgeVec[1] = id2;
            } else {
                edgeVec[0] = id2;
                edgeVec[1] = id1;
            }

            FiniteElement edge(edgeVec);
            edges->addEdge(edge, i);
        }
    }
    // we do not need elementsGlobal anymore
    elementsGlobal.reset();

    vec2D_GO_Type combinedEdgeElements;
    FEDD_TIMER_START(edgeListUniqueTimer, " : MeshReader : Make Edge List Unique");
    edges->sortUniqueAndSetGlobalIDs(combinedEdgeElements);
    FEDD_TIMER_STOP(edgeListUniqueTimer);
    // Next we need to communicate all edge information. This will not scale at all!

    edges->setElementsEdges(combinedEdgeElements);

    mesh->setEdgeElements(edges);
};

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::setLocalEdgeIndices(vec2D_int_Type &localEdgeIndices) {
    if (dim_ == 2) {
        localEdgeIndices.resize(3, vec_int_Type(2, -1));
        localEdgeIndices.at(0).at(0) = 0;
        localEdgeIndices.at(0).at(1) = 1;
        localEdgeIndices.at(1).at(0) = 0;
        localEdgeIndices.at(1).at(1) = 2;
        localEdgeIndices.at(2).at(0) = 1;
        localEdgeIndices.at(2).at(1) = 2;
    } else if (dim_ == 3) {
        localEdgeIndices.resize(6, vec_int_Type(2, -1));
        localEdgeIndices.at(0).at(0) = 0;
        localEdgeIndices.at(0).at(1) = 1;
        localEdgeIndices.at(1).at(0) = 0;
        localEdgeIndices.at(1).at(1) = 2;
        localEdgeIndices.at(2).at(0) = 1;
        localEdgeIndices.at(2).at(1) = 2;
        localEdgeIndices.at(3).at(0) = 0;
        localEdgeIndices.at(3).at(1) = 3;
        localEdgeIndices.at(4).at(0) = 1;
        localEdgeIndices.at(4).at(1) = 3;
        localEdgeIndices.at(5).at(0) = 2;
        localEdgeIndices.at(5).at(1) = 3;
    }
}

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::makeContinuousElements(ElementsPtr_Type elements, vec_idx_Type &eind_vec,
                                                             vec_idx_Type &eptr_vec) {

    int nodesPerElement = elements->nodesPerElement();

    int elcounter = 0;
    for (int i = 0; i < elements->numberElements(); i++) {
        for (int j = 0; j < nodesPerElement; j++) {
            eind_vec.push_back(elements->getElement(i).getNode(j));
        }
        eptr_vec.push_back(elcounter);
        elcounter += nodesPerElement;
    }
    eptr_vec.push_back(elcounter);
}

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::setLocalSurfaceEdgeIndices(vec2D_int_Type &localSurfaceEdgeIndices,
                                                                 int edgesElementOrder) {

    if (dim_ == 3) {

        if (edgesElementOrder == 2) { // P1
            localSurfaceEdgeIndices.resize(6, vec_int_Type(2, -1));
            localSurfaceEdgeIndices.at(0).at(0) = 0;
            localSurfaceEdgeIndices.at(0).at(1) = 1;
            localSurfaceEdgeIndices.at(1).at(0) = 0;
            localSurfaceEdgeIndices.at(1).at(1) = 2;
            localSurfaceEdgeIndices.at(2).at(0) = 0;
            localSurfaceEdgeIndices.at(2).at(1) = 3;
            localSurfaceEdgeIndices.at(3).at(0) = 1;
            localSurfaceEdgeIndices.at(3).at(1) = 2;
            localSurfaceEdgeIndices.at(4).at(0) = 1;
            localSurfaceEdgeIndices.at(4).at(1) = 3;
            localSurfaceEdgeIndices.at(5).at(0) = 2;
            localSurfaceEdgeIndices.at(5).at(1) = 3;
        }
    }
}

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::findAndSetSurfaceEdges(vec2D_int_Type &edgeElements_vec,
                                                             vec_int_Type &edgeElementsFlag_vec, FiniteElement &element,
                                                             vec2D_int_Type &permutation,
                                                             MapConstPtr_Type mapRepeated) {

    int loc, id1Glob, id2Glob;
    if (dim_ == 3){
        for (int j=0; j<permutation.size(); j++) {           
            id1Glob = mapRepeated->getGlobalElement( element.getNode( permutation.at(j).at(0) ) );
            id2Glob = mapRepeated->getGlobalElement( element.getNode( permutation.at(j).at(1) ) );
            vec_int_Type tmpEdge(0);
            if (id2Glob > id1Glob)
                tmpEdge = {id1Glob, id2Glob};
            else
                tmpEdge = {id2Glob , id1Glob};
            
            loc = searchInSurfaces( edgeElements_vec, tmpEdge );
            
            if (loc>-1) {
                
                int id1 = element.getNode( permutation.at(j).at(0) );
                int id2 = element.getNode( permutation.at(j).at(1) );
                vec_int_Type tmpEdgeLocal(0);
                if (id2 > id1)
                    tmpEdgeLocal = {id1, id2};
                else
                    tmpEdgeLocal = { id2 , id1 };
                
                // If no partition was performed, all information is still global at this point. We still use the function below and partition the mesh and surfaces later.
                FiniteElement feEdge( tmpEdgeLocal, edgeElementsFlag_vec[loc] );
                // In some cases an edge is the only part of the surface of an Element. In that case there does not exist a triangle subelement. 
                // We then have to initialize the edge as subelement.     
                // In very coarse meshes it is even possible that an interior element has multiple surface edges or nodes connected to the surface, which is why we might even set interior edges as subelements                  
                if ( !element.subElementsInitialized() ){
                    element.initializeSubElements( "P1", 1 ); // only P1 for now                
                    element.addSubElement( feEdge );
                }
                else{
                    ElementsPtr_Type surfaces = element.getSubElements();
                    if(surfaces->getDimension() == 2)   // We set the edge to the corresponding surface element(s)
                        surfaces->setToCorrectElement( feEdge ); // Case we have surface subelements
                    else{ // simply adding the edge as subelement
                        element.addSubElement( feEdge ); // Case we dont have surface subelements
                        // Comment: Theoretically edges as subelement are only truely relevant when we apply a neuman bc on an edge which is currently not the case. 
                        // This is just in case!
                    }
                }                                
            }
        }
    }
}

template <class SC, class LO, class GO, class NO>
int MeshPartitioner<SC, LO, GO, NO>::searchInSurfaces(vec2D_int_Type &surfaces, vec_int_Type searchSurface) {

    int loc = -1;

    vec2D_int_Type::iterator it = find(surfaces.begin(), surfaces.end(), searchSurface);

    if (it != surfaces.end())
        loc = distance(surfaces.begin(), it);

    return loc;
}

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::setLocalSurfaceIndices(vec2D_int_Type &localSurfaceIndices,
                                                             int surfaceElementOrder) {

    if (dim_ == 2) {

        if (surfaceElementOrder == 2) { // P1
            localSurfaceIndices.resize(3, vec_int_Type(3, -1));
            localSurfaceIndices.at(0).at(0) = 0;
            localSurfaceIndices.at(0).at(1) = 1;
            localSurfaceIndices.at(1).at(0) = 0;
            localSurfaceIndices.at(1).at(1) = 2;
            localSurfaceIndices.at(2).at(0) = 1;
            localSurfaceIndices.at(2).at(1) = 2;
        } else
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "No permutation for this surface yet.");
    } else if (dim_ == 3) {
        if (surfaceElementOrder == 3) {
            localSurfaceIndices.resize(4, vec_int_Type(3, -1));
            localSurfaceIndices.at(0).at(0) = 0;
            localSurfaceIndices.at(0).at(1) = 1;
            localSurfaceIndices.at(0).at(2) = 2;
            localSurfaceIndices.at(1).at(0) = 0;
            localSurfaceIndices.at(1).at(1) = 1;
            localSurfaceIndices.at(1).at(2) = 3;
            localSurfaceIndices.at(2).at(0) = 1;
            localSurfaceIndices.at(2).at(1) = 2;
            localSurfaceIndices.at(2).at(2) = 3;
            localSurfaceIndices.at(3).at(0) = 0;
            localSurfaceIndices.at(3).at(1) = 2;
            localSurfaceIndices.at(3).at(2) = 3;
        } else
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "No permutation for this surface yet.");
    }
}

// ######################## Nonlinear Schwarz related methods ##########################

/**
 * \brief Reads mesh data from a .mesh file for each domain (e.g. velocity and pressure)
 *
 */
template <class SC, class LO, class GO, class NO> void MeshPartitioner<SC, LO, GO, NO>::readMesh(const int volumeID) {
    if (volumeID != 10) {
        if (this->comm_->getRank() == 0) {
            cout << " #### WARNING: The volumeID was set manually and is no longer 10. Please make sure your volumeID "
                    "corresponds to the volumeID in your mesh file. #### "
                 << endl;
        }
    }
    const auto delimiter = pList_->get("Delimiter", " ");
    for (int i = 0; i < domains_.size(); i++) {
        const auto meshName = pList_->get("Mesh " + std::to_string(i + 1) + " Name", "noName");
        TEUCHOS_TEST_FOR_EXCEPTION(meshName == "noName", std::runtime_error, "No mesh name given.");
        domains_[i]->initializeUnstructuredMesh(domains_[i]->getDimension(), "P1",
                                                volumeID); // we only allow to read P1 meshes.
        domains_[i]->readMeshSize(meshName, delimiter);
    }

    // Determine an allocation of ranks to meshes
    determineRanks();

    for (int i = 0; i < domains_.size(); i++) {
        auto meshUnstr = Teuchos::rcp_dynamic_cast<MeshUnstr_Type>(domains_[i]->getMesh());
        // Reading nodes
        meshUnstr->readMeshEntity("node");
        // Reset repeated elements. They are determined when partitioning the mesh.
        /* meshUnstr->pointsRep_.reset(); */
        // Reading elements
        meshUnstr->readMeshEntity("element");
        // Reading surfaces
        meshUnstr->readMeshEntity("surface");
        // Reading line segments
        meshUnstr->readMeshEntity("line");
    }
}

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::buildDualGraph(const int meshNumber) {

    typedef Teuchos::OrdinalTraits<GO> OTGO;
    const auto myRank = this->comm_->getRank();

#ifdef UNDERLYING_LIB_TPETRA
    const Xpetra::UnderlyingLib underlyingLibType = Xpetra::UseTpetra;
#endif
    const auto meshUnstr = Teuchos::rcp_dynamic_cast<MeshUnstr_Type>(this->domains_[meshNumber]->getMesh());
    TEUCHOS_TEST_FOR_EXCEPTION(meshUnstr.is_null(), std::runtime_error, "Mesh is not of type unstructured.");

    // Number of elements in the mesh
    auto ne = Teuchos::implicit_cast<idx_t>(meshUnstr->getNumElementsGlobal());
    // Number of nodes in the mesh
    auto nn = Teuchos::implicit_cast<idx_t>(meshUnstr->getNumGlobalNodes());

    // Setup for paritioning with metis
    vec_idx_Type eptrVec(0); // Vector for local elements ptr (local is still global at this point)
    vec_idx_Type eindVec(0); // Vector for local elements ids
    // Serially distributed elements
    const auto elementsMesh = meshUnstr->getElementsC();
    // Fill the arrays
    makeContinuousElements(elementsMesh, eindVec, eptrVec);
    auto eptr = &eptrVec.at(0);
    auto eind = &eindVec.at(0);

    // Number of common nodes required to classify two elements as neighbours: min(ncommon, n1-1, n2-1)
    idx_t ncommon;
    const auto dim = meshUnstr->getDimension();
    const auto FEType = domains_[meshNumber]->getFEType();
    if (dim == 2) {
        if (FEType == "P1") {
            ncommon = 2;
        } else if (FEType == "P2") {
            ncommon = 3;
        }
    } else if (dim == 3) {
        if (FEType == "P1") {
            ncommon = 3;
        } else if (FEType == "P2") {
            ncommon = 6;
        }
    } else
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Wrong Dimension.");

    idx_t numflag = 0;

    // Arrays for storing the partition
    // METIS allocates these using malloc
    idx_t *xadj;
    idx_t *adjncy;

    if (myRank == 0) {
        cout << "--- Building dual graph with METIS ...";
    }
    const auto returnCode = METIS_MeshToDual(&ne, &nn, eptr, eind, &ncommon, &numflag, &xadj, &adjncy);

    if (myRank == 0) {
        cout << "\n--\t Metis return code: " << returnCode;
        cout << "\n--- done" << endl;
    }

    const auto locReplMap = Xpetra::MapFactory<LO, GO, NO>::createLocalMap(underlyingLibType, ne, this->comm_);
    meshUnstr->elementMap_.reset(new Map<LO, GO, NO>(locReplMap));

    // Copy idx_t arrays to ArrayRCP arrays required by graph constructor
    auto xadjArrayRCP = Teuchos::ArrayRCP<size_t>(ne + 1);
    for (auto i = 0; i < xadjArrayRCP.size(); i++) {
        xadjArrayRCP[i] = xadj[i];
    }
    // Size of adjncy is stored in last entry of xadj
    auto adjncyArrayRCP = Teuchos::ArrayRCP<LO>(xadj[ne]);
    for (auto i = 0; i < adjncyArrayRCP.size(); i++) {
        adjncyArrayRCP[i] = adjncy[i];
    }

    // Both row and column maps are unity
    // All entries are on all ranks
    const auto xpetraElementMap = meshUnstr->getElementMap()->getXpetraMap();
    meshUnstr->dualGraph_ =
        Xpetra::CrsGraphFactory<LO, GO, NO>::Build(xpetraElementMap, xpetraElementMap, xadjArrayRCP, adjncyArrayRCP);
    meshUnstr->dualGraph_->fillComplete();
    METIS_Free(xadj);
    METIS_Free(adjncy);
}

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::partitionDualGraphWithOverlap(const int meshNumber, const int overlap) {

    auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
    typedef Teuchos::OrdinalTraits<GO> OTGO;
#ifdef UNDERLYING_LIB_TPETRA
    const string underlyingLib = "Tpetra";
    const Xpetra::UnderlyingLib underlyingLibType = Xpetra::UseTpetra;
#endif

    const auto myRank = this->comm_->getRank();
    const auto meshUnstr = Teuchos::rcp_dynamic_cast<MeshUnstr_Type>(this->domains_[meshNumber]->getMesh());

    TEUCHOS_TEST_FOR_EXCEPTION(meshUnstr.is_null(), std::runtime_error, "Mesh is not of type unstructured.");
    TEUCHOS_TEST_FOR_EXCEPTION(meshUnstr->dualGraph_.is_null(), std::runtime_error,
                               "Attempting to partition non-existent dual graph. Ensure dual graph has been "
                               "constructed e.g. with buildDualGraph()");
    // Ensure that dual graph is not yet partitioned
    TEUCHOS_TEST_FOR_EXCEPTION(meshUnstr->elementMap_->getXpetraMap()->isDistributed(), std::runtime_error,
                               "Dual graph is already partitioned. Can only partition locally replicated graph.");

    const int indexBase = 0;
    // Number of elements in the mesh = number of vertices in the dual graph
    auto nvtxs = Teuchos::implicit_cast<idx_t>(meshUnstr->getNumElementsGlobal());
    // Balancing constraints. Simple vertex balancing here i.e. one weight per vertex
    idx_t ncon = 1;

    // Row indices of the CRS format
    const auto ptrTpetraCrsGraph = Teuchos::rcp_dynamic_cast<Xpetra::TpetraCrsGraph<LO, GO, NO>>(meshUnstr->dualGraph_);
    TEUCHOS_TEST_FOR_EXCEPTION(ptrTpetraCrsGraph.is_null(), std::runtime_error,
                               "Using dual graph requires Tpetra. It seems you are using Epetra.")
    const auto xadjRCPVec = ptrTpetraCrsGraph->getNodeRowPtrs();

    // Note that using std::move here might be more efficient
    vec_idx_Type xadjVec(xadjRCPVec.begin(), xadjRCPVec.end());

    // Column indices of the CRS format
    // Size is stored at the end of the row indices
    vec_idx_Type adjncyVec(xadjVec.back());
    Teuchos::ArrayView<const LO> tempRowView;

    for (auto i = 0; i < nvtxs; i++) {
        meshUnstr->dualGraph_->getLocalRowView(i, tempRowView);
        // Not using std::move so that dualGraph remains valid for later use
        std::copy(tempRowView.begin(), tempRowView.end(), adjncyVec.begin() + xadjVec.at(i));
    }

    idx_t *xadj = xadjVec.data();
    idx_t *adjncy = adjncyVec.data();

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    // Edge cut minimization
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    // Random matching during coarsening
    options[METIS_OPTION_CTYPE] = METIS_CTYPE_RM;
    // Initial partitioning algo
    /* options[METIS_OPTION_IPTYPE] */
    // Refinement algo
    /* options[METIS_OPTION_RTYPE] */
    // Perform two hop matching if normal matching does not work
    /* options[METIS_OPTION_NO2HOP] */
    // Specify how many times to partition. Best partition is chosen. Default = 1
    /* options[METIS_OPTION_NCUTS] */
    // Iterations of refinement algos at each uncoarsening step. Default = 10
    options[METIS_OPTION_NITER] = 50;
    // Allowed load imbalance
    /* options[METIS_OPTION_UFACTOR] */
    // Minimize maximum connectivity between subdomains. 1 = explicitly minimize
    options[METIS_OPTION_MINCONN] = 0;
    // Try to produce contiguous partitions i.e. no reordering of the connectivity matrix
    options[METIS_OPTION_CONTIG] = pList_->get("Contiguous", false);
    // Seed for random number generator
    options[METIS_OPTION_SEED] = 555;
    // Start at zero or one
    options[METIS_OPTION_NUMBERING] = indexBase;
    // Verbosity level during execution of the algo
    /* options[METIS_OPTION_DBGLVL] = METIS_DBG_INFO; */

    idx_t edgecut = 0;
    vec_idx_Type partVec(nvtxs, -1);
    auto part = partVec.data();

    // Number of available ranks = number of parts to partition into
    idx_t nparts = get<1>(rankRanges_[meshNumber]) - get<0>(rankRanges_[meshNumber]) + 1;

    if (myRank == 0) {
        cout << "--- Partitioning dual graph with METIS ... \n";
    }

    if (nparts > 1) {
        idx_t returnCode = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, NULL, NULL, NULL, &nparts, NULL, NULL,
                                                    options, &edgecut, part);
        if (myRank == 0) {
            cout << "\n--\t Metis return code: " << returnCode;
        }
        const auto [min, max] = std::minmax_element(partVec.begin(), partVec.end());
        TEUCHOS_TEST_FOR_EXCEPTION(nparts - 1 != *max - *min, std::runtime_error,
                                   "METIS did not manage to partition dual graph into requested number of partitions.");
    } else {
        for (auto &temp : partVec) {
            temp = 0;
        }
    }

    if (myRank == 0) {
        cout << "\n--\t objval: " << edgecut;
        cout << "\n-- done!" << endl;
    }
    // We need the distributed dual graph to be able to perform assembly on each subdomain
    // The dual graph map is the same as the element map, so update that here
    vec_GO_Type locepart(0);
    for (auto i = 0; i < nvtxs; i++) {
        // Subtract start of the rank range to map required ranks to interval starting at zero
        if (part[i] == this->comm_->getRank() - get<0>(this->rankRanges_[meshNumber])) {
            locepart.push_back(i);
        }
    }
    // elementsGlobalMapping -> elements per Processor i.e. global indices of the elements owned by the current rank
    // Pass invalid since the global number of elements is unknown (includes repetition across the ranks)
    auto elementsGlobalMapping = Teuchos::arrayViewFromVector(locepart);
    auto newElementMap = Xpetra::MapFactory<LO, GO, NO>::Build(underlyingLibType, OTGO::invalid(),
                                                               elementsGlobalMapping, indexBase, this->comm_);

    // Create an export object since the target map is one to one but the source map may not be
    const auto exporter =
        Xpetra::ExportFactory<LO, GO, NO>::Build(meshUnstr->elementMap_->getXpetraMap(), newElementMap);

    // New graph into which the previous graph will be imported
    auto newDualGraph = Xpetra::CrsGraphFactory<LO, GO, NO>::Build(newElementMap, newElementMap->getLocalNumElements());
    newDualGraph->doExport(*meshUnstr->dualGraph_, *exporter, Xpetra::INSERT);
    newDualGraph->fillComplete();

    // Export operation complete so we can overwrite the old element mapRepeated
    meshUnstr->elementMap_.reset(
        new Map<LO, GO, NO>(underlyingLib, OTGO::invalid(), elementsGlobalMapping, indexBase, this->comm_));

    /* logGreen("Before extending:", comm_); */
    /* newDualGraph->describe(*out, Teuchos::VERB_EXTREME); */

    // Cast to pointer-to-const for FROSch function
    auto graphExtended = Teuchos::rcp_implicit_cast<const Xpetra::CrsGraph<LO, GO, NO>>(newDualGraph);

    // Extend overlap by specified number of times
    for (auto i = 0; i < overlap; i++) {
        ExtendOverlapByOneLayer(graphExtended, graphExtended);
    }
    /* logGreen("After extending, before sorting:", comm_); */
    /* graphExtended->describe(*out, Teuchos::VERB_EXTREME); */

    // Store the interior of the subdomain to differentiate the border
    auto extendedElementMap = FROSch::SortMapByGlobalIndex(graphExtended->getRowMap());

    // Build graph with sorted map
    newDualGraph =
        Xpetra::CrsGraphFactory<LO, GO, NO>::Build(extendedElementMap, extendedElementMap->getLocalNumElements());
    auto importer = Xpetra::ImportFactory<LO, GO, NO>::Build(graphExtended->getRowMap(), extendedElementMap);
    newDualGraph->doImport(*graphExtended, *importer, Xpetra::ADD);
    newDualGraph->fillComplete(graphExtended->getDomainMap(), graphExtended->getRangeMap());

    meshUnstr->dualGraph_ = newDualGraph;
}

template <class SC, class LO, class GO, class NO>
void MeshPartitioner<SC, LO, GO, NO>::buildSubdomainFEsAndNodeLists(const int meshNumber) {

    typedef Teuchos::OrdinalTraits<GO> OTGO;
#ifdef UNDERLYING_LIB_TPETRA
    const string underlyingLib = "Tpetra";
    const Xpetra::UnderlyingLib underlyingLibType = Xpetra::UseTpetra;
#endif

    const auto myRank = comm_->getRank();
    const auto meshUnstr = Teuchos::rcp_dynamic_cast<MeshUnstr_Type>(domains_.at(meshNumber)->getMesh());
    const auto dim = meshUnstr->getDimension();
    const auto FEType = domains_.at(meshNumber)->getFEType();
    const auto elementMap = meshUnstr->getElementMap();
    const auto elementMapOverlappingInterior = meshUnstr->dualGraph_->getRowMap();
    const int indexBase = 0;
    vec_int_Type elementsOverlappingIndices(0);

    // Build index vectors for the elements which can then be used to build the elements in the current subdomain.
    vec_idx_Type eptrVec(0); // Vector for local elements ptr (local is still global at this point)
    vec_idx_Type eindVec(0); // Vector for local elements ids
    // Serially distributed elements
    const auto elementsMesh = meshUnstr->getElementsC();
    // Fill the arrays
    makeContinuousElements(elementsMesh, eindVec, eptrVec);

    // Fill repeated and overlapping maps of nodes
    vec_GO_Type pointsRepIndices(0);
    // Also keep track of which nodes belong to the subdomain interior
    // This is needed to set zero Dirichlet boundary conditions on the edge nodes later on
    vec_GO_Type pointsOverlappingInteriorIndices(0);

    // ==================== Build subdomain point index lists ====================
    for (auto i = 0; i < elementMap->getNodeNumElements(); i++) {
        // Get the global index of i
        auto globalID = elementMap->getGlobalElement(i);
        // For each element add the indices corresponding to that element
        for (int j = eptrVec.at(globalID); j < eptrVec.at(globalID + 1); j++) {
            pointsRepIndices.push_back(eindVec.at(j)); // Ids of element nodes, globalIDs
        }
    }
    // Do the same for elementMapOverlappingInterior
    for (auto i = 0; i < elementMapOverlappingInterior->getLocalNumElements(); i++) {
        // Get the global index of i
        auto globalID = elementMapOverlappingInterior->getGlobalElement(i);
        // Start filling indices of overlapping elements
        elementsOverlappingIndices.push_back(globalID);
        // For each element add the indices corresponding to that element
        for (int j = eptrVec.at(globalID); j < eptrVec.at(globalID + 1); j++) {
            pointsOverlappingInteriorIndices.push_back(eindVec.at(j));
        }
    }

    // make_unique also sorts in ascending order
    make_unique(pointsRepIndices);
    make_unique(pointsOverlappingInteriorIndices);

    // Extend by an extra layer, sometimes denoted "ghost layer"
    // This is required to facilitate local assembly of a Dirichlet problem. The current global solution is prescribed
    // as a Dirichlet boundary condition on the ghost layer while solving. This is equivalent to assembling the Neumann
    // matrix on the subdomain including ghost layer and subsequently extracting the submatrix corresponding to interior
    // nodes for solving.
    // This is implemented here and not in partitionDualGraphWithOverlap() since the subdomain points list is required
    // which is not built in the latter.

    vec_GO_Type pointsOverlappingIndices(pointsOverlappingInteriorIndices);

    // Mark all elements that have node in overlappingInterior.
    auto graphImporter = Xpetra::ImportFactory<LO, GO, NO>::Build(meshUnstr->dualGraph_->getRowMap(),
                                                                  meshUnstr->dualGraph_->getRowMap());
    // Build a copy of dualGraph_ here. Using an importer that imports to the same map is the simplest way to do this
    auto ghostDualGraphOld = Teuchos::rcp_implicit_cast<const Xpetra::CrsGraph<LO, GO, NO>>(
        Xpetra::CrsGraphFactory<LO, GO, NO>::Build(meshUnstr->dualGraph_, *graphImporter));
    auto ghostDualGraphNew = Teuchos::rcp_implicit_cast<const Xpetra::CrsGraph<LO, GO, NO>>(
        Xpetra::CrsGraphFactory<LO, GO, NO>::Build(meshUnstr->dualGraph_->getRowMap()));

    int globalNumElementsAddedToGhosts = 1;
    bool ghostLayerComplete = false;
    while (globalNumElementsAddedToGhosts != 0) {
        // TODO: generate list of only those elements added. Iterate over that list and count how many elements are in
        // centre for stopping criterion
        ExtendOverlapByOneLayer(ghostDualGraphOld, ghostDualGraphNew);
        int localNumElementsAddedToGhosts = 0;
        if (!ghostLayerComplete) {
            auto previousElements = Teuchos::createVector(ghostDualGraphOld->getRowMap()->getLocalElementList());
            auto newElements = Teuchos::createVector(ghostDualGraphNew->getRowMap()->getLocalElementList());

            std::sort(previousElements.begin(), previousElements.end());
            std::sort(newElements.begin(), newElements.end());

            std::vector<GO> addedElements(0);
            // This requires sorted inputs and returns a sorted output
            std::set_difference(newElements.begin(), newElements.end(), previousElements.begin(),
                                previousElements.end(), std::back_inserter(addedElements));

            for (const auto i : addedElements) {
                // At this stage elementsC_ still contains all elements i.e. it is not partitioned yet
                auto elementNodes = meshUnstr->elementsC_->getElement(i).getVectorNodeList();
                auto allNodesAreExterior = true;
                for (auto j = 0; j < elementNodes.size(); j++) {
                    // Since pointsOverlappingInteriorIndices is sorted, can use binary search
                    auto indexInSubdomain =
                        std::binary_search(pointsOverlappingInteriorIndices.begin(),
                                           pointsOverlappingInteriorIndices.end(), elementNodes.at(j));
                    allNodesAreExterior = allNodesAreExterior && !indexInSubdomain;
                }
                if (!allNodesAreExterior) {
                    pointsOverlappingIndices.insert(pointsOverlappingIndices.end(), elementNodes.begin(),
                                                    elementNodes.end());
                    elementsOverlappingIndices.push_back(i);
                    localNumElementsAddedToGhosts++;
                }
            }
            if (localNumElementsAddedToGhosts == 0) {
                ghostLayerComplete = true;
            }
            auto tempDualGraph = ghostDualGraphOld;
            ghostDualGraphOld = ghostDualGraphNew;
            ghostDualGraphNew = tempDualGraph;
        }
        Teuchos::reduceAll(*comm_, Teuchos::REDUCE_SUM, 1, &localNumElementsAddedToGhosts,
                           &globalNumElementsAddedToGhosts);
    }
    make_unique(pointsOverlappingIndices);

    auto pointsRepIndicesView = Teuchos::arrayViewFromVector(pointsRepIndices);
    auto pointsOverlappingIndicesView = Teuchos::arrayViewFromVector(pointsOverlappingIndices);
    auto pointsOverlappingInteriorIndicesView = Teuchos::arrayViewFromVector(pointsOverlappingInteriorIndices);

    // ==================== Build repeated, unique and overlapping maps ====================
    meshUnstr->mapRepeated_.reset(
        new Map<LO, GO, NO>(underlyingLib, OTGO::invalid(), pointsRepIndicesView, indexBase, comm_));
    meshUnstr->mapUnique_ = meshUnstr->mapRepeated_->buildUniqueMap(rankRanges_.at(meshNumber));
    meshUnstr->mapOverlapping_.reset(
        new Map<LO, GO, NO>(underlyingLib, OTGO::invalid(), pointsOverlappingIndicesView, indexBase, comm_));
    meshUnstr->mapOverlappingInterior_.reset(
        new Map<LO, GO, NO>(underlyingLib, OTGO::invalid(), pointsOverlappingInteriorIndicesView, indexBase, comm_));

    // ==================== Set repeated points and BC flags ====================
    // pointsRep_ contains all points after reading from mesh
    vec2D_dbl_Type points = *meshUnstr->getPointsRepeated();
    // At this point bcFlagRep_ constains all BC flags of the mesh, not only those belonging to the current subdomain
    vec_int_Type flags = *meshUnstr->getBCFlagRepeated();
    meshUnstr->pointsRep_.reset(new std::vector<std::vector<double>>(meshUnstr->mapRepeated_->getNodeNumElements(),
                                                                     std::vector<double>(dim, -1.)));
    meshUnstr->bcFlagRep_.reset(new std::vector<int>(meshUnstr->mapRepeated_->getNodeNumElements(), 0));

    int pointIDcont;
    for (int i = 0; i < pointsRepIndices.size(); i++) {
        pointIDcont = pointsRepIndices.at(i);
        for (int j = 0; j < dim; j++)
            meshUnstr->pointsRep_->at(i).at(j) = points.at(pointIDcont).at(j);
        meshUnstr->bcFlagRep_->at(i) = flags.at(pointIDcont);
    }

    // ==================== Set overlapping points and BC flags ====================
    meshUnstr->pointsOverlapping_.reset(new std::vector<std::vector<double>>(
        meshUnstr->mapOverlapping_->getNodeNumElements(), std::vector<double>(dim, -1.)));
    meshUnstr->bcFlagOverlapping_.reset(new std::vector<int>(meshUnstr->mapOverlapping_->getNodeNumElements(), 0));

    auto interiorIt = pointsOverlappingInteriorIndices.begin();
    for (int i = 0; i < pointsOverlappingIndices.size(); i++) {
        pointIDcont = pointsOverlappingIndices.at(i);
        for (int j = 0; j < dim; j++) {
            meshUnstr->pointsOverlapping_->at(i).at(j) = points.at(pointIDcont).at(j);
        }
        meshUnstr->bcFlagOverlapping_->at(i) = flags.at(pointIDcont);
        // This only works because index lists are ordered
        if (*interiorIt != pointIDcont) {
            // Only set ghost flag for points that are not on the real Dirichlet boundary
            // Further volume flags must be added here if used in the mesh
            meshUnstr->bcFlagOverlapping_->at(i) = -99;
        } else {
            interiorIt++;
        }
    }

    // ==================== Set unique points and BC flags ====================
    meshUnstr->pointsUni_.reset(new std::vector<std::vector<double>>(meshUnstr->mapUnique_->getNodeNumElements(),
                                                                     std::vector<double>(dim, -1.)));
    meshUnstr->bcFlagUni_.reset(new std::vector<int>(meshUnstr->mapUnique_->getNodeNumElements(), 0));
    GO indexGlobal;
    MapConstPtr_Type map = meshUnstr->getMapRepeated();
    vec2D_dbl_ptr_Type pointsRep = meshUnstr->pointsRep_;
    for (int i = 0; i < meshUnstr->mapUnique_->getNodeNumElements(); i++) {
        indexGlobal = meshUnstr->mapUnique_->getGlobalElement(i);
        for (int j = 0; j < dim; j++) {
            meshUnstr->pointsUni_->at(i).at(j) = pointsRep->at(map->getLocalElement(indexGlobal)).at(j);
        }
        meshUnstr->bcFlagUni_->at(i) = meshUnstr->bcFlagRep_->at(map->getLocalElement(indexGlobal));
    }
    // ==================== Build repeated (elementsC_) and overlapping elements ====================
    meshUnstr->elementsOverlapping_.reset(new Elements(FEType, dim));
    meshUnstr->elementsC_.reset(new Elements(FEType, dim));

    for (auto i = 0; i < elementsOverlappingIndices.size(); i++) {
        // Build local element and save
        std::vector<int> tmpElementC;
        std::vector<int> tmpElementOverlapping;
        auto globalID = elementsOverlappingIndices.at(i);
        auto nonOverlappingElement = elementMap->getXpetraMap()->isNodeGlobalElement(globalID);
        // Iterate over the nodes in an element
        for (int j = eptrVec.at(globalID); j < eptrVec.at(globalID + 1); j++) {
            // Map the node index from global to local and save
            int indexOverlapping = meshUnstr->mapOverlapping_->getLocalElement(eindVec.at(j));
            tmpElementOverlapping.push_back(indexOverlapping);
            // Build nonoverlapping elements
            if (nonOverlappingElement) {
                int indexC = meshUnstr->mapRepeated_->getLocalElement(eindVec.at(j));
                tmpElementC.push_back(indexC);
            }
        }
        FiniteElement feOverlapping(tmpElementOverlapping, elementsMesh->getElement(globalID).getFlag());
        // NOTE KHo Surfaces are not added here for now. Since they probably will not be needed.
        meshUnstr->elementsOverlapping_->addElement(feOverlapping);
        if (nonOverlappingElement) {
            FiniteElement feC(tmpElementC, elementsMesh->getElement(globalID).getFlag());
            meshUnstr->elementsC_->addElement(feC);
        }
    }
}

} // namespace FEDD
#endif
