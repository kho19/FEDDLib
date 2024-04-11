#ifndef MeshPartitioner_decl_hpp
#define MeshPartitioner_decl_hpp

#include "feddlib/core/FE/Domain.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include <Xpetra_CrsGraphFactory.hpp>

#define FEDD_HAVE_METIS
#define FEDD_HAVE_PARMETIS

#ifdef FEDD_HAVE_METIS
#include "metis.h"
#endif
#ifdef FEDD_HAVE_PARMETIS
#include "parmetis.h"
#endif

/*!
 Defintion of MeshPartitioner

 @brief  MeshPartitioner
 @author Christian Hochmuth
 @version 1.0
 @copyright CH


 */

namespace FEDD {
     /*!
    \class Meshpartitioner
    \brief This class partitions a given input P1-Mesh and distributes it among the different processors. 

    \tparam SC The scalar type. So far, this is always double, but having it as a template parameter would allow flexibily, e.g., for using complex instead
    \tparam LO The local ordinal type. The is the index type for local indices
    \tparam GO The global ordinal type. The is the index type for global indices
    \tparam NO The Kokkos Node type. This would allow for performance portibility when using Kokkos. Currently, this is not used.

    General Structure of partitioning:
    - Mesh entities are read 
    - Partitioned via Metis (METIS_PartMeshDual)
    - Corresponding Node- and Elementlists are distributed accordingly

    Mesh entities must include the nodes and elements. Additional information of edges and surfaces (in 3D) can be necessary for problems where complex boundary conditions are set on surfaces or edges.

    In general elements follow a hierarchical structure. For example a 3D tetrahedral element owns all 2D and 1D entities (triangle/surface, edges) that are part of the surface. 

    As of now each surface and element node list is sorted to improve find/search operations. This influences for example the calculation of surface normals.
    
    */
template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class MeshPartitioner {
    
public:
    typedef Domain<SC, LO, GO, NO> Domain_Type;
    typedef typename Domain_Type::DomainPtr_Type DomainPtr_Type;
    typedef typename Domain_Type::CommConstPtr_Type CommConstPtr_Type;
    typedef std::vector<DomainPtr_Type> DomainPtrArray_Type;
    typedef typename Domain_Type::MeshUnstr_Type MeshUnstr_Type;
    typedef typename Domain_Type::MeshUnstrPtr_Type MeshUnstrPtr_Type;
    typedef typename Domain_Type::MapConstPtr_Type MapConstPtr_Type;
    typedef Elements Elements_Type;
    typedef Teuchos::RCP<Elements_Type> ElementsPtr_Type;
    typedef EdgeElements EdgeElements_Type;
    typedef Teuchos::RCP<EdgeElements_Type> EdgeElementsPtr_Type;
    // Nonlinear Schwarz related
    typedef typename MeshUnstr_Type::Graph_Type Graph_Type;
    typedef typename MeshUnstr_Type::GraphPtr_Type GraphPtr_Type;
    typedef typename MeshUnstr_Type::GraphFactory_Type GraphFactory_Type;

    typedef std::vector<idx_t> vec_idx_Type; //Metis
    
    MeshPartitioner();

    MeshPartitioner( DomainPtrArray_Type domains, ParameterListPtr_Type pL, std::string feType, int dimension );
    
    ~MeshPartitioner();
    
	/*! 
		\brief Main Function of partitioner. Called with volume ID in order to set in case it is not equal to ten. Always make sure the volumeID corresponds to the
		highest given flag. 
	*/
    void readAndPartition(int volumeID=10);
        
    /*! \brief Only used in 3D to set the edges as subelements to surfaces*/
    void setEdgesToSurfaces(int meshNumber);
    
	/*! 
		\brief Setting surfaces, i.e. edges in 2D and triangles in 3D, as subelements to the corresponding elements
	*/
    void setSurfacesToElements(int meshNumber);
    
	/*! 
		\brief Main function, that reads and partions and distributes the mesh to the different processors. 
		Here all necessary maps and lists are created
	*/
    void partitionMesh( MeshUnstrPtr_Type& mesh, int meshNumber );

	/*! 
		\brief Making the edge list parallel
	*/
    void buildEdgeListParallel( MeshUnstrPtr_Type mesh, ElementsPtr_Type elementsGlobal );
    
	/*! 
		\brief Building the edge list
	*/
    void buildEdgeList( MeshUnstrPtr_Type mesh, ElementsPtr_Type& elementsGlobal );

	/*! 
		\brief Setting surfaces, i.e. edges in 2D and triangles in 3D, as subelements to the corresponding elements
	*/
    void setLocalEdgeIndices(vec2D_int_Type &localEdgeIndices );

    void determineRanks();

    void determineRanksFromNumberRanks(vec_int_Type& ranks);
    
    void determineRanksFromFractions(vec_int_Type& fractions);
    
    void makeContinuousElements(ElementsPtr_Type elements, vec_idx_Type& eind_vec, vec_idx_Type& eptr_vec );

	/*! 
		\brief Finding the surfaces corresponding to a specfic element and then setting subelements
	*/
    void findAndSetSurfacesPartitioned( vec2D_int_Type& surfElements_vec,vec2D_int_Type& surfElements_vec_unsorted, vec_int_Type& surfElementsFlag_vec, FiniteElement& element, vec2D_int_Type& permutation, vec_GO_Type& linearSurfacePartitionOffset, int globalElID);

	/*! 
		\brief Setting local IDs to the edges in 3D case with respect to the local numbering of elements
	*/
    void setLocalSurfaceEdgeIndices( vec2D_int_Type &localSurfaceEdgeIndices, int edgesElementOrder );
    
	/*! 
		\brief Only relevant in 3D. Finding the edges corresponding to the specfic element and then setting as subsubelement.
	*/
    void findAndSetSurfaceEdges( vec2D_int_Type& edgeElements_vec, vec_int_Type& edgeElementsFlag_vec, FiniteElement& element, vec2D_int_Type& permutation, MapConstPtr_Type mapRepeated);
    
	/*! 
		\brief Searching on particular surface in a surface list.
	*/
    int searchInSurfaces( vec2D_int_Type& surfaces, vec_int_Type searchSurface);
    
    void setLocalSurfaceIndices(vec2D_int_Type &localSurfaceIndices, int surfaceElementOrder );
    /* ######################### Nonlinear Schwarz related methods ############################################# */
    /**
     * \brief Reads mesh and stores it in the mesh_ property of the corresponding domain. Mesh exists on all
     * ranks after a call to this function. Called with volume ID in order to set in case it is not equal to ten. Always
     * make sure the volumeID corresponds to the highest given flag.
     * This function combines the functionality of readAndPartition() and the first part of readAndPartitionMesh().
     * After it has been called the mesh data should be available on all ranks.
     */
    void readMesh(const int volumeID = 10);

    /**
     * \brief Build dual graph of the mesh provided
     * Essentialy a wrapper of the metis function METIS_MeshToDual()
     * Note: CSR (compressed sparse row), CRS (compressed row storage), Yale format are all the same
     */
    void buildDualGraph(const int meshNumber);

    /**
     * \brief partitions a locally replicated dual graph using metis and adds the specified overlapping
     * this->dualGraph_ must be initialised before calling partitionDualGraph()
     * Rather than using METIS to partition the mesh directly, the dual graph is used for nonlinear Schwarz to make
     * increasing the overlap and subsequent assembly easy. If the mesh was partitioned directly and the overlap
     * increased node-wise, assembly would be more difficult since the elements corresponding to the local subdomain
     * vertices would have to be determined.
     */
    void partitionDualGraphWithOverlap(const int meshNumber, const int overlap = 0);

    /**
     * \brief Fills the elementsC_ and the pointsRep_ member of a mesh corresponding to a (potentially overlapping)
     * partition of the dual graph
     */
    void buildSubdomainFEsAndNodeLists(const int meshNumber);

    /**
     * \brief Returns the indices corresponding to a complete layer of vertices around the domain described redundantly
     * by indicesIn and dualGraph
     */
    void buildGhostLayer(vec_GO_Type &pointIndices, vec_int_Type &elementIndices,
                         GraphPtr_Type dualGraph, ElementsPtr_Type elementList);

private:    
    
	/*! 
		\brief Function called internally to read and partition mesh i of domain i
	*/
    void readAndPartitionMesh( int meshNumber );
    
    ParameterListPtr_Type pList_;
    DomainPtrArray_Type domains_;
    CommConstPtr_Type comm_;
    std::string feType_;
    std::vector< tuple_intint_Type > rankRanges_;
    int dim_;
    };
}

#endif
