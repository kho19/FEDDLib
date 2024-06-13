#ifndef Mesh_decl_hpp
#define Mesh_decl_hpp

#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include <Xpetra_CrsGraphFactory.hpp>
#define FULL_Mesh_TIMER

#include "feddlib/core/FE/Elements.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/MultiVector.hpp"
#include "feddlib/core/FE/Elements.hpp"
#include "feddlib/core/FE/FiniteElement.hpp"
#include "feddlib/core/FE/Helper.hpp"
#include "feddlib/core/Mesh/AABBTree.hpp"

/*!
Defintion of Mesh

@brief  Mesh
@author Christian Hochmuth
@version 1.0
@copyright CH
*/

// TODO: kho public data fields and corresponding getter functions are superfluous. Either make members private or remove
// getters
namespace FEDD {
template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no> class Mesh {

  public:
    typedef Elements Elements_Type;
    typedef FiniteElement FiniteElement_Type;
    typedef Teuchos::RCP<FiniteElement_Type> FiniteElementPtr_Type;
    typedef Teuchos::RCP<Elements_Type> ElementsPtr_Type;

    typedef Teuchos::RCP<Mesh> Mesh_ptr_Type;

    typedef Teuchos::RCP<Teuchos::Comm<int>> CommPtr_Type;
    typedef Teuchos::RCP<const Teuchos::Comm<int>> CommConstPtr_Type;
    typedef const CommConstPtr_Type CommConstPtrConst_Type;

    typedef Map<LO, GO, NO> Map_Type;
    typedef Teuchos::RCP<Map_Type> MapPtr_Type;
    typedef Teuchos::RCP<const Map_Type> MapConstPtr_Type;
    typedef Teuchos::RCP<const Map_Type> MapConstPtrConst_Type;

    typedef MultiVector<SC, LO, GO, NO> MultiVector_Type;
    typedef Teuchos::RCP<MultiVector_Type> MultiVectorPtr_Type;
    typedef Xpetra::CrsGraph<LO, GO, NO> Graph_Type;
    typedef Teuchos::RCP<Graph_Type> GraphPtr_Type;
    typedef Xpetra::CrsGraphFactory<LO, GO, NO> GraphFactory_Type;

    typedef AABBTree<SC, LO, GO, NO> AABBTree_Type;
    typedef Teuchos::RCP<AABBTree_Type> AABBTreePtr_Type;

    /* ###################################################################### */
    //
    Mesh();

    Mesh(CommConstPtrConst_Type &comm);

    ~Mesh();

    /*!
     Delete all member variables
     */
    void deleteData();
    
    /// @brief Setting input parameterlist to be parameterlist here
    /// @param pL 
    void setParameterList( ParameterListPtr_Type& pL );

    /// @brief Getter for paramaeterlist that is set here
    /// @return pL
    ParameterListConstPtr_Type getParameterList( ) const;
    
    /// @brief Getter for element flags
    /// @return elements flag vector -- not sure if this is used anywhere
    vec_int_ptr_Type getElementsFlag() const;
    
    /// @brief Getter for unique node map
    /// @return mapUnique_
    MapConstPtr_Type getMapUnique() const;
    
    /// @brief Getter for repeated node mal
    /// @return mapRepeated_
    MapConstPtr_Type getMapRepeated() const;

    /// @brief Getter for unique node P2 map. Dont know what this is for exactly. Think this object is empty
    /// @return 
    MapConstPtr_Type getMapUniqueP2() const;
    
    /// @brief Getter for repeated node P2 map. Dont know what this is for exactly. Think this object is empty
    /// @return 
    MapConstPtr_Type getMapRepeatedP2() const;
    
    /// @brief Getter for element map 
    /// @return 
    MapConstPtr_Type getElementMap() const;
	
    /// @brief Getter for edge map
    /// @return 
    MapConstPtr_Type getEdgeMap() const; // Edge Map
    
    /// @brief getter for list of repeated points with x,y,z coordinates in each row
    /// @return pointsRep_
    vec2D_dbl_ptr_Type getPointsRepeated() const;

    /// @brief getter for list of unique points with x,y,z coordinates in each row
    /// @return pointsUni_
    vec2D_dbl_ptr_Type getPointsUnique() const;
    
    /// @brief Getter for flags corresponting to repeated points
    /// @return bcFlagRep_
    vec_int_ptr_Type getBCFlagRepeated() const;
    
    /// @brief Getter for flags corresponting to unique points
    /// @return bcFlagUni_
    vec_int_ptr_Type getBCFlagUnique() const;

    virtual void dummy() = 0;
    
    /// @brief Returns element list as c-object
    /// @return elementsC_
    ElementsPtr_Type getElementsC() const;

    /// @brief Getter for surface elements. Probably set in mesh partitioner. They are generally the dim-1 surface elements
    /// @return surfaceElements_
    ElementsPtr_Type getSurfaceElements();
    
    /// @brief 
    /// @return dim_ 
    int getDimension();
    
    /// @brief Global number of elements
    /// @return numElementsGlob_
    GO getNumElementsGlobal();
    
    /// @brief Local number of elements
    /// @return 
    LO getNumElements();
    
    /// @brief Get local (LO) number of points either in unique or repeated version
    /// @param type LO (local ordinal)
    /// @return numer of points
    LO getNumPoints(std::string type="Unique");
    
    /// @brief 
    /// @return 
    int getOrderElement();

    /// @brief Communicator object
    /// @return 
    CommConstPtrConst_Type getComm(){return comm_;};
    
    /// @brief This is done in meshStructured. Maybe we should move it here or delete this.
    /// @param flags 
    /// @return 
    int setStructuredMeshFlags(int flags){return 0;};
    
    /// @brief Something for TPM. Can be deprecated soon.
    /// @param type 
    void setElementFlags(std::string type="");
    
    /// @brief Setting current coordinates as reference configuration. Should only be called once :D
    void setReferenceConfiguration();
    
    /// @brief Moving mesh according to displacement based on reference configuration.
    /// @param displacementUnique displacement in unqiue dist.
    /// @param displacementRepeated displacement in repeated dist.
    void moveMesh( MultiVectorPtr_Type displacementUnique, MultiVectorPtr_Type displacementRepeated );
    
    // Creates an AABBTree from own vertice- and elementlist.
    void create_AABBTree();

    vec_int_ptr_Type findElemsForPoints(vec2D_dbl_ptr_Type query_points);

    vec_dbl_Type getBaryCoords(vec_dbl_Type point, int element);

    bool isPointInElem(vec_dbl_Type point, int element);

    /// @brief 
    /// @return 
    tuple_intint_Type getRankRange() const {return rankRange_;};
    
    /// @brief Deleting surface elements, called after reading input mesh. 
    void deleteSurfaceElements(){ surfaceElements_.reset(); };

    /// @brief Correcting the normal direction of all surface normals set as subelements of volume elements to be outward normals
    void correctNormalDirections();

    /// @brief Correct the element orientation of all elements to have positive volume / det when doint transformation
    void correctElementOrientation();    

    /*!
            \brief Returns elements as a vector type
    */
    vec2D_int_ptr_Type getElements();

    // ##################### Nonlinear Schwarz related functions ###############
    MapConstPtr_Type getMapOverlappingGhosts() const;
    MapConstPtr_Type getMapOverlapping() const;
    ElementsPtr_Type getElementsOverlappingGhosts() const;
    vec_int_ptr_Type getBCFlagOverlappingGhosts() const;
    GraphPtr_Type getDualGraph() const;

    // Have to make these const and the maps mutable to fit to the const structure of problem <- domain <- mesh
    void setElementsC(ElementsPtr_Type newElements) const;
    void replaceRepeatedMembers(const MapPtr_Type newMap, const vec2D_dbl_ptr_Type newPoints,
                                const vec_int_ptr_Type newBCs) const;
    void replaceUniqueMembers(const MapPtr_Type newMap, const vec2D_dbl_ptr_Type newPoints,
                              const vec_int_ptr_Type newBCs) const;
	    /* ###################################################################### */

    int dim_;
    long long numElementsGlob_;

    std::string FEType_;
    // Unique partition of vertices based on mapRepeated_
    mutable MapPtr_Type mapUnique_;
    // Nonoverlapping partition of elements. Repeated because subdomain interface vertices are shared.
    mutable MapPtr_Type mapRepeated_;
    mutable vec2D_dbl_ptr_Type pointsRep_;
    mutable vec2D_dbl_ptr_Type pointsUni_;
    mutable vec_int_ptr_Type bcFlagRep_;
    mutable vec_int_ptr_Type bcFlagUni_;

    ElementsPtr_Type surfaceElements_;

    // Object containing all elements on current rank
    // Exposes a number of utility functions
    mutable ElementsPtr_Type elementsC_;
    MapPtr_Type elementMap_;
    MapPtr_Type edgeMap_;

    mutable CommConstPtr_Type comm_;

    // 2D int vector (vector of int vectors) that stores element point indices
    // Rarely used and simpler than elementsC_
    vec2D_int_ptr_Type elementsVec_;

    vec2D_dbl_ptr_Type pointsRepRef_; // Repeated reference configuration
    vec2D_dbl_ptr_Type pointsUniRef_; // Unique reference configuration

    MapPtr_Type mapUniqueP2Map_;
    MapPtr_Type mapRepeatedP2Map_;

    ParameterListPtr_Type pList_;

    int elementOrder_;
    int surfaceElementOrder_;
    int edgesElementOrder_;

    AABBTreePtr_Type AABBTree_;

    tuple_intint_Type rankRange_;

    // ######################## Nonlinear Schwarz related member variables
    // TODO: kho at some point it could be an idea to move these variables to the nonlinear Schwarz solver
    // dual graph of the interior (excluding ghost layer) of the overlapping subdomain
    GraphPtr_Type dualGraph_;
    // Overlapping partition of nodes for nonlinear Schwarz method
    MapPtr_Type mapOverlappingGhosts_;
    // Only interior nodes of the subdomain
    MapPtr_Type mapOverlapping_;
    // List of points in the overlapping subdomain
    vec2D_dbl_ptr_Type pointsOverlappingGhosts_;
    // BC flags corresponding to pointsOverlappingGhosts_
    vec_int_ptr_Type bcFlagOverlappingGhosts_;
    // Elements in overlapping subdomain
    ElementsPtr_Type elementsOverlappingGhosts_;

    /* ###################################################################### */
private:

    void flipSurface(ElementsPtr_Type subEl, int surfaceNumber);
    void flipElement(ElementsPtr_Type elements, int elementNumber);
};
} // namespace FEDD

#endif
