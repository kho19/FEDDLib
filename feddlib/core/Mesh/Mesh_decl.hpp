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

// TODO public data fields and corresponding getter functions are superfluous. Either make members private or remove
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

    void setParameterList(ParameterListPtr_Type &pL);

    ParameterListConstPtr_Type getParameterList() const;

    vec_int_ptr_Type getElementsFlag() const;

    MapConstPtr_Type getMapUnique() const;

    MapConstPtr_Type getMapRepeated() const;

    GraphPtr_Type getDualGraph() const;

    MapConstPtr_Type getMapUniqueP2() const;

    MapConstPtr_Type getMapRepeatedP2() const;

    MapConstPtr_Type getElementMap();

    MapConstPtr_Type getEdgeMap(); // Edge Map

    vec2D_dbl_ptr_Type getPointsRepeated() const;

    vec2D_dbl_ptr_Type getPointsUnique() const;

    vec_int_ptr_Type getBCFlagRepeated() const;

    vec_int_ptr_Type getBCFlagUnique() const;

    virtual void dummy() = 0;

    ElementsPtr_Type getElementsC();

    ElementsPtr_Type getSurfaceElements();

    int getDimension();

    GO getNumElementsGlobal();

    LO getNumElements();

    LO getNumPoints(std::string type = "Unique");

    int getOrderElement();

    CommConstPtrConst_Type getComm() { return comm_; };

    int setStructuredMeshFlags(int flags) { return 0; };

    void setElementFlags(std::string type = "");

    void setReferenceConfiguration();

    void moveMesh(MultiVectorPtr_Type displacementUnique, MultiVectorPtr_Type displacementRepeated);

    // Creates an AABBTree from own vertice- and elementlist.
    void create_AABBTree();

    vec_int_ptr_Type findElemsForPoints(vec2D_dbl_ptr_Type query_points);

    vec_dbl_Type getBaryCoords(vec_dbl_Type point, int element);

    bool isPointInElem(vec_dbl_Type point, int element);

    tuple_intint_Type getRankRange() const {return rankRange_;};
    
    void deleteSurfaceElements(){ surfaceElements_.reset(); };

    void correctNormalDirections();

    void correctElementOrientation();    

    /*!
            \brief Returns elements as a vector type
    */
    vec2D_int_ptr_Type getElements();


    //##################### Nonlinear Schwarz related functions ###############
    MapConstPtr_Type getElementMapOverlapping();
    MapConstPtr_Type getMapOverlapping();

    /* ###################################################################### */

    int dim_;
    long long numElementsGlob_;

    std::string FEType_;
    // Unique partition of vertices based on mapRepeated_
    MapPtr_Type mapUnique_;
    // Nonoverlapping partition of elements. Repeated because subdomain interface vertices are shared.
    MapPtr_Type mapRepeated_;
    vec2D_dbl_ptr_Type pointsRep_;
    vec2D_dbl_ptr_Type pointsUni_;
    vec_int_ptr_Type bcFlagRep_;
    vec_int_ptr_Type bcFlagUni_;

    ElementsPtr_Type surfaceElements_;

    // Object containing all elements on current rank
    // Exposes a number of utility functions
    ElementsPtr_Type elementsC_;
    MapPtr_Type elementMap_;
    MapPtr_Type edgeMap_;

    CommConstPtrConst_Type comm_;

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

    // Nonlinear Schwarz related member variables
    // dualGraph_ does not need its own map since its row map = elementMapOverlapping_
    GraphPtr_Type dualGraph_;
    // contents of elementsC_ built according to this map in nonlinear Schwarz
    MapPtr_Type elementMapOverlapping_;
    // Overlapping partition of nodes for nonlinear Schwarz method
    MapPtr_Type mapOverlapping_;

    /* ###################################################################### */
private:

    void flipSurface(FiniteElement_Type feSub);
};
} // namespace FEDD

#endif
