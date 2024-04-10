#ifndef Mesh_def_hpp
#define Mesh_def_hpp

#include "Mesh_decl.hpp"
#include "feddlib/core/FE/Elements.hpp"
#include "feddlib/core/FEDDCore.hpp"

/*!
Definition of Mesh

@brief  Mesh
@author Christian Hochmuth
@version 1.0
@copyright CH
*/
using Teuchos::reduceAll;
using Teuchos::REDUCE_SUM;
using Teuchos::outArg;

using namespace std;
namespace FEDD {
template <class SC, class LO, class GO, class NO>
Mesh<SC, LO, GO, NO>::Mesh()
    : dim_(-1), numElementsGlob_(0), FEType_(""), mapUnique_(), mapRepeated_(), pointsRep_(), pointsUni_(),
      bcFlagRep_(), bcFlagUni_(), surfaceElements_(new Elements()), elementsC_(new Elements()), elementMap_(new Map()),
      edgeMap_(new Map()), comm_(), elementsVec_(), pointsRepRef_(), pointsUniRef_(), mapUniqueP2Map_(),
      mapRepeatedP2Map_(), elementOrder_(-1), surfaceElementOrder_(-1), edgesElementOrder_(-1), AABBTree_(),
      rankRange_(-1, -1), dualGraph_(), elementMapOverlapping_(), mapOverlapping_(), elementMapOverlappingInterior_(),
      pointsOverlapping_(), bcFlagOverlapping_(), elementsOverlapping_(new Elements()) {}

template <class SC, class LO, class GO, class NO>
Mesh<SC, LO, GO, NO>::Mesh(CommConstPtrConst_Type &comm)
    : dim_(-1), numElementsGlob_(0), FEType_(""), mapUnique_(), mapRepeated_(), pointsRep_(), pointsUni_(),
      bcFlagRep_(), bcFlagUni_(), surfaceElements_(new Elements()), elementsC_(new Elements()), elementMap_(new Map()),
      edgeMap_(new Map()), comm_(comm), elementsVec_(), pointsRepRef_(), pointsUniRef_(), mapUniqueP2Map_(),
      mapRepeatedP2Map_(), elementOrder_(-1), surfaceElementOrder_(-1), edgesElementOrder_(-1), AABBTree_(),
      rankRange_(-1, -1), dualGraph_(), elementMapOverlapping_(), mapOverlapping_(), elementMapOverlappingInterior_(),
      pointsOverlapping_(), bcFlagOverlapping_(), elementsOverlapping_(new Elements()) {}

template <class SC, class LO, class GO, class NO> Mesh<SC, LO, GO, NO>::~Mesh() {}

template <class SC, class LO, class GO, class NO> void Mesh<SC, LO, GO, NO>::setElementFlags(std::string type) {

    ElementsPtr_Type elements = this->getElementsC();
    //    this->elementFlag_.reset( new vec_int_Type( elements->numberElements(), 0 ) );
    if (type == "TPM_square") {
        double xRef, yRef;

        for (int i = 0; i < elements->numberElements(); i++) {
            xRef = (this->pointsRep_->at(elements->getElement(i).getNode(0))[0] +
                    this->pointsRep_->at(elements->getElement(i).getNode(1))[0] +
                    this->pointsRep_->at(elements->getElement(i).getNode(2))[0]) /
                   3.;
            yRef = (this->pointsRep_->at(elements->getElement(i).getNode(0))[1] +
                    this->pointsRep_->at(elements->getElement(i).getNode(1))[1] +
                    this->pointsRep_->at(elements->getElement(i).getNode(2))[1]) /
                   3.;
            if (xRef >= 0.3 && xRef <= 0.7) {
                if (yRef >= 0.6) {
                    elements->getElement(i).setFlag(1);
                    //                    this->elementFlag_->at(i) = 1;
                }
            }
        }
    } else if (type == "Excavation1") {
    } else {
    }
}

template <class SC, class LO, class GO, class NO>
void Mesh<SC, LO, GO, NO>::setParameterList(ParameterListPtr_Type &pL) {
    pList_ = pL;
}

template <class SC, class LO, class GO, class NO>
ParameterListConstPtr_Type Mesh<SC, LO, GO, NO>::getParameterList() const {
    return pList_;
}

template <class SC, class LO, class GO, class NO> vec_int_ptr_Type Mesh<SC, LO, GO, NO>::getElementsFlag() const {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "we are not using the correct flags here. use the flags of elementC_.");
    vec_int_ptr_Type tmp;
    return tmp;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::MapConstPtr_Type Mesh<SC, LO, GO, NO>::getMapUnique() const {

    return mapUnique_;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::MapConstPtr_Type Mesh<SC, LO, GO, NO>::getMapRepeated() const {

    return mapRepeated_;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::GraphPtr_Type Mesh<SC, LO, GO, NO>::getDualGraph() const {
    return dualGraph_;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::MapConstPtr_Type Mesh<SC, LO, GO, NO>::getMapUniqueP2() const {

    return mapUniqueP2Map_;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::MapConstPtr_Type Mesh<SC, LO, GO, NO>::getMapRepeatedP2() const {

    return mapRepeatedP2Map_;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::MapConstPtr_Type Mesh<SC, LO, GO, NO>::getElementMap() const {
    TEUCHOS_TEST_FOR_EXCEPTION(elementMap_.is_null(), std::runtime_error, "Element map of mesh does not exist.");
    return elementMap_;
}

// edgeMap
template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::MapConstPtr_Type Mesh<SC, LO, GO, NO>::getEdgeMap() {
    TEUCHOS_TEST_FOR_EXCEPTION(edgeMap_.is_null(), std::runtime_error, "Element map of mesh does not exist.");
    return edgeMap_;
}

template <class SC, class LO, class GO, class NO> vec2D_dbl_ptr_Type Mesh<SC, LO, GO, NO>::getPointsRepeated() const {

    return pointsRep_;
}

template <class SC, class LO, class GO, class NO> vec2D_dbl_ptr_Type Mesh<SC, LO, GO, NO>::getPointsUnique() const {

    return pointsUni_;
}

template <class SC, class LO, class GO, class NO> vec_int_ptr_Type Mesh<SC, LO, GO, NO>::getBCFlagRepeated() const {

    return bcFlagRep_;
}

template <class SC, class LO, class GO, class NO> vec_int_ptr_Type Mesh<SC, LO, GO, NO>::getBCFlagUnique() const {

    return bcFlagUni_;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::ElementsPtr_Type Mesh<SC, LO, GO, NO>::getElementsC() const {
    return elementsC_;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::ElementsPtr_Type Mesh<SC, LO, GO, NO>::getSurfaceElements() {
    return surfaceElements_;
}

template <class SC, class LO, class GO, class NO> int Mesh<SC, LO, GO, NO>::getDimension() { return dim_; }

template <class SC, class LO, class GO, class NO> GO Mesh<SC, LO, GO, NO>::getNumElementsGlobal() {

    return numElementsGlob_;
}

template <class SC, class LO, class GO, class NO> LO Mesh<SC, LO, GO, NO>::getNumElements() {
    TEUCHOS_TEST_FOR_EXCEPTION(this->elementsC_.is_null(), std::runtime_error, "Elements do not exist.");
    return this->elementsC_->numberElements();
}

template <class SC, class LO, class GO, class NO> LO Mesh<SC, LO, GO, NO>::getNumPoints(std::string type) {
    if (!type.compare("Unique"))
        return pointsUni_->size();
    else if (!type.compare("Repeated"))
        return pointsRep_->size();

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Select valid map type: unique or repeated.");
    return 0;
}

template <class SC, class LO, class GO, class NO> int Mesh<SC, LO, GO, NO>::getOrderElement() {

    switch (dim_) {
    case 2:
        if (!FEType_.compare("P1"))
            return 3;
        else if (!FEType_.compare("P1-disc") || !FEType_.compare("P1-disc-global")) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "P1-disc only available in 3D.");
        } else if (!FEType_.compare("P2"))
            return 6;
        else if (!FEType_.compare("P2-CR")) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "P2-CR only available in 3D.");
        } else if (!FEType_.compare("Q2-20")) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Q2-20 only available in 3D.");
        } else if (!FEType_.compare("Q2")) {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Q2 only available in 3D.");
        }
        break;
    case 3:
        if (!FEType_.compare("P1"))
            return 4;
        if (!FEType_.compare("P1-disc") || !FEType_.compare("P1-disc-global"))
            return 4;
        else if (!FEType_.compare("P2"))
            return 10;
        else if (!FEType_.compare("P2-CR"))
            return 15;
        else if (!FEType_.compare("Q2-20"))
            return 20;
        else if (!FEType_.compare("Q2"))
            return 27;
        break;
    default:
        return -1;
        break;
    }
    return -1;
}

template <class SC, class LO, class GO, class NO> void Mesh<SC, LO, GO, NO>::setReferenceConfiguration() {
    // Bemerkung: Repeated und Unique sind unterschiedlich lang!!! => zwei Schleifen

    // Setze zunaechst alles auf Null, andernfalls kann man nicht drauf zugreifen
    //    vec2D_dbl_ptr_Type zeroRep(new vec2D_dbl_Type(pointsRep_->size(),vec_dbl_Type(pointsRep_->at(0).size(),0.0)));
    //    vec2D_dbl_ptr_Type zeroUni(new vec2D_dbl_Type(pointsUni_->size(),vec_dbl_Type(pointsUni_->at(0).size(),0.0)));

    pointsRepRef_.reset(new vec2D_dbl_Type());
    pointsUniRef_.reset(new vec2D_dbl_Type());
    // zeroRep und zeroUni leben nur hier drinnen, weswegen wir Pointer gleich Pointer setzen koennen.
    // TODO: *PointsRepRef_ = *zeroRep funktioniert nicht.
    *pointsRepRef_ = *pointsRep_;
    *pointsUniRef_ = *pointsUni_;

    //    // Repeated
    //    for(int i = 0; i < pointsRep_->size(); i++)
    //    {
    //        for(int j = 0; j < pointsRep_->at(0).size(); j++)
    //        {
    //            pointsRepRef_->at(i).at(j) = PointsRep_->at(i).at(j);
    //        }
    //    }
    //
    //    // Unique
    //    for(int i = 0; i < PointsUni_->size(); i++)
    //    {
    //        for(int j = 0; j < PointsUni_->at(0).size(); j++)
    //        {
    //            PointsUniRef_->at(i).at(j) = PointsUni_->at(i).at(j);
    //        }
    //    }
}

template <class SC, class LO, class GO, class NO>
void Mesh<SC, LO, GO, NO>::moveMesh(MultiVectorPtr_Type displacementUnique, MultiVectorPtr_Type displacementRepeated) {
    // Bemerkung: Repeated und Unique sind unterschiedlich lang!!! => zwei Schleifen
    TEUCHOS_TEST_FOR_EXCEPTION(displacementRepeated.is_null(), std::runtime_error,
                               " displacementRepeated in moveMesh is null.")
    TEUCHOS_TEST_FOR_EXCEPTION(displacementUnique.is_null(), std::runtime_error,
                               " displacementRepeated in moveMesh is null.")
    // Repeated
    Teuchos::ArrayRCP<const SC> values = displacementRepeated->getData(0); // only 1 MV
    for (int i = 0; i < pointsRepRef_->size(); i++) {
        for (int j = 0; j < pointsRepRef_->at(0).size(); j++) {
            // Sortierung von DisplacementRepeated ist x-y-x-y-x-y-x-y bzw. x-y-z-x-y-z-x-y-z
            // Achtung: DisplacementRepeated ist ein Pointer der mit (*) dereferenziert werden muss.
            // Operator[] kann nicht auf einen Pointer angewendet werden!!!
            // Es sei denn es ist ein Array.
            pointsRep_->at(i).at(j) = pointsRepRef_->at(i).at(j) + values[dim_ * i + j];
        }
    }

    // Unique
    values = displacementUnique->getData(0); // only 1 MV
    for (int i = 0; i < pointsUniRef_->size(); i++) {
        for (int j = 0; j < pointsUniRef_->at(0).size(); j++) {
            // Sortierung von DisplacementRepeated ist x-y-x-y-x-y-x-y bzw. x-y-z-x-y-z-x-y-z
            // Erklaerung: DisplacementUnique ist ein Vector-Pointer, wo in jedem Eintrag ein MultiVector-Pointer drin
            // steht (std::vector<MultiVector_ptr_Type>) Greife mit ->at auf den Eintrag des Vektors zu (hier nur ein
            // Eintrag vorhanden), dereferenziere den damit erhaltenen MultiVector-Pointer (als Referenz) um einen
            // MultiVector zu erhalten.
            // Greife dann mit [] auf das entsprechende Array (double *&) im MultiVector zu (hier gibt es nur einen)
            // und anschliessend mit [] auf den Wert des Arrays.
            // Beachte falls x ein Array ist (also z.B. double *), dann ist x[i] := *(x+i)!!!
            // Liefert also direkt den Wert und keinen Pointer auf einen double.
            // Achtung: MultiVector[] liefert double* wohingegen MultiVector() Epetra_Vector* zurueck liefert
            pointsUni_->at(i).at(j) = pointsUniRef_->at(i).at(j) + values[dim_ * i + j];
        }
    }
}

template <class SC, class LO, class GO, class NO> void Mesh<SC, LO, GO, NO>::create_AABBTree() {
    if (AABBTree_.is_null()) {
        AABBTree_.reset(new AABBTree_Type());
    }
    AABBTree_->createTreeFromElements(getElementsC(), getPointsRepeated());
}

template <class SC, class LO, class GO, class NO>
vec_int_ptr_Type Mesh<SC, LO, GO, NO>::findElemsForPoints(vec2D_dbl_ptr_Type queryPoints) {
    int numPoints = queryPoints->size();

    // Return vector. -1 means that point is in no elem, otherwise entry is the
    // elem the point is in
    vec_int_ptr_Type pointToElem(new vec_int_Type(numPoints, -1));

    // Create tree if it is empty
    if (AABBTree_->isEmpty()) {
        AABBTree_->createTreeFromElements(getElementsC(), getPointsRepeated(), false);
    }

    // Query the AABBTree
    map<int, list<int>> treeToItem;
    map<int, list<int>> itemToTree;
    tie(treeToItem, itemToTree) = AABBTree_->scanTree(queryPoints, false);

    // FIXME: put this in a function of AABBTree?
    // unnest the returned answer for each query_point
    int point = -1;
    bool found = false;
    list<int> rectangles;
    list<int> elements;
    for (auto keyValue : itemToTree) {
        // FIXME: put this in a function of AABBTree?
        // rectangles is a list<int> of all rectangles point is in
        // find the element(s) that is/are in all of said rectangles.
        // If there is only one element that is the element the point is in,
        // if not we have to query all remaining elements
        point = keyValue.first;
        rectangles = keyValue.second;

        // query all remaining elements
        for (auto rectangle : rectangles) {
            elements = AABBTree_->getElements(rectangle);
            for (auto element : elements) {
                found = isPointInElem(queryPoints->at(point), element);
                if (found) {
                    pointToElem->at(point) = element;
                    break;
                }
            }
            if (found) {
                // we already found the element, no need to check additional rectangles
                break;
            }
        }
    }
    return pointToElem;
}

template <class SC, class LO, class GO, class NO>
vec_dbl_Type Mesh<SC, LO, GO, NO>::getBaryCoords(vec_dbl_Type point, int element) {
    vec_int_Type localNodes = elementsC_->getElement(element).getVectorNodeList();
    vec2D_dbl_Type coords(localNodes.size(), vec_dbl_Type(2, 0.0) // FIXME: this depends on the dimension
    );
    int node = 0;
    for (int localNode = 0; localNode < localNodes.size(); localNode++) {
        node = localNodes.at(localNode);             // get global node_id
        coords.at(localNode) = pointsRep_->at(node); // get global coordinates
    }

    double px, py, x1, x2, x3, y1, y2, y3;
    px = point.at(0);
    py = point.at(1);
    x1 = coords.at(0).at(0);
    y1 = coords.at(0).at(1);
    x2 = coords.at(1).at(0);
    y2 = coords.at(1).at(1);
    x3 = coords.at(2).at(0);
    y3 = coords.at(2).at(1);

    // baryzentric coordinates
    double det_T = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);

    vec_dbl_Type baryCoords(3, 0.0);
    baryCoords[0] = (y2 - y3) * (px - x3) + (x3 - x2) * (py - y3);
    baryCoords[0] = baryCoords[0] / det_T;

    baryCoords[1] = (y3 - y1) * (px - x3) + (x1 - x3) * (py - y3);
    baryCoords[1] = baryCoords[1] / det_T;

    baryCoords[2] = 1 - baryCoords[1] - baryCoords[0];
    return baryCoords;
}

template <class SC, class LO, class GO, class NO>
bool Mesh<SC, LO, GO, NO>::isPointInElem(vec_dbl_Type point, int element) {
    // FIXME: This is only valid for a triangle
    vec_dbl_Type baryCoords;
    baryCoords = getBaryCoords(point, element);

    if (baryCoords[0] >= 0 && baryCoords[1] >= 0 && baryCoords[2] >= 0) {
        return true;
    }
    return false;
}

template <class SC, class LO, class GO, class NO> vec2D_int_ptr_Type Mesh<SC, LO, GO, NO>::getElements() {
    this->elementsVec_ = Teuchos::rcp(new vec2D_int_Type(this->elementsC_->numberElements()));
    for (int i = 0; i < this->elementsVec_->size(); i++)
        this->elementsVec_->at(i) = this->elementsC_->getElement(i).getVectorNodeList();

    return this->elementsVec_;
}

template <class SC, class LO, class GO, class NO>
void Mesh<SC,LO,GO,NO>::correctNormalDirections(){

    int outwardNormals = 0;
    int inwardNormals = 0;
    for (UN T=0; T<elementsC_->numberElements(); T++) {
        FiniteElement fe = elementsC_->getElement( T );
        ElementsPtr_Type subEl = fe.getSubElements(); // might be null
        for (int surface=0; surface<fe.numSubElements(); surface++) {
            FiniteElement feSub = subEl->getElement( surface  );
            vec_int_Type nodeListElement = fe.getVectorNodeList();
            if(subEl->getDimension() == dim_-1 ){
                vec_int_Type nodeList = feSub.getVectorNodeListNonConst();
                int numNodes_T = nodeList.size();
                
                vec_dbl_Type v_E(dim_,1.);
                double norm_v_E=1.;
                LO id0 = nodeList[0];

                Helper::computeSurfaceNormal(dim_, pointsRep_,nodeList,v_E,norm_v_E);

                std::sort(nodeList.begin(), nodeList.end());
                std::sort(nodeListElement.begin(), nodeListElement.end());

                std::vector<int> v_symDifference;

                std::set_symmetric_difference(
                    nodeList.begin(), nodeList.end(),
                    nodeListElement.begin(), nodeListElement.end(),
                    std::back_inserter(v_symDifference));

                LO id1 = v_symDifference[0];

                vec_dbl_Type p0(dim_,0.);
                for(int i=0; i< dim_; i++)
                    p0[i] = pointsRep_->at(id1)[i] - pointsRep_->at(id0)[i];

                double sum = 0.;
                for(int i=0; i< dim_; i++)
                    sum += p0[i] * v_E[i];
                
                if(sum<=0){
                    outwardNormals++;
                }
                if(sum>0){
                    inwardNormals++;
                }
                if(sum>0)
                    flipSurface(feSub);
                    

            }
        }
    }
    reduceAll<int, int> (*this->getComm(), REDUCE_SUM, inwardNormals, outArg (inwardNormals));
    reduceAll<int, int> (*this->getComm(), REDUCE_SUM, outwardNormals, outArg (outwardNormals));

    if(this->getComm()->getRank() == 0){
        cout << " ############################################ " << endl;
        cout << " Mesh Orientation Statistic " << endl;
        cout << " Number of outward normals " << outwardNormals << endl;
        cout << " Number of inward normals " << inwardNormals << endl;
        cout << " ############################################ " << endl;
    }

}
template <class SC, class LO, class GO, class NO>
void Mesh<SC,LO,GO,NO>::correctElementOrientation(){

    for (UN T=0; T<elementsC_->numberElements(); T++) {
        FiniteElement fe = elementsC_->getElement( T );
        ElementsPtr_Type subEl = fe.getSubElements(); // might be null
        for (int surface=0; surface<fe.numSubElements(); surface++) {
            FiniteElement feSub = subEl->getElement( surface  );
            if(subEl->getDimension() == dim_-1 ){
                vec_int_Type nodeList = feSub.getVectorNodeListNonConst ();
                int numNodes_T = nodeList.size();
                
                vec_dbl_Type v_E(dim_,1.);
                double norm_v_E=1.;

                Helper::computeSurfaceNormal(dim_, pointsRep_,nodeList,v_E,norm_v_E);


            }
        }
    }

}

// We allways want a outward normal direction
template <class SC, class LO, class GO, class NO>
void Mesh<SC,LO,GO,NO>::flipSurface(FiniteElement_Type feSub){

    vec_LO_Type surfaceElements_vec = feSub.getVectorNodeList();

    if(dim_ == 2){

    }
    else if(dim_ == 3){

        if(FEType_ == "P1"){
            LO id1,id2,id3,id4,id5,id6;
            id1= surfaceElements_vec[0];
            id2= surfaceElements_vec[1];
            id3= surfaceElements_vec[2];
           
            surfaceElements_vec[0] = id1;
            surfaceElements_vec[1] = id3;
            surfaceElements_vec[2] = id2;           
        }
        else if(FEType_ == "P2"){
            LO id1,id2,id3,id4,id5,id6;
            id1= surfaceElements_vec[0];
            id2= surfaceElements_vec[1];
            id3= surfaceElements_vec[2];
            id4= surfaceElements_vec[3];
            id5= surfaceElements_vec[4];
            id6= surfaceElements_vec[5];

            surfaceElements_vec[0] = id1;
            surfaceElements_vec[1] = id3;
            surfaceElements_vec[2] = id2;
            surfaceElements_vec[3] = id6;
            surfaceElements_vec[4] = id5;
            surfaceElements_vec[5] = id4;
        }
        else    
            TEUCHOS_TEST_FOR_EXCEPTION( true, std::runtime_error, "We can only flip normals for P1 or P2 elements. Invalid " << FEType_ << " " );

    }   

}
// ################# Nonlinear Schwarz related functions ##################
template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::MapConstPtr_Type Mesh<SC, LO, GO, NO>::getElementMapOverlapping() const {
    TEUCHOS_TEST_FOR_EXCEPTION(elementMapOverlapping_.is_null(), std::runtime_error,
                               "Overlapping element map of mesh does not exist.");
    return elementMapOverlapping_;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::MapConstPtr_Type Mesh<SC, LO, GO, NO>::getElementMapOverlappingInterior() const {
    TEUCHOS_TEST_FOR_EXCEPTION(elementMapOverlappingInterior_.is_null(), std::runtime_error,
                               "Interior overlapping element map of mesh does not exist.");
    return elementMapOverlappingInterior_;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::MapConstPtr_Type Mesh<SC, LO, GO, NO>::getMapOverlapping() const {
    TEUCHOS_TEST_FOR_EXCEPTION(elementMapOverlapping_.is_null(), std::runtime_error,
                               "Overlapping element map of mesh does not exist.");
    return mapOverlapping_;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::MapConstPtr_Type Mesh<SC, LO, GO, NO>::getMapOverlappingInterior() const {
    TEUCHOS_TEST_FOR_EXCEPTION(elementMapOverlappingInterior_.is_null(), std::runtime_error,
                               "Overlapping interior element map of mesh does not exist.");
    return mapOverlappingInterior_;
}

template <class SC, class LO, class GO, class NO>
typename Mesh<SC, LO, GO, NO>::ElementsPtr_Type Mesh<SC, LO, GO, NO>::getElementsOverlapping() const {
    TEUCHOS_TEST_FOR_EXCEPTION(elementsOverlapping_.is_null(), std::runtime_error,
                               "Overlapping elements have not been constructed.");
    return elementsOverlapping_;
}

template <class SC, class LO, class GO, class NO>
void Mesh<SC, LO, GO, NO>::setElementsC(ElementsPtr_Type newElements) const {
    elementsC_ = newElements;
}
// Replace all unique and repeated members in these functions as required by the application.
template <class SC, class LO, class GO, class NO>
void Mesh<SC, LO, GO, NO>::replaceRepeatedMembers(const MapPtr_Type newMap, const vec2D_dbl_ptr_Type newPoints,
                                                  const vec_int_ptr_Type newBCs) const {
    // Ensure that all members being replaced have the same number of local elements
    TEUCHOS_TEST_FOR_EXCEPTION(newMap->getNodeNumElements() != newPoints->size(), std::runtime_error,
                               "New memembers must have the same number of local elements");
    this->mapRepeated_ = newMap;
    this->pointsRep_ = newPoints;
    this->bcFlagRep_ = newBCs;
}

template <class SC, class LO, class GO, class NO>
void Mesh<SC, LO, GO, NO>::replaceUniqueMembers(const MapPtr_Type newMap, const vec2D_dbl_ptr_Type newPoints,
                                                const vec_int_ptr_Type newBCs) const {
    // Ensure that all members being replaced have the same number of local elements
    TEUCHOS_TEST_FOR_EXCEPTION(newMap->getNodeNumElements() != newPoints->size(), std::runtime_error,
                               "New memembers must have the same number of local elements");
    this->mapUnique_ = newMap;
    this->pointsUni_ = newPoints;
    this->bcFlagUni_ = newBCs;
}


}
#endif
