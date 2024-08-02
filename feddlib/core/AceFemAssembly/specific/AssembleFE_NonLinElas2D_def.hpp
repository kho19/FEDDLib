#ifndef AssembleFE_NonLinElas2D_DEF_hpp
#define AssembleFE_NonLinElas2D_DEF_hpp

#include "AssembleFE_NonLinElas2D_decl.hpp"
#include "feddlib/core/FEDDCore.hpp"
#include <iostream>
#include <vector>
// #ifdef FEDD_HAVE_ACEGENINTERFACE
#include "aceinterface.hpp"
// #endif

namespace FEDD {

/*!

 \brief Constructor for  AssembleFE_NonLinElas2D

@param[in] flag Flag of element
@param[in] nodesRefConfig Nodes of element in reference configuration
@param[in] params Parameterlist for current problem
@param[in} tuple Vector of tuples with Discretization information

*/
template <class SC, class LO, class GO, class NO>
AssembleFE_NonLinElas2D<SC, LO, GO, NO>::AssembleFE_NonLinElas2D(int flag, vec2D_dbl_Type nodesRefConfig,
                                                                 ParameterListPtr_Type params,
                                                                 tuple_disk_vec_ptr_Type tuple)
    : AssembleFE<SC, LO, GO, NO>(flag, nodesRefConfig, params, tuple) {
    // Youngs modulus
    E_ = this->params_->sublist("Parameter").get("E", 1000.0);
    density_ = this->params_->sublist("Parameter").get("Density", 1.);
    poissonRatio_ = this->params_->sublist("Parameter").get("Poisson Ratio", 0.4e-0);
    forceY_ = this->params_->sublist("Parameter").get("Volume force", 0.01);
    forceX_ = 0;

    FEType_ = std::get<1>(this->diskTuple_->at(0));
    dofs_ = std::get<2>(this->diskTuple_->at(0));
    numNodes_ = std::get<3>(this->diskTuple_->at(0));
    dofsElement_ = dofs_ * numNodes_;
}

// Both assembleJacobian() and assembleRHS() do the same, sufficient to just call one of them. Implemented like this
// because the AceGenInterface only permits assembling the Jacobian and residual at the same time
template <class SC, class LO, class GO, class NO> void AssembleFE_NonLinElas2D<SC, LO, GO, NO>::assembleJacobian() {

    SmallMatrixPtr_Type elementMatrix = Teuchos::rcp(new SmallMatrix_Type(dofsElement_));
    assemblyNonLinElas2D(elementMatrix);
    this->jacobian_ = elementMatrix;
}

template <class SC, class LO, class GO, class NO> void AssembleFE_NonLinElas2D<SC, LO, GO, NO>::assembleRHS() {
    // assemblyNonLinElas();
}
template <class SC, class LO, class GO, class NO>
void AssembleFE_NonLinElas2D<SC, LO, GO, NO>::assemblyNonLinElas2D(SmallMatrixPtr_Type &elementMatrix) {

    // #ifdef FEDD_HAVE_ACEGENINTERFACE

    this->rhsVec_.reset(new vec_dbl_Type(dofsElement_, 0.));
    // x,y coords of the nodes of the current element
    std::vector<double> positions(dofsElement_);
    // the current solution at the nodes of the current element
    std::vector<double> displacements(dofsElement_);
    // Parameters: [Youngs mod. E, Poisson ratio nu, body force x, body force y, density rho, thickness]
    std::vector<double> domainData(6);

    for (int i = 0; i < numNodes_; i++) {
        for (int j = 0; j < dofs_; j++) {
            positions.at(i * dofs_ + j) = this->getNodesRefConfig()[i][j];
        }
    }

    for (int i = 0; i < dofsElement_; i++) {
        displacements.at(i) = (*this->solution_)[i];
    }

    domainData[0] = this->E_;
    domainData[1] = this->poissonRatio_;
    // domainData[2] = forceX_;
    // domainData[3] = forceY_;
    domainData[2] = 0;
    domainData[3] = 0;
    domainData[4] = density_;
    // TODO: kho what is a suitable thickness to use?
    domainData[5] = 1.;
    // domainData = {3000, 0.2, 0.0, 0.0, 2.0, 1.1};
    // domainData = {3000, 0.2, forceX_, forceY_, 2.0, 1.1};

    AceGenInterface::NeoHookeTriangle2D3PlaneStress neoHookeElement(positions, displacements, domainData,
                                                                   this->getGlobalElementID());
    int error = neoHookeElement.compute();
    TEUCHOS_TEST_FOR_EXCEPTION(error != 0, std::runtime_error, "Residual and stiffness matrix computation failed");

    auto residual = neoHookeElement.getResiduum();
    TEUCHOS_TEST_FOR_EXCEPTION(residual.size() == 0, std::runtime_error, "Residual computation failed");
    for (int i = 0; i < residual.size(); i++) {
        (*this->rhsVec_).at(i) = residual.at(i);
    }

    auto stiffnessMatrix = neoHookeElement.getStiffnessMatrix();
    for (UN i = 0; i < this->dofsElement_; i++) {
        for (UN j = 0; j < this->dofsElement_; j++) {
            (*elementMatrix)[i][j] = stiffnessMatrix[i][j];
        }
    }
    // #endif // FEDD_HAVE_ACEGENINTERFACE
}

} // namespace FEDD
#endif
