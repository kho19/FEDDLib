#ifndef SIMPLEOVERLAPPINGOPERATOR_DECL_HPP
#define SIMPLEOVERLAPPINGOPERATOR_DECL_HPP

#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/Mesh/Mesh_decl.hpp"
#include "feddlib/problems/abstract/NonLinearProblem_decl.hpp"
#include <FROSch_OverlappingOperator_decl.hpp>
#include <FROSch_SchwarzOperator_def.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Xpetra_Matrix.hpp>
#include <stdexcept>

/*!
 Declaration of ASPENOverlappingOperator

 @brief Implements the ASPEN tangent $D\mathcal{F}(u) = \sum P_i(R_iDF(u_i)P_i)^{-1}R_iDF(u_i)$ from the nonlinear
 Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class SimpleOverlappingOperator : public OverlappingOperator<SC, LO, GO, NO> {

  protected:
    using CommPtr = typename SchwarzOperator<SC, LO, GO, NO>::CommPtr;

    using XMap = typename SchwarzOperator<SC, LO, GO, NO>::XMap;
    using XMapPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMapPtr;
    using ConstXMapPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMapPtr;
    using XMapPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMapPtrVecPtr;
    using ConstXMapPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMapPtrVecPtr;
    using ConstXMapPtrVecPtr2D = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMapPtrVecPtr2D;

    using XMatrixPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMatrixPtr;
    using ConstXMatrixPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMatrixPtr;

    using XCrsGraph = typename SchwarzOperator<SC, LO, GO, NO>::XCrsGraph;
    using GraphPtr = typename SchwarzOperator<SC, LO, GO, NO>::GraphPtr;
    using ConstXCrsGraphPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXCrsGraphPtr;

    using XMultiVector = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVector;
    using XMultiVectorPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVectorPtr;
    using XMultiVectorPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVectorPtrVecPtr;
    using ConstXMultiVectorPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMultiVectorPtr;
    using ConstXMultiVectorPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMultiVectorPtrVecPtr;

    using XImport = typename SchwarzOperator<SC, LO, GO, NO>::XImport;
    using XImportPtr = typename SchwarzOperator<SC, LO, GO, NO>::XImportPtr;
    using XImportPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::XImportPtrVecPtr;

    using XExport = typename SchwarzOperator<SC, LO, GO, NO>::XExport;
    using XExportPtr = typename SchwarzOperator<SC, LO, GO, NO>::XExportPtr;
    using XExportPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::XExportPtrVecPtr;

    using ParameterListPtr = typename SchwarzOperator<SC, LO, GO, NO>::ParameterListPtr;

    using SolverPtr = typename SchwarzOperator<SC, LO, GO, NO>::SolverPtr;
    using SolverFactoryPtr = typename SchwarzOperator<SC, LO, GO, NO>::SolverFactoryPtr;

    using UN = typename SchwarzOperator<SC, LO, GO, NO>::UN;

    using IntVec = typename SchwarzOperator<SC, LO, GO, NO>::IntVec;
    using IntVec2D = typename SchwarzOperator<SC, LO, GO, NO>::IntVec2D;

    using GOVec = typename SchwarzOperator<SC, LO, GO, NO>::GOVec;
    using GOVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::GOVecPtr;

    using LOVec = typename SchwarzOperator<SC, LO, GO, NO>::LOVec;
    using LOVecPtr2D = typename SchwarzOperator<SC, LO, GO, NO>::LOVecPtr2D;

    using SCVec = typename SchwarzOperator<SC, LO, GO, NO>::SCVec;

    using ConstLOVecView = typename SchwarzOperator<SC, LO, GO, NO>::ConstLOVecView;

    using ConstGOVecView = typename SchwarzOperator<SC, LO, GO, NO>::ConstGOVecView;

    using ConstSCVecView = typename SchwarzOperator<SC, LO, GO, NO>::ConstSCVecView;

    using MeshPtrFEDD = typename Teuchos::RCP<FEDD::Mesh<SC, LO, GO, NO>>;
    using NonLinearProblemPtrFEDD = typename Teuchos::RCP<FEDD::NonLinearProblem<SC, LO, GO, NO>>;
    using BlockMatrixPtrFEDD = typename Teuchos::RCP<FEDD::BlockMatrix<SC, LO, GO, NO>>;
    using BlockMultiVectorPtrFEDD = typename Teuchos::RCP<FEDD::BlockMultiVector<SC, LO, GO, NO>>;
    using MapConstPtrFEDD = typename Teuchos::RCP<const FEDD::Map<LO, GO, NO>>;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    SimpleOverlappingOperator(NonLinearProblemPtrFEDD problem, ParameterListPtr parameterList);

    ~SimpleOverlappingOperator() = default;

    int initialize() override {
        TEUCHOS_TEST_FOR_EXCEPTION(
            true, std::runtime_error,
            "SimpleOverlappingOperator requires local Jacobians, local and global maps and serial "
            "and mpi comms during initialization");
    };
    int initialize(CommPtr serialComm, ConstXMatrixPtr jacobianGhosts, ConstXMapPtr overlappingMap,
                   ConstXMapPtr overlappingGhostsMap, ConstXMapPtr uniqueMap,
                   FEDD::vec_int_ptr_Type bcFlagOverlappingGhosts);

    int compute() override;

    void apply(const XMultiVector &x, XMultiVector &y, ETransp mode = NO_TRANS, SC alpha = ScalarTraits<SC>::one(),
               SC beta = ScalarTraits<SC>::zero()) const override;

    void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly, ETransp mode = NO_TRANS,
               SC alpha = ScalarTraits<SC>::one(), SC beta = ScalarTraits<SC>::zero()) const override;

    void describe(FancyOStream &out, const EVerbosityLevel verbLevel = Describable::verbLevel_default) const override;

    string description() const override;

  protected:
    // Do nothing op in this case since the local overlapping matrices are already known
    int updateLocalOverlappingMatrices() override { return 0; }

  private:
    // Tangent of the nonlinear Schwarz operator is saved in this->OverlappingMatrix_ which lives on
    // serial version of OverlappingMap_
    // This operator does not know whether the system it is being applied to is a block system or not. This information
    // is only necessary in operators that do assembly. The following maps are intitialized to the correct dof maps

    // Distributed maps
    // GhostsMap is stored in this->OverlappingMap_
    ConstXMapPtr uniqueMap_;
    // Importers
    XImportPtr importerUniqueToGhosts_;

    // Temp. vectors for local results
    mutable XMultiVectorPtr x_Ghosts_;
    mutable XMultiVectorPtr y_unique_;
    mutable XMultiVectorPtr y_Ghosts_;

    // Boundary condition flags for recognizing the ghost boundary
    FEDD::vec_int_ptr_Type bcFlagOverlappingGhosts_;

    // We need to know how many dofs per node there are. This is stored in the problem object. We keep a pointer to this
    // object here to avoid having to copy the vector around.
    NonLinearProblemPtrFEDD problem_;
    // Recombination mode. [Restricted, Averaging, Addition]
    /* RecombinationMode recombinationMode_; */
    /* BlockMultiVectorPtrFEDD multiplicity_; */
};

} // namespace FROSch

#endif
