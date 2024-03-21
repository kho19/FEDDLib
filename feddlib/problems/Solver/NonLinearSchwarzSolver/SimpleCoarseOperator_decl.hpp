#ifndef SIMPLECOARSEOPERATOR_DECL_HPP
#define SIMPLECOARSEOPERATOR_DECL_HPP

#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/Mesh/Mesh_decl.hpp"
#include "feddlib/problems/abstract/NonLinearProblem_decl.hpp"
#include <FROSch_CoarseOperator_decl.hpp>
#include <FROSch_SchwarzOperator_def.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_TestForException.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Xpetra_Matrix.hpp>

/*!
 Declaration of simple coarse operator
 This class is just a wrapper around an existing NonlinearCoarseOperator providing it with an alternative apply() method
 that evaluates the tanget. This design choice facilitates reusing the underlying CoarseOperator for evaluation of the
 coarse problem and its tangent.

 @brief Implements the coarse tangent $D\mathcal{F}(u) = P_0(R_0DF(u_0)P_0)^{-1}R_0DF(u_0)$ from the nonlinear
 Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class SimpleCoarseOperator : public CoarseOperator<SC, LO, GO, NO> {

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

    using CoarseSpacePtr = typename SchwarzOperator<SC, LO, GO, NO>::CoarseSpacePtr;
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
    SimpleCoarseOperator(ConstXMatrixPtr k, ParameterListPtr parameterList);

    ~SimpleCoarseOperator() = default;

    int initialize() override;
    int initialize(Teuchos::RCP<CoarseOperator<SC, LO, GO, NO>> inputOp);

    int compute() override;

    ConstXMapPtr computeCoarseSpace(CoarseSpacePtr coarseSpace) override;

    // These methods must be overriden, but are not used by the simple coarse operator
    int buildElementNodeList() override;
    int buildGlobalGraph(Teuchos::RCP<DDInterface<SC, LO, GO, NO>> theDDInterface_) override;
    XMapPtr BuildRepeatedMapCoarseLevel(ConstXMapPtr &nodesMap, UN dofsPerNode, ConstXMapPtrVecPtr dofsMaps,
                                        UN partition) override;

    int buildCoarseGraph() override;

    void describe(FancyOStream &out, const EVerbosityLevel verbLevel = Describable::verbLevel_default) const override;

    string description() const override;

  protected:
    void extractLocalSubdomainMatrix_Symbolic() override;

  private:
};

} // namespace FROSch

#endif
