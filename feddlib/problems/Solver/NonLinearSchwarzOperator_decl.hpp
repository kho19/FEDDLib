#ifndef NONLINEARSCHWARZOPERATOR_DECL_HPP
#define NONLINEARSCHWARZOPERATOR_DECL_HPP

#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/Mesh/Mesh_decl.hpp"
#include "feddlib/problems/abstract/NonLinearProblem_decl.hpp"
#include <FROSch_SchwarzOperator_def.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Xpetra_Matrix.hpp>

/*!
 Declaration of NonLinearSchwarzOperator

 @brief Implements the surrogate problem $\mathcal{F}(u)$ from the nonlinear Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class NonLinearSchwarzOperator : public SchwarzOperator<SC, LO, GO, NO> {

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
    using BlockMultiVectorPtrFEDD = typename Teuchos::RCP<FEDD::BlockMultiVector<SC, LO, GO, NO>>;
    using MapConstPtrFEDD = typename Teuchos::RCP<const FEDD::Map<LO, GO, NO>>;
   
   

  public:
    NonLinearSchwarzOperator(CommPtr mpiComm, ParameterListPtr parameterList, NonLinearProblemPtrFEDD problem);

    ~NonLinearSchwarzOperator();

    int initialize() {
        initialize(0);
        return 0;
    };
    int initialize(int overlap);

    int compute();

    void apply(const BlockMultiVectorPtrFEDD x, BlockMultiVectorPtrFEDD y, SC alpha = ScalarTraits<SC>::one(),
               SC beta = ScalarTraits<SC>::zero());

    // This apply method must be overridden but does not make sense in the context of nonlinear operators
    void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly, ETransp mode = NO_TRANS,
               SC alpha = ScalarTraits<SC>::one(), SC beta = ScalarTraits<SC>::zero()) const;

    void describe(FancyOStream &out, const EVerbosityLevel verbLevel = Describable::verbLevel_default) const;

    string description() const;

  private:
    void replaceMapAndExportProblem();

    // FEDDLib problem object. (will need to be changed for interoperability)
    NonLinearProblemPtrFEDD problem_;
    // Current point of evaluation. Null if none has been passed
    BlockMultiVectorPtrFEDD x_;
    // Current output. Null if no valid output stored.
    BlockMultiVectorPtrFEDD y_;

    // Newtons method params
    double newtonTol_;
    int maxNumIts_;

    // Maps for saving the mpiComm maps of the problems domain when replacing them with serial maps
    ConstXMapPtr mapRepeatedMpiTmp_;
    ConstXMapPtr mapUniqueMpiTmp_;
    ConstXMapPtr elementMapMpiTmp_;
    ConstXMapPtr elementMapOverlappingMpiTmp_;
    ConstXMapPtr mapOverlappingMpiTmp_;
    // Vectors for saving repeated and unique points
    FEDD::vec2D_dbl_ptr_Type pointsRepTmp_;
    FEDD::vec2D_dbl_ptr_Type pointsUniTmp_;
    // Vectors for saving the boundary conditios
    FEDD::vec_int_ptr_Type bcFlagRepTmp_;
    FEDD::vec_int_ptr_Type bcFlagUniTmp_;
    // Vector of elements for saving elementsC_
    Teuchos::RCP<FEDD::Elements> elementsCTmp_;
    // Current global solution of the problem
    BlockMultiVectorPtrFEDD solutionTmp_;
};

} // namespace FROSch

#endif
