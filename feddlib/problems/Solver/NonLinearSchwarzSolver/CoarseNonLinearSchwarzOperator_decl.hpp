#ifndef COARSENONLINEARSCHWARZOPERATOR_DECL_HPP
#define COARSENONLINEARSCHWARZOPERATOR_DECL_HPP

#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/BlockMatrix_decl.hpp"
#include "feddlib/core/LinearAlgebra/BlockMultiVector_decl.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/Mesh/Mesh_decl.hpp"
#include "feddlib/problems/Solver/NonLinearSchwarzSolver/NonLinearOperator_decl.hpp"
#include "feddlib/problems/abstract/NonLinearProblem_decl.hpp"
#include <FROSch_IPOUHarmonicCoarseOperator_decl.hpp>
#include <FROSch_SchwarzOperator_def.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_ScalarTraitsDecl.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Xpetra_Matrix.hpp>

/*!
 Declaration of CoarseNonLinearSchwarzOperator

 @brief Implements the coarse correction T_0 from the nonlinear Schwarz approach
 @author Kyrill Ho
 @version 1.0
 @copyright KH
 */

namespace FROSch {

template <class SC = default_sc, class LO = default_lo, class GO = default_go, class NO = default_no>
class CoarseNonLinearSchwarzOperator : public IPOUHarmonicCoarseOperator<SC, LO, GO, NO>,
                                       public NonLinearOperator<SC, LO, GO, NO> {

  protected:
    using CommPtr = typename SchwarzOperator<SC, LO, GO, NO>::CommPtr;

    using XMapPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMapPtr;
    using ConstXMapPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMapPtr;
    using XMapPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMapPtrVecPtr;
    using ConstXMapPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMapPtrVecPtr;
    using XMapPtrVecPtr2D = typename SchwarzOperator<SC, LO, GO, NO>::XMapPtrVecPtr2D;
    using ConstXMapPtrVecPtr2D = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMapPtrVecPtr2D;

    using XMatrixPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMatrixPtr;
    using ConstXMatrixPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMatrixPtr;

    using XMultiVector = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVector;
    using XMultiVectorPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVectorPtr;
    using ConstXMultiVectorPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMultiVectorPtr;
    using XMultiVectorPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::XMultiVectorPtrVecPtr;
    using ConstXMultiVectorPtrVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXMultiVectorPtrVecPtr;

    using XCrsGraph = typename SchwarzOperator<SC, LO, GO, NO>::XCrsGraph;
    using GraphPtr = typename SchwarzOperator<SC, LO, GO, NO>::GraphPtr;
    using ConstXCrsGraphPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstXCrsGraphPtr;
    using ParameterListPtr = typename SchwarzOperator<SC, LO, GO, NO>::ParameterListPtr;

    using DDInterfacePtr = typename SchwarzOperator<SC, LO, GO, NO>::DDInterfacePtr;

    using EntitySetPtr = typename SchwarzOperator<SC, LO, GO, NO>::EntitySetPtr;

    using InterfaceEntityPtr = typename SchwarzOperator<SC, LO, GO, NO>::InterfaceEntityPtr;

    using CoarseSpacePtr = typename SchwarzOperator<SC, LO, GO, NO>::CoarseSpacePtr;

    using PartitionOfUnityPtr = typename SchwarzOperator<SC, LO, GO, NO>::PartitionOfUnityPtr;
    using InterfacePartitionOfUnityPtr = typename SchwarzOperator<SC, LO, GO, NO>::InterfacePartitionOfUnityPtr;

    using LocalPartitionOfUnityBasisPtr = typename SchwarzOperator<SC, LO, GO, NO>::LocalPartitionOfUnityBasisPtr;

    using SolverPtr = typename SchwarzOperator<SC, LO, GO, NO>::SolverPtr;
    using SolverFactoryPtr = typename SchwarzOperator<SC, LO, GO, NO>::SolverFactoryPtr;

    using UN = typename SchwarzOperator<SC, LO, GO, NO>::UN;
    using UNVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::UNVecPtr;

    using LOVec = typename SchwarzOperator<SC, LO, GO, NO>::LOVec;
    using LOVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::LOVecPtr;
    using LOVecPtr2D = typename SchwarzOperator<SC, LO, GO, NO>::LOVecPtr2D;

    using GOVec = typename SchwarzOperator<SC, LO, GO, NO>::GOVec;
    using GOVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::GOVecPtr;
    using GOVecView = typename SchwarzOperator<SC, LO, GO, NO>::GOVecView;
    using GOVecPtr2D = typename SchwarzOperator<SC, LO, GO, NO>::GOVecPtr2D;

    using SCVec = typename SchwarzOperator<SC, LO, GO, NO>::SCVec;
    using SCVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::SCVecPtr;
    using ConstSCVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::ConstSCVecPtr;

    using BoolVecPtr = typename SchwarzOperator<SC, LO, GO, NO>::BoolVecPtr;

    using MeshPtrFEDD = typename Teuchos::RCP<FEDD::Mesh<SC, LO, GO, NO>>;
    using NonLinearProblemPtrFEDD = typename Teuchos::RCP<FEDD::NonLinearProblem<SC, LO, GO, NO>>;
    using BlockMatrixPtrFEDD = typename Teuchos::RCP<FEDD::BlockMatrix<SC, LO, GO, NO>>;
    using BlockMultiVectorPtrFEDD = typename Teuchos::RCP<FEDD::BlockMultiVector<SC, LO, GO, NO>>;
    using MapConstPtrFEDD = typename Teuchos::RCP<const FEDD::Map<LO, GO, NO>>;
    using ST = typename Teuchos::ScalarTraits<SC>;

  public:
    explicit CoarseNonLinearSchwarzOperator(NonLinearProblemPtrFEDD problem, ParameterListPtr parameterList);

    ~CoarseNonLinearSchwarzOperator() = default;

    int initialize() override;

    // the compute method is implemented in FROSch_CoarseOperator_def

    void apply(const BlockMultiVectorPtrFEDD x, BlockMultiVectorPtrFEDD y, SC alpha = ScalarTraits<SC>::one(),
               SC beta = ScalarTraits<SC>::zero());

    void apply(const XMultiVector &x, XMultiVector &y, SC alpha = ScalarTraits<SC>::one(),
               SC beta = ScalarTraits<SC>::zero()) override;

    // This apply method must be overridden but does not make sense in the context of nonlinear operators
    void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly, ETransp mode = NO_TRANS,
               SC alpha = ScalarTraits<SC>::one(), SC beta = ScalarTraits<SC>::zero()) const override;

    void describe(FancyOStream &out, const EVerbosityLevel verbLevel = Describable::verbLevel_default) const override;

    string description() const override;

    BlockMatrixPtrFEDD getCoarseJacobian() const;

  private:
    // FEDDLib problem object. (will need to be changed for interoperability)
    NonLinearProblemPtrFEDD problem_;
    // Current point of evaluation. Null if none has been passed
    BlockMultiVectorPtrFEDD x_;
    // Current output. Null if no valid output stored.
    BlockMultiVectorPtrFEDD y_;
    // Tangent of the nonlinear problem R_iDF(u_i)P_i as used in ASPEN
    BlockMatrixPtrFEDD coarseJacobian_;

    // Newtons method params
    double newtonTol_;
    int maxNumIts_;
    std::string criterion_;
    // Temp. problem state params
    BlockMultiVectorPtrFEDD solutionTmp_;
    BlockMatrixPtrFEDD systemTmp_;
};

} // namespace FROSch

#endif
