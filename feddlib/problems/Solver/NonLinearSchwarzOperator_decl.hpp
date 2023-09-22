#ifndef NONLINEARSCHWARZOPERATOR_DECL_HPP
#define NONLINEARSCHWARZOPERATOR_DECL_HPP

#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include <FROSch_SchwarzOperator_def.hpp>
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

  public:
    NonLinearSchwarzOperator(ConstXMatrixPtr k, ParameterListPtr parameterList);

    ~NonLinearSchwarzOperator();

    int initialize() { initialize(0); };
    int initialize(int overlap);

    int compute();

    void apply(const XMultiVector &x, XMultiVector &y, bool usePreconditionerOnly, ETransp mode = NO_TRANS,
               SC alpha = ScalarTraits<SC>::one(), SC beta = ScalarTraits<SC>::zero()) const;

    // TODO might need this
    int buildElementNodeList();

  private:
    void assemble(XMatrixPtr localJacobian, XMultiVectorPtr localRHS);

    // Current point of evaluation. Null if none has been passed
    mutable XMultiVectorPtr x_;
    // Current output. Null if no valid output stored.
    mutable XMultiVectorPtr fX_;
    // Temp Vectors for apply()
    mutable XMultiVectorPtr xTmp_;
    mutable XMultiVectorPtr yTmp_;

    // Newtons method params
    double newtonTol_;
    int maxNumIts_;
    GraphPtr elementNodeList_;
};

} // namespace FROSch

#endif
