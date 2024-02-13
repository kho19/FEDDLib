#ifndef SimpleOVERLAPPINGOPERATPR_DEF_HPP
#define SimpleOVERLAPPINGOPERATPR_DEF_HPP

#include "SimpleOverlappingOperator_decl.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_EReductionType.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_implicit_cast.hpp>
#include <Xpetra_ConfigDefs.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_Map_decl.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <stdexcept>
#include <string>

namespace FROSch {

template <class SC, class LO, class GO, class NO>
SimpleOverlappingOperator<SC, LO, GO, NO>::SimpleOverlappingOperator(ConstXMatrixPtr k, ParameterListPtr parameterList)
    : OverlappingOperator<SC, LO, GO, NO>(k, parameterList), overlappingWithGhostsMap_(), overlappingColMap_(),
      uniqueMap_(), overlappingWithGhostsMatrix_() {
    // Override the combine mode of the FROSch operator base object from the nonlinear Schwarz configuration
    if (!this->ParameterList_->get("Combine Mode", "Restricted").compare("Averaging")) {
        this->Combine_ = this->CombinationType::Averaging;
    } else if (!this->ParameterList_->get("Combine Mode", "Restricted").compare("Full")) {
        this->Combine_ = this->CombinationType::Full;
    } else if (!this->ParameterList_->get("Combine Mode", "Restricted").compare("Restricted")) {
        this->Combine_ = this->CombinationType::Restricted;
    }
}

template <class SC, class LO, class GO, class NO>
int SimpleOverlappingOperator<SC, LO, GO, NO>::initialize(CommPtr serialComm, ConstXMatrixPtr localJacobian,
                                                          ConstXMapPtr overlappingWithGhostsMap,
                                                          ConstXMapPtr overlappingColMap, ConstXMapPtr overlappingMap,
                                                          ConstXMapPtr uniqueMap) {
    // AlgebraicOverlappingOperator does: calculates overlap multiplicity if needed and does symbolic extraction
    // of local subdomain matrix and initialization of solver (symbolic factorization)
    // Here we just read in the localSubdomainMatrix since it already exists

    // Need to set the following as AlgebraicOverlappingOperator initialize()
    // x OverlappingMap_ (global)
    // x OverlappingMatrix_ (global and local)
    // x Multiplicity_
    // x SubdomainSolver_
    // x IsInitialzed_
    // x IsComputed_
    // - subdomainMatrix_ (NO: only used as an intermediate overlapping matrix from which values are taken for
    // localSubdomainMatrix_)
    // - localSubdomainMatrix_ (NO: this is the non-const predecessor of OverlappingMatrix_ used for extraction and
    // factorization in AlgebraicOverlappingOperator)
    // - ExtractLocalSubdomainMatrix_Symbolic_Done flag (NO: should not need this since not doing any symbolic
    // extraction)

    // Need to pass the serial communicator on which localJacobian lives
    this->SerialComm_ = serialComm;
    overlappingWithGhostsMap_ = overlappingWithGhostsMap;
    overlappingColMap_ = overlappingColMap;
    this->OverlappingMap_ = overlappingMap;
    uniqueMap_ = uniqueMap;

    buildOverlappingMatrices(localJacobian);
    //  Calculate overlap multiplicity if needed
    this->initializeOverlappingOperator();

    // Start debugging
    auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
    FEDD::logGreen("Multiplicity_", this->MpiComm_);
    this->Multiplicity_->describe(*out, Teuchos::VERB_EXTREME);
    // End debugging
    // Compute symbolic factorization
    this->initializeSubdomainSolver(this->OverlappingMatrix_);
    this->IsInitialized_ = true;
    this->IsComputed_ = false;
    return 0;
}

template <class SC, class LO, class GO, class NO> int SimpleOverlappingOperator<SC, LO, GO, NO>::compute() {
    // AlgebraicOverlappingOperator does: gets values of the local subdomain matrices and computes the numerical
    // factorization
    // Here we do not need to fill values into the sparsity pattern since they are already there
    // ==> updateLocalOverlappingMatrices() is a no-op
    TEUCHOS_TEST_FOR_EXCEPTION(!this->IsInitialized_, std::runtime_error,
                               "ASPENOverlappingOperator must be initialized before calling compute()");
    this->computeOverlappingOperator();
    return 0;
}

template <class SC, class LO, class GO, class NO>
void SimpleOverlappingOperator<SC, LO, GO, NO>::apply(const XMultiVector &x, XMultiVector &y, ETransp mode, SC alpha,
                                                      SC beta) const {
    // move the input to the local serial overlapping maps including ghost layer (localJacobian rowMap)
    // apply the local OverlappingIncludingGhostsMatrix_
    // restrict to the local overlapping subdomain (remove ghosts) corresponds to (this->OverlappingMap_)
    // apply the local solution
}

// NOTE: It may be more efficient to locally assemble these matrices in nonlinearSchwarzOperator, avoiding the
// communication overhead of this function
template <class SC, class LO, class GO, class NO>
void SimpleOverlappingOperator<SC, LO, GO, NO>::buildOverlappingMatrices(ConstXMatrixPtr localJacobian) {

    TEUCHOS_TEST_FOR_EXCEPTION(
        overlappingWithGhostsMap_.is_null(), std::runtime_error,
        "overlappingWithGhostsMatrix_ lives on mapOverlappingGlobal_ and cannot be built without.");
    TEUCHOS_TEST_FOR_EXCEPTION(
        overlappingColMap_.is_null(), std::runtime_error,
        "overlappingColMap_ is required for correct local insertion of off rank values on ghost layer.");
    TEUCHOS_TEST_FOR_EXCEPTION(this->OverlappingMap_.is_null(), std::runtime_error,
                               "OverlappingMap_ is required for correct assembly of local Jacobians.");
    TEUCHOS_TEST_FOR_EXCEPTION(uniqueMap_.is_null(), std::runtime_error,
                               "uniqueMap_ is required for correct communication of off rank values.");

    // Get the global max number of elements per row across all serially distributed matrices
    int localMaxNumRowEntries = localJacobian->getLocalMaxNumRowEntries();
    int globalMaxNumRowEntries = 0;
    Teuchos::reduceAll(*this->MpiComm_, Teuchos::REDUCE_MAX, 1, &localMaxNumRowEntries, &globalMaxNumRowEntries);

    auto globalUniqueMatrix =
        Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(uniqueMap_, static_cast<GO>(globalMaxNumRowEntries));

    Teuchos::Array<GO> tempIndices;
    // Local and global number of elements is the same (as is expected)
    for (auto i = 0; i < globalUniqueMatrix->getMap()->getLocalNumElements(); i++) {
        Teuchos::ArrayView<const LO> indices;
        Teuchos::ArrayView<const SC> values;
        // Map local unique index to local overlapping index
        auto localOverlappingIndex = overlappingWithGhostsMap_->getLocalElement(uniqueMap_->getGlobalElement(i));
        localJacobian->getLocalRowView(localOverlappingIndex, indices, values);
        Teuchos::Array<GO> globalIndices(indices.size());

        // Map local indices to global indices for insertGlobalValues
        // Column map from global assembly is valid for interior nodes since interior nodes "see" all surrounding
        // nodes even on serially distributed subdomains. In constrast, ghost nodes will not "see" globally adjacent
        // nodes in other subdomains during serial assembly leading to different number of row entries in globally
        // assembled and locally assembled and then combined matrices.
        // insertLocalValues cannot be used since column map of globalOverlappingWithGhostsMatrix unknown
        for (auto j = 0; j < indices.size(); j++) {
            globalIndices.at(j) = overlappingColMap_->getGlobalElement(indices[j]);
        }

        globalUniqueMatrix->insertGlobalValues(uniqueMap_->getGlobalElement(i), globalIndices, values);
    }

    // Correct values will be broadcasted during fill complete since values are distributed according to uniqueMap_ and
    // otherwise matrix is filled with zeros
    globalUniqueMatrix->fillComplete();

    auto importer = Xpetra::ImportFactory<LO, GO>::Build(uniqueMap_, overlappingWithGhostsMap_);
    auto globalOverlappingWithGhostsMatrix =
        Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(globalUniqueMatrix, *importer);

    overlappingWithGhostsMatrix_ = Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(
        localJacobian->getRowMap(), localJacobian->getLocalMaxNumRowEntries());

    // Fill the local overlappingMatrixWithGhosts
    // Alternative way of copying to local matrix used below
    for (auto i = 0; i < overlappingWithGhostsMap_->getLocalNumElements(); i++) {
        Teuchos::ArrayView<const LO> indices;
        Teuchos::ArrayView<const SC> values;
        globalOverlappingWithGhostsMatrix->getLocalRowView(i, indices, values);
        Teuchos::Array<GO> localIndices;
        Teuchos::Array<SC> localValues;
        for (auto j = 0; j < indices.size(); j++) {
            // Rows of globalOverlappingWithGhostsMatrix corresponding to the ghost layer contain entries for nodes in
            // adjacent subdomains. These entries should not be added to local overlappingWithGhostsMatrix_.
            // If global index is in the local row map, then the corresponding node is in the subdomain
            auto globalIndex = globalOverlappingWithGhostsMatrix->getColMap()->getGlobalElement(indices[j]);
            if (overlappingWithGhostsMap_->isNodeGlobalElement(globalIndex)) {
                localIndices.push_back(indices[j]);
                localValues.push_back(values[j]);
            }
        }
        overlappingWithGhostsMatrix_->insertGlobalValues(i, localIndices, localValues);
    }
    overlappingWithGhostsMatrix_->fillComplete();

    // Restrict localJacobian to the interior of the subdomain (excluding ghosts)
    importer = Xpetra::ImportFactory<LO, GO>::Build(uniqueMap_, this->OverlappingMap_);
    auto globalOverlappingMatrix = Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(
        this->OverlappingMap_, globalOverlappingWithGhostsMatrix->getGlobalMaxNumRowEntries());
    // Not calling fillComplete() here since matrix will be discarded after filling local matrix
    // Have to use getGlobalRowView() instead of getLocalRowView() since no column map exists without calling
    // fillComplete()
    globalOverlappingMatrix->doImport(*globalUniqueMatrix, *importer, Xpetra::CombineMode::INSERT);

    auto overlappingSerialMap = Xpetra::MapFactory<LO, GO, NO>::Build(
        this->OverlappingMap_->lib(), this->OverlappingMap_->getLocalNumElements(),
        this->OverlappingMap_->getLocalNumElements(), ST::zero(), this->SerialComm_);

    // Required for construction since this->OverlappingMatrix_ is const
    auto overlappingMatrix =
        Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(overlappingSerialMap, localJacobian->getLocalMaxNumRowEntries());

    for (auto i = 0; i < overlappingSerialMap->getLocalNumElements(); i++) {
        Teuchos::ArrayView<const GO> indices;
        Teuchos::ArrayView<const SC> values;
        globalOverlappingMatrix->getGlobalRowView(this->OverlappingMap_->getGlobalElement(i), indices, values);

        if (indices.size() > 0) {
            Teuchos::Array<GO> localIndices;
            Teuchos::Array<SC> localValues;
            for (auto j = 0; j < indices.size(); j++) {
                // If node is in interior map (excluding ghosts), add to OverlappingMatrix_
                auto localIndex = this->OverlappingMap_->getLocalElement(indices[j]);
                if (localIndex >= 0) {
                    localIndices.push_back(localIndex);
                    localValues.push_back(values[j]);
                }
            }
            // inserting local and global elements does the same thing here since the object is serial
            overlappingMatrix->insertGlobalValues(i, localIndices, localValues);
        }
    }
    overlappingMatrix->fillComplete();
    this->OverlappingMatrix_ = overlappingMatrix;

    // Start debugging
    // ##################### Print functions ###################
    /* auto out = Teuchos::VerboseObjectBase::getDefaultOStream(); */
    /* FEDD::logGreen("localJacobian", this->MpiComm_); */
    /* localJacobian->describe(*out, Teuchos::VERB_EXTREME); */
    /* FEDD::logGreen("OverlappingMatrix_", this->MpiComm_); */
    /* this->OverlappingMatrix_->describe(*out, Teuchos::VERB_EXTREME); */
    /* FEDD::logGreen("globalOverlappingMatrix", this->MpiComm_); */
    /* globalOverlappingMatrix->fillComplete(); */
    /* globalOverlappingMatrix->describe(*out, Teuchos::VERB_EXTREME); */
    /* FEDD::logGreen("overlappingMatrixWithGhosts_", this->MpiComm_); */
    /* overlappingMatrixWithGhosts_->describe(*out, Teuchos::VERB_EXTREME); */
    /* FEDD::logGreen("globalOverlappingMatrixWithGhosts", this->MpiComm_); */
    /* globalOverlappingMatrixWithGhosts->describe(*out, Teuchos::VERB_EXTREME); */
    /* FEDD::logGreen("uniqueMatrix", this->MpiComm_); */
    /* globalUniqueMatrixWithGhosts->describe(*out, Teuchos::VERB_EXTREME); */
    // End debugging
}

// TODO: a lot of this implementation could be moved to FROSch
template <class SC, class LO, class GO, class NO>
void SimpleOverlappingOperator<SC, LO, GO, NO>::describe(FancyOStream &out, const EVerbosityLevel verbLevel) const {
    using std::endl;
    using std::setw;
    using Teuchos::ArrayView;
    using Teuchos::Comm;
    using Teuchos::RCP;
    using Teuchos::TypeNameTraits;
    using Teuchos::VERB_DEFAULT;
    using Teuchos::VERB_EXTREME;
    using Teuchos::VERB_HIGH;
    using Teuchos::VERB_LOW;
    using Teuchos::VERB_MEDIUM;
    using Teuchos::VERB_NONE;

    const Teuchos::EVerbosityLevel vl = (verbLevel == VERB_DEFAULT) ? VERB_LOW : verbLevel;

    if (vl == VERB_NONE) {
        return; // Don't print anything at all
    }

    // By convention, describe() always begins with a tab.
    Teuchos::OSTab tab0(out);

    RCP<const Comm<int>> comm = this->MpiComm_;
    const int myRank = comm->getRank();
    const int numProcs = comm->getSize();
    size_t width = 1;
    for (size_t dec = 10; dec < this->OverlappingMatrix_->getGlobalNumRows(); dec *= 10) {
        ++width;
    }
    width = std::max<size_t>(width, static_cast<size_t>(11)) + 2;
    // set boolean printing to true/false
    cout << std::boolalpha;

    //    none: print nothing
    //     low: print O(1) info from node 0
    //  medium: print O(P) info, num entries per process
    //    high: print O(N) info, num entries per row
    // extreme: print O(NNZ) info: print indices and values
    //
    // for medium and higher, print constituent objects at specified verbLevel
    if (myRank == 0) {
        out << "Precomputed Overlapping Schwarz Operator:" << endl;
    }
    Teuchos::OSTab tab1(out);

    if (myRank == 0) {
        if (this->getObjectLabel() != "") {
            out << "Label: \"" << this->getObjectLabel() << "\", ";
        }
        {
            out << "Template parameters:" << endl;
            Teuchos::OSTab tab2(out);
            out << "Scalar: " << TypeNameTraits<SC>::name() << endl
                << "LocalOrdinal: " << TypeNameTraits<LO>::name() << endl
                << "GlobalOrdinal: " << TypeNameTraits<GO>::name() << endl
                << "Node: " << TypeNameTraits<NO>::name() << endl;
        }
        if (this->OverlappingMatrix_->isFillComplete()) {
            out << "isFillComplete: true" << endl
                << "Global dimensions: [" << this->OverlappingMatrix_->getGlobalNumRows() << ", "
                << this->OverlappingMatrix_->getGlobalNumCols() << "]" << endl
                << "Global number of entries: " << this->OverlappingMatrix_->getGlobalNumEntries() << endl
                << endl
                << "Global max number of entries in a row: " << this->OverlappingMatrix_->getGlobalMaxNumRowEntries()
                << endl;
        } else {
            out << "isFillComplete: false" << endl
                << "Global dimensions: [" << this->OverlappingMatrix_->getGlobalNumRows() << ", "
                << this->OverlappingMatrix_->getGlobalNumCols() << "]" << endl;
        }
        out << "isInitialized: " << this->IsInitialized_ << endl;
        out << "isComputed: " << this->IsComputed_ << endl;
    }

    if (vl < VERB_MEDIUM) {
        return; // all done!
    }

    // Describe the row Map.
    if (myRank == 0) {
        out << endl << "Row Map:" << endl;
    }
    if (this->OverlappingMatrix_->getRowMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->OverlappingMatrix_->getRowMap()->describe(out, vl);
    }

    // Describe the column Map.
    if (myRank == 0) {
        out << "Column Map: ";
    }
    if (this->OverlappingMatrix_->getColMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else if (this->OverlappingMatrix_->getColMap() == this->OverlappingMatrix_->getRowMap()) {
        if (myRank == 0) {
            out << "same as row Map" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->OverlappingMatrix_->getColMap()->describe(out, vl);
    }

    // Describe the domain Map.
    if (myRank == 0) {
        out << "Domain Map: ";
    }
    if (this->OverlappingMatrix_->getDomainMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else if (this->OverlappingMatrix_->getDomainMap() == this->OverlappingMatrix_->getRowMap()) {
        if (myRank == 0) {
            out << "same as row Map" << endl;
        }
    } else if (this->OverlappingMatrix_->getDomainMap() == this->OverlappingMatrix_->getColMap()) {
        if (myRank == 0) {
            out << "same as column Map" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->OverlappingMatrix_->getDomainMap()->describe(out, vl);
    }

    // Describe the range Map.
    if (myRank == 0) {
        out << "Range Map: ";
    }
    if (this->OverlappingMatrix_->getRangeMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else if (this->OverlappingMatrix_->getRangeMap() == this->OverlappingMatrix_->getDomainMap()) {
        if (myRank == 0) {
            out << "same as domain Map" << endl;
        }
    } else if (this->OverlappingMatrix_->getRangeMap() == this->OverlappingMatrix_->getRowMap()) {
        if (myRank == 0) {
            out << "same as row Map" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->OverlappingMatrix_->getRangeMap()->describe(out, vl);
    }

    if (vl < VERB_HIGH) {
        return; // all done!
    }

    // O(N) and O(NNZ) data
    for (int curRank = 0; curRank < numProcs; ++curRank) {
        if (myRank == curRank) {
            out << std::setw(width) << "Proc Rank" << std::setw(width) << "Global Row" << std::setw(width)
                << "Num Entries";
            if (vl == VERB_EXTREME) {
                out << std::setw(width) << "(Index,Value)";
            }
            out << endl;
            for (size_t r = 0; r < this->OverlappingMatrix_->getLocalNumRows(); ++r) {
                const size_t nE = this->OverlappingMatrix_->getNumEntriesInLocalRow(r);
                GO gid = this->OverlappingMatrix_->getRowMap()->getGlobalElement(r);
                out << std::setw(width) << myRank << std::setw(width) << gid << std::setw(width) << nE;
                if (vl == VERB_EXTREME) {
                    if (this->OverlappingMatrix_->isGloballyIndexed()) {
                        ArrayView<const GO> rowinds;
                        ArrayView<const SC> rowvals;
                        this->OverlappingMatrix_->getGlobalRowView(gid, rowinds, rowvals);
                        for (size_t j = 0; j < nE; ++j) {
                            out << " (" << rowinds[j] << ", " << rowvals[j] << ") ";
                        }
                    } else if (this->OverlappingMatrix_->isLocallyIndexed()) {
                        ArrayView<const LO> rowinds;
                        ArrayView<const SC> rowvals;
                        this->OverlappingMatrix_->getLocalRowView(r, rowinds, rowvals);
                        for (size_t j = 0; j < nE; ++j) {
                            out << " (" << this->OverlappingMatrix_->getColMap()->getGlobalElement(rowinds[j]) << ", "
                                << rowvals[j] << ") ";
                        }
                    } // globally or locally indexed
                }     // vl == VERB_EXTREME
                out << endl;
            } // for each row r on this process
        }     // if (myRank == curRank)

        // Give output time to complete
        comm->barrier();
        comm->barrier();
        comm->barrier();
    } // for each process p
}

template <class SC, class LO, class GO, class NO>
std::string SimpleOverlappingOperator<SC, LO, GO, NO>::description() const {
    return "ASPEN Overlapping Operator";
}

} // namespace FROSch

#endif // SimpleOVERLAPPINGOPERATPR_DEF_HPP
