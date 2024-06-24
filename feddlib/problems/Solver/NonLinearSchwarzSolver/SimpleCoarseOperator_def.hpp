#ifndef SIMPLECOARSEOPERATPR_DEF_HPP
#define SIMPLECOARSEOPERATPR_DEF_HPP

#include "SimpleCoarseOperator_decl.hpp"
#include "feddlib/core/Utils/FEDDUtils.hpp"
#include <FROSch_CoarseOperator_decl.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultSerialComm.hpp>
#include <Teuchos_EReductionType.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_implicit_cast.hpp>
#include <Xpetra_ConfigDefs.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_ExportFactory.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_Map_decl.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <Xpetra_MultiVectorFactory_decl.hpp>
#include <stdexcept>
#include <string>
#include <trilinos_amd.h>

namespace FROSch {

template <class SC, class LO, class GO, class NO>
SimpleCoarseOperator<SC, LO, GO, NO>::SimpleCoarseOperator(ConstXMatrixPtr k, ParameterListPtr parameterList)
    : CoarseOperator<SC, LO, GO, NO>(k, parameterList) {}

template <class SC, class LO, class GO, class NO> int SimpleCoarseOperator<SC, LO, GO, NO>::initialize() {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "SimpleCoarseOperator requires coarse Jacobians, fine and coarse maps and mpi and "
                               "coarse solve comms during initialization");
};

template <class SC, class LO, class GO, class NO>
int SimpleCoarseOperator<SC, LO, GO, NO>::initialize(const Teuchos::RCP<const CoarseOperator<SC, LO, GO, NO>> inputOp) {
    // Shallow copy into underlying CoarseOperator object
    // For this to work FROSch has to be modified (considering master branch at commit
    // 986c0264d5bc81983e78227ed9375dd1105bde10) by making LevelID_ in FROSch_SchwarzOperator_decl.hpp non-const. This
    // ensures that default assignment operators are generated by the compiler (and can be used here).
    CoarseOperator<SC, LO, GO, NO>::operator=(*inputOp);
    return 0;
}

template <class SC, class LO, class GO, class NO> int SimpleCoarseOperator<SC, LO, GO, NO>::compute() { return 0; }

template <class SC, class LO, class GO, class NO>
typename SimpleCoarseOperator<SC, LO, GO, NO>::ConstXMapPtr
SimpleCoarseOperator<SC, LO, GO, NO>::computeCoarseSpace(CoarseSpacePtr coarseSpace) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "SimpleCoarseOperator is not able to compute the coarse space");
    return null;
}

// These methods must be overriden, but are not used by the simple coarse operator
template <class SC, class LO, class GO, class NO> int SimpleCoarseOperator<SC, LO, GO, NO>::buildElementNodeList() {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "SimpleCoarseOperator is not able to build the element node list");
    return 0;
}
template <class SC, class LO, class GO, class NO>
int SimpleCoarseOperator<SC, LO, GO, NO>::buildGlobalGraph(Teuchos::RCP<DDInterface<SC, LO, GO, NO>> theDDInterface_) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "SimpleCoarseOperator is not able to build the global graph");
    return 0;
}

template <class SC, class LO, class GO, class NO>
typename SimpleCoarseOperator<SC, LO, GO, NO>::XMapPtr
SimpleCoarseOperator<SC, LO, GO, NO>::BuildRepeatedMapCoarseLevel(ConstXMapPtr &nodesMap, UN dofsPerNode,
                                                                  ConstXMapPtrVecPtr dofsMaps, UN partition) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "SimpleCoarseOperator is not able to build the coarse repeated map");
    return null;
}

template <class SC, class LO, class GO, class NO> int SimpleCoarseOperator<SC, LO, GO, NO>::buildCoarseGraph() {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "SimpleCoarseOperator is not able to build the coarse graph");
    return 0;
}

// TODO: a lot of this implementation could be moved to FROSch
template <class SC, class LO, class GO, class NO>
void SimpleCoarseOperator<SC, LO, GO, NO>::describe(FancyOStream &out, const EVerbosityLevel verbLevel) const {
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
    for (size_t dec = 10; dec < this->CoarseMatrix_->getGlobalNumRows(); dec *= 10) {
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
        if (this->CoarseMatrix_->isFillComplete()) {
            out << "isFillComplete: true" << endl
                << "Global dimensions: [" << this->CoarseMatrix_->getGlobalNumRows() << ", "
                << this->CoarseMatrix_->getGlobalNumCols() << "]" << endl
                << "Global number of entries: " << this->CoarseMatrix_->getGlobalNumEntries() << endl
                << endl
                << "Global max number of entries in a row: " << this->CoarseMatrix_->getGlobalMaxNumRowEntries()
                << endl;
        } else {
            out << "isFillComplete: false" << endl
                << "Global dimensions: [" << this->CoarseMatrix_->getGlobalNumRows() << ", "
                << this->CoarseMatrix_->getGlobalNumCols() << "]" << endl;
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
    if (this->CoarseMatrix_->getRowMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->CoarseMatrix_->getRowMap()->describe(out, vl);
    }

    // Describe the column Map.
    if (myRank == 0) {
        out << "Column Map: ";
    }
    if (this->CoarseMatrix_->getColMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else if (this->CoarseMatrix_->getColMap() == this->CoarseMatrix_->getRowMap()) {
        if (myRank == 0) {
            out << "same as row Map" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->CoarseMatrix_->getColMap()->describe(out, vl);
    }

    // Describe the domain Map.
    if (myRank == 0) {
        out << "Domain Map: ";
    }
    if (this->CoarseMatrix_->getDomainMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else if (this->CoarseMatrix_->getDomainMap() == this->CoarseMatrix_->getRowMap()) {
        if (myRank == 0) {
            out << "same as row Map" << endl;
        }
    } else if (this->CoarseMatrix_->getDomainMap() == this->CoarseMatrix_->getColMap()) {
        if (myRank == 0) {
            out << "same as column Map" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->CoarseMatrix_->getDomainMap()->describe(out, vl);
    }

    // Describe the range Map.
    if (myRank == 0) {
        out << "Range Map: ";
    }
    if (this->CoarseMatrix_->getRangeMap().is_null()) {
        if (myRank == 0) {
            out << "null" << endl;
        }
    } else if (this->CoarseMatrix_->getRangeMap() == this->CoarseMatrix_->getDomainMap()) {
        if (myRank == 0) {
            out << "same as domain Map" << endl;
        }
    } else if (this->CoarseMatrix_->getRangeMap() == this->CoarseMatrix_->getRowMap()) {
        if (myRank == 0) {
            out << "same as row Map" << endl;
        }
    } else {
        if (myRank == 0) {
            out << endl;
        }
        this->CoarseMatrix_->getRangeMap()->describe(out, vl);
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
            for (size_t r = 0; r < this->CoarseMatrix_->getLocalNumRows(); ++r) {
                const size_t nE = this->CoarseMatrix_->getNumEntriesInLocalRow(r);
                GO gid = this->CoarseMatrix_->getRowMap()->getGlobalElement(r);
                out << std::setw(width) << myRank << std::setw(width) << gid << std::setw(width) << nE;
                if (vl == VERB_EXTREME) {
                    if (this->CoarseMatrix_->isGloballyIndexed()) {
                        ArrayView<const GO> rowinds;
                        ArrayView<const SC> rowvals;
                        this->CoarseMatrix_->getGlobalRowView(gid, rowinds, rowvals);
                        for (size_t j = 0; j < nE; ++j) {
                            out << " (" << rowinds[j] << ", " << rowvals[j] << ") ";
                        }
                    } else if (this->CoarseMatrix_->isLocallyIndexed()) {
                        ArrayView<const LO> rowinds;
                        ArrayView<const SC> rowvals;
                        this->CoarseMatrix_->getLocalRowView(r, rowinds, rowvals);
                        for (size_t j = 0; j < nE; ++j) {
                            out << " (" << this->CoarseMatrix_->getColMap()->getGlobalElement(rowinds[j]) << ", "
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
std::string SimpleCoarseOperator<SC, LO, GO, NO>::description() const {
    return "ASPEN Coarse Operator";
}

template <class SC, class LO, class GO, class NO>
void SimpleCoarseOperator<SC, LO, GO, NO>::extractLocalSubdomainMatrix_Symbolic() {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                               "SimpleCoarseOperator is not able to do local symbolic matrix extraction");
}

} // namespace FROSch

#endif // SIMPLECOARSEOPERATPR_DEF_HPP
