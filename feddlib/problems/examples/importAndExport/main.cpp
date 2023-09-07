#include "feddlib/core/FEDDCore.hpp"
#include "feddlib/core/General/DefaultTypeDefs.hpp"
#include "feddlib/core/LinearAlgebra/Map_decl.hpp"
#include "feddlib/core/LinearAlgebra/Matrix_decl.hpp"

#include <Teuchos_ArrayRCPDecl.hpp>
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_VerbosityLevel.hpp>
#include <Teuchos_implicit_cast.hpp>
#include <Tpetra_CombineMode.hpp>
#include <Tpetra_Core.hpp>
#include <Xpetra_ConfigDefs.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#include <Xpetra_ExportFactory.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_MapFactory_decl.hpp>
#include <Xpetra_MatrixFactory.hpp>
#include <cstdlib>
#include <stdexcept>
#include <vector>

typedef unsigned UN;
typedef default_sc SC;
typedef default_lo LO;
typedef default_go GO;
typedef default_no NO;

using namespace FEDD;
using namespace Teuchos;

/**
 * xpetraImportExport() and feddlibImportExport() are basic setups for testing/understanding import and export behaviour
 * of distributed matrices in the Xpetra and the FEDDLib implmentations respectively. Most testing is done in
 * tpetraImportExport as this gives access to REPLACE, ADD, INSERT, ABSMAX, ZERO, ADD_ASSIGN, while Xpetra only
 * defines ADD and INSERT
 */
void xpetraImportExport(int argc, char *argv[]) {

    const Xpetra::UnderlyingLib lib = Xpetra::UseTpetra;
    Teuchos::oblackholestream blackhole;
    const Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);
    const auto out = Teuchos::VerboseObjectBase::getDefaultOStream();

    const Teuchos::RCP<const Teuchos::Comm<int>> comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();
    const int size = comm->getSize();
    TEUCHOS_TEST_FOR_EXCEPTION(size > 2, std::runtime_error, "This test requires one or two ranks");

    const auto myRank = comm->getRank();
    const auto zero = Teuchos::implicit_cast<GO>(0);
    const auto four = Teuchos::implicit_cast<GO>(4);

    // Element lists for overlapping map
    auto elementList = Teuchos::ArrayRCP<GO>(3);
    for (auto i = 0; i < elementList.size(); i++) {
        elementList[i] = i + myRank;
    }
    Teuchos::ArrayView<const GO> elementListConstView = elementList();

    // One-to-one map
    Teuchos::RCP<const Xpetra::Map<LO, GO, NO>> uniqueMap =
        Xpetra::MapFactory<LO, GO, NO>::createUniformContigMap(lib, four, comm);
    // Both middle rows are owned by both processes
    Teuchos::RCP<const Xpetra::Map<LO, GO, NO>> overlappingMap =
        Xpetra::MapFactory<LO, GO, NO>::Build(lib, four, elementListConstView, zero, comm);

    // Build importer and exporter
    const auto uniqueToOverlappingImporter = Xpetra::ImportFactory<LO, GO, NO>::Build(uniqueMap, overlappingMap);
    const auto overlappingToUniqueExporter = Xpetra::ExportFactory<LO, GO, NO>::Build(overlappingMap, uniqueMap);
    // Values for prefilling matrices
    auto colidxVec = std::vector<GO>{0, 1, 2, 3};
    auto colidx = Teuchos::ArrayRCP<const GO>(colidxVec.data(), 0, 4, false);
    auto valZeroVec = std::vector<SC>{0, 0, 0, 0};
    auto valZero = Teuchos::ArrayRCP<SC>(valZeroVec.data(), 0, 4, false);
    auto valOneVec = std::vector<SC>{1, 1, 1, 1};
    auto valOne = Teuchos::ArrayRCP<SC>(valOneVec.data(), 0, 4, false);

    /* auto zerosOverlapping = rcp(new Xpetra::TpetraCrsMatrix<SC, LO, GO, NO>(overlappingMap, 4)); */
    auto zerosOverlapping = Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(overlappingMap, 4);
    for (int i = 0; i < 3; i++) {
        zerosOverlapping->insertGlobalValues(i + myRank, colidx(), valOne());
    }
    /* zerosOverlapping->fillComplete(); */

    /* auto onesUnique = rcp(new Xpetra::TpetraCrsMatrix<SC, LO, GO, NO>(uniqueMap, 4)); */
    auto onesUnique = Xpetra::MatrixFactory<SC, LO, GO, NO>::Build(uniqueMap, 4);
    for (int i = 0; i < 2; i++) {
        onesUnique->insertGlobalValues(i + 2 * myRank, colidx(), valZero());
    }
    onesUnique->fillComplete();
    onesUnique->describe(*out, Teuchos::VERB_EXTREME);

    // Test 1: Import from unique to overlapping distributions
    // Does not work after calling fillComplete() followed by resumeFill()
    // Xpetra::ADD and INSERT show the same behaviour
    /* zerosOverlapping->describe(*out, Teuchos::VERB_EXTREME); */
    /* zerosOverlapping->resumeFill(); */
    zerosOverlapping->doImport(*onesUnique, *uniqueToOverlappingImporter, Xpetra::INSERT);
    zerosOverlapping->fillComplete();
    zerosOverlapping->describe(*out, Teuchos::VERB_EXTREME);

    std::cout << "Size of communicator: " << size << std::endl;
}

void feddlibImportExport(int argc, char *argv[]) {

    const Xpetra::UnderlyingLib lib = Xpetra::UseTpetra;
    const string libString = "Tpetra";
    Teuchos::oblackholestream blackhole;
    const Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);
    const auto out = Teuchos::VerboseObjectBase::getDefaultOStream();

    const Teuchos::RCP<const Teuchos::Comm<int>> comm = Xpetra::DefaultPlatform::getDefaultPlatform().getComm();
    const int size = comm->getSize();
    TEUCHOS_TEST_FOR_EXCEPTION(size > 2, std::runtime_error, "This test requires one or two ranks");

    const auto myRank = comm->getRank();
    const auto zero = Teuchos::implicit_cast<GO>(0);
    const auto two = Teuchos::implicit_cast<GO>(2);
    const auto four = Teuchos::implicit_cast<GO>(4);

    // Element lists for overlapping map
    auto elementList = Teuchos::ArrayRCP<GO>(3);
    for (auto i = 0; i < elementList.size(); i++) {
        elementList[i] = i + myRank;
    }
    Teuchos::ArrayView<const GO> elementListConstView = elementList();

    // One-to-one map
    auto uniqueMap = rcp(new Map<LO, GO, NO>(libString, four, two, zero, comm));
    // Both middle rows are owned by both processes
    auto overlappingMap = rcp(new Map<LO, GO, NO>(libString, four, elementListConstView, zero, comm));

    // Values for prefilling matrices
    auto colidxVec = std::vector<GO>{0, 1, 2, 3};
    auto colidx = Teuchos::ArrayRCP<const GO>(colidxVec.data(), 0, 4, false);
    auto valZeroVec = std::vector<SC>{0, 0, 0, 0};
    auto valZero = Teuchos::ArrayRCP<SC>(valZeroVec.data(), 0, 4, false);
    auto valOneVec = std::vector<SC>{1, 1, 1, 1};
    auto valOne = Teuchos::ArrayRCP<SC>(valOneVec.data(), 0, 4, false);

    /* auto zerosOverlapping = rcp(new Xpetra::TpetraCrsMatrix<SC, LO, GO, NO>(overlappingMap, 4)); */
    auto zerosOverlapping = rcp(new Matrix<SC, LO, GO, NO>(overlappingMap, 4));
    /* for (int i = 0; i < 3; i++) { */
    /*     zerosOverlapping->insertGlobalValues(Teuchos::implicit_cast<GO>(i + myRank), colidx(), valOne()); */
    /* } */
    /* zerosOverlapping->fillComplete(); */
    /* zerosOverlapping->print(); */

    auto onesUnique = rcp(new Matrix<SC, LO, GO, NO>(uniqueMap, 4));
    for (int i = 0; i < 2; i++) {
        onesUnique->insertGlobalValues(i + 2 * myRank, colidx(), valZero());
    }
    onesUnique->fillComplete();
    onesUnique->print();

    // Test 1: Import from unique to overlapping distributions
    /* zerosOverlapping->resumeFill(); */
    zerosOverlapping->importFromVector(onesUnique);
    zerosOverlapping->fillComplete();
    zerosOverlapping->print();

    std::cout << "Size of communicator: " << size << std::endl;
}

void tpetraImportExport(int argc, char *argv[]) {

    Teuchos::oblackholestream blackhole;
    const Teuchos::GlobalMPISession mpiSession(&argc, &argv, &blackhole);
    const auto out = Teuchos::VerboseObjectBase::getDefaultOStream();

    const auto comm = Tpetra::getDefaultComm();
    const int size = comm->getSize();
    TEUCHOS_TEST_FOR_EXCEPTION(size > 2, std::runtime_error, "This test requires one or two ranks");

    const auto myRank = comm->getRank();
    // Operations: doImport() and doExport()
    // Objects: importer and exporter
    // Modes: INSERT and ADD
    // Distributions: overlapping and unique
    const auto zero = Teuchos::implicit_cast<GO>(0);
    const auto three = Teuchos::implicit_cast<size_t>(3);
    const auto four = Teuchos::implicit_cast<size_t>(4);
    const auto six = Teuchos::implicit_cast<size_t>(6);
    const auto eight = Teuchos::implicit_cast<size_t>(8);

    // Element lists for overlapping map
    std::vector<GO> elementList{myRank, myRank + 1, myRank + 2};
    Teuchos::ArrayView<const GO> elementListConstView = Teuchos::arrayViewFromVector(elementList);
    // Element list for column map
    std::vector<GO> columnMapList{0, 1, 2, 3};
    Teuchos::ArrayView<const GO> columnMapListConstView = Teuchos::arrayViewFromVector(columnMapList);

    // One-to-one map
    Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> uniqueMap = rcp(new Tpetra::Map<LO, GO, NO>(four, zero, comm));
    // Both middle rows are owned by both ranks
    Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> overlappingMap =
        rcp(new Tpetra::Map<LO, GO, NO>(six, elementListConstView, zero, comm));
    // All columns are owned by both ranks
    Teuchos::RCP<const Tpetra::Map<LO, GO, NO>> columnMap =
        rcp(new Tpetra::Map<LO, GO, NO>(eight, columnMapListConstView, zero, comm));

    /* uniqueMap->describe(*out, Teuchos::VERB_EXTREME); */
    /* overlappingMap->describe(*out, Teuchos::VERB_EXTREME); */
    /* columnMap->describe(*out, Teuchos::VERB_EXTREME); */

    // Build four importers and four exporters each from and to the different maps
    const auto uniqueToOverlappingImporter = Tpetra::Import<LO, GO, NO>(uniqueMap, overlappingMap);

    const auto overlappingToUniqueExporter = Tpetra::Export<LO, GO, NO>(overlappingMap, uniqueMap, out);

    // Values for prefilling matrices
    std::vector<GO> globalColIdxVec{0, 1, 2, 3};
    auto globalColIdx = Teuchos::arrayViewFromVector(globalColIdxVec);
    std::vector<LO> localColIdxVec{0, 1, 2, 3};
    auto localColIdx = Teuchos::arrayViewFromVector(localColIdxVec);
    std::vector<SC> valZeroVec{0, 0, 0, 0};
    auto valZero = Teuchos::arrayViewFromVector(valZeroVec);
    std::vector<SC> valOneVec{1, 1, 1, 1};
    auto valOne = Teuchos::arrayViewFromVector(valOneVec);
    std::vector<SC> valTwoVec{2, 2, 2, 2};
    auto valTwo = Teuchos::arrayViewFromVector(valTwoVec);

    // ----------------------- Test 1: Import from unique to overlapping distributions ----------------------------
    // NOTE calling fillComplete() followed by resumeFill() followed by doImport()/doExport()
    // results in undefined behaviour i.e. importing/exporting can only be carried out on matrices that are "pristine".
    // Values can be inserted immediately before, but without calling fillComplete() after insertion.
    // Mode specifiers:
    //  - REPLACE (does not work when the matrix has a dynamic graph i.e. it owns and can change the graph)
    //  - ADD (No difference from INSERT for doImport. Documentation says INSERT is more efficient if no value exists at
    //  the insert location)
    //  - INSERT (Inserts values. Existing value is summed with the incoming value.)
    //  - ABSMAX (same as REPLACE)
    //  - ZERO (does not place zeros as documented. Instead seems to make the map ignore overlap)
    //  - ADD_ASSIGN (Does not work at all, throwing "Should never get here!" error)
    auto overlappingMat = rcp(new Tpetra::CrsMatrix<SC, LO, GO, NO>(overlappingMap, 4));
    /* for (int i = 0; i < 3; i++) { */
    /*     overlappingMat->insertGlobalValues(i + myRank, globalColIdx(), valOne()); */
    /* } */
    /**/
    auto uniqueMat = rcp(new Tpetra::CrsMatrix<SC, LO, GO, NO>(uniqueMap, 4));
    /* for (int i = 0; i < 2; i++) { */
    /*     uniqueMat->insertGlobalValues(i + 2 * myRank, globalColIdx, valTwo); */
    /* } */
    /* uniqueMat->fillComplete(); */
    /* uniqueMat->describe(*out, Teuchos::VERB_EXTREME); */
    /**/
    /* overlappingMat->doImport(*uniqueMat, uniqueToOverlappingImporter, Tpetra::ADD); */
    /* overlappingMat->fillComplete(); */
    /* overlappingMat->describe(*out, Teuchos::VERB_EXTREME); */

    // ----------------------- Test 2: insertGlobalValues() test -----------------------
    // Take care when using insertGlobalValues() on overlapping matrices. Values on nonowned rows are communicated
    // during fillComplete(). Values on owned rows are not communicated, so inserting values into rows that are multiply
    // owned could lead to unexpected results
    /* overlappingMat = rcp(new Tpetra::CrsMatrix<SC, LO, GO, NO>(overlappingMap, 4)); */
    /* if (myRank == 1) { */
    /*     for (int i = 0; i < 4; i++) { */
    /*         overlappingMat->insertGlobalValues(i, colidx, valOne); */
    /*     } */
    /* } */
    /* overlappingMat->fillComplete(); */
    /* overlappingMat->describe(*out, Teuchos::VERB_EXTREME); */

    // ----------------------- Test 3: Export from overlapping to unique distributions -----------------------
    // ADD and INSERT also behave the same. Probably meant to differentiate behaviour of a different operation.
    overlappingMat = rcp(new Tpetra::CrsMatrix<SC, LO, GO, NO>(overlappingMap, columnMap, 4));
    for (int i = 0; i < 3; i++) {
        overlappingMat->insertLocalValues(i, localColIdx, valOne);
    }
    overlappingMat->fillComplete();
    /* overlappingMat->describe(*out, Teuchos::VERB_EXTREME); */

    uniqueMat = rcp(new Tpetra::CrsMatrix<SC, LO, GO, NO>(uniqueMap, columnMap, 4));
    /* for (int i = 0; i < 2; i++) { */
    /*     uniqueMat->insertLocalValues(i, localColIdx, valTwo); */
    /* } */
    // The following code causes undefined behaviour. doExport and doImport can only be called on pristine matrices
    // TODO discuss this
    /* uniqueMat->fillComplete(); */
    /* uniqueMat->describe(*out, Teuchos::VERB_EXTREME); */
    /* uniqueMat->resumeFill(); */
    uniqueMat->doExport(*overlappingMat, overlappingToUniqueExporter, Tpetra::ADD);
    uniqueMat->fillComplete();
    uniqueMat->describe(*out, Teuchos::VERB_EXTREME);

    // -------------- Test 4: Modify values in recieving matrix and then do reverse operation ------------------
    // Use the matrices from Test 3
    // TODO why does this not work? Should be able to insert local values anywhere in local row zero since all columns
    // are owned. replaceLocalValues does work, so changing the map seems to be the problem?
    /* uniqueMat->resumeFill(); */
    /* if (myRank == 0) { */
    /*     uniqueMat->replaceLocalValues(0, localColIdx, valZero); */
    /* } */
    /* uniqueMat->fillComplete(); */
    /* uniqueMat->describe(*out, Teuchos::VERB_EXTREME); */
    /* overlappingMat->resumeFill(); */
    /* overlappingMat->doImport(*uniqueMat, overlappingToUniqueExporter, Tpetra::INSERT); */

    // ----------------------- Test 5: The FEDDLib case of both sides overlapping -----------------------
    // https://docs.trilinos.org/dev/packages/tpetra/doc/html/Tpetra_Lesson05.html it says overlapping to overlapping is
    // not allowed but here it works.
    /* auto overlappingMat1 = rcp(new Tpetra::CrsMatrix<SC, LO, GO, NO>(overlappingMap, 4)); */
    /* for (int i = 0; i < 3; i++) { */
    /*     overlappingMat1->insertGlobalValues(i + myRank, globalColIdx, valOne); */
    /* } */
    /* overlappingMat1->fillComplete(); */
    /* overlappingMat1->describe(*out, Teuchos::VERB_EXTREME); */
    /**/
    /* std::vector<GO> elementList2{0, 1, 2, 3}; */
    /* Teuchos::ArrayView<const GO> elementList2ConstView = Teuchos::arrayViewFromVector(elementList2); */
    /* auto overlappingMap2 = rcp(new Tpetra::Map<LO, GO, NO>(eight, elementList2ConstView, zero, comm)); */
    /* auto overlappingMat2 = rcp(new Tpetra::CrsMatrix<SC, LO, GO, NO>(overlappingMap2, 4)); */
    /* const auto overlappingToOverlappingImporter = Tpetra::Import<LO, GO, NO>(overlappingMap, overlappingMap2); */
    /**/
    /* overlappingMat2->doImport(*overlappingMat1, overlappingToOverlappingImporter, Tpetra::INSERT); */
    /* overlappingMat2->fillComplete(); */
    /* overlappingMat2->describe(*out, Teuchos::VERB_EXTREME); */
}

int main(int argc, char *argv[]) {
    /* xpetraImportExport(argc, argv); */
    /* feddlibImportExport(argc, argv); */
    tpetraImportExport(argc, argv);
    return (EXIT_SUCCESS);
}
