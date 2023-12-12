#ifndef FEDDUTILS_hpp
#define FEDDUTILS_hpp

#include "feddlib/core/FEDDCore.hpp"
#include <Xpetra_CrsGraphFactory.hpp>
#include <Xpetra_Map.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <fstream>
#include <vector>


#ifndef FEDD_TIMER_START
#define FEDD_TIMER_START(A,S) Teuchos::RCP<Teuchos::TimeMonitor> A = Teuchos::rcp(new Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer(std::string("FEDD") + std::string(S))));
#endif

#ifndef FEDD_TIMER_STOP
#define FEDD_TIMER_STOP(A) A.reset();
#endif


namespace FEDD{

#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
    
template<class ForwardIt, class GO>
ForwardIt uniqueWithCombines(ForwardIt first, ForwardIt last, std::vector<std::vector<GO> >& combines)
{
    
    if (first == last)
        return last;
    
    ForwardIt firstForDistance = first;
    ForwardIt result = first;
    combines[ distance( firstForDistance, result ) ].push_back( (GO) distance( firstForDistance, first ) );
    while (++first != last) {
        if (!(*result == *first) && ++result != first) {
            *result = std::move(*first);
            // also add the element which is the final unique element (the first in the sorted list)
            combines[ distance( firstForDistance, result ) ].push_back( (GO) distance( firstForDistance, first ) );
        }
        else{
            combines[ distance( firstForDistance, result ) ].push_back( (GO) distance( firstForDistance, first ) );
        }
    }
    return ++result;
};

template <typename T>
std::vector<T> sort_from_ref(
                        std::vector<T> const& in,
                        std::vector<int> const& reference
                        ) {
    std::vector<T> ret(in.size());
    
    int const size = in.size();
    for (int i = 0; i < size; ++i)
        ret[i] = in[reference[i]];
    
    return ret;
};

template <typename T>
std::vector<T> sort_from_ref(
                             std::vector<T> const& in,
                             std::vector<long long> const& reference
                             ) {
    std::vector<T> ret(in.size());
    
    int const size = in.size();
    for (long long i = 0; i < size; ++i)
        ret[i] = in[reference[i]];
    
    return ret;
};

    
template <typename T>
void sort2byFirst( std::vector<std::vector<T> >& in, std::vector<T>& in2 )
{

    std::vector<int> index(in.size(), 0);
    for (int i = 0 ; i != index.size() ; i++)
        index[i] = i;
    
    std::sort(index.begin(), index.end(),
              [&](const int& a, const int& b) {
                  return  in[a] < in[b];
              }
              );
    in = sort_from_ref( in, index );
    in2 = sort_from_ref( in2, index );
        
}
    
template <typename T, class GO>
void make_unique( std::vector<std::vector<T> >& in, vec2D_GO_Type& combinedElements, std::vector<GO>& globaIDs )
{
    {
        std::vector<int> index(in.size(), 0);
        for (int i = 0 ; i != index.size() ; i++)
            index[i] = i;
        
        std::sort(index.begin(), index.end(),
             [&](const int& a, const int& b) {
                 return  in[a] < in[b];
             }
             );
        in = sort_from_ref( in, index );
        globaIDs = sort_from_ref( globaIDs, index );
    }
    {
        std::vector<int> index(in.size(), 0);
        for (int i = 0 ; i != index.size() ; i++)
            index[i] = i;
        
        combinedElements.resize( in.size() );
        
        auto it = uniqueWithCombines( in.begin(), in.end(), combinedElements );
        
        in.resize( distance( in.begin(), it ) );
        combinedElements.resize( in.size() );
    }
};
    
template <typename T>
void make_unique( std::vector<std::vector<T> >& in )
{
    {
        std::vector<int> index(in.size(), 0);
        for (int i = 0 ; i != index.size() ; i++)
            index[i] = i;

        std::sort(index.begin(), index.end(),
             [&](const int& a, const int& b) {
                 return  in[a] < in[b];
             }
             );
        in = sort_from_ref( in, index );
    }
    {

        auto it = std::unique( in.begin(), in.end() );

        in.resize( distance( in.begin(), it ) );
    }
};

template <typename T>
void make_unique( std::vector<std::vector<T> >& in, vec2D_GO_Type& combinedElements )
{
    {
        std::vector<int> index(in.size(), 0);
        for (int i = 0 ; i != index.size() ; i++)
            index[i] = i;
        
        std::sort(index.begin(), index.end(),
             [&](const int& a, const int& b) {
                 return  in[a] < in[b];
             }
             );
        in = sort_from_ref( in, index );
        
    }
    {
        std::vector<int> index(in.size(), 0);
        for (int i = 0 ; i != index.size() ; i++)
            index[i] = i;
        
        combinedElements.resize( in.size() );
        
        auto it = uniqueWithCombines( in.begin(), in.end(), combinedElements );
        
        in.resize( distance( in.begin(), it ) );
        
        combinedElements.resize( in.size() );
    }
};
    
template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());
    
    std::vector<T> result;
    result.reserve(a.size());
    
    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
};
    
template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());
    
    std::vector<T> result;
    result.reserve(a.size());
    
    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::minus<T>());
    return result;
};
    
template <typename T>
void make_unique( std::vector<T>& in )
{
    std::sort( in.begin(), in.end() );
    auto it = unique( in.begin(), in.end() );
    in.erase( it, in.end() );
};

// ################# Nonlinear Schwarz related functions
template <class LO, class GO, class NO>
int ExtendOverlapByOneLayer(Teuchos::RCP<const Xpetra::CrsGraph<LO, GO, NO>> inputGraph,
                            Teuchos::RCP<const Xpetra::CrsGraph<LO, GO, NO>> &outputGraph) {
    // In the adjacency matrix of the graph, connectivity of node i is given by row i. The column map of row i
    // corresponds to the connectivity and by assigning ownership of all rows referenced by the column map to the rank,
    // the local subdomain is extended by one layer of connectivity
    Teuchos::RCP<Xpetra::CrsGraph<LO, GO, NO>> tmpGraph =
        Xpetra::CrsGraphFactory<LO, GO, NO>::Build(inputGraph->getColMap(), inputGraph->getGlobalMaxNumRowEntries());
    Teuchos::RCP<Xpetra::Import<LO, GO, NO>> scatter =
        Xpetra::ImportFactory<LO, GO, NO>::Build(inputGraph->getRowMap(), inputGraph->getColMap());
    tmpGraph->doImport(*inputGraph, *scatter, Xpetra::ADD);
    tmpGraph->fillComplete(inputGraph->getDomainMap(), inputGraph->getRangeMap());

    outputGraph = tmpGraph.getConst();
    return 0;
}

template <typename T> void waitForGdbAttach() {
    volatile T i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    auto fs = std::fstream{"pid_for_debugger.txt", std::ios::out | std::ios::trunc};
    fs << getpid();
    fs.close();
    fflush(stdout);
    while (0 == i)
            sleep(5);
}

inline void logGreen(const std::string &s, const Teuchos::RCP<const Teuchos::Comm<int>> &comm) {
    if (comm->getRank() == 0) {
            std::cout << GREEN << '\n' << "==> " << s << '\n' << RESET;
    }
}

inline void logSimple(const std::string &s, const Teuchos::RCP<const Teuchos::Comm<int>> &comm) {
    if (comm->getRank() == 0) {
            std::cout << "==> " << s << '\n';
    }
}

template <typename T> void logVec(const std::vector<T> vec, const Teuchos::RCP<const Teuchos::Comm<int>> &comm) {
    if (comm->getRank() == 0) {
        std::cout << "[ ";
        for (auto it = vec.begin(); it != vec.end() - 1; it++) {
            std::cout << *it << ", ";
        }
        std::cout << vec.back() << "]" << std::endl;
    }
}
}
#endif
