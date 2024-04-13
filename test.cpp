#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
    std::cout << "this is going to be a crasy long line of code that should be "
                 "formated by "
              << std::endl;
  std::vector<double> vecd{1.2,2.3,3.4,4.5,5.6};
  for (auto entry : vecd) {
    std::cout << entry << ", ";
  }
  std::cout << std::endl;
    return 0;
}
