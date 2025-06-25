#include "FFNN.hpp"
#include "Utils.cpp"
#include <stdexcept>

namespace FFNN {

template<typename T>
bool loadBinary(const std::string& filename, std::vector<T>& out) {
    return Utils::readBinary(filename, out);
}

// Explicit instantiation
template bool loadBinary<double>(const std::string&, std::vector<double>&);

} // namespace FFNN
