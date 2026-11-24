#include <cmath>
#include <vector>
extern "C" double hyp(double x, double y) { std::vector<double> v{x, y}; return std::sqrt(v[0] * v[0] + v[1] * v[1]); }
