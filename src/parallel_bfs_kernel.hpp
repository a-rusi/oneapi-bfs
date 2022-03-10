#include <CL/sycl.hpp>

using namespace sycl;

vector<int> parallel_bfs(sycl::queue& q, vector<vector<int>>& graph, int source);