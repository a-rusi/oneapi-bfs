#include <CL/sycl.hpp>

using namespace sycl;

vector<int> parallel_bfs_pipes(sycl::queue& q, vector<vector<int>>& graph, int source);