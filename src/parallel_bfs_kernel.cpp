#include <chrono>
#include <CL/sycl.hpp>
#include <vector>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

using namespace sycl;
using namespace std;
using namespace chrono;

#include "dpc_common.hpp"
#include <iostream>
#include <fstream>

#include "parallel_bfs_kernel.hpp"


const int N = 124599;

#define NS (1000000000.0) // number of nanoseconds in a second

class BFSKernel;

vector<int> parallel_bfs(sycl::queue& q, vector<vector<int>>& graph, int source)

{
    //simple graph init
    int nodes = graph.size();

    // profiling utils
    chrono::high_resolution_clock::time_point t1_host,
        t2_host;
    sycl::cl_ulong t1_kernel, t2_kernel;
    double time_kernel;

    //create parent vector
    vector<int> parent(nodes, -1);

    parent[source] = source;

    buffer graph_buffer(graph);                 // graph buffer, to access in algorithm
    buffer parent_buffer(parent);               // parent buffer, to wait until algorithm is done to print result

    event bfs_kernel;

    std::cout << "Starting BFS" << std::endl;
    bfs_kernel = q.submit([&](handler &h) {
        accessor graph_access(graph_buffer, h, read_only);
        accessor parents_access(parent_buffer, h);

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // create and initialize on-chip structures
            int parent[N];
            bool frontier[N];
            bool new_frontier[N];
            bool visited[N];

            for (int i = 0; i < N; i++) {
                visited[i] = false;
                frontier[i] = false;
                new_frontier[i] = false;
                parent[i] = -1;
            }
            frontier[source] = true;
            parent[source] = source;
            visited[source] = true;
            
            bool left_to_visit = false;
            for (int iter = 0; iter < nodes; iter++){
                // read neighbors
                int neighbors_amount = 0;
                left_to_visit = false;
                for (int i = 0; i < nodes; i++) {
                    if (frontier[i]) {
                        for (int neighbor : graph_access[i]) {
                            if (!visited[neighbor]) {
                                visited[neighbor] = true;
                                new_frontier[neighbor] = true;
                                parent[neighbor] = i;
                                left_to_visit = true;
                            }
                        }
                    }
                }

                //clean slate
                for (int i = 0; i < nodes; i++) {
                    frontier[i] = new_frontier[i];
                    new_frontier[i] = false;
                }
                if (!left_to_visit) {
                    break;
                }
            };

            for (int i = 0; i < nodes; i++) {
                parents_access[i] = parent[i];
            }
        });
    });
    bfs_kernel.wait();

    q.wait();

    // Report kernel execution time and throughput
    t1_kernel = bfs_kernel.get_profiling_info<sycl::info::event_profiling::command_start>();
    t2_kernel = bfs_kernel.get_profiling_info<sycl::info::event_profiling::command_end>();
    time_kernel = (t2_kernel - t1_kernel) / NS;
    std::cout << "Kernel execution time: " << time_kernel << " seconds" << std::endl;

    host_accessor read_parent(parent_buffer, read_only);
    for (int i = 0; i < parent.size(); i++)
    {
        std::cout << read_parent[i] << ",";
    }

    return parent;
}