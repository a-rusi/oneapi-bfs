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
        stream out(1024, 256, h);

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            out << "Starting\n";
            // create and initialize on-chip structures
            int queue[N];
            int parent[N];
            bool visited[N];
            bool currently_visited[N];
            int possible_parent[N];

            for (int i = 0; i < N; i++) {
                queue[i] = -1;
                parent[i] = -1;
                visited[i] = false;
                currently_visited[i] = false;
                possible_parent[i] = -1;
            }
            out << "Finished Creating structs\n";
            queue[0] = source;
            parent[source] = source;
            visited[source] = true;
            
            int queue_size = 1;
            for (int iter = 0; iter < nodes && queue_size != 0; iter++) {
                // read neighbors
                for (int i = 0; i < queue_size; i++) {
                    int origin = queue[i];
                    for (int neighbor : graph_access[origin]) {
                        out << "Visiting: " << neighbor << "\n";
                        currently_visited[neighbor] = true;
                        possible_parent[neighbor] = origin;
                    }
                }
                out << "Finished reading neighbors\n";
                //clean slate
                queue_size = 0;
                for (int i = 0; i < nodes; i++) {
                    if (currently_visited[i] && !visited[i]) {
                        out << "Reading about: " << i << "\n";
                        visited[i] = true;
                        parent[i] = possible_parent[i];
                        queue[queue_size] = i;
                        queue_size++;
                    }
                }
                out << "New queue size is: " << queue_size << "\n";
            }
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