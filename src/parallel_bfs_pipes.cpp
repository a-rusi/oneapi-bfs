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

#include "parallel_bfs_pipes.hpp"

struct parent_child_nodes
{
    int parent;
    int child;
};

using node_pipe = pipe<class visit_pipe, parent_child_nodes, 8>;

const int N = 124599;

#define NS (1000000000.0) // number of nanoseconds in a second

class BFSKernel;

vector<int> parallel_bfs_pipes(sycl::queue& q, vector<vector<int>>& graph, int source)

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

    event visit_kernel;
    event compute_kernel;

    std::cout << "Starting BFS" << std::endl;
    queue q1;
    queue q2;
    visit_kernel = q.submit([&](handler &h) {
        accessor graph_access(graph_buffer, h, read_only);
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            bool frontier[N];
            bool new_frontier[N];
            bool visited[N];

            for (int i = 0; i < nodes; i++) {
                frontier[i] = false;
                new_frontier[i] = false;
                visited[i] = false;
            }
            frontier[source] = true;
            visited[source] = true;
            int nodes_processed = 1;

            do {
                // read neighbors
                for (int i = 0; i < nodes; i++) {
                    if (frontier[i]) {
                        for (int neighbor : graph_access[i]) {
                            node_pipe::write({i, neighbor});
                            new_frontier[neighbor] = true;
                        }
                    }
                }

                for (int i = 0; i < nodes; i++) {
                    frontier[i] = false;
                    if (new_frontier[i] && !visited[i]) {
                        visited[i] = true;
                        frontier[i] = new_frontier[i];
                        nodes_processed++;
                        new_frontier[i] = false;
                    }
                }
            } while (nodes_processed < nodes);
        });
    });

    compute_kernel = q.submit([&](handler &h) {
        accessor parents_access(parent_buffer, h);

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            bool visited[N];
            for (int i = 0; i < N; i++) {
                visited[i] = false;
            }
            int nodes_computed = 1;

            do {
                parent_child_nodes p = node_pipe::read();
                if (parents_access[p.child] == -1) {
                    parents_access[p.child] = p.parent;
                    nodes_computed++;
                }
            } while(nodes_computed < nodes);

        });
    });
    q.wait();

    // Report kernel execution time and throughput
    t1_kernel = visit_kernel.get_profiling_info<sycl::info::event_profiling::command_start>();
    t2_kernel = visit_kernel.get_profiling_info<sycl::info::event_profiling::command_end>();
    time_kernel = (t2_kernel - t1_kernel) / NS;
    std::cout << "Kernel execution time: " << time_kernel << " seconds" << std::endl;

    host_accessor read_parent(parent_buffer, read_only);
    for (int i = 0; i < parent.size(); i++)
    {
        std::cout << read_parent[i] << ",";
    }

    return parent;
}