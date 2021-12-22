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
#include <string>
#include "sequential_bfs.h"

#define NS (1000000000.0) // number of nanoseconds in a second

struct edge
{
    int n;
    int m;
};

vector<vector<int>> create_graph(string filename)
{

    std::ifstream file;
    file.open(filename);
    int nodes;
    int n;
    int m;

    file >> nodes;
    vector<edge> edges;
    while (file >> n >> m)
    {
        edges.push_back({n, m});
    }
    file.close();

    vector<vector<int>> graph(nodes, vector<int>(0));
    for (int i = 0; i < edges.size(); i++)
    {
        int start = edges.at(i).n;
        int end = edges.at(i).m;
        graph[start].push_back(end);
    }
    std::cout << "graph created" << std::endl;

    return graph;
}

vector<int> parallel_bfs(vector<vector<int>> &graph)

{
    //simple graph init
    int nodes = graph.size();

    // profiling utils
    chrono::high_resolution_clock::time_point t1_host,
        t2_host;
    sycl::cl_ulong t1_kernel, t2_kernel;
    double time_kernel;
    auto my_property_list = property_list{sycl::property::queue::enable_profiling()};

    // create queue
#if defined(FPGA_EMULATOR)
    sycl::INTEL::fpga_emulator_selector device_selector;
#else
    sycl::INTEL::fpga_selector device_selector;
#endif
    sycl::queue q(device_selector, NULL, my_property_list);
    platform platform = q.get_context().get_platform();
    device my_device = q.get_device();
    std::cout << "Platform name: " << platform.get_info<sycl::info::platform::name>().c_str() << std::endl;
    std::cout << "Device name: " << my_device.get_info<sycl::info::device::name>().c_str() << std::endl;
    event queue_event;

    //create parent vector
    vector<int> parent(nodes, -1);

    // start algorithm
    int source = 0;
    parent[source] = source;
    vector<int> frontier(nodes, -1);
    vector<int> next_frontier(nodes, -1);
    frontier[source] = 1;
    vector<int> finish_vector(1, -1);

    buffer graph_buffer(graph);                 // graph buffer, to access in algorithm
    buffer parent_buffer(parent);               // parent buffer, to wait until algorithm is done to print result
    buffer frontier_buffer(frontier);           // frontier buffer, reads its nodes in parallel
    buffer next_frontier_buffer(next_frontier); // next frontier buffer, writes permission needed
    buffer finish_vector_buffer(finish_vector); // vector just to flag if execution should be stopped

    bool all_nodes_visited = false;
    auto nodes_range = range(frontier.size());

    queue_event = q.submit([&](handler &h) {
        accessor graph_access(graph_buffer, h, read_only);
        accessor parents_access(parent_buffer, h);
        accessor frontier_access(frontier_buffer, h);
        accessor next_frontier_access(next_frontier_buffer, h);

        // iterate over the frontier, update node parents

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            bool all_nodes_visited = false;
            for (int iteration = 0; iteration < nodes && !all_nodes_visited; iteration++)
            {
                all_nodes_visited = true;
                for (int i = 0; i < nodes; i++)
                {
                    if (frontier_access[i] != -1)
                    {
                        for (auto &neighbor : graph_access[i])
                        {
                            if (parents_access[neighbor] == -1)
                            {
                                parents_access[neighbor] = i;
                                next_frontier_access[neighbor] = 1;
                            }
                        }
                    }
                }
                for (int i = 0; i < nodes; i++)
                {
                    if (parents_access[i] == -1)
                    {
                        all_nodes_visited = false;
                    }
                    frontier_access[i] = next_frontier_access[i];
                    next_frontier_access[i] = -1;
                }
            }
        });
    });
    q.wait();

    // Report kernel execution time and throughput
    t1_kernel = queue_event.get_profiling_info<sycl::info::event_profiling::command_start>();
    t2_kernel = queue_event.get_profiling_info<sycl::info::event_profiling::command_end>();
    time_kernel = (t2_kernel - t1_kernel) / NS;
    std::cout << "Kernel execution time: " << time_kernel << " seconds" << std::endl;

    host_accessor read_parent(parent_buffer, read_only);
    for (int i = 0; i < parent.size(); i++)
    {
        std::cout << read_parent[i] << ",";
    }

    return parent;
}

int main()
{
    vector<vector<int>> graph = create_graph("artist_edges.csv");
    vector<int> parallel_result = parallel_bfs(graph);
    auto start = high_resolution_clock::now();

    for (int i = 0; i < parallel_result.size(); i++)
    {
        int parent = parallel_result[i];
        if (parent != i && parent != -1)
        {
            vector<int> parent_edges = graph[parent];
            if (std::find(parent_edges.begin(), parent_edges.end(), i) == parent_edges.end())
            {
                std::cout << "Invalid parent found!" << std::endl;
            }
        }
    }
}