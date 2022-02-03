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
#include <unordered_map>
#include "sequential_bfs.h"


const int N = 50514;

struct edge
{
    int start;
    int end;
};

#define NS (1000000000.0) // number of nanoseconds in a second

//using node_visit_pipe = pipe<class visit_pipe, int, 100>;
using node_paint_pipe = pipe<class paint_pipe, tuple<int,int>, 100>;


vector<vector<int>> create_graph(string filename)
{

    std::ifstream file;
    file.open(filename);
    int n;
    int m;
    int latest_id = 0;
    int node_amount = 0;
    unordered_map<int, int> node_id_map;

    vector<edge> edges;
    while (file >> n >> m)
    {
        if (node_id_map.find(n) != node_id_map.end()) {
            n = node_id_map[n];
        }
        else {
            n = latest_id;
            node_id_map[n] = n;
            latest_id++;
            node_amount++;
        }

        if (node_id_map.find(m) != node_id_map.end()) {
            m = node_id_map[m];
        }
        else {
            m = latest_id;
            node_id_map[m] = m;
            latest_id++;
            node_amount++;
        }
        edges.push_back({n, m});
    }
    file.close();

    vector<vector<int>> graph(node_amount, vector<int>(0));
    for (int i = 0; i < edges.size(); i++)
    {
        int start = edges.at(i).start;
        int end = edges.at(i).end;
        graph[start].push_back(end);
    }

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

    auto nodes_range = range(frontier.size());

    event node_visit_kernel;
    event node_paint_kernel;
    event bfs_kernel;

    std::cout << "Starting BFS" << std::endl;

    node_visit_kernel = q.submit([&](handler &h) {
        accessor graph_access(graph_buffer, h, read_only);
        accessor frontier_access(frontier_buffer, h);
        accessor next_frontier_access(next_frontier_buffer, h);
        accessor finish_access(finish_vector_buffer, h, read_only);
        accessor parents_access(parent_buffer, h);

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // create and initialize on-chip structures
            int queue[N];
            int new_queue[N];
            tuple<int,int> neighbors[N/2];
            bool visited[N];

            for (int i = 0; i < N; i++) {
                queue[i] = -1;
                new_queue[i] = -1;
                visited[i] = false;
            }
            queue[0] = source;
            
            int nodes_visited = 1;
            int queue_size = 1;
            int new_queue_size = 0;
            do {
                int neighbors_amount = 0;
                // read neighbors
                for (int i = 0; i < queue_size; i++) {
                    int origin = queue[i];
                    for (int neighbor : graph_access[origin]) {
                        if (!visited[neighbor]){
                            visited[neighbor] = true;
                            neighbors[neighbors_amount] = make_tuple(origin,neighbor);
                            neighbors_amount += 1;
                        }
                    }
                }
                // process neighbors
                for (int i = 0; i < neighbors_amount; i++) {
                    tuple<int,int> e = neighbors[i];
                    int start = get<0>(e);
                    int end = get<1>(e);
                    parents_access[end] = start;
                    new_queue[new_queue_size] = end;
                    new_queue_size += 1;
                }
                //clean slate
                queue_size = new_queue_size;
                new_queue_size = 0;
                for (int i = 0; i < queue_size; i++) {
                    queue[i] = new_queue[i];
                    new_queue[i] = -1;
                }
            } while(queue_size != 0);
        });
    });

    /*bfs_kernel = q.submit([&](handler &h) {
        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // create queue in device
            int nodes_visited = 1;
            int queue_size = 1;
            vector<int> queue(nodes, -1);
            queue[0] = source;
            do {
                nodes_visited += 1;
            } while(nodes_visited < nodes);
        });
    });*/

    /*node_visit_kernel = q.submit([&](handler &h) {
        accessor graph_access(graph_buffer, h, read_only);
        accessor frontier_access(frontier_buffer, h);
        accessor next_frontier_access(next_frontier_buffer, h);
        accessor finish_access(finish_vector_buffer, h, read_only);

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            int k = 0;
            do
            {
                for (int i = 0; i < nodes; i++) {
                    if (frontier_access[i] != -1) {
                        for (auto &neighbor : graph_access[i]) {
                            node_paint_pipe::write(make_tuple(i, neighbor));
                            next_frontier_access[neighbor] = 1;
                        }
                    }
                }
                for (int i = 0; i < nodes; i++) {
                    frontier_access[i] = next_frontier_access[i];
                    next_frontier_access[i] = -1;
                }
                k++;
            } while (k < 10);
        });
    });

    node_paint_kernel = q.submit([&](handler &h) {
        accessor parents_access(parent_buffer, h);
        accessor finish_access(finish_vector_buffer, h);
        // podria crear vector de parents aca, settear source y listo
        // uso este para leer y el otro para escribir

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            int nodes_visited = 1;
            do
            {
                tuple<int,int> e = node_paint_pipe::read();
                int start = get<0>(e);
                int end = get<1>(e);
                if (parents_access[end] != -1) {
                    parents_access[end] = start;
                    nodes_visited += 1;
                }
            } while (nodes_visited < nodes);
            finish_access[0] = 1;
        });
    });*/

    q.wait();

    // Report kernel execution time and throughput
    /*t1_kernel = node_visit_kernel.get_profiling_info<sycl::info::event_profiling::command_start>();
    t2_kernel = node_visit_kernel.get_profiling_info<sycl::info::event_profiling::command_end>();
    time_kernel = (t2_kernel - t1_kernel) / NS;
    std::cout << "Kernel execution time: " << time_kernel << " seconds" << std::endl;*/

    host_accessor read_parent(parent_buffer, read_only);
    for (int i = 0; i < parent.size(); i++)
    {
        std::cout << read_parent[i] << ",";
    }

    return parent;
}

int main()
{
    vector<vector<int>> graph = create_graph("./src/facebook_combined.txt");
    try {
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
    } catch (sycl::exception const &e) {
        // Catches exceptions in the host code
        std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

        std::terminate();
    }
}