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
#include <map>
#include "sequential_bfs.h"


const int N = 60514;

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
    map<int, int> node_id_map;

    vector<edge> edges;
    map<int, int>::iterator it;

    while (file >> n >> m)
    {
        it = node_id_map.find(n);
        if (it != node_id_map.end()) {
            n = node_id_map[n];
        }
        else {
            node_id_map[n] = latest_id;
            n = latest_id;
            latest_id++;
            node_amount++;
        }

        if (node_id_map.find(m) != node_id_map.end()) {
            m = node_id_map[m];

        }
        else {
            node_id_map[m] = latest_id;
            m = latest_id;
            latest_id++;
            node_amount++;
        }
        edges.push_back({n, m});
    }
    file.close();

    std::cout << "Creating graph" << std::endl;
    std::cout << "node amount: " << node_amount << std::endl;

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
        accessor finish_access(finish_vector_buffer, h, read_only);
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
            for (int iter = 0; iter < nodes; iter++) {
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
                if(queue_size == 0) {
                    break;
                }
            }
            for (int i = 0; i < nodes; i++) {
                parents_access[i] = parent[i];
            }
        });
    });
    node_visit_kernel.wait();

    // Report kernel execution time and throughput
    t1_kernel = node_visit_kernel.get_profiling_info<sycl::info::event_profiling::command_start>();
    t2_kernel = node_visit_kernel.get_profiling_info<sycl::info::event_profiling::command_end>();
    time_kernel = (t2_kernel - t1_kernel) / NS;
    std::cout << "Kernel execution time: " << time_kernel << " seconds" << std::endl;

    host_accessor read_parent(parent_buffer, read_only);
    for (int i = 0; i < parent.size(); i++)
    {
        std::cout << read_parent[i] << ",";
    }

    return parent;
}

int main(int argc, char** argv)
{
    std::cout << argv[1] << std::endl;
    vector<vector<int>> graph = create_graph(argv[1]);
    try {
        vector<int> parallel_result = parallel_bfs(graph);
        vector<int> sequential_result = sequential_bfs(graph);

        auto start = high_resolution_clock::now();
        bool invalid_parent = false;
        bool invalid_unvisited = false;
        int invalid_parent_amount = 0;
        int invalid_unvisited_amount = 0;

        for (int i = 0; i < parallel_result.size(); i++)
        {
            int parent = parallel_result[i];
            if (parent != i && parent != -1)
            {
                vector<int> parent_edges = graph[parent];

                if (std::find(parent_edges.begin(), parent_edges.end(), i) == parent_edges.end())
                {
                    invalid_parent = true;
                    invalid_parent_amount += 1;
                }

            }
            if (parent == -1 && sequential_result[i] != -1) {
                invalid_unvisited = true;
                invalid_unvisited_amount += 1;
            }
        }
        if (invalid_parent) {
            std::cout << "Invalid parent found! Amount:" << invalid_parent_amount << std::endl;
        }
        if (invalid_unvisited) {
            std::cout << "Univisted nodes!: Amount: " << invalid_unvisited_amount << std::endl;
        }
    } catch (sycl::exception const &e) {
        // Catches exceptions in the host code
        std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

        std::terminate();
    }
}