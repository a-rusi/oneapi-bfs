#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include <string>
#include <map>

using namespace sycl;
using namespace std;
using namespace chrono;


#include "sequential_bfs.h"
#include "parallel_bfs_kernel.hpp"
#include "parallel_bfs_pipes.hpp"
#include "csr.cpp"



struct edge
{
    int start;
    int end;
};

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
        edges.push_back({m, n});
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

int main(int argc, char** argv)
{
    // Create graph from input filename
    std::cout << argv[1] << std::endl;
    vector<vector<int>> graph = create_graph(argv[1]);
    CSR csr;
    csr.from_graph(graph, graph.size());

    vector<int> begin_position = csr.begin_position;
    for (int i = 0; i < begin_position.size(); i++) {
        std::cout << begin_position[i] << std::endl;
    }

    // Try to run kernel code
    try {

#if defined(FPGA_EMULATOR)
  sycl::INTEL::fpga_emulator_selector device_selector;
#else
  sycl::INTEL::fpga_selector device_selector;
#endif
        auto my_property_list = property_list{sycl::property::queue::enable_profiling()};
        sycl::queue q(device_selector, NULL, my_property_list);

        platform platform = q.get_context().get_platform();
        device my_device = q.get_device();
        std::cout << "Platform name: " << platform.get_info<sycl::info::platform::name>().c_str() << std::endl;
        std::cout << "Device name: " << my_device.get_info<sycl::info::device::name>().c_str() << std::endl;


        vector<int> parallel_result = parallel_bfs_pipes(q, graph, 0);

        bool invalid_parent = false;
        int invalid_parent_amount = 0;

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
        }
        if (invalid_parent) {
            std::cout << "Invalid parent found! Amount:" << invalid_parent_amount << std::endl;
        }
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
        std::terminate();
    }
}