#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include <string>
#include <map>

using namespace sycl;
using namespace std;
using namespace chrono;


#include "sequential_bfs.h"
#include "parallel_bfs_frontier.hpp"
#include "csr.cpp"
#include "producer.cpp"




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

    // Create CSR for optimal storage
    CSR csr;
    csr.from_graph(graph, graph.size());

    vector<int> begin_position_csr = csr.begin_position;
    vector<int> adjacency_list_csr = csr.adjacency_list;

    // Try to run kernel code
    try {

#if defined(FPGA_EMULATOR)
  sycl::INTEL::fpga_emulator_selector device_selector;
#else
  sycl::INTEL::fpga_selector device_selector;
#endif
        auto my_property_list = property_list{sycl::property::queue::enable_profiling()};
        sycl::queue q(device_selector, NULL, my_property_list);

        // make sure the device supports USM device allocations
        auto d = q.get_device();
        if (!d.get_info<info::device::usm_device_allocations>()) {
            std::cerr << "ERROR: The selected device does not support USM device"
                    << " allocations\n";
            std::terminate();
        }

        // make sure the device support USM host allocations if we chose to use them
        if (!d.get_info<info::device::usm_host_allocations>()) {
            std::cerr << "ERROR: The selected device does not support USM host"
                    << " allocations\n";
            std::terminate();
        }

        platform platform = q.get_context().get_platform();
        device my_device = q.get_device();
        std::cout << "Platform name: " << platform.get_info<sycl::info::platform::name>().c_str() << std::endl;
        std::cout << "Device name: " << my_device.get_info<sycl::info::device::name>().c_str() << std::endl;

        std::cout << "Creating host structures" << std::endl;

        int * begin_position;
        int * adjacency_list;
        int * output;
        vector<int> output_host(graph.size(), -1);


        // using USM host allocations
        if ((begin_position = malloc_device<int>(begin_position_csr.size(), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'begin_position' using "
                    << "malloc_host\n";
        std::terminate();
        }
        if ((adjacency_list = malloc_device<int>(adjacency_list_csr.size(), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'adjacency_list' using "
                    << "malloc_host\n";
        std::terminate();
        }
        if ((output = malloc_device<int>(graph.size(), q)) == nullptr) {
        std::cerr << "ERROR: could not allocate space for 'output' using "
                    << "malloc_host\n";
        std::terminate();
        }

        q.memcpy(begin_position, begin_position_csr.data(), begin_position_csr.size() * sizeof(int)).wait();
        q.memcpy(adjacency_list, adjacency_list_csr.data(), adjacency_list_csr.size() * sizeof(int)).wait();
        q.memcpy(output, output_host.data(), output_host.size() * sizeof(int)).wait();

        for (int i = 0; i < begin_position_csr.size(); i++) {
            std::cout << begin_position[i] << std::endl;
        }

        std::cout << "Copied structures to device" << std::endl;

        BFSRun(q, begin_position, adjacency_list, output, graph.size(), 0);

        sycl::free(begin_position, q);
        sycl::free(adjacency_list, q);
        sycl::free(output, q);


       /* vector<int> parallel_result = parallel_bfs_frontier(q, graph, 0);

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
        }*/
    } catch (sycl::exception const &e) {
        std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
        std::terminate();
    }
}