#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl.hpp>

class BFSInPipeID;
class BFSOutPipeID;
class BFSQueuePipeID;
class InputKernelID;

const int N = 124599;


double BFSRun(sycl::queue& q, int* begin_position_ptr, int* adjacency_list_ptr, int* out_ptr, int graph_size, int source) {
    // BFS input and output pipes
    using BFSInPipe =
        sycl::INTEL::pipe<BFSInPipeID, sycl::vec<int, 8>>;
    using BFSOutPipe =
        sycl::INTEL::pipe<BFSOutPipeID, sycl::vec<int, 8>>;
    using BFSQueuePipe =  sycl::INTEL::pipe<BFSQueuePipeID, int, 8>;

    const auto padding_element = -1;


    // input kernel to BFSInPipe
    auto input_kernel_event =
        q.single_task<InputKernelID>([=]() [[intel::kernel_args_restrict]] {
        // read from the input pointer and write it to the sorter's input pipe
        device_ptr<int> begin_position(begin_position_ptr);
        device_ptr<int> adjacency_list(adjacency_list_ptr);
        BFSQueuePipe::write(source);
        int nodes_visited = 1;
        bool visited[N];
        for (int i = 0; i < N; i++) {
            visited[i] = false;
        }
        visited[source] = true;

        while (nodes_visited < graph_size) {
            int node = BFSQueuePipe::read();
            nodes_visited = visited[node] ? nodes_visited : nodes_visited + 1;

            // build the input pipe data
            sycl::vec<int, 8> data;
            int start = begin_position[source];
            int end = begin_position[source + 1];
            #pragma unroll
            for (unsigned char j = 0; j < 8; j++) {
                data[j] = -1;
            //data[j] = start + j < graph_size ? adjacency_list[start + j] : padding_element;
            }

            // write it into the sorter
            BFSInPipe::write(data);
        }
        });

    // output kernel that reads from BFSOutPipe

    // launch BFS

    // wait for input and output kernels to finish

    // wait for BFS kernels to finish

    // free memory

    // return the duration of the sort in milliseconds

    return 0;
}