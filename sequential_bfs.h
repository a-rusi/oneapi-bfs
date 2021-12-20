
using namespace std;
#include <vector>
#include <queue>
#include <iostream>

vector<int> sequential_bfs(vector<vector<int>> &graph)
{
    //simple graph init
    int nodes = graph.size();

    //create parent vector
    vector<int> parent(nodes, -1);

    // start algorithm
    int source = 0;
    parent[source] = source;

    std::queue<int> queue;
    queue.push(source);
    while (!queue.empty())
    {
        int node = queue.front();
        queue.pop();
        for (auto &neighbor : graph[node])
        {
            if (parent[neighbor] < 0)
            {
                parent[neighbor] = node;
                queue.push(neighbor);
            }
        }
    }

    return parent;
}