class CSR {
    public: 
        void from_graph(vector<vector<int>>& graph, int nodes) {
            begin_position.reserve(nodes);
            begin_position.push_back(0);
            int total = 0;
            for (int i = 1; i < graph.size(); i++) {
                total += graph[i-1].size();
                begin_position.push_back(total);
                for (int j = 0; j < graph[i].size(); j++) {
                    adjacency_list.push_back(graph[i][j]);
                }
            }
        }
        vector<int> begin_position;
        vector<int> adjacency_list;
};