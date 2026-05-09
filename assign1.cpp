#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void dfs_util(int v, vector<int>& visited) {
        visited[v] = 1;
        for (int u : adj[v]) {
            if (!visited[u]) {
                dfs_util(u, visited);
            }
        }
    }

    void dfs(int start) {
        vector<int> visited(V, 0);
        dfs_util(start, visited);
    }

    void parallel_dfs_util(int v, vector<int>& visited) {
        for (int u : adj[v]) {
            
            if (!visited[u]) {
                bool spawn_task = false;
                
             
                #pragma omp critical
                {
                    if (!visited[u]) {
                        visited[u] = 1;
                        spawn_task = true;
                    }
                }
                
               
                if (spawn_task) {
                    #pragma omp task shared(visited)
                    parallel_dfs_util(u, visited);
                }
            }
        }
    }

    void parallel_dfs(int start) {
        vector<int> visited(V, 0);
        visited[start] = 1; 
        
        #pragma omp parallel
        {
            #pragma omp single 
            {
                #pragma omp task shared(visited)
                parallel_dfs_util(start, visited);
            }
        }
    }

    void bfs(int start) {
        vector<int> visited(V, 0);
        queue<int> q;

        visited[start] = 1;
        q.push(start);

        while (!q.empty()) {
            int node = q.front();
            q.pop();

            for (int u : adj[node]) {
                if (!visited[u]) {
                    visited[u] = 1;
                    q.push(u);
                }
            }
        }
    }

    void parallel_bfs(int start) {
        vector<int> visited(V, 0);
        vector<int> current;

        visited[start] = 1;
        current.push_back(start);

        while (!current.empty()) {
            vector<int> next;

            #pragma omp parallel 
            {
                vector<int> local_next;
                
                #pragma omp for 
                for (int i = 0; i < current.size(); i++) {
                    int node = current[i];

                    for (int u : adj[node]) {
                        if (!visited[u]) {
                            bool push_queue = false;
                            
                            #pragma omp critical
                            {
                                if (!visited[u]) {
                                    visited[u] = 1;
                                    push_queue = true;
                                }
                            }
                            
                            if (push_queue) {
                                local_next.push_back(u);
                            }
                        }
                    }
                }
                
                #pragma omp critical
                {
                    next.insert(next.end(), local_next.begin(), local_next.end());
                }
            }
            current = next;
        }
    }
};



int main() {
   
    int V = 30000; 
    Graph g(V);

    cout << "Generating dense graph..." << endl;
    srand(42); 
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < 30; j++) {
            int random_node = rand() % V;
            if (i != random_node) {
                g.addEdge(i, random_node);
            }
        }
    }
    cout << "Graph generated. Starting traversal..." << endl << endl;

    double t1, t2;

    t1 = omp_get_wtime();
    g.dfs(0);
    t2 = omp_get_wtime();
    cout << "Sequential DFS Time: " << fixed << t2 - t1 << " sec\n";

    t1 = omp_get_wtime();
    g.parallel_dfs(0);
    t2 = omp_get_wtime();
    cout << "Parallel DFS Time:   " << fixed << t2 - t1 << " sec\n";

    t1 = omp_get_wtime();
    g.bfs(0);
    t2 = omp_get_wtime();
    cout << "Sequential BFS Time: " << fixed << t2 - t1 << " sec\n";

    t1 = omp_get_wtime();
    g.parallel_bfs(0);
    t2 = omp_get_wtime();
    cout << "Parallel BFS Time:   " << fixed << t2 - t1 << " sec\n";

    return 0;
}