#include <bits/stdc++.h>
#include "mcqd.h"
using namespace std;
int main(int argc, char const *argv[])
{
    if(argc != 2)
    {
        return 0;
    }
    FILE *f1 = freopen(argv[1], "r", stdin);
    int n;
    scanf("%d", &n);
    bool ** conn = new bool *[n];
    for(int i = 0;i < n;i++)
    {
        conn[i] = new bool[n];
        for(int j = 0;j < n;j++)
            cin >> conn[i][j];
    }
    fclose(f1);
    f1 = freopen(argv[1], "w", stdout);
    int *qmax;
    int qsize;
    Maxclique m(conn, n);
    m.mcqdyn(qmax, qsize);
    fclose(f1);
    return qsize;
}