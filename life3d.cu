/*-----------------------------------------------
 * 请在此处填写你的个人信息
 * 学号: SA24229109
 * 姓名: 于慧木
 * 邮箱: curealb@gmail.com
 ------------------------------------------------*/

 #include <chrono>
 #include <thread>
 #include <cstring>
 #include <fstream>
 #include <iostream>
 #include <string>
 #include <cuda_runtime.h>
 #include <device_launch_parameters.h>
 #include <omp.h>
 #include "helper_cuda.h"
 
 #define AT(x, y, z) universe[(x) * N * N + (y) * N + z]
 #define BLOCK_SIZE 4
 
 using std::cin, std::cout, std::endl;
 using std::ifstream, std::ofstream;
 
 // 检查是否算错
 bool check_universe(char* a,char* b,int N){
 
     for (int i = 0;i < N * N * N;i++) {
         if (a[i] != b[i])return false;
     }
 
     return true;
 }
 __global__ void print_test(){
     printf("Hello world from block %d, thread %d\n", blockIdx.x, threadIdx.x);
 }
 
 // 存活细胞数
 int population(int N, char *universe)
 {
     int result = 0;
     for (int i = 0; i < N * N * N; i++)
         result += universe[i];
     return result;
 }
 
 // 打印世界状态
 void print_universe(int N, char *universe)
 {
     // 仅在N较小(<= 32)时用于Debug
     if (N > 32)
         return;
     for (int x = 0; x < N; x++)
     {
         for (int y = 0; y < N; y++)
         {
             for (int z = 0; z < N; z++)
             {
                 if (AT(x, y, z))
                     cout << "O ";
                 else
                     cout << "* ";
             }
             cout << endl;
         }
         cout << endl;
     }
     cout << "population: " << population(N, universe) << endl;
 }
 
 __global__ void update_kernel(char* universe, char* next, int N) {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     int z = blockIdx.z * blockDim.z + threadIdx.z;
     printf("%d %d %d\t", x,y,z);
     // if (x >= N || y >= N || z >= N)return;
 
     // // 计算邻居的存活个数
     // int alive = 0;
     // for (int dx = -1;dx <= 1;dx++) {
     //     for (int dy = -1;dy <= 1;dy++) {
     //         for (int dz = -1;dz <= 1;dz++) {
     //             if (dx == 0 && dy == 0 && dz == 0)continue;
     //             int nx = (x + dx + N) % N;
     //             int ny = (y + dy + N) % N;
     //             int nz = (z + dz + N) % N;
     //             alive += AT(nx, ny, nz);
     //         }
     //     }
     // }
     
     // // 判断更新方式
     // if (AT(x, y, z) && (alive < 5 || alive > 7)) {
     //     next[x * N * N + y * N + z] = 0;
     // }
     // else if ((!AT(x, y, z) && alive == 6)) {
     //     next[x * N * N + y * N + z] = 1;
     // }
     // else {
     //     next[x * N * N + y * N + z] = AT(x, y, z);
     // }
 }
 
 void life3d_run_cuda(int N, char* universe, int T) {
     size_t size = N * N * N * sizeof(char);
 
     char* device_universe, * device_next;
 
     cudaMalloc((void**)&device_universe, size);
     cudaMalloc((void**)&device_next, size);
 
     cudaMemcpy(device_universe, universe, size, cudaMemcpyHostToDevice);
 
     dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
     dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y, (N + block.z - 1) / block.z);
 
     for (int t = 0;t < T;t++) {
         
         update_kernel << <grid,block >> > (device_universe, device_next, N);
         cudaDeviceSynchronize();
 
 
         // 交换指向地址
         char* temp = device_universe;
         device_universe = device_next;
         device_next = temp;
     }
 
     cudaMemcpy(universe, device_universe, size, cudaMemcpyDeviceToHost);
 
     cudaFree(device_universe);
     cudaFree(device_next);
 }
 
 // 核心计算代码，将世界向前推进T个时刻
 void life3d_run(int N, char *universe, int T)
 {
     char *next = (char *)malloc(N * N * N);
     for (int t = 0; t < T; t++)
     {
         // outerloop: iter universe
         for (int x = 0; x < N; x++)
             for (int y = 0; y < N; y++)
                 for (int z = 0; z < N; z++)
                 {
                     // inner loop: stencil
                     int alive = 0;
                     for (int dx = -1; dx <= 1; dx++)
                         for (int dy = -1; dy <= 1; dy++)
                             for (int dz = -1; dz <= 1; dz++)
                             {
                                 if (dx == 0 && dy == 0 && dz == 0)
                                     continue;
                                 int nx = (x + dx + N) % N;
                                 int ny = (y + dy + N) % N;
                                 int nz = (z + dz + N) % N;
                                 alive += AT(nx, ny, nz);
                             }
                     if (AT(x, y, z) && (alive < 5 || alive > 7))
                         next[x * N * N + y * N + z] = 0;
                     else if (!AT(x, y, z) && alive == 6)
                         next[x * N * N + y * N + z] = 1;
                     else
                         next[x * N * N + y * N + z] = AT(x, y, z);
                 }
         memcpy(universe, next, N * N * N);
     }
     free(next); 
 }
 
 // 读取输入文件
 void read_file(char *input_file, char *buffer)
 {
     ifstream file(input_file, std::ios::binary | std::ios::ate);
     if (!file.is_open())
     {
         cout << "Error: Could not open file " << input_file << std::endl;
         exit(1);
     }
     std::streamsize file_size = file.tellg();
     file.seekg(0, std::ios::beg);
     if (!file.read(buffer, file_size))
     {
         std::cerr << "Error: Could not read file " << input_file << std::endl;
         exit(1);
     }
     file.close();
 }
 
 // 写入输出文件
 void write_file(char *output_file, char *buffer, int N)
 {
     ofstream file(output_file, std::ios::binary | std::ios::trunc);
     if (!file)
     {
         cout << "Error: Could not open file " << output_file << std::endl;
         exit(1);
     }
     file.write(buffer, N * N * N);
     file.close();
 }
 
 int main(int argc, char **argv)
 {
     // cmd args
     if (argc < 5)
     {
         cout << "usage: ./life3d N T input output" << endl;
         return 1;
     }
     int N = std::stoi(argv[1]);
     int T = std::stoi(argv[2]);
     char *input_file = argv[3];
     char *output_file = argv[4];
 
     char* universe = (char*)malloc(N * N * N);
     char *universe1 = (char *)malloc(N * N * N);
     read_file(input_file, universe);
     read_file(input_file, universe1);
 
     cout << "开始执行串行程序:" << endl;
     int start_pop = population(N, universe);
     auto start_time = std::chrono::high_resolution_clock::now();
     life3d_run(N, universe, T);
     auto end_time = std::chrono::high_resolution_clock::now();
     std::chrono::duration<double> duration = end_time - start_time;
     int final_pop = population(N, universe);
     write_file(output_file, universe, N);
 
     cout << "start population: " << start_pop << endl;
     cout << "final population: " << final_pop << endl;
     double time = duration.count();
     cout << "time: " << time << "s" << endl;
     cout << "cell per sec: " << T / time * N * N * N << endl;
 
 //------------------------------ --------------------------------//
     cout << "开始执行cuda程序:" << endl;
     start_pop = population(N, universe1);
     start_time = std::chrono::high_resolution_clock::now();
     life3d_run_cuda(N, universe1, T);
     end_time = std::chrono::high_resolution_clock::now();
     duration = end_time - start_time;
     final_pop = population(N, universe1);
     if (check_universe(universe, universe1, N)) {
         cout << "结果相同" << endl;
     }
     else {
         cout << "结果不同" << endl;
     }
 
 
     cout << "start population: " << start_pop << endl;
     cout << "final population: " << final_pop << endl;
     time = duration.count();
     cout << "time: " << time << "s" << endl;
     cout << "cell per sec: " << T / time * N * N * N << endl;
 
     
 
     free(universe);
     free(universe1);
     return 0;
 }
 