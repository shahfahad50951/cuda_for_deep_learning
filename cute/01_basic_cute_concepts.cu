#include <iostream>
#include <vector>
#include <cute/tensor.hpp>

using namespace std;

int main(){
    // Create standard vector as raw memory (backing storage) of the cute::Tensor
    vector<int> raw_memory{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    cout << "Raw Memory\n";
    for(int i{0}; i < 12; i++) cout << raw_memory[i] << ' ';
    cout << "\n\n";

    // Define the matrix layout i.e cute::Layout
    // 3x4 Matrix, Column-Major
    // Here, auto would resolve to a very complex / unreadable type. This is why auto keyword is helpful. 
    // cute::Layout<cute::tuple<cute::C<3>,cute::C<4>>, cute::tuple<cute::C<1>,cute::C<3>>>

    auto layout = cute::make_layout(
        cute::make_shape(cute::_3{}, cute::_4{}),
        cute::make_stride(cute::_1{}, cute::_3{})
    );
    
    // Define a cute::Tensor that binds the raw_memory to the cute::Layout
    auto tensor = cute::make_tensor(
        raw_memory.data(),
        layout
    );

    // Now we can access the 2D matrix generically and all the mappings from coordinate space
    // within the logical shape of the matrix to index space of raw data would be handled 
    // automatically by Layout
    cout << "Tensor: Shape: (3,4), Stride: (1,3), Column-Major\n";
    for(int row{0}; row < 3; row++){
        for(int col{0}; col < 4; col++) {
            // cout << "Tensor[" << row << "," << col << "]: " << tensor(row, col) << " -- ";
            cout << tensor(row, col) << ' ';
        }
        cout << '\n';
    }

    return 0;
}

/*
Clone cutlass in this directory to get access to the cuTe header files
Compilaton Command
    nvcc --std=c++17 -I ./cutlass/include basic.cu
*/