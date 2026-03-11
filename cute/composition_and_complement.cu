#include <iostream>
#include <vector>
#include <cute/tensor.hpp>

using namespace std;

int main(){
    vector<float> raw_data;
    for(int i{0}; i < 24; i++) raw_data.push_back(i);

    // Layout from which we want to select elements / logical coordiantes from
    auto layoutA = cute::make_layout(
        cute::make_shape(cute::_6{}, cute::_4{}),
        cute::make_stride(cute::_2{}, cute::_12{})
    );

    // Tiler i.e the Layout that would select the elements from the Layout A
    auto layoutB = cute::make_layout(
        cute::make_shape(cute::_2{}, cute::_2{}),
        cute::make_stride(cute::_1{}, cute::_6{})
    );

    // Compute the composition of the Layouts i.e use LayoutB to select logical elements of layoutA
    auto layoutA_o_B  = cute::composition(layoutA, layoutB);

    // Compute the Complement i.e the "Layout of Shifts" or "Layout of Repetitions of the Tile" 
    // such that it covers all the logical elements of the ShapeA when we shift the tiles according to the Complement Layout
    // Keep in mind that the Complement Layout is defined in terms of the logical coordinates space of the shapeA and not the 
    // index space of the LayoutA and this is the reason we don't need to pass the LayoutA as cotarget, just the ShapeA
    auto complement_layoutB_wrt_shapeA = cute::complement(layoutB, shape(layoutA));

    cout << "LayoutA\n"; cute::print_layout(layoutA);
    cout << "\nLayoutB\n"; cute::print_layout(layoutB);
    cout << "\nLayout AoB\n"; cute::print_layout(layoutA_o_B);
    cout << "\nComplement Layout of LayoutB (tiler) wrt ShapeA\n"; cute::print_layout(complement_layoutB_wrt_shapeA);

    return 0;
}

/*
    Compilation Command:
    nvcc ++ -I ./cutlass/include --std=c++17 composition_and_complement.cu
*/