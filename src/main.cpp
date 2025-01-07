#include <cat_and_dogs.h>
#include <iostream>
#include <array>

#include <MNIST.h>

int main(int argc, const char** argv)
{
    fp::run_mnist(argc, argv);
    // fp::run_cat_and_dogs();
    return 0;
}
