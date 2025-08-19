/** @file example.cu
 *  @brief Usage example for the CUDA C++ API
 */

#include "sphericart_sycl.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "sphericart.hpp"
int main() {
    /* ===== set up the calculation ===== */

    // hard-coded parameters for the example
    size_t n_samples = 2;
    size_t l_max = 6;

    // initializes samples
    auto xyz = std::vector<double>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz[i] = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
    }

    // to avoid unnecessary allocations, calculators can use pre-allocated
    // memory, one also can provide uninitialized vectors that will be
    // automatically reshaped
    auto sph = std::vector<double>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph = std::vector<double>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph = std::vector<double>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

    // the class is templated, so one can also use 32-bit double operations
    auto xyz_f = std::vector<double>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz_f[i] = (double)xyz[i];
    }
    auto sph_f = std::vector<double>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph_f = std::vector<double>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph_f = std::vector<double>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

    DEVICE_INIT(double, xyz_device, xyz_f.data(), xyz_f.size())
    DEVICE_INIT(double, sph_device, sph_f.data(), sph_f.size())
    DEVICE_INIT(double, dsph_device, dsph_f.data(), dsph_f.size())
    DEVICE_INIT(double, ddsph_device, ddsph_f.data(), ddsph_f.size())
    /* ===== API calls ===== */

    // internal buffers and numerical factors are initalized at construction
    sphericart::intel::SphericalHarmonics<double> calculator_sycl(l_max);
    auto calculator = sphericart::SphericalHarmonics<double>(l_max);
//
//    // allcate device memory
//    // calculation examples
    calculator_sycl.compute(xyz_device, n_samples, sph_device); // no gradients
    calculator.compute(xyz, sph);                      // no gradients
    DEVICE_GET(double, sph_f.data(), sph_device, sph_f.size())
    int size2 =
       (l_max + 1) * (l_max + 1); // Size of the second+third dimensions in derivative arrays
    for (size_t i_sample = 0; i_sample < n_samples; i_sample++) {
       for (size_t l = 0; l < (l_max + 1); l++) {
           for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {

                   printf(
                       "SPH: %e , %e\n",
                       sph_f[size2 * i_sample + l * l + l + m],
                       sph[size2 * i_sample + l * l + l + m]
                   );
           }
       }
    }
    printf("computing gradients \n");
    calculator.compute_with_gradients(
        xyz, sph, dsph
    ); // with gradients
    calculator_sycl.compute_with_gradients(
        xyz_device, n_samples, sph_device, dsph_device
    ); // with gradients00
    DEVICE_GET(double, sph_f.data(), sph_device, sph_f.size())
    DEVICE_GET(double, dsph_f.data(), dsph_device, dsph_f.size())
    /* ===== check results ===== */

    double dsph_error = 0.0, dsph_norm = 0.0;
    int n_sph = (l_max + 1) * (l_max + 1);
    for (size_t alpha = 0; alpha < 3; alpha++) {
        for (size_t i_sample = 0; i_sample < n_samples; i_sample++) {
            for (int i_sph = 0; i_sph < n_sph; i_sph++) {
                 dsph_error += 
                         dsph[3 * n_sph * i_sample + n_sph * alpha + i_sph]
                        - dsph_f[3 * n_sph * i_sample + n_sph * alpha + i_sph];
                 dsph_norm += 
                         dsph[3 * n_sph * i_sample + n_sph * alpha + i_sph]
                        * dsph[3 * n_sph * i_sample + n_sph * alpha + i_sph];
            }
        }
    }
    printf("Float vs double relative error: %12.8e\n", sqrt(dsph_error / dsph_norm));
//

    return 0;
}
