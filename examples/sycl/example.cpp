/** @file example.cu
 *  @brief Usage example for the CUDA C++ API
 */

#include "sphericart_sycl.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#define DTYPE double

#include "sphericart.hpp"
#define TOLERANCE 1e-4 // High tolerance: finite differences are inaccurate for second
int main() {
    /* ===== set up the calculation ===== */

    // hard-coded parameters for the example
    size_t n_samples = 2;
    size_t l_max = 6;

    // initializes samples
    auto xyz = std::vector<DTYPE>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz[i] = (DTYPE)rand() / (DTYPE)RAND_MAX * 2.0 - 1.0;
    }

    // to avoid unnecessary allocations, calculators can use pre-allocated
    // memory, one also can provide uninitialized vectors that will be
    // automatically reshaped
    auto sph = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph = std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph = std::vector<DTYPE>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

    // the class is templated, so one can also use 32-bit DTYPE operations
    auto xyz_f = std::vector<DTYPE>(n_samples * 3, 0.0);
    for (size_t i = 0; i < n_samples * 3; ++i) {
        xyz_f[i] = (DTYPE)xyz[i];
    }
    auto sph_f = std::vector<DTYPE>(n_samples * (l_max + 1) * (l_max + 1), 0.0);
    auto dsph_f = std::vector<DTYPE>(n_samples * 3 * (l_max + 1) * (l_max + 1), 0.0);
    auto ddsph_f = std::vector<DTYPE>(n_samples * 3 * 3 * (l_max + 1) * (l_max + 1), 0.0);

    DEVICE_INIT(DTYPE, xyz_device, xyz_f.data(), xyz_f.size())
    DEVICE_INIT(DTYPE, sph_device, sph_f.data(), sph_f.size())
    DEVICE_INIT(DTYPE, dsph_device, dsph_f.data(), dsph_f.size())
    DEVICE_INIT(DTYPE, ddsph_device, ddsph_f.data(), ddsph_f.size())
    /* ===== API calls ===== */

    // internal buffers and numerical factors are initalized at construction
    sphericart::intel::SphericalHarmonics<DTYPE> calculator_sycl(l_max);
    auto calculator = sphericart::SphericalHarmonics<DTYPE>(l_max);
//
//    // allcate device memory
//    // calculation examples
    //calculator_sycl.compute(xyz_device, n_samples, sph_device); // no gradients
    //calculator.compute(xyz, sph);                      // no gradients
    printf("computing gradients \n");
    calculator.compute_with_gradients(
        xyz, sph, dsph
    ); // with gradients
    calculator_sycl.compute_with_gradients(
        xyz_device, n_samples, sph_device, dsph_device
    ); // with gradients00

    /* ===== check results ===== */
    DEVICE_GET(DTYPE, sph_f.data(), sph_device, sph_f.size())

    int size2 = (l_max + 1) * (l_max + 1); // Size of the second+third dimensions in derivative arrays
    DTYPE sph_error = 0.0, sph_norm = 0.0;
    for (size_t i = 0; i < n_samples * (l_max + 1) * (l_max + 1); ++i) {
            sph_error += (sph_f[i] - sph[i]) * (sph_f[i] - sph[i]);
            sph_norm += sph[i] * sph[i];
            //printf("SPHERR: %e ,SPHERNOR %e\n", sph_f[i] - sph[i], sph_norm);
   }

    printf("CPU vs GPU relative error SPH: %12.8e\n", sqrt(sph_error / sph_norm));

    /* ===== check results ===== */
    DEVICE_GET(DTYPE, dsph_f.data(), dsph_device, dsph_f.size());

    DTYPE dsph_error = 0.0, dsph_norm = 0.0;
    int n_sph = (l_max + 1) * (l_max + 1);
    for (size_t alpha = 0; alpha < 3; alpha++) {
       for (size_t i_sample = 0; i_sample < n_samples; i_sample++) {
           for (int i_sph = 0; i_sph < n_sph; i_sph++) {
               DTYPE d0 = dsph[3 * n_sph * i_sample + n_sph * alpha + i_sph];
               DTYPE d0_f = dsph_f[3 * n_sph * i_sample + n_sph * alpha + i_sph];
               dsph_error += (d0 - d0_f)*(d0 - d0_f);
               dsph_norm += d0 * d0;
                   if (std::abs(d0 / d0_f - 1.0) > TOLERANCE) {
                       std::cout << "Wrong first derivative: " << d0 << " vs " << d0_f
                                 << std::endl;
                   }
           }
       }
    }
    printf("CPU vs GPU relative error DSPH: %12.8e\n", sqrt(dsph_error / dsph_norm));
//

    return 0;
}
