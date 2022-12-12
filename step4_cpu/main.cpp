/**
 * @file      main.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     PCG Assignment 2
 *            N-Body simulation in ACC
 *
 * @version   2022
 *
 * @date      11 November  2020, 11:22 (created) \n
 * @date      15 November  2022, 14:03 (revised) \n
 *
 */

#include <chrono>
#include <cstdio>
#include <cmath>

#include "nbody.h"
#include "h5Helper.h"

/**
 * Main routine of the project
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {
    // Parse command line parameters
    if (argc != 7) {
        printf("Usage: nbody <N> <dt> <steps> <write intesity> <input> <output>\n");
        exit(EXIT_FAILURE);
    }

    const int N = std::stoi(argv[1]);
    const float dt = std::stof(argv[2]);
    const int steps = std::stoi(argv[3]);
    const int writeFreq = (std::stoi(argv[4]) > 0) ? std::stoi(argv[4]) : 0;

    printf("N: %d\n", N);
    printf("dt: %f\n", dt);
    printf("steps: %d\n", steps);

    const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                         Code to be implemented                                                   //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 1.  Memory allocation on CPU
    Particles particles_curr = Particles(N);
    // 2. Create memory descriptor
    /*
     * Caution! Create only after CPU side allocation
     * parameters:
     *                                    Stride of two               Offset of the first
     *             Data pointer           consecutive elements        element in floats,
     *                                    in floats, not bytes        not bytes
    */
    MemDesc md(
            particles_curr.pos_x, 1, 0,            // Position in X
            particles_curr.pos_y, 1, 0,            // Position in Y
            particles_curr.pos_z, 1, 0,            // Position in Z
            particles_curr.vel_x, 1, 0,            // Velocity in X
            particles_curr.vel_y, 1, 0,            // Velocity in Y
            particles_curr.vel_z, 1, 0,            // Velocity in Z
            particles_curr.weight, 1, 0,            // Weight
            N,                                                                // Number of particles
            recordsNum);                                                      // Number of records in output file



    H5Helper h5Helper(argv[5], argv[6], md);

    // Read data
    try {
        h5Helper.init();
        h5Helper.readParticleData();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // 3. Copy data to GPU
    Particles particles_next = particles_curr;

    particles_curr.copyToGPU();
    particles_next.copyToGPU();

    // Start the time
    auto startTime = std::chrono::high_resolution_clock::now();


    float4 comOnGPULoop = {0.0f, 0.0f, 0.0f, 0.f};
    // 4. Run the loop - calculate new Particle positions.
    for (int s = 0; s < steps; s++) {
#pragma acc wait(VEL_STREAM)
        comOnGPULoop = centerOfMassGPU((steps % 2 ? particles_next : particles_curr), N);


        calculate_velocity(s % 2 ? particles_next : particles_curr,
                           s % 2 ? particles_curr : particles_next, N, dt);

        /// In step 4 - fill in the code to store Particle snapshots.
        if (writeFreq > 0 && (s % writeFreq == 0)) {

            (steps % 2 ? particles_next : particles_curr).copyToCPU();

            // since memory descriptor is attached to curr arr, there is need to copy values to properly calculate COM on CPU
            if (steps % 2 > 0) {
                particles_curr.copy(particles_next);
            }

            h5Helper.writeParticleData(s / writeFreq);
            h5Helper.writeCom(comOnGPULoop.x, comOnGPULoop.y, comOnGPULoop.z, comOnGPULoop.w, s / writeFreq);
        }
    }// for s ...
#pragma acc wait
    // 5. In steps 3 and 4 -  Compute center of gravity
    float4 comOnGPU = centerOfMassGPU((steps % 2 ? particles_next : particles_curr), N);


    // Stop watchclock
    const auto endTime = std::chrono::high_resolution_clock::now();
    const double time = (endTime - startTime) / std::chrono::microseconds(1);
    printf("Time: %f s\n", time / 10e6);


    // 5. Copy data from GPU back to CPU.
    (steps % 2 ? particles_next : particles_curr).copyToCPU();

    // since memory descriptor is attached to curr arr, there is need to copy values to properly calculate COM on CPU
    if (steps % 2 > 0) {
        particles_curr.copy(particles_next);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// Calculate center of gravity
    float4 comOnCPU = centerOfMassCPU(md);


    std::cout << "Center of mass on CPU:" << std::endl
              << comOnCPU.x << ", "
              << comOnCPU.y << ", "
              << comOnCPU.z << ", "
              << comOnCPU.w
              << std::endl;

    std::cout << "Center of mass on GPU:" << std::endl
              << comOnGPU.x << ", "
              << comOnGPU.y << ", "
              << comOnGPU.z << ", "
              << comOnGPU.w
              << std::endl;

    // Store final positions of the particles into a file
    h5Helper.writeComFinal(comOnGPU.x, comOnGPU.y, comOnGPU.z, comOnGPU.w);
    h5Helper.writeParticleDataFinal();

    return EXIT_SUCCESS;
}// end of main
//----------------------------------------------------------------------------------------------------------------------

