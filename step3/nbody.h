/**
 * @file      nbody.h
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
 * @date      15 November  2022, 14:10 (revised) \n
 *
 */

#ifndef __NBODY_H__
#define __NBODY_H__

#include <cstdlib>
#include <cstdio>
#include  <cmath>
#include "h5Helper.h"
#include <cstring>

/// Gravity constant
constexpr float G = 6.67384e-11f;

/// Collision distance threshold
constexpr float COLLISION_DISTANCE = 0.01f;
constexpr float COLLISION_DISTANCE_INVERSE = 100.0f;
constexpr float COLLISION_DISTANCE_INVERSE_POW3 = 1e6f;
constexpr float INVERSE_SQRT_POW3 = -3.0 / 2.0;

/**
 * @struct float4
 * Structure that mimics CUDA float4
 */
struct float4 {
    float x;
    float y;
    float z;
    float w;

};

/// Define sqrtf from CUDA libm library
#pragma acc routine(sqrtf) seq

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Declare following structs / classes                                          //
//                                  If necessary, add your own classes / routines                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Structure with particle data
 */
struct Particles {
    float *pos_x;
    float *pos_y;
    float *pos_z;
    float *vel_x;
    float *vel_y;
    float *vel_z;
    float *weight;

    size_t p_count;
    // Fill the structure holding the particle/s data
    // It is recommended to implement constructor / destructor and copyToGPU and copyToCPU routines

    Particles(size_t p_count) : p_count(p_count) {
        pos_x = new float[p_count];
        pos_y = new float[p_count];
        pos_z = new float[p_count];
        vel_x = new float[p_count];
        vel_y = new float[p_count];
        vel_z = new float[p_count];
        weight = new float[p_count];

#pragma acc enter data copyin(this)
#pragma acc enter data create(pos_x[0:p_count], pos_y[0:p_count], pos_z[0:p_count], vel_x[0:p_count], vel_y[0:p_count], vel_z[0:p_count], weight[0:p_count])
    }

    Particles(const Particles &p) {
        p_count = p.p_count;

        pos_x = new float[p_count];
        pos_y = new float[p_count];
        pos_z = new float[p_count];
        vel_x = new float[p_count];
        vel_y = new float[p_count];
        vel_z = new float[p_count];
        weight = new float[p_count];

        size_t arr_byte_size = sizeof(float) * p_count;

        std::memcpy(weight, p.weight, arr_byte_size);

#pragma acc enter data copyin(this)
#pragma acc enter data copyin(weight[0:p_count])
#pragma acc enter data create(pos_x[0:p_count], pos_y[0:p_count], pos_z[0:p_count], vel_x[0:p_count], vel_y[0:p_count], vel_z[0:p_count])
    }

    ~Particles() {
#pragma acc exit data delete(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, weight)
#pragma acc exit data delete(this)
        delete[] pos_x;
        delete[] pos_y;
        delete[] pos_z;
        delete[] vel_x;
        delete[] vel_y;
        delete[] vel_z;
        delete[] weight;
    }

    void copyToGPU() {
#pragma acc update device(pos_x[0:p_count], pos_y[0:p_count], pos_z[0:p_count], vel_x[0:p_count], vel_y[0:p_count], vel_z[0:p_count], weight[0:p_count])
    }

    void copyToCPU() {
#pragma acc update host(pos_x[0:p_count], pos_y[0:p_count], pos_z[0:p_count], vel_x[0:p_count], vel_y[0:p_count], vel_z[0:p_count], weight[0:p_count])
    }

};// end of Particles
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute velocity
 * @param [in]  p_curr        - Particles current
 * @param [out] p_next  - Particles next
 * @param [in ] N        - Number of particles
 * @param [in]  dt       - Time step size
 */
void calculate_velocity(const Particles &p_curr,
                        Particles &p_next,
                        const int N,
                        const float dt);


/**
 * Compute center of gravity - implement in steps 3 and 4.
 * @param [in] p - Particles
 * @param [in] N - Number of particles
 * @return Center of Mass [x, y, z] and total weight[w]
 */
float4 centerOfMassGPU(const Particles &p,
                       const int N);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Compute center of mass on CPU
 * @param memDesc
 * @return centre of gravity
 */
float4 centerOfMassCPU(MemDesc &memDesc);

#endif /* __NBODY_H__ */
