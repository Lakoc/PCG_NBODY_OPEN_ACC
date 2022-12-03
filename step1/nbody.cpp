/**
 * @file      nbody.cpp
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

#include <math.h>
#include <cfloat>
#include "nbody.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                       Declare following structs / classes                                          //
//                                  If necessary, add your own classes / routines                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compute gravitation velocity
void calculate_gravitation_velocity(const Particles &p,
                                    Velocities &tmp_vel,
                                    const int N,
                                    const float dt) {
// Loop over all particles
#pragma acc parallel loop  present(p, tmp_vel) gang, worker, vector
    for (unsigned p1_index = 0; p1_index < N; p1_index++) {
        // Load particle_1 positions
        float p1_pos_x = p.pos_x[p1_index];
        float p1_pos_y = p.pos_y[p1_index];
        float p1_pos_z = p.pos_z[p1_index];

        // Init aux velocity vector and other aux variables
        float v_temp_x = .0f;
        float v_temp_y = .0f;
        float v_temp_z = .0f;

        float dx, dy, dz, ir3, Fg_dt_m2_r;
// Loop over other particles to keep p1 data cached and avoid global memory access
#pragma acc loop seq
        for (unsigned p2_index = 0; p2_index < N; p2_index++) {
            // Load particle_2 positions and weight
            float p2_pos_x = p.pos_x[p2_index];
            float p2_pos_y = p.pos_y[p2_index];
            float p2_pos_z = p.pos_z[p2_index];
            float p2_weight = p.weight[p2_index];
            // Calculate per axis distance
            // Reverted order to save up 1 more unary operation (-G  -> G)
            dx = p2_pos_x - p1_pos_x;
            dy = p2_pos_y - p1_pos_y;
            dz = p2_pos_z - p1_pos_z;

            // Calculate inverse of Euclidean distance pow3
            ir3 = powf(dx * dx + dy * dy + dz * dz, INVERSE_SQRT_POW3) + FLT_MIN;

            // Simplified from CPU implementation
            Fg_dt_m2_r = G * dt * ir3 * p2_weight;

            bool not_colliding = ir3 < COLLISION_DISTANCE_INVERSE_POW3;

            // If there is no collision, add local velocities to temporal vector
            v_temp_x += not_colliding ? Fg_dt_m2_r * dx : 0.0f;
            v_temp_y += not_colliding ? Fg_dt_m2_r * dy : 0.0f;
            v_temp_z += not_colliding ? Fg_dt_m2_r * dz : 0.0f;
        }

        tmp_vel.vel_x[p1_index] = v_temp_x;
        tmp_vel.vel_y[p1_index] = v_temp_y;
        tmp_vel.vel_z[p1_index] = v_temp_z;
    }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

void calculate_collision_velocity(const Particles &p,
                                  Velocities &tmp_vel,
                                  const int N,
                                  const float dt) {

#pragma acc parallel loop  present(p, tmp_vel) gang, worker, vector
    for (unsigned p1_index = 0; p1_index < N; p1_index++) {
        // Load particle_1 positions, velocities and weight
        float p1_pos_x = p.pos_x[p1_index];
        float p1_pos_y = p.pos_y[p1_index];
        float p1_pos_z = p.pos_z[p1_index];
        float p1_vel_x = p.vel_x[p1_index];
        float p1_vel_y = p.vel_y[p1_index];
        float p1_vel_z = p.vel_z[p1_index];
        float p1_weight = p.weight[p1_index];

        // Init aux velocity vector and other aux variables
        float v_temp_x = .0f;
        float v_temp_y = .0f;
        float v_temp_z = .0f;

        bool colliding;

        float r, dx, dy, dz, weight_difference, weight_sum, double_m2;
#pragma acc loop seq
        for (unsigned p2_index = 0; p2_index < N; p2_index++) {
            // Load particle_2 positions and weight
            float p2_pos_x = p.pos_x[p2_index];
            float p2_pos_y = p.pos_y[p2_index];
            float p2_pos_z = p.pos_z[p2_index];
            float p2_vel_x = p.vel_x[p2_index];
            float p2_vel_y = p.vel_y[p2_index];
            float p2_vel_z = p.vel_z[p2_index];
            float p2_weight = p.weight[p2_index];

            // Calculate per axis distance
            // Reverted order to save up 1 more unary operation (-G  -> G)
            dx = p2_pos_x - p1_pos_x;
            dy = p2_pos_y - p1_pos_y;
            dz = p2_pos_z - p1_pos_z;

            // Calculate inverse of Euclidean distance pow3
            r = sqrt(dx * dx + dy * dy + dz * dz) + FLT_MIN;

            // Save values below to registers to save accesses to memory and multiple calculations of same code
            weight_difference = p1_weight - p2_weight;
            weight_sum = p1_weight + p2_weight;
            double_m2 = p2_weight * 2.0f;

            // Inverse condition, inverse distance is equal to infinity if it's calculated between same point
            colliding = r < COLLISION_DISTANCE;


            // If colliding add to temporal vector current velocities
            // Application of distributive law of *,+ operations in Real field => p1.weight* p1.vel_x - p2.weight *p1.vel_x  - > p1.vel_x * (weight_difference)
            v_temp_x += colliding ? ((p1_vel_x * weight_difference + double_m2 * p2_vel_x) / weight_sum) - p1_vel_x
                                  : 0.0f;
            v_temp_y+= colliding ? ((p1_vel_y * weight_difference + double_m2 * p2_vel_y) / weight_sum) - p1_vel_y
                                  : 0.0f;
            v_temp_z += colliding ? ((p1_vel_z * weight_difference + double_m2 * p2_vel_z) / weight_sum) - p1_vel_z
                                  : 0.0f;
        }

        // Update values in global context
        tmp_vel.vel_x[p1_index] += v_temp_x;
        tmp_vel.vel_y[p1_index] += v_temp_y;
        tmp_vel.vel_z[p1_index] += v_temp_z;
    }
}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/// Update particle position
void update_particle(Particles &p,
                     Velocities &tmp_vel,
                     const int N,
                     const float dt) {
#pragma acc parallel loop present(p, tmp_vel)
    for (unsigned i = 0; i < N; i++) {
        p.vel_x[i] += tmp_vel.vel_x[i];
        p.vel_y[i] += tmp_vel.vel_y[i];
        p.vel_z[i] += tmp_vel.vel_z[i];
        p.pos_x[i] += p.vel_x[i] * dt;
        p.pos_y[i] += p.vel_y[i] * dt;
        p.pos_z[i] += p.vel_z[i] * dt;
    }


}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------



/// Compute center of gravity
float4 centerOfMassGPU(const Particles &p,
                       const int N) {

    return {0.0f, 0.0f, 0.0f, 0.0f};
}// end of centerOfMassGPU
//----------------------------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compute center of mass on CPU
float4 centerOfMassCPU(MemDesc &memDesc) {
    float4 com = {0, 0, 0, 0};

    for (int i = 0; i < memDesc.getDataSize(); i++) {
        // Calculate the vector on the line connecting points and most recent position of center-of-mass
        const float dx = memDesc.getPosX(i) - com.x;
        const float dy = memDesc.getPosY(i) - com.y;
        const float dz = memDesc.getPosZ(i) - com.z;

        // Calculate weight ratio only if at least one particle isn't massless
        const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                         ? (memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

        // Update position and weight of the center-of-mass according to the weight ration and vector
        com.x += dx * dw;
        com.y += dy * dw;
        com.z += dz * dw;
        com.w += memDesc.getWeight(i);
    }
    return com;
}// end of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
