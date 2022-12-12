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

/// Compute velocity
void calculate_velocity(const Particles &p_curr,
                        Particles &p_next,
                        const int N,
                        const float dt) {
// Loop over all particles
#pragma acc parallel loop  present(p_curr, p_next) gang worker vector async(1)
    for (unsigned p1_index = 0; p1_index < N; p1_index++) {
        // Load particle_1 data
        float p1_pos_x = p_curr.pos_x[p1_index];
        float p1_pos_y = p_curr.pos_y[p1_index];
        float p1_pos_z = p_curr.pos_z[p1_index];
        float p1_vel_x = p_curr.vel_x[p1_index];
        float p1_vel_y = p_curr.vel_y[p1_index];
        float p1_vel_z = p_curr.vel_z[p1_index];
        float p1_weight = p_curr.weight[p1_index];

        // Init aux velocity vector and other aux variables
        float v_temp_x = 0.0f;
        float v_temp_y = 0.0f;
        float v_temp_z = 0.0f;

        float dx, dy, dz, ir3, Fg_dt_m2_r;

        bool not_colliding;
// Loop over other particles to keep p1 data cached and avoid global memory access
#pragma acc loop seq
        for (unsigned p2_index = 0; p2_index < N; p2_index++) {
            // Load particle_2 data
            float p2_pos_x = p_curr.pos_x[p2_index];
            float p2_pos_y = p_curr.pos_y[p2_index];
            float p2_pos_z = p_curr.pos_z[p2_index];
            float p2_weight = p_curr.weight[p2_index];

            // Calculate per axis distance
            // Reverted order to save up 1 more unary operation (-G  -> G)
            dx = p2_pos_x - p1_pos_x;
            dy = p2_pos_y - p1_pos_y;
            dz = p2_pos_z - p1_pos_z;

            // Calculate inverse of Euclidean distance pow3, could be applied since 1. operand is always positive
            ir3 = powf(dx * dx + dy * dy + dz * dz, INVERSE_SQRT_POW3);
            Fg_dt_m2_r = G * dt * ir3 * p2_weight;

            not_colliding = ir3 < COLLISION_DISTANCE_INVERSE_POW3;
            v_temp_x += not_colliding ? Fg_dt_m2_r * dx : 0.0f;
            v_temp_y += not_colliding ? Fg_dt_m2_r * dy : 0.0f;
            v_temp_z += not_colliding ? Fg_dt_m2_r * dz : 0.0f;

            // Since there are usually not many collision, if branch is faster than ternary
            if (ir3 >= COLLISION_DISTANCE_INVERSE_POW3 && p1_index != p2_index) {
                float p1_weight_ratio = p1_weight / (p1_weight + p2_weight);
                float p2_weight_ratio = 1 - p1_weight_ratio;
                float p1_vel_ratio = p1_weight_ratio - p2_weight_ratio - 1;
                float double_p2_weight_ratio = 2 * p2_weight_ratio;
                float p2_vel_x = p_curr.vel_x[p2_index];
                float p2_vel_y = p_curr.vel_y[p2_index];
                float p2_vel_z = p_curr.vel_z[p2_index];

                v_temp_x += p1_vel_x * p1_vel_ratio + double_p2_weight_ratio * p2_vel_x;
                v_temp_y += p1_vel_y * p1_vel_ratio + double_p2_weight_ratio * p2_vel_y;
                v_temp_z += p1_vel_z * p1_vel_ratio + double_p2_weight_ratio * p2_vel_z;
            }

        }

        // Update locally
        p1_vel_x = p1_vel_x + v_temp_x;
        p1_vel_y = p1_vel_y + v_temp_y;
        p1_vel_z = p1_vel_z + v_temp_z;

        // Flush to global mem
        p_next.vel_x[p1_index] = p1_vel_x;
        p_next.vel_y[p1_index] = p1_vel_y;
        p_next.vel_z[p1_index] = p1_vel_z;

        // Update_positions
        p_next.pos_x[p1_index] = p1_pos_x + p1_vel_x * dt;
        p_next.pos_y[p1_index] = p1_pos_y + p1_vel_y * dt;
        p_next.pos_z[p1_index] = p1_pos_z + p1_vel_z * dt;
    }
}// end of calculate_velocity
//----------------------------------------------------------------------------------------------------------------------

/// Compute center of gravity
float4 centerOfMassGPU(const Particles &p,
                       const int N) {
    float weighted_sum_pos_x = 0.0f;
    float weighted_sum_pos_y = 0.0f;
    float weighted_sum_pos_z = 0.0f;
    float com_w = 0.0f;
#pragma acc parallel loop copy(weighted_sum_pos_x, weighted_sum_pos_y, weighted_sum_pos_z, com_w) present(p)  reduction(+:weighted_sum_pos_x, weighted_sum_pos_y, weighted_sum_pos_z, com_w) gang worker vector
    for (unsigned particle_index = 0; particle_index < N; particle_index++) {
        float particle_weight = p.weight[particle_index];
        weighted_sum_pos_x += p.pos_x[particle_index] * particle_weight;
        weighted_sum_pos_y += p.pos_y[particle_index] * particle_weight;
        weighted_sum_pos_z += p.pos_z[particle_index] * particle_weight;
        com_w += particle_weight;
    }
    return {weighted_sum_pos_x / com_w, weighted_sum_pos_y / com_w, weighted_sum_pos_z / com_w, com_w};
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
