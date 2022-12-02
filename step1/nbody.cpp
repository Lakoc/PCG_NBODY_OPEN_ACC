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

#pragma acc parallel loop  gang, vector present(p, p.pos[N], p.vel[N], tmp_vel, tmp_vel.vel[N]) tile(16, 16)
    for (unsigned p1_index = 0; p1_index < N; p1_index++) {
        // Load particle_1 positions and weight
        float4 pos_p1 = p.pos[p1_index];

        float3 v_temp = {0.0f, 0.0f, 0.0f};

        float dx, dy, dz, ir3, Fg_dt_m2_r;;
//#pragma acc loop
        for (unsigned p2_index = 0; p2_index < N; p2_index++) {
            // Load particle_2 positions and weight
            float4 pos_p2 = p.pos[p2_index];

            // Calculate per axis distance
            // Reverted order to save up 1 more unary operation (-G  -> G)
            dx = pos_p2.x - pos_p1.x;
            dy = pos_p2.y - pos_p1.y;
            dz = pos_p2.z - pos_p1.z;

            // Calculate inverse of Euclidean distance pow3
            ir3 = powf(dx * dx + dy * dy + dz * dz, INVERSE_SQRT_POW3) + FLT_MIN;

            // Simplified from CPU implementation
            Fg_dt_m2_r = G * dt * ir3 * pos_p2.w;

            bool not_colliding = ir3 < COLLISION_DISTANCE_INVERSE_POW3;

            // If there is no collision, add local velocities to temporal vector
            v_temp.x += not_colliding ? Fg_dt_m2_r * dx : 0.0f;
            v_temp.y += not_colliding ? Fg_dt_m2_r * dy : 0.0f;
            v_temp.z += not_colliding ? Fg_dt_m2_r * dz : 0.0f;
        }

        tmp_vel.vel[p1_index].x = v_temp.x;
        tmp_vel.vel[p1_index].y = v_temp.y;
        tmp_vel.vel[p1_index].z = v_temp.z;
    }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

void calculate_collision_velocity(const Particles &p,
                                  Velocities &tmp_vel,
                                  const int N,
                                  const float dt) {

#pragma acc parallel loop  gang, vector present(p, p.pos[N], p.vel[N], tmp_vel, tmp_vel.vel[N]) tile(16, 16)
    for (unsigned p1_index = 0; p1_index < N; p1_index++) {
        // Load particle_1 positions and weight
        float4 pos_p1 = p.pos[p1_index];
        float3 vel_p1 = p.vel[p1_index];

        float3 v_temp = {0.0f, 0.0f, 0.0f};
        bool colliding;

        float r, dx, dy, dz, weight_difference, weight_sum, double_m2;
//#pragma acc loop
        for (unsigned p2_index = 0; p2_index < N; p2_index++) {
            // Load particle_2 positions and weight
            float4 pos_p2 = p.pos[p2_index];
            float3 vel_p2 = p.vel[p2_index];

            // Calculate per axis distance
            // Reverted order to save up 1 more unary operation (-G  -> G)
            dx = pos_p2.x - pos_p1.x;
            dy = pos_p2.y - pos_p1.y;
            dz = pos_p2.z - pos_p1.z;

            // Calculate inverse of Euclidean distance pow3
            r = sqrt(dx * dx + dy * dy + dz * dz) + FLT_MIN;

            // Save values below to registers to save accesses to memory and multiple calculations of same code
            weight_difference = pos_p1.w - pos_p2.w;
            weight_sum = pos_p1.w + pos_p2.w;
            double_m2 = pos_p2.w * 2.0f;

            // Inverse condition, inverse distance is equal to infinity if it's calculated between same point
            colliding = r < COLLISION_DISTANCE;


            // If colliding add to temporal vector current velocities
            // Application of distributive law of *,+ operations in Real field => p1.weight* p1.vel_x - p2.weight *p1.vel_x  - > p1.vel_x * (weight_difference)
            v_temp.x += colliding ? ((vel_p1.x * weight_difference + double_m2 * vel_p2.x) / weight_sum) - vel_p1.x
                                  : 0.0f;
            v_temp.y += colliding ? ((vel_p1.y * weight_difference + double_m2 * vel_p2.y) / weight_sum) - vel_p1.y
                                  : 0.0f;
            v_temp.z += colliding ? ((vel_p1.z * weight_difference + double_m2 * vel_p2.z) / weight_sum) - vel_p1.z
                                  : 0.0f;
        }

        // Update values in global context
        tmp_vel.vel[p1_index].x += v_temp.x;
        tmp_vel.vel[p1_index].y += v_temp.y;
        tmp_vel.vel[p1_index].z += v_temp.z;
    }


}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/// Update particle position
void update_particle(Particles &p,
                     Velocities &tmp_vel,
                     const int N,
                     const float dt) {
#pragma acc parallel loop present(p, p.pos[N], p.vel[N], tmp_vel, tmp_vel.vel[N])
    for (unsigned i = 0; i < N; i++) {
        p.vel[i].x += tmp_vel.vel[i].x;
        p.vel[i].y += tmp_vel.vel[i].y;
        p.vel[i].z += tmp_vel.vel[i].z;
        p.pos[i].x += p.vel[i].x * dt;
        p.pos[i].y += p.vel[i].y * dt;
        p.pos[i].z += p.vel[i].z * dt;
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
