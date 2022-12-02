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
void calculate_gravitation_velocity(const Particles& p,
                                    Velocities&      tmp_vel,
                                    const int        N,
                                    const float      dt)
{

#pragma acc parallel loop present(p,tmp_vel)
    for (unsigned p1_index = 0; p1_index < N; p1_index++)
    {
        // Load particle_1 positions and weight
        float4 pos_p1 = p.data[p1_index];

        float3 v_temp = {0.0f, 0.0f, 0.0f};

        float ir, dx, dy, dz, ir3, Fg_dt_m2_r;;
#pragma acc loop
        for (unsigned p2_index = p1_index; p2_index < N; p2_index++)
        {
            // Load particle_2 positions and weight
            float4 pos_p2 = p.data[p2_index];

            // Calculate per axis distance
            // Reverted order to save up 1 more unary operation (-G  -> G)
            dx = pos_p2.x - pos_p1.x;
            dy = pos_p2.y - pos_p1.y;
            dz = pos_p2.z - pos_p1.z;

            // Calculate inverse of Euclidean distance between two particles, get rid of division
            ir = 1.0f / sqrt(dx * dx + dy * dy + dz * dz);
            ir3 = ir * ir * ir + FLT_MIN;


            // Simplified from CPU implementation
            Fg_dt_m2_r = G * dt * ir3 * pos_p2.w;

            bool not_colliding = ir < COLLISION_DISTANCE_INVERSE;

            // If there is no collision, add local velocities to temporal vector
            v_temp.x += not_colliding ? Fg_dt_m2_r * dx : 0.0f;
            v_temp.y += not_colliding ? Fg_dt_m2_r * dy : 0.0f;
            v_temp.z += not_colliding ? Fg_dt_m2_r * dz : 0.0f;
        }

        tmp_vel.data[p1_index].x = v_temp.x;
        tmp_vel.data[p1_index].y = v_temp.y;
        tmp_vel.data[p1_index].z = v_temp.z;
    }
}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

void calculate_collision_velocity(const Particles& p,
                                  Velocities&      tmp_vel,
                                  const int        N,
                                  const float      dt)
{



}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/// Update particle position
void update_particle(const Particles& p,
                     Velocities&      tmp_vel,
                     const int        N,
                     const float      dt)
{



}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------



/// Compute center of gravity
float4 centerOfMassGPU(const Particles& p,
                       const int        N)
{

  return {0.0f, 0.0f, 0.0f, 0.0f};
}// end of centerOfMassGPU
//----------------------------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Compute center of mass on CPU
float4 centerOfMassCPU(MemDesc& memDesc)
{
  float4 com = {0 ,0, 0, 0};

  for(int i = 0; i < memDesc.getDataSize(); i++)
  {
    // Calculate the vector on the line connecting points and most recent position of center-of-mass
    const float dx = memDesc.getPosX(i) - com.x;
    const float dy = memDesc.getPosY(i) - com.y;
    const float dz = memDesc.getPosZ(i) - com.z;

    // Calculate weight ratio only if at least one particle isn't massless
    const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                          ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += dx * dw;
    com.y += dy * dw;
    com.z += dz * dw;
    com.w += memDesc.getWeight(i);
  }
  return com;
}// end of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------
