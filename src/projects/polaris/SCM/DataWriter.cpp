// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2022 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Radu Serban
// =============================================================================
//
// Output data writer for Polaris on SPH terrain system
//
// =============================================================================

#include <array>

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"

#include "DataWriter.h"

using namespace chrono;
using namespace chrono::vehicle;

using std::cout;
using std::cin;
using std::endl;

DataWriter::DataWriter(ChSystem* sysFSI, int num_sample_boxes)
    : m_sys(sysFSI),
      m_mbs_output(true),
      m_verbose(true) 
      {
    //////m_indices.resize(m_num_sample_boxes);
}

DataWriter::~DataWriter() {
    m_mbs_stream.close();
}


// void DataWriter::SetSamplingVolume(const ChVector<>& offset, const ChVector2<>& size) {
//     m_box_size.x() = size.x();
//     m_box_size.y() = size.y();
//     m_box_offset = offset;
// }

void DataWriter::Initialize(const std::string& dir,
                            double major_FPS,
                            double step_size) {
    m_dir = dir;
    m_major_skip = (int)std::round((1.0 / major_FPS) / step_size);
    m_major_frame = -1;
    m_last_major = -1;
    m_minor_frame = 0;

    // Resize vectors
    m_mbs_outputs.resize(GetNumChannelsMBS());

    std::string filename = m_dir + "/mbs.csv";
    m_mbs_stream.open(filename, std::ios_base::trunc);

    // cout << "Sampling box size:   " << m_box_size << endl;
    // cout << "Sampling box offset: " << m_box_offset << endl;
    cout << "Major skip: " << m_major_skip << endl;
}

void DataWriter::Process(int sim_frame, double time) {
    // Collect data from all MBS channels and run through filters if requested
    CollectDataMBS();

    if (sim_frame % m_major_skip == 0) {
        m_last_major = sim_frame;
        m_major_frame++;
        m_minor_frame = 0;
        if (m_verbose)
        { 
            cout << "Start collection " << m_major_frame << endl;
            cout << "    Output data " << m_major_frame << "/"  << "  time: " << time << endl;
            Write();
        }
    }
    if (m_verbose)
        cout << std::flush;
}


void DataWriter::Write() {

    if (m_mbs_output) {
        std::string filename =
            m_dir + "/mbs_" + std::to_string(m_major_frame) + "_" +  ".csv";
        WriteDataMBS(filename);
    }

    {
        // Write line to global MBS output file
        m_mbs_stream << m_sys->GetChTime() << "    ";
        for (int i = 0; i < GetNumChannelsMBS(); i++)
            m_mbs_stream << m_mbs_outputs[i] << "  ";
        m_mbs_stream << "\n";
    }
}

// struct print_particle_pos {
//     print_particle_pos(std::ofstream* stream) : m_stream(stream) {}
//     __host__ void operator()(const Real4 p) { (*m_stream) << p.x << ", " << p.y << ", " << p.z << "\n"; }
//     std::ofstream* m_stream;
// };

// struct print_particle_pos_vel_acc_frc {
//     print_particle_pos_vel_acc_frc(std::ofstream* stream) : m_stream(stream) {}
//     template <typename T>
//     __host__ void operator()(const T pvf) {
//         auto p = thrust::get<0>(pvf);
//         auto v = thrust::get<1>(pvf);
//         auto a = thrust::get<2>(pvf);
//         auto f = thrust::get<3>(pvf);
//         (*m_stream) << p.x << ", " << p.y << ", " << p.z << ", "  //
//                     << v.x << ", " << v.y << ", " << v.z << ", "  //
//                     << a.x << ", " << a.y << ", " << a.z << ", "  //
//                     << f.x << ", " << f.y << ", " << f.z << "\n";
//     }
//     std::ofstream* m_stream;
// };

// void DataWriter::WriteDataParticles(const thrust::device_vector<int>& indices_D, const std::string& filename) {
//     if (m_particle_output == ParticleOutput::POSITIONS) {
//         // Get particle positions on device
//         auto pos_D = m_sysFSI.GetParticlePositions(indices_D);

//         // Copy vector to host
//         thrust::host_vector<Real4> pos_H = pos_D;

//         // Write output file
//         std::ofstream stream;
//         stream.open(filename, std::ios_base::trunc);
//         thrust::for_each(thrust::host, pos_H.begin(), pos_H.end(), print_particle_pos(&stream));
//         stream.close();
//     } else {
//         // Get particle positions, velocities, accelerations, and forces on device
//         auto pos_D = m_sysFSI.GetParticlePositions(indices_D);
//         auto vel_D = m_sysFSI.GetParticleVelocities(indices_D);
//         auto acc_D = m_sysFSI.GetParticleAccelerations(indices_D);
//         auto frc_D = m_sysFSI.GetParticleForces(indices_D);

//         // Copy vectors to host
//         thrust::host_vector<Real4> pos_H = pos_D;
//         thrust::host_vector<Real3> vel_H = vel_D;
//         thrust::host_vector<Real4> acc_H = acc_D;
//         thrust::host_vector<Real4> frc_H = frc_D;

//         // Write output file
//         std::ofstream stream;
//         stream.open(filename, std::ios_base::trunc);
//         thrust::for_each(thrust::host,                                                                         //
//                          thrust::make_zip_iterator(                                                            //
//                              thrust::make_tuple(pos_H.begin(), vel_H.begin(), acc_H.begin(), frc_H.begin())),  //
//                          thrust::make_zip_iterator(                                                            //
//                              thrust::make_tuple(pos_H.end(), vel_H.end(), acc_H.end(), frc_H.end())),          //
//                          print_particle_pos_vel_acc_frc(&stream)                                               //
//         );
//         stream.close();
//     }
// }

// --------------------------------------------------------------------------------------------------------------------

DataWriterVehicle::DataWriterVehicle(ChSystem* sysFSI, std::shared_ptr<WheeledVehicle> vehicle, SCMDeformableTerrain& terrain)
    : DataWriter(sysFSI, 4), m_vehicle(vehicle), m_terrain(terrain) {
    m_wheels[0] = vehicle->GetWheel(0, LEFT);
    m_wheels[1] = vehicle->GetWheel(0, RIGHT);
    m_wheels[2] = vehicle->GetWheel(1, LEFT);
    m_wheels[3] = vehicle->GetWheel(1, RIGHT);


    m_vel_channels = {
        7,  8,  9,  10, 11, 12,  // chassis
        20, 21, 22, 23, 24, 25,  // wheel FL
        33, 34, 35, 36, 37, 38,  // wheel FR
        46, 47, 48, 49, 50, 51,  // wheel RL
        59, 60, 61, 62, 63, 64   // wheel RR
    };

    m_acc_channels = {
        65, 66, 67, 68, 69, 70,  // wheel FL
        71, 72, 73, 74, 75, 76,  // wheel FR
        77, 78, 79, 80, 81, 82,  // wheel RL
        83, 84, 85, 86, 87, 88   // wheel RR
    };
}

void DataWriterVehicle::CollectDataMBS() {
    size_t start = 0;

    auto v_pos = m_vehicle->GetPos();
    m_mbs_outputs[start + 0] = v_pos.x();
    m_mbs_outputs[start + 1] = v_pos.y();
    m_mbs_outputs[start + 2] = v_pos.z();
    start += 3;

    auto v_rot = m_vehicle->GetRot();
    m_mbs_outputs[start + 0] = v_rot.e0();
    m_mbs_outputs[start + 1] = v_rot.e1();
    m_mbs_outputs[start + 2] = v_rot.e2();
    m_mbs_outputs[start + 3] = v_rot.e3();
    start += 4;

    auto v_vel = m_vehicle->GetPointVelocity(ChVector<>(0, 0, 0));
    m_mbs_outputs[start + 0] = v_vel.x();
    m_mbs_outputs[start + 1] = v_vel.y();
    m_mbs_outputs[start + 2] = v_vel.z();
    start += 3;

    auto v_omg = m_vehicle->GetChassisBody()->GetWvel_par();
    m_mbs_outputs[start + 0] = v_omg.x();
    m_mbs_outputs[start + 1] = v_omg.y();
    m_mbs_outputs[start + 2] = v_omg.z();
    start += 3;

    for (int i = 0; i < 4; i++) {
        auto w_state = m_wheels[i]->GetState();

        m_mbs_outputs[start + 0] = w_state.pos.x();
        m_mbs_outputs[start + 1] = w_state.pos.y();
        m_mbs_outputs[start + 2] = w_state.pos.z();
        start += 3;

        m_mbs_outputs[start + 0] = w_state.rot.e0();
        m_mbs_outputs[start + 1] = w_state.rot.e1();
        m_mbs_outputs[start + 2] = w_state.rot.e2();
        m_mbs_outputs[start + 3] = w_state.rot.e3();
        start += 4;

        m_mbs_outputs[start + 0] = w_state.lin_vel.x();
        m_mbs_outputs[start + 1] = w_state.lin_vel.y();
        m_mbs_outputs[start + 2] = w_state.lin_vel.z();
        start += 3;

        m_mbs_outputs[start + 0] = w_state.ang_vel.x();
        m_mbs_outputs[start + 1] = w_state.ang_vel.y();
        m_mbs_outputs[start + 2] = w_state.ang_vel.z();
        start += 3;
    }

    for (int i = 0; i < 4; i++) {
        // const auto& t_force = m_wheels[i]->GetSpindle()->Get_accumulated_force();
        const auto& t_force = m_terrain.GetContactForce(m_wheels[i]->GetSpindle());
        m_mbs_outputs[start + 0] = t_force.force.x();
        m_mbs_outputs[start + 1] = t_force.force.y();
        m_mbs_outputs[start + 2] = t_force.force.z();
        start += 3;

        // const auto& t_torque = m_wheels[i]->GetSpindle()->Get_accumulated_torque();
        const auto& t_torque = m_terrain.GetContactForce(m_wheels[i]->GetSpindle());
        m_mbs_outputs[start + 0] = t_torque.moment.x();
        m_mbs_outputs[start + 1] = t_torque.moment.y();
        m_mbs_outputs[start + 2] = t_torque.moment.z();
        start += 3;
    }
}

void DataWriterVehicle::WriteDataMBS(const std::string& filename) {
    const auto& o = m_mbs_outputs;
    std::ofstream stream;
    stream.open(filename, std::ios_base::trunc);

    size_t start = 0;

    // Vehicle position, orientation, linear and angular velocities
    for (int j = 0; j < 13; j++)
        stream << o[start + j] << ", ";
    stream << "\n";
    start += 13;

    // Wheel position, orientation, linear and angular velocities
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 13; j++)
            stream << o[start + j] << ", ";
        stream << "\n";
        start += 13;
    }

    // Tire force and moment
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 6; j++)
            stream << o[start + j] << ", ";
        stream << "\n";
        start += 6;
    }

    stream.close();
}


// // --------------------------------------------------------------------------------------------------------------------

// DataWriterObject::DataWriterObject(ChSystemFsi& sysFSI, std::shared_ptr<ChBody> body, const ChVector<>& body_size)
//     : DataWriter(sysFSI, 1), m_body(body), m_body_size(body_size) {
//     m_box_size = 2.0 * body_size;
//     m_box_offset = VNULL;

//     m_vel_channels = {7, 8, 9, 10, 11, 12};
//     m_acc_channels = {13, 14, 15, 16, 17, 18};
// }

// void DataWriterObject::CollectDataMBS() {
//     size_t start = 0;

//     auto v_pos = m_body->GetPos();
//     m_mbs_outputs[start + 0] = v_pos.x();
//     m_mbs_outputs[start + 1] = v_pos.y();
//     m_mbs_outputs[start + 2] = v_pos.z();
//     start += 3;

//     auto v_rot = m_body->GetRot();
//     m_mbs_outputs[start + 0] = v_rot.e0();
//     m_mbs_outputs[start + 1] = v_rot.e1();
//     m_mbs_outputs[start + 2] = v_rot.e2();
//     m_mbs_outputs[start + 3] = v_rot.e3();
//     start += 4;

//     auto v_vel = m_body->GetPos_dt();
//     m_mbs_outputs[start + 0] = v_vel.x();
//     m_mbs_outputs[start + 1] = v_vel.y();
//     m_mbs_outputs[start + 2] = v_vel.z();
//     start += 3;

//     auto v_omg = m_body->GetWvel_par();
//     m_mbs_outputs[start + 0] = v_omg.x();
//     m_mbs_outputs[start + 1] = v_omg.y();
//     m_mbs_outputs[start + 2] = v_omg.z();
//     start += 3;

//     const auto& t_force = m_body->Get_accumulated_force();
//     m_mbs_outputs[start + 0] = t_force.x();
//     m_mbs_outputs[start + 1] = t_force.y();
//     m_mbs_outputs[start + 2] = t_force.z();
//     start += 3;

//     const auto& t_torque = m_body->Get_accumulated_torque();
//     m_mbs_outputs[start + 0] = t_torque.x();
//     m_mbs_outputs[start + 1] = t_torque.y();
//     m_mbs_outputs[start + 2] = t_torque.z();
//     start += 3;
// }

// void DataWriterObject::WriteDataMBS(const std::string& filename) {
//     const auto& o = m_mbs_outputs;
//     std::ofstream stream;
//     stream.open(filename, std::ios_base::trunc);

//     size_t start = 0;

//     // Body position, orientation, linear and angular velocities
//     for (int j = 0; j < 13; j++)
//         stream << o[start + j] << ", ";
//     stream << "\n";
//     start += 13;

//     // Body force and moment
//     for (int j = 0; j < 6; j++)
//         stream << o[start + j] << ", ";
//     stream << "\n";
//     start += 6;

//     stream.close();
// }

// ChFrame<> DataWriterObject::GetSampleBoxFrame(int box_id) const {
//     auto pos = m_body->GetPos();
//     auto normal = m_body->GetRot().GetYaxis();
//     auto hheight = m_body_size.z() / 2;

//     ChVector<> Z_dir(0, 0, 1);
//     ChVector<> X_dir = Vcross(normal, ChVector<>(0, 0, 1)).GetNormalized();
//     ChVector<> Y_dir = Vcross(Z_dir, X_dir);
//     ChMatrix33<> box_rot(X_dir, Y_dir, Z_dir);
//     ChVector<> box_pos = pos + box_rot * (m_box_offset - ChVector<>(0, 0, hheight));

//     return ChFrame<>(box_pos, box_rot);
// }