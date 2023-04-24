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

// using namespace chrono;
// using namespace chrono::vehicle;

using std::cout;
using std::cin;
using std::endl;

DataWriter::DataWriter(chrono::ChSystem* sys)
:m_sys(sys)
      {
}

DataWriter::~DataWriter() {
    m_mbs_stream.close();
}

void DataWriter::Initialize(const std::string& dir,
                            double step_size) {
    m_dir = dir;
    m_major_frame = -1;
    // Resize vectors
    m_mbs_outputs.resize(GetNumChannelsMBS());

    std::string filename = m_dir + "/mbs.csv";
    m_mbs_stream.open(filename, std::ios_base::trunc);

    // // cout << "Sampling box size:   " << m_box_size << endl;
    // // cout << "Sampling box offset: " << m_box_offset << endl;
    // cout << "Major skip: " << m_major_skip << endl;
}

void DataWriter::Process(int sim_frame, double time, const std::array<chrono::vehicle::TerrainForce, 4>& tire_forces) {
    // Collect data from all MBS channels and run through filters if requested
    m_major_frame++;
    m_tire_forces=tire_forces;
    CollectDataMBS();

    cout << "Start collection " << m_major_frame << endl;
    cout << "    Output data " << m_major_frame << "/"  << "  time: " << time << endl;
    Write();
    cout << std::flush;
}

void DataWriter::Write() {

    std::string filename =
        m_dir + "/mbs_" + std::to_string(m_major_frame) + "_" +  ".csv";
    WriteDataMBS(filename);
    

    // Write line to global MBS output file
    m_mbs_stream << m_sys->GetChTime() << "    ";
    for (int i = 0; i < GetNumChannelsMBS(); i++)
        m_mbs_stream << m_mbs_outputs[i] << "  ";
    m_mbs_stream << "\n";
    
}

DataWriterVehicle::DataWriterVehicle(chrono::ChSystem* sys, chrono::vehicle::WheeledVehicle* vehicle, chrono::vehicle::ChTerrain& terrain)
    : DataWriter(sys), m_vehicle(vehicle), m_terrain(terrain){
    m_wheels[0] = m_vehicle->GetWheel(0, chrono::vehicle::LEFT);
    m_wheels[1] = m_vehicle->GetWheel(0, chrono::vehicle::RIGHT);
    m_wheels[2] = m_vehicle->GetWheel(1, chrono::vehicle::LEFT);
    m_wheels[3] = m_vehicle->GetWheel(1, chrono::vehicle::RIGHT);
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

    auto v_vel = m_vehicle->GetPointVelocity(chrono::ChVector<>(0, 0, 0));
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
        const auto& t_force = m_tire_forces[i].force;
        m_mbs_outputs[start + 0] = t_force.x();
        m_mbs_outputs[start + 1] = t_force.y();
        m_mbs_outputs[start + 2] = t_force.z();
        start += 3;

        const auto& t_torque = m_tire_forces[i].moment;
        m_mbs_outputs[start + 0] = t_torque.x();
        m_mbs_outputs[start + 1] = t_torque.y();
        m_mbs_outputs[start + 2] = t_torque.z();
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



