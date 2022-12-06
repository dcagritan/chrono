// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban, Asher Elmquist
// =============================================================================
//
// A driver model that uses a path to bring the vehicle up to speed. It then
// switches to an open loop controller based on steering data provided in a
// text file. The vehicle will still try to maintain the original
// target velocity
//
// =============================================================================

#include "chrono/assets/ChLineShape.h"
#include "chrono/core/ChMathematics.h"
#include "chrono/geometry/ChLineBezier.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_models/vehicle/wvp/WVP_FollowerDataDriver.h"

namespace chrono {
namespace vehicle {
namespace wvp {

double steeringFactor = -.001282;  // ratio of steering wheel full turn to -1:1

WVP_FollowerDataDriver::WVP_FollowerDataDriver(ChVehicle& vehicle,
                                               const std::string& steering_filename,
                                               const std::string& speed_filename,
                                               std::shared_ptr<ChBezierCurve> path,
                                               const std::string& path_name,
                                               double target_speed,
                                               const std::string data_input_file,
                                               int time_column,
                                               int steering_column,
                                               double time_til_steady)
    : ChDriver(vehicle),
      m_steeringPID(steering_filename, path),
      m_speedPID(speed_filename),
      m_pathName(path_name),
      m_target_speed(target_speed),
      m_throttle_threshold(0.2),
      m_time_til_steady(time_til_steady) {
    ParseCSV(data_input_file, time_column, steering_column);
    Create();
}

void WVP_FollowerDataDriver::Create() {
    // Reset the steering and speed controllers
    m_steeringPID.Reset(m_vehicle);
    m_speedPID.Reset(m_vehicle);

    // Create a fixed body to carry a visualization asset for the path
    auto road = std::shared_ptr<ChBody>(m_vehicle.GetSystem()->NewBody());
    road->SetBodyFixed(true);
    m_vehicle.GetSystem()->AddBody(road);

    auto path_asset = chrono_types::make_shared<ChLineShape>();
    path_asset->SetLineGeometry(chrono_types::make_shared<geometry::ChLineBezier>(m_steeringPID.GetPath()));
    path_asset->SetColor(ChColor(0.0f, 0.8f, 0.0f));
    path_asset->SetName(m_pathName);
    road->AddVisualShape(path_asset);
}

void WVP_FollowerDataDriver::Reset() {
    m_steeringPID.Reset(m_vehicle);
    m_speedPID.Reset(m_vehicle);
}

void WVP_FollowerDataDriver::Advance(double step) {
    // Set the throttle and braking values based on the output from the speed controller.
    double out_speed = m_speedPID.Advance(m_vehicle, m_target_speed, step);
    ChClampValue(out_speed, -1.0, 1.0);

    if (out_speed > 0) {
        // Vehicle moving too slow
        m_braking = 0;
        m_throttle = out_speed;
    } else if (m_throttle > m_throttle_threshold) {
        // Vehicle moving too fast: reduce throttle
        m_braking = 0;
        m_throttle = 1 + out_speed;
    } else {
        // Vehicle moving too fast: apply brakes
        m_braking = -out_speed;
        m_throttle = 0;
    }
    double time = m_vehicle.GetChTime();

    if (time < m_time_til_steady) {
        // Set the steering value based on the output from the steering controller.
        double out_steering = m_steeringPID.Advance(m_vehicle, step);
        ChClampValue(out_steering, -1.0, 1.0);
        m_steering = out_steering;
    } else {
        m_steering = m_steering_map.Get_y(time - m_time_til_steady + m_dataStartTime);
        m_steering = m_steering * steeringFactor;
    }
}

void WVP_FollowerDataDriver::ExportPathPovray(const std::string& out_dir) {
    utils::WriteCurvePovray(*m_steeringPID.GetPath(), m_pathName, out_dir, 0.04, ChColor(0.8f, 0.5f, 0.0f));
}

void WVP_FollowerDataDriver::ParseCSV(std::string dataFile, int timeCol, int steeringCol) {
    // std::cout<<"parsing CSV"<<std::endl;
    std::ifstream file(dataFile);
    std::string line;

    // std::cout<<"file at: "<<dataFile<<std::endl;

    // skip the first 8 lines
    for (int i = 0; i < 8; i++) {
        // std::cout<<"skipped line"<<std::endl;
        std::getline(file, line);
        // std::cout<<line<<std::endl;
    }
    bool recStartTime = true;
    //add 0 steering point to map\
    //m_steering_map.AddPoint(0.0, 0.0);
    double startSteeringVal = 0;

    while (std::getline(file, line)) {
        std::vector<std::string> parts;
        std::stringstream linestring(line);
        std::string val;

        while (std::getline(linestring, val, ',')) {
            parts.push_back(val);
        }
        std::stringstream convertTime(parts[timeCol]);
        std::stringstream convertSteer(parts[steeringCol]);

        double tempTime;
        double tempSteer;

        convertTime >> tempTime;
        convertSteer >> tempSteer;

        if (recStartTime) {
            m_dataStartTime = tempTime;
            startSteeringVal = tempSteer;
            recStartTime = false;
        }

        m_steering_map.AddPoint(tempTime, tempSteer-startSteeringVal);
        /*std::cout<<"added point: "<<tempTime<<", "<<tempSteer<<std::endl;*/
        /*std::cout<<line<<std::endl;*/
        /*std::cout<<"read in another line"<<std::endl;*/
    }
}

}  // end namespace wvp
}  // end namespace vehicle
}  // end namespace chrono
