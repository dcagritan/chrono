// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban
// =============================================================================
//
// =============================================================================

#include "truckAV.h"
#include "framework.h"

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::vehicle::hmmwv;

namespace av {

TruckAV::TruckAV(Framework* framework, const chrono::ChCoordsys<>& init_pos) : Vehicle(framework) {
    m_hmmwv = std::make_shared<HMMWV_Reduced>(framework->m_system);

    m_hmmwv->SetChassisFixed(false);
    m_hmmwv->SetChassisCollisionType(ChassisCollisionType::NONE);
    m_hmmwv->SetPowertrainType(PowertrainModelType::SIMPLE);
    m_hmmwv->SetDriveType(DrivelineType::RWD);
    m_hmmwv->SetTireType(TireModelType::RIGID);
    m_hmmwv->SetTireStepSize(framework->m_step);
    m_hmmwv->SetVehicleStepSize(framework->m_step);
    m_hmmwv->SetInitPosition(init_pos);
    m_hmmwv->Initialize();

    m_hmmwv->SetChassisVisualizationType(VisualizationType::PRIMITIVES);
    m_hmmwv->SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    m_hmmwv->SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    m_hmmwv->SetWheelVisualizationType(VisualizationType::NONE);
    m_hmmwv->SetTireVisualizationType(VisualizationType::PRIMITIVES);
}

TruckAV::~TruckAV() {
    //
}

ChCoordsys<> TruckAV::GetPosition() const {
    return ChCoordsys<>(m_hmmwv->GetVehicle().GetVehiclePos(), m_hmmwv->GetVehicle().GetVehicleRot());
}

ChVehicle& TruckAV::GetVehicle() const {
    return m_hmmwv->GetVehicle();
}

ChPowertrain& TruckAV::GetPowertrain() const {
    return m_hmmwv->GetPowertrain();
}

void TruckAV::Synchronize(double time) {
    m_hmmwv->Synchronize(time, m_steering, m_braking, m_throttle, *m_framework->m_terrain);
}

void TruckAV::Advance(double step) {
    m_hmmwv->Advance(step);
    AdvanceDriver(step);
}

}  // end namespace av