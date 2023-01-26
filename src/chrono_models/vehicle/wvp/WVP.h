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
// Wrapper classes for modeling an entire WVP vehicle assembly
// (including the vehicle itself, the powertrain, and the tires).
//
// =============================================================================

#ifndef WVP_H
#define WVP_H

#include <array>
#include <string>

#include "chrono_models/ChApiModels.h"
#include "chrono_models/vehicle/wvp/WVP_Vehicle.h"
#include "chrono_models/vehicle/wvp/WVP_SimpleMapPowertrain.h"
#include "chrono_models/vehicle/wvp/WVP_FialaTire.h"
#include "chrono_models/vehicle/wvp/WVP_Pac89Tire.h"
#include "chrono_models/vehicle/wvp/WVP_Powertrain.h"
#include "chrono_models/vehicle/wvp/WVP_RigidTire.h"
#include "chrono_models/vehicle/wvp/WVP_TMeasyTire.h"

namespace chrono {
namespace vehicle {
namespace wvp {

class CH_MODELS_API WVP {
  public:
    WVP();
    WVP(ChSystem* system);

    ~WVP();

    void SetContactMethod(ChContactMethod val) { m_contactMethod = val; }

    void SetChassisFixed(bool val) { m_fixed = val; }
    void SetChassisCollisionType(CollisionType val) { m_chassisCollisionType = val; }

    void SetTireType(TireModelType val) { m_tireType = val; }
    void SetTireCollisionType(ChTire::CollisionType collision_type) { m_tire_collision_type = collision_type; }

    void setSteeringType(SteeringTypeWV val) { m_steeringType = val; }

    void SetInitPosition(const ChCoordsys<>& pos) { m_initPos = pos; }
    void SetInitFwdVel(double fwdVel) { m_initFwdVel = fwdVel; }
    void SetInitWheelAngVel(const std::vector<double>& omega) { m_initOmega = omega; }

    void SetTireStepSize(double step_size) { m_tire_step_size = step_size; }

    void DisconnectPowertrain() { m_powertrain_connected = false; }

    ChSystem* GetSystem() const { return m_vehicle->GetSystem(); }
    ChWheeledVehicle& GetVehicle() const { return *m_vehicle; }
    std::shared_ptr<ChChassis> GetChassis() const { return m_vehicle->GetChassis(); }
    std::shared_ptr<ChBodyAuxRef> GetChassisBody() const { return m_vehicle->GetChassisBody(); }

    void Initialize();

    void SetAerodynamicDrag(double Cd, double area, double air_density);

    void SetChassisVisualizationType(VisualizationType vis) { m_vehicle->SetChassisVisualizationType(vis); }
    void SetSuspensionVisualizationType(VisualizationType vis) { m_vehicle->SetSuspensionVisualizationType(vis); }
    void SetSteeringVisualizationType(VisualizationType vis) { m_vehicle->SetSteeringVisualizationType(vis); }
    void SetWheelVisualizationType(VisualizationType vis) { m_vehicle->SetWheelVisualizationType(vis); }
    void SetTireVisualizationType(VisualizationType vis);

    void Synchronize(double time, const DriverInputs& driver_inputs, const ChTerrain& terrain);
    void Advance(double step);

    void LogHardpointLocations() { m_vehicle->LogHardpointLocations(); }
    void DebugLog(int what) { m_vehicle->DebugLog(what); }

  protected:
    ChContactMethod m_contactMethod;
    CollisionType m_chassisCollisionType;
    bool m_fixed;

    TireModelType m_tireType;
    ChTire::CollisionType m_tire_collision_type;
 
    double m_tire_step_size;

    SteeringTypeWV m_steeringType;

    ChCoordsys<> m_initPos;
    double m_initFwdVel;
    std::vector<double> m_initOmega;

    bool m_apply_drag;
    double m_Cd;
    double m_area;
    double m_air_density;

    bool m_powertrain_connected;

    ChSystem* m_system;
    WVP_Vehicle* m_vehicle;

    double m_tire_mass;
};

}  // end namespace wvp
}  // end namespace vehicle
}  // end namespace chrono

#endif
