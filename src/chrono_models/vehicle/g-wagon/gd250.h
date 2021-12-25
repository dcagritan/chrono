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
// Wrapper classes for modeling an entire Mercedes GD250 vehicle assembly
// (including the vehicle itself, the powertrain, and the tires).
//
// =============================================================================

#ifndef GD250_H
#define GD250_H

#include <array>
#include <string>

#include "chrono_vehicle/wheeled_vehicle/tire/ChPacejkaTire.h"

#include "chrono_models/ChApiModels.h"
#include "chrono_models/vehicle/g-wagon/gd250_Vehicle.h"
#include "chrono_models/vehicle/g-wagon/gd250_SimpleMapPowertrain.h"
#include "chrono_models/vehicle/g-wagon/gd250_RigidTire.h"
#include "chrono_models/vehicle/g-wagon/gd250_TMeasyTire.h"
#include "chrono_models/vehicle/g-wagon/gd250_Pac02Tire.h"

namespace chrono {
    namespace vehicle {
        namespace gwagon {

/// @addtogroup vehicle_models_gwagon
/// @{

/// Definition of the UAZ assembly.
/// This class encapsulates a concrete wheeled vehicle model with parameters corresponding to
/// a UAZ vehicle, the powertrain model, and the 4 tires.
            class CH_MODELS_API GD250{
                    public:
                    GD250();
                    GD250(ChSystem* system);

                    ~GD250();

                    void SetContactMethod(ChContactMethod val) {
                        m_contactMethod = val;
                    }

                    void SetChassisFixed(bool val) {
                        m_fixed = val;
                    }
                    void SetChassisCollisionType(CollisionType val) {
                        m_chassisCollisionType = val;
                    }

                    void SetBrakeType(BrakeType brake_type) {
                        m_brake_type = brake_type;
                    }
                    void SetTireType(TireModelType val) {
                        m_tireType = val;
                    }

                    // void setSteeringType(SteeringTypeWV val) { m_steeringType = val; }

                    void SetInitPosition(const ChCoordsys<>& pos) {
                        m_initPos = pos;
                    }
                    void SetInitFwdVel(double fwdVel) {
                        m_initFwdVel = fwdVel;
                    }
                    void SetInitWheelAngVel(const std::vector<double>& omega) {
                        m_initOmega = omega;
                    }

                    void SetTireStepSize(double step_size) {
                        m_tire_step_size = step_size;
                    }

                    void EnableBrakeLocking(bool lock) {
                        m_brake_locking = lock;
                    }

                    ChSystem* GetSystem() const {
                        return m_vehicle->GetSystem();
                    }
                    ChWheeledVehicle& GetVehicle() const {
                        return *m_vehicle;
                    }
                    std::shared_ptr<ChChassis> GetChassis() const {
                        return m_vehicle->GetChassis();
                    }
                    std::shared_ptr<ChBodyAuxRef> GetChassisBody() const {
                        return m_vehicle->GetChassisBody();
                    }
                    std::shared_ptr<ChPowertrain> GetPowertrain() const {
                        return m_vehicle->GetPowertrain();
                    }
                    double GetTotalMass() const;

                    void Initialize();

                    void LockAxleDifferential(int axle, bool lock) {
                        m_vehicle->LockAxleDifferential(axle, lock);
                    }
                    void LockCentralDifferential(int which, bool lock) {
                        m_vehicle->LockCentralDifferential(which, lock);
                    }

                    void SetAerodynamicDrag(double Cd, double area, double air_density);

                    void SetChassisVisualizationType(VisualizationType vis) {
                        m_vehicle->SetChassisVisualizationType(vis);
                    }
                    void SetSuspensionVisualizationType(VisualizationType vis) {
                        m_vehicle->SetSuspensionVisualizationType(vis);
                    }
                    void SetSteeringVisualizationType(VisualizationType vis) {
                        m_vehicle->SetSteeringVisualizationType(vis);
                    }
                    void SetWheelVisualizationType(VisualizationType vis) {
                        m_vehicle->SetWheelVisualizationType(vis);
                    }
                    void SetTireVisualizationType(VisualizationType vis) {
                        m_vehicle->SetWheelVisualizationType(vis);
                    }

                    void Synchronize(double time, const ChDriver::Inputs& driver_inputs, const ChTerrain& terrain);
                    void Advance(double step);

                    void LogHardpointLocations() {
                        m_vehicle->LogHardpointLocations();
                    }
                    void DebugLog(int what) {
                        m_vehicle->DebugLog(what);
                    }

                    protected:
                    ChSystem* m_system;
                    GD250_Vehicle* m_vehicle;

                    ChContactMethod m_contactMethod;
                    CollisionType m_chassisCollisionType;
                    bool m_fixed;
                    bool m_brake_locking;

                    BrakeType m_brake_type;
                    TireModelType m_tireType;

                    double m_tire_step_size;

                    SteeringTypeWV m_steeringType;

                    ChCoordsys<> m_initPos;
                    double m_initFwdVel;
                    std::vector<double> m_initOmega;

                    bool m_apply_drag;
                    double m_Cd;
                    double m_area;
                    double m_air_density;

                    double m_tire_mass;
            };

/// @} vehicle_models_gwagon

        }  // end namespace gwagon
    }  // end namespace vehicle
}  // end namespace chrono

#endif
