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
// Authors: Radu Serban, Mike Taylor
// =============================================================================
//
// Simple powertrain model for the M113 vehicle.
// - both power and torque limited
// - no torque converter
// - no transmission box
//
// =============================================================================

#ifndef WVP_SIMPLEMAPPOWERTRAIN_H
#define WVP_SIMPLEMAPPOWERTRAIN_H

#include "chrono_vehicle/ChVehicle.h"
#include "chrono_vehicle/ChPowertrain.h"
#include "chrono/core/ChCubicSpline.h"

#include "chrono_models/ChApiModels.h"

namespace chrono {
namespace vehicle {
namespace wvp {

/// @addtogroup vehicle_models_generic
/// @{

/// Custom powertrain model for a generic vehicle.
/// Note that this does not use one of the Chrono::vehicle templates, but rather inherits directly
/// from ChPowertrain.
class CH_MODELS_API WVP_SimpleMapPowertrain : public ChPowertrain {
  public:
    WVP_SimpleMapPowertrain();

    ~WVP_SimpleMapPowertrain() {}

    /// Return the current engine speed.
    virtual double GetMotorSpeed() const override { return m_motorSpeed; }

    /// Return the current engine torque.
    virtual double GetMotorTorque() const override { return m_motorTorque; }

    /// Return the value of slippage in the torque converter.
    /// This simplified model does not have a torque converter.
    virtual double GetTorqueConverterSlippage() const override { return 0; }

    /// Return the input torque to the torque converter.
    /// This simplified model does not have a torque converter.
    virtual double GetTorqueConverterInputTorque() const override { return 0; }

    /// Return the output torque from the torque converter.
    /// This simplified model does not have a torque converter.
    virtual double GetTorqueConverterOutputTorque() const override { return 0; }

    /// Return the current transmission gear
    /// This simplified model does not have a transmission box.
    virtual int GetCurrentTransmissionGear() const override { return 1; }

    /// Return the output torque from the powertrain.
    /// This is the torque that is passed to a vehicle system, thus providing the
    /// interface between the powertrain and vehicle co-simulation modules.
    /// Since a ShaftsPowertrain is directly connected to the vehicle's driveline,
    /// this function returns 0.
    virtual double GetOutputTorque() const override { return m_shaftTorque; }

    /// Use this function to set the mode of automatic transmission.
    /// This simplified model does not have a transmission box.
    virtual void SetDriveMode(ChPowertrain::DriveMode mmode) override;

    /// Use this function to shift from one gear to another.
    /// A zero latency shift is assumed.
    /// Note, index starts from 0.
    void SetSelectedGear(int igear);

    /// Initialize the powertrain system.
    virtual void Initialize(std::shared_ptr<ChBody> chassis,     ///< [in] chassis o the associated vehicle
                            std::shared_ptr<ChShaft> driveshaft  ///< [in] shaft connection to the vehicle driveline
                            ) override;

    /// Update the state of this powertrain system at the current time.
    /// The powertrain system is provided the current driver throttle input, a
    /// value in the range [0,1], and the current angular speed of the transmission
    /// shaft (from the driveline).
    virtual void Synchronize(double time,        ///< [in] current time
                             double throttle,    ///< [in] current throttle input [0,1]
                             double shaft_speed  ///< [in] current angular speed of the transmission shaft
                             ) override;

    /// Advance the state of this powertrain system by the specified time step.
    /// This function does nothing for this simplified powertrain model.
    virtual void Advance(double step) override {}

  private:
    double m_current_gear_ratio;
    double m_motorSpeed;
    double m_motorTorque;
    double m_shaftTorque;

    int m_current_gear;
    std::vector<double> m_gear_ratios;

    // ChCubicSpline m_zeroThrottleMap;
    // ChCubicSpline m_fullThrottleMap;
    ChFunction_Recorder m_zeroThrottleFunction;
    ChFunction_Recorder m_fullThrottleFunction;

    void CheckShift();


};

/// @} vehicle_models_generic

}  // end namespace generic
}  // end namespace vehicle
}  // end namespace chrono

#endif
