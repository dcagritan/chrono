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
// Authors: Radu Serban, Justin Madsen
// =============================================================================
//
// GD250 wheel subsystem
//
// =============================================================================

#ifndef GD250_WHEEL_H
#define GD250_WHEEL_H

#include "chrono/assets/ChTriangleMeshShape.h"

#include "chrono_vehicle/ChSubsysDefs.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/wheeled_vehicle/ChWheel.h"

#include "chrono_models/ChApiModels.h"

namespace chrono {
    namespace vehicle {
        namespace gwagon {

/// @addtogroup vehicle_models_gwagon
/// @{

/// GD250 wheel (can be used on any axle, left or right).
            class CH_MODELS_API GD250_Wheel : public ChWheel {
            public:
            GD250_Wheel(const std::string& name);
            ~GD250_Wheel() {}

            virtual double GetMass() const override { return m_mass; }
        virtual ChVector<> GetInertia() const override { return m_inertia; }
    virtual double GetRadius() const override { return m_radius; }
virtual double GetWidth() const override { return m_width; }

protected:
static const double m_radius;
static const double m_width;
static const double m_mass;
static const ChVector<> m_inertia;
};

/// @} vehicle_models_gwagon

}  // end namespace gwagon
}  // end namespace vehicle
}  // end namespace chrono

#endif
