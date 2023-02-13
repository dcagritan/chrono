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
// Author: Radu Serban
// =============================================================================
//
// =============================================================================

#ifndef AV_IRRAPP_H
#define AV_IRRAPP_H

#include "chrono_vehicle/ChVehicleVisualSystemIrrlicht.h"
#include "vehicle.h"

namespace av {

class Framework;
class EventReceiver;

class IrrApp : public chrono::vehicle::ChVehicleVisualSystemIrrlicht {
  public:
    IrrApp(Framework* framework, unsigned int width = 1000, unsigned int height = 800);

  private:
    void ChangeVehicle(int index);
    virtual void renderOtherStats(int left, int top) override;

    Framework* m_framework;
    EventReceiver* m_evrec;

    friend class EventReceiver;
};

}  // end namespace av

#endif