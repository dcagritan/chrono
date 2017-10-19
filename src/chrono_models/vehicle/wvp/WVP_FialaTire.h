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
// WVP Fiala tire subsystem
//
// =============================================================================

#ifndef WVP_FIALA_TIRE_H
#define WVP_FIALA_TIRE_H

#include "chrono/assets/ChTriangleMeshShape.h"

#include "chrono_vehicle/wheeled_vehicle/tire/ChFialaTire.h"

#include "chrono_models/ChApiModels.h"

namespace chrono {
namespace vehicle {
namespace wvp {

class CH_MODELS_API WVP_FialaTire : public ChFialaTire {
  public:
    WVP_FialaTire(const std::string& name);
    ~WVP_FialaTire() {}

    virtual double GetNormalStiffnessForce(double depth) const override;
    virtual double GetNormalDampingForce(double depth, double velocity) const override {
        return m_normalDamping * velocity;
    }

    virtual double GetVisualizationWidth() const override { return 0.25; }

    virtual void SetFialaParams() override;

    virtual void AddVisualizationAssets(VisualizationType vis) override;
    virtual void RemoveVisualizationAssets() override final;

  private:
    static const double m_normalDamping;

    static const std::string m_meshName;
    static const std::string m_meshFile;
    std::shared_ptr<ChTriangleMeshShape> m_trimesh_shape;
};

}  // end namespace wvp
}  // end namespace vehicle
}  // end namespace chrono

#endif