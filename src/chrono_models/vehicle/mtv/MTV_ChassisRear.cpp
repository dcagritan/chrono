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
// Authors: Radu Serban, Rainer Gericke
// =============================================================================
//
// MTV cargo truck (5 tons) rear chassis subsystems.
//
// =============================================================================

#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChVehicleModelData.h"

#include "chrono_models/vehicle/mtv/MTV_ChassisRear.h"

namespace chrono {
namespace vehicle {
namespace fmtv {

// Static variables

const double MTV_ChassisRear::m_mass = 3338.333;
const ChVector<> MTV_ChassisRear::m_inertiaXX(2.861e3, 2.8605e3, 3.6300e3);
const ChVector<> MTV_ChassisRear::m_inertiaXY(0, -0.1055e3, 0);
const ChVector<> MTV_ChassisRear::m_COM_loc(-3.4919, 0, 0.8404);
const ChVector<> MTV_ChassisRear::m_connector_loc(-1.85, 0, 0.45);

const double MTV_ChassisConnector::m_torsion_stiffness = 7085;

// -----------------------------------------------------------------------------

MTV_ChassisRear::MTV_ChassisRear(const std::string& name, ChassisCollisionType chassis_collision_type)
    : ChRigidChassisRear(name) {
    m_inertia(0, 0) = m_inertiaXX.x();
    m_inertia(1, 1) = m_inertiaXX.y();
    m_inertia(2, 2) = m_inertiaXX.z();

    m_inertia(0, 1) = m_inertiaXY.x();
    m_inertia(0, 2) = m_inertiaXY.y();
    m_inertia(1, 2) = m_inertiaXY.z();
    m_inertia(1, 0) = m_inertiaXY.x();
    m_inertia(2, 0) = m_inertiaXY.y();
    m_inertia(2, 1) = m_inertiaXY.z();

    //// TODO:
    //// A more appropriate contact shape from primitives
    //// Add collision shapes for rear body

    double joint_pos_x = m_connector_loc.x();
    double joint_pos_z = m_connector_loc.z();
    double widthFrame = 0.905;
    double heightFrame = 0.2;

    ChVector<> rearBoxPos((-5.5 + joint_pos_x) / 2, 0, joint_pos_z);
    ChRigidChassisGeometry::BoxShape box(rearBoxPos, ChQuaternion<>(1, 0, 0, 0),
                                         ChVector<>(joint_pos_x + 5.5, widthFrame, heightFrame));
    ChRigidChassisGeometry::CylinderShape cyl_torsion(m_connector_loc, Q_from_AngZ(CH_C_PI_2), 0.1, 0.2);

    m_geometry.m_has_primitives = true;
    m_geometry.m_vis_boxes.push_back(box);
    m_geometry.m_vis_cylinders.push_back(cyl_torsion);
    m_geometry.m_color = ChColor(0.4f, 0.2f, 0.2f);

    m_geometry.m_has_mesh = true;
    m_geometry.m_vis_mesh_file = "mtv/meshes/m1083_rear.obj";

    m_geometry.m_has_collision = (chassis_collision_type != ChassisCollisionType::NONE);
    switch (chassis_collision_type) {
        case ChassisCollisionType::MESH:
            // For now, fall back to using primitive collision shapes
        case ChassisCollisionType::PRIMITIVES:
            box.m_matID = 0;
            m_geometry.m_coll_boxes.push_back(box);
            break;
        default:
            break;
    }
}

void MTV_ChassisRear::CreateContactMaterials(ChContactMethod contact_method) {
    // This model uses a single material with default properties.
    MaterialInfo minfo;
    m_geometry.m_materials.push_back(minfo.CreateMaterial(contact_method));
}

// -----------------------------------------------------------------------------

MTV_ChassisConnector::MTV_ChassisConnector(const std::string& name) : ChChassisConnectorTorsion(name) {}

}  // namespace fmtv
}  // end namespace vehicle
}  // end namespace chrono
