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
// Authors: Rainer Gericke, Radu Serban
// =============================================================================
//
// Base class for a link guided solid axle suspension.
//
// This class is meant for modelling a nonsteerable solid axle. This type
// of suspension is used by Mercedes G-models. Like in the real vehicle
// the longitudinal links use bushings for connection to the axle tube.
// Bushing parameters are taken from industry solutions for machine/ground
// isolation elements.
//
// The suspension subsystem is modeled with respect to a right-handed frame,
// with X pointing towards the front, Y to the left, and Z up (ISO standard).
// The suspension reference frame is assumed to be always aligned with that of
// the vehicle.  When attached to a chassis, only an offset is provided.
//
// All point locations are assumed to be given for the left half of the
// suspension and will be mirrored (reflecting the y coordinates) to construct
// the right side.
//
// =============================================================================

#include "chrono/assets/ChCylinderShape.h"
#include "chrono/assets/ChPointPointShape.h"

#include "chrono_models/vehicle/g-wagon/base/ChGAxle.h"

namespace chrono {
namespace vehicle {

// -----------------------------------------------------------------------------
// Static variables
// -----------------------------------------------------------------------------
const std::string ChGAxle::m_pointNames[] = {"SHOCK_A    ", "SHOCK_C    ", "SPRING_A   ", "SPRING_C   ",
                                             "SPINDLE    ", "PANHARD_A  ", "PANHARD_C  ", "LONGLINK_O ",
                                             "LONGLINK_I ", "LONGLINK_C ", "ANTIROLL_A ", "ANTIROLL_C "};

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
ChGAxle::ChGAxle(const std::string& name) : ChSuspension(name) {}

ChGAxle::~ChGAxle() {
    auto sys = m_axleTube->GetSystem();
    if (sys) {
        sys->Remove(m_axleTube);
        sys->Remove(m_axleTubeGuide);
        for (int i = 0; i < 2; i++) {
            sys->Remove(m_shock[i]);
            sys->Remove(m_spring[i]);
        }
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void ChGAxle::Initialize(std::shared_ptr<ChChassis> chassis,
                         std::shared_ptr<ChSubchassis> subchassis,
                         std::shared_ptr<ChSteering> steering,
                         const ChVector<>& location,
                         double left_ang_vel,
                         double right_ang_vel) {
    m_parent = chassis;
    m_rel_loc = location;

    // Unit vectors for orientation matrices.
    ChVector<> u;
    ChVector<> v;
    ChVector<> w;
    ChMatrix33<> rot;

    // Express the suspension reference frame in the absolute coordinate system.
    ChFrame<> suspension_to_abs(location);
    suspension_to_abs.ConcatenatePreTransformation(chassis->GetBody()->GetFrame_REF_to_abs());

    // Transform the location of the axle body COM to absolute frame.
    ChVector<> axleCOM_local = getAxleTubeCOM();
    ChVector<> axleCOM = suspension_to_abs.TransformLocalToParent(axleCOM_local);

    // Calculate end points on the axle body, expressed in the absolute frame
    // (for visualization)
    ////ChVector<> midpoint_local = 0.0;
    ////ChVector<> outer_local(axleCOM_local.x(), midpoint_local.y(), axleCOM_local.z());
    ChVector<> outer_local(getLocation(SPINDLE));
    m_axleOuterL = suspension_to_abs.TransformPointLocalToParent(outer_local);
    outer_local.y() = -outer_local.y();
    m_axleOuterR = suspension_to_abs.TransformPointLocalToParent(outer_local);

    ChVector<> panhardAxle_local(getLocation(PANHARD_A));
    m_ptPanhardAxle = suspension_to_abs.TransformPointLocalToParent(panhardAxle_local);
    ChVector<> panhardChassis_local(getLocation(PANHARD_C));
    m_ptPanhardChassis = suspension_to_abs.TransformPointLocalToParent(panhardChassis_local);
    ChVector<> ptPanhardCom = 0.5 * (m_ptPanhardAxle + m_ptPanhardChassis);

    ChVector<> linkO_local(getLocation(LONGLINK_O));
    m_ptLongLinkAxle[LEFT] = suspension_to_abs.TransformPointLocalToParent(linkO_local);
    linkO_local.y() *= -1.0;
    m_ptLongLinkAxle[RIGHT] = suspension_to_abs.TransformPointLocalToParent(linkO_local);
    ChVector<> linkC_local(getLocation(LONGLINK_C));
    m_ptLongLinkChassis[LEFT] = suspension_to_abs.TransformPointLocalToParent(linkC_local);
    linkC_local.y() *= -1.0;
    m_ptLongLinkChassis[RIGHT] = suspension_to_abs.TransformPointLocalToParent(linkC_local);

    ChVector<> arbC_local(getLocation(ANTIROLL_C));
    m_ptARBChassis[LEFT] = suspension_to_abs.TransformPointLocalToParent(arbC_local);
    arbC_local.y() *= -1.0;
    m_ptARBChassis[RIGHT] = suspension_to_abs.TransformPointLocalToParent(arbC_local);

    ChVector<> arbA_local(getLocation(ANTIROLL_A));
    m_ptARBAxle[LEFT] = suspension_to_abs.TransformPointLocalToParent(arbA_local);
    arbA_local.y() *= -1.0;
    m_ptARBAxle[RIGHT] = suspension_to_abs.TransformPointLocalToParent(arbA_local);

    m_ptARBCenter = 0.5 * (m_ptARBChassis[LEFT] + m_ptARBChassis[RIGHT]);

    // Create and initialize the axle body.
    m_axleTube = std::shared_ptr<ChBody>(chassis->GetBody()->GetSystem()->NewBody());
    m_axleTube->SetNameString(m_name + "_axleTube");
    m_axleTube->SetPos(axleCOM);
    m_axleTube->SetRot(chassis->GetBody()->GetFrame_REF_to_abs().GetRot());
    m_axleTube->SetMass(getAxleTubeMass());
    m_axleTube->SetInertiaXX(getAxleTubeInertia());
    chassis->GetBody()->GetSystem()->AddBody(m_axleTube);

    // Fix the axle body to the chassis
    m_axleTubeGuide = chrono_types::make_shared<ChLinkLockFree>();
    m_axleTubeGuide->SetNameString(m_name + "_freeAxleTube");
    const ChQuaternion<>& guideRot = chassis->GetBody()->GetFrame_REF_to_abs().GetRot();
    m_axleTubeGuide->Initialize(chassis->GetBody(), m_axleTube,
                                ChCoordsys<>(axleCOM, guideRot * Q_from_AngY(CH_C_PI_2)));
    chassis->GetBody()->GetSystem()->AddLink(m_axleTubeGuide);

    // Create and initialize the axle body.
    m_panhardRod = std::shared_ptr<ChBody>(chassis->GetBody()->GetSystem()->NewBody());
    m_panhardRod->SetNameString(m_name + "_panhardRod");
    m_panhardRod->SetPos(ptPanhardCom);
    m_panhardRod->SetRot(chassis->GetBody()->GetFrame_REF_to_abs().GetRot());
    m_panhardRod->SetMass(getPanhardRodMass());
    m_panhardRod->SetInertiaXX(getPanhardRodInertia());
    chassis->GetBody()->GetSystem()->AddBody(m_panhardRod);

    // connect the panhard rod to the chassis
    m_sphPanhardChassis = chrono_types::make_shared<ChVehicleJoint>(
        ChVehicleJoint::Type::SPHERICAL, m_name + "_sphericalPanhardChassis", chassis->GetBody(), m_panhardRod,
        ChCoordsys<>(m_ptPanhardChassis, QUNIT));
    chassis->AddJoint(m_sphPanhardChassis);

    // connect the panhard rod to the axle tube
    m_sphPanhardAxle =
        chrono_types::make_shared<ChVehicleJoint>(ChVehicleJoint::Type::SPHERICAL, m_name + "_sphericalPanhardAxle",
                                                  m_axleTube, m_panhardRod, ChCoordsys<>(m_ptPanhardAxle, QUNIT));
    chassis->AddJoint(m_sphPanhardAxle);

    // Transform all hardpoints to absolute frame.
    m_pointsL.resize(NUM_POINTS);
    m_pointsR.resize(NUM_POINTS);
    for (int i = 0; i < NUM_POINTS; i++) {
        ChVector<> rel_pos = getLocation(static_cast<PointId>(i));
        m_pointsL[i] = suspension_to_abs.TransformLocalToParent(rel_pos);
        rel_pos.y() = -rel_pos.y();
        m_pointsR[i] = suspension_to_abs.TransformLocalToParent(rel_pos);
    }

    // Initialize left and right sides.
    std::shared_ptr<ChBody> scbeamL = (subchassis == nullptr) ? chassis->GetBody() : subchassis->GetBeam(LEFT);
    std::shared_ptr<ChBody> scbeamR = (subchassis == nullptr) ? chassis->GetBody() : subchassis->GetBeam(RIGHT);
    InitializeSide(LEFT, chassis, scbeamL, m_pointsL, left_ang_vel);
    InitializeSide(RIGHT, chassis, scbeamR, m_pointsR, right_ang_vel);
}

void ChGAxle::InitializeSide(VehicleSide side,
                             std::shared_ptr<ChChassis> chassis,
                             std::shared_ptr<ChBody> scbeam,
                             const std::vector<ChVector<>>& points,
                             double ang_vel) {
    std::string suffix = (side == LEFT) ? "_L" : "_R";

    auto chassisBody = chassis->GetBody();

    // Unit vectors for orientation matrices.
    ChVector<> u;
    ChVector<> v;
    ChVector<> w;
    ChMatrix33<> rot;

    // Chassis orientation (expressed in absolute frame)
    // Recall that the suspension reference frame is aligned with the chassis.
    ChQuaternion<> chassisRot = chassisBody->GetFrame_REF_to_abs().GetRot();

    // Create and initialize spindle body (same orientation as the chassis)
    m_spindle[side] = std::shared_ptr<ChBody>(chassis->GetSystem()->NewBody());
    m_spindle[side]->SetNameString(m_name + "_spindle" + suffix);
    m_spindle[side]->SetPos(points[SPINDLE]);
    m_spindle[side]->SetRot(chassisRot);
    m_spindle[side]->SetWvel_loc(ChVector<>(0, ang_vel, 0));
    m_spindle[side]->SetMass(getSpindleMass());
    m_spindle[side]->SetInertiaXX(getSpindleInertia());
    chassis->GetSystem()->AddBody(m_spindle[side]);

    // Create and initialize longitudinal link body (same orientation as the chassis)
    m_longLink[side] = std::shared_ptr<ChBody>(chassis->GetSystem()->NewBody());
    m_longLink[side]->SetNameString(m_name + "_longLink" + suffix);
    m_longLink[side]->SetPos(0.5 * (points[LONGLINK_C] + points[LONGLINK_O]));
    m_longLink[side]->SetRot(chassisRot);
    m_longLink[side]->SetMass(getLongLinkMass());
    m_longLink[side]->SetInertiaXX(getLongLinkInertia());
    chassis->GetSystem()->AddBody(m_longLink[side]);

    // connect the long link to the chassis
    m_sphLongLinkChassis[side] = chrono_types::make_shared<ChVehicleJoint>(
        ChVehicleJoint::Type::SPHERICAL, m_name + "_sphericalLongLinkChassis", chassisBody, m_longLink[side],
        ChCoordsys<>(points[LONGLINK_C], QUNIT));
    chassis->AddJoint(m_sphLongLinkChassis[side]);

    // connect the long link to the outer axle connection point
    m_sphLongLinkAxleOuter[side] = chrono_types::make_shared<ChVehicleJoint>(
        ChVehicleJoint::Type::SPHERICAL, m_name + "_sphericalLongLinkAxleOuter", m_axleTube, m_longLink[side],
        ChCoordsys<>(points[LONGLINK_O], QUNIT), getBushingData());
    chassis->AddJoint(m_sphLongLinkAxleOuter[side]);

    // connect the long link to the inner axle connection point
    m_sphLongLinkAxleInner[side] = chrono_types::make_shared<ChVehicleJoint>(
        ChVehicleJoint::Type::SPHERICAL, m_name + "_sphericalLongLinkAxleInner", m_axleTube, m_longLink[side],
        ChCoordsys<>(points[LONGLINK_I], QUNIT), getBushingData());
    chassis->AddJoint(m_sphLongLinkAxleInner[side]);

    // Create and initialize the revolute joint between axle tube and spindle.
    ChCoordsys<> rev_csys(points[SPINDLE], chassisRot * Q_from_AngAxis(CH_C_PI / 2.0, VECT_X));
    m_revolute[side] = chrono_types::make_shared<ChLinkLockRevolute>();
    m_revolute[side]->SetNameString(m_name + "_revolute" + suffix);
    m_revolute[side]->Initialize(m_spindle[side], m_axleTube, rev_csys);
    chassis->GetSystem()->AddLink(m_revolute[side]);

    // Create and initialize the shock damper
    m_shock[side] = chrono_types::make_shared<ChLinkTSDA>();
    m_shock[side]->SetNameString(m_name + "_shock" + suffix);
    m_shock[side]->Initialize(chassisBody, m_axleTube, false, points[SHOCK_C], points[SHOCK_A]);
    m_shock[side]->RegisterForceFunctor(getShockForceFunctor());
    chassis->GetSystem()->AddLink(m_shock[side]);

    // Create and initialize the spring
    m_spring[side] = chrono_types::make_shared<ChLinkTSDA>();
    m_spring[side]->SetNameString(m_name + "_spring" + suffix);
    m_spring[side]->Initialize(scbeam, m_axleTube, false, points[SPRING_C], points[SPRING_A]);
    m_spring[side]->SetRestLength(getSpringRestLength());
    m_spring[side]->RegisterForceFunctor(getSpringForceFunctor());
    chassis->GetSystem()->AddLink(m_spring[side]);

    // Create and initialize the axle shaft and its connection to the spindle. Note that the
    // spindle rotates about the Y axis.
    m_axle[side] = chrono_types::make_shared<ChShaft>();
    m_axle[side]->SetNameString(m_name + "_axle" + suffix);
    m_axle[side]->SetInertia(getAxleInertia());
    m_axle[side]->SetPos_dt(-ang_vel);
    chassis->GetSystem()->AddShaft(m_axle[side]);

    m_axle_to_spindle[side] = chrono_types::make_shared<ChShaftsBody>();
    m_axle_to_spindle[side]->SetNameString(m_name + "_axle_to_spindle" + suffix);
    m_axle_to_spindle[side]->Initialize(m_axle[side], m_spindle[side], ChVector<>(0, -1, 0));
    chassis->GetSystem()->Add(m_axle_to_spindle[side]);

    m_arb[side] = std::shared_ptr<ChBody>(chassis->GetSystem()->NewBody());
    m_arb[side]->SetNameString(m_name + "_arb" + suffix);
    m_arb[side]->SetPos(0.5 * (points[ANTIROLL_C] + m_ptARBCenter));
    m_arb[side]->SetRot(chassisRot);
    m_arb[side]->SetMass(getARBMass());
    m_arb[side]->SetInertiaXX(getARBInertia());
    chassis->GetSystem()->AddBody(m_arb[side]);

    if (side == LEFT) {
        m_revARBChassis = chrono_types::make_shared<ChVehicleJoint>(
            ChVehicleJoint::Type::REVOLUTE, m_name + "_revARBchassis", chassisBody, m_arb[side],
            ChCoordsys<>(m_ptARBCenter, chassisRot * Q_from_AngAxis(CH_C_PI / 2.0, VECT_X)));
        chassis->AddJoint(m_revARBChassis);
    } else {
        m_revARBLeftRight = chrono_types::make_shared<ChLinkLockRevolute>();
        m_revARBLeftRight->SetNameString(m_name + "_revARBleftRight");
        m_revARBLeftRight->Initialize(m_arb[LEFT], m_arb[RIGHT],
                                      ChCoordsys<>(m_ptARBCenter, chassisRot * Q_from_AngAxis(CH_C_PI / 2.0, VECT_X)));
        chassis->GetSystem()->AddLink(m_revARBLeftRight);

        m_revARBLeftRight->GetForce_Rz().SetActive(1);
        m_revARBLeftRight->GetForce_Rz().SetK(getARBStiffness());
        m_revARBLeftRight->GetForce_Rz().SetR(getARBDamping());
    }

    m_slideARB[side] = chrono_types::make_shared<ChVehicleJoint>(
        ChVehicleJoint::Type::POINTPLANE, m_name + "_revARBslide" + suffix, m_arb[side], m_axleTube,
        ChCoordsys<>(m_ptARBAxle[side], chassisRot * QUNIT));
    chassis->AddJoint(m_slideARB[side]);
}

void ChGAxle::InitializeInertiaProperties() {
    m_mass = getAxleTubeMass() + getPanhardRodMass() + 2 * (getARBMass() + getSpindleMass() + getLongLinkMass());
}

void ChGAxle::UpdateInertiaProperties() {
    m_parent->GetTransform().TransformLocalToParent(ChFrame<>(m_rel_loc, QUNIT), m_xform);

    // Calculate COM and inertia expressed in global frame
    utils::CompositeInertia composite;
    composite.AddComponent(m_axleTube->GetFrame_COG_to_abs(), m_axleTube->GetMass(), m_axleTube->GetInertia());
    composite.AddComponent(m_panhardRod->GetFrame_COG_to_abs(), m_panhardRod->GetMass(), m_panhardRod->GetInertia());
    composite.AddComponent(m_spindle[LEFT]->GetFrame_COG_to_abs(), m_spindle[LEFT]->GetMass(),
                           m_spindle[LEFT]->GetInertia());
    composite.AddComponent(m_spindle[RIGHT]->GetFrame_COG_to_abs(), m_spindle[RIGHT]->GetMass(),
                           m_spindle[RIGHT]->GetInertia());
    composite.AddComponent(m_longLink[LEFT]->GetFrame_COG_to_abs(), m_longLink[LEFT]->GetMass(),
                           m_longLink[LEFT]->GetInertia());
    composite.AddComponent(m_longLink[RIGHT]->GetFrame_COG_to_abs(), m_longLink[RIGHT]->GetMass(),
                           m_longLink[RIGHT]->GetInertia());
    composite.AddComponent(m_arb[LEFT]->GetFrame_COG_to_abs(), m_arb[LEFT]->GetMass(), m_arb[LEFT]->GetInertia());
    composite.AddComponent(m_arb[RIGHT]->GetFrame_COG_to_abs(), m_arb[RIGHT]->GetMass(), m_arb[RIGHT]->GetInertia());

    // Express COM and inertia in subsystem reference frame
    m_com.coord.pos = m_xform.TransformPointParentToLocal(composite.GetCOM());
    m_com.coord.rot = QUNIT;

    m_inertia = m_xform.GetA().transpose() * composite.GetInertia() * m_xform.GetA();
}

// -----------------------------------------------------------------------------
// Get the wheel track using the spindle local position.
// -----------------------------------------------------------------------------
double ChGAxle::GetTrack() {
    return 2 * getLocation(SPINDLE).y();
}

// -----------------------------------------------------------------------------
// Return current suspension forces
// -----------------------------------------------------------------------------
std::vector<ChSuspension::ForceTSDA> ChGAxle::ReportSuspensionForce(VehicleSide side) const {
    std::vector<ChSuspension::ForceTSDA> forces(2);

    forces[0] = ChSuspension::ForceTSDA("Spring", m_spring[side]->GetForce(), m_spring[side]->GetLength(),
                                        m_spring[side]->GetVelocity());
    forces[1] = ChSuspension::ForceTSDA("Shock", m_shock[side]->GetForce(), m_shock[side]->GetLength(),
                                        m_shock[side]->GetVelocity());

    return forces;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void ChGAxle::LogHardpointLocations(const ChVector<>& ref, bool inches) {
    double unit = inches ? 1 / 0.0254 : 1.0;

    for (int i = 0; i < NUM_POINTS; i++) {
        ChVector<> pos = ref + unit * getLocation(static_cast<PointId>(i));

        GetLog() << "   " << m_pointNames[i].c_str() << "  " << pos.x() << "  " << pos.y() << "  " << pos.z() << "\n";
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void ChGAxle::LogConstraintViolations(VehicleSide side) {
    {
        ChVectorDynamic<> C = m_axleTubeGuide->GetConstraintViolation();
        GetLog() << "Axle tube prismatic       ";
        GetLog() << "  " << C(0) << "  ";
        GetLog() << "  " << C(1) << "  ";
        GetLog() << "  " << C(2) << "\n";
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void ChGAxle::AddVisualizationAssets(VisualizationType vis) {
    ChSuspension::AddVisualizationAssets(vis);

    if (vis == VisualizationType::NONE)
        return;

    AddVisualizationLink(m_axleTube, m_axleOuterL, m_axleOuterR, getAxleTubeRadius(), ChColor(0.7f, 0.7f, 0.7f));
    AddVisualizationLink(m_panhardRod, m_ptPanhardChassis, m_ptPanhardAxle, getPanhardRodRadius(),
                         ChColor(0.5f, 0.7f, 0.3f));

    AddVisualizationLink(m_longLink[LEFT], m_ptLongLinkAxle[LEFT], m_ptLongLinkChassis[LEFT], getLongLinkRadius(),
                         ChColor(0.3f, 0.7f, 0.5f));
    AddVisualizationLink(m_longLink[RIGHT], m_ptLongLinkAxle[RIGHT], m_ptLongLinkChassis[RIGHT], getLongLinkRadius(),
                         ChColor(0.3f, 0.7f, 0.5f));

    AddVisualizationLink(m_arb[LEFT], m_ptARBAxle[LEFT], m_ptARBChassis[LEFT], getARBRadius(),
                         ChColor(0.5f, 7.0f, 0.5f));
    AddVisualizationLink(m_arb[LEFT], m_ptARBCenter, m_ptARBChassis[LEFT], getARBRadius(), ChColor(0.5f, 0.7f, 0.5f));

    AddVisualizationLink(m_arb[RIGHT], m_ptARBAxle[RIGHT], m_ptARBChassis[RIGHT], getARBRadius(),
                         ChColor(0.7f, 0.5f, 0.5f));
    AddVisualizationLink(m_arb[RIGHT], m_ptARBCenter, m_ptARBChassis[RIGHT], getARBRadius(), ChColor(0.7f, 0.5f, 0.5f));

    // Add visualization for the springs and shocks
    m_spring[LEFT]->AddVisualShape(chrono_types::make_shared<ChSpringShape>(0.06, 150, 15));
    m_spring[RIGHT]->AddVisualShape(chrono_types::make_shared<ChSpringShape>(0.06, 150, 15));

    m_shock[LEFT]->AddVisualShape(chrono_types::make_shared<ChSegmentShape>());
    m_shock[RIGHT]->AddVisualShape(chrono_types::make_shared<ChSegmentShape>());
}

void ChGAxle::RemoveVisualizationAssets() {
    ChPart::RemoveVisualizationAssets(m_axleTube);
    ChPart::RemoveVisualizationAssets(m_panhardRod);

    ChPart::RemoveVisualizationAssets(m_spring[LEFT]);
    ChPart::RemoveVisualizationAssets(m_spring[RIGHT]);

    ChPart::RemoveVisualizationAssets(m_shock[LEFT]);
    ChPart::RemoveVisualizationAssets(m_shock[RIGHT]);

    ChSuspension::RemoveVisualizationAssets();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void ChGAxle::AddVisualizationLink(std::shared_ptr<ChBody> body,
                                   const ChVector<> pt_1,
                                   const ChVector<> pt_2,
                                   double radius,
                                   const ChColor& color) {
    // Express hardpoint locations in body frame.
    ChVector<> p_1 = body->TransformPointParentToLocal(pt_1);
    ChVector<> p_2 = body->TransformPointParentToLocal(pt_2);

    auto cyl = chrono_types::make_shared<ChCylinderShape>();
    cyl->GetCylinderGeometry().p1 = p_1;
    cyl->GetCylinderGeometry().p2 = p_2;
    cyl->GetCylinderGeometry().rad = radius;
    cyl->SetColor(color);
    body->AddVisualShape(cyl);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void ChGAxle::ExportComponentList(rapidjson::Document& jsonDocument) const {
    ChPart::ExportComponentList(jsonDocument);

    std::vector<std::shared_ptr<ChBody>> bodies;
    bodies.push_back(m_spindle[0]);
    bodies.push_back(m_spindle[1]);
    bodies.push_back(m_axleTube);
    ChPart::ExportBodyList(jsonDocument, bodies);

    std::vector<std::shared_ptr<ChShaft>> shafts;
    shafts.push_back(m_axle[0]);
    shafts.push_back(m_axle[1]);
    ChPart::ExportShaftList(jsonDocument, shafts);

    std::vector<std::shared_ptr<ChLink>> joints;
    joints.push_back(m_revolute[0]);
    joints.push_back(m_revolute[1]);
    ChPart::ExportJointList(jsonDocument, joints);

    std::vector<std::shared_ptr<ChLinkTSDA>> springs;
    springs.push_back(m_spring[0]);
    springs.push_back(m_spring[1]);
    springs.push_back(m_shock[0]);
    springs.push_back(m_shock[1]);
    ChPart::ExportLinSpringList(jsonDocument, springs);
}

void ChGAxle::Output(ChVehicleOutput& database) const {
    if (!m_output)
        return;

    std::vector<std::shared_ptr<ChBody>> bodies;
    bodies.push_back(m_spindle[0]);
    bodies.push_back(m_spindle[1]);
    bodies.push_back(m_axleTube);
    database.WriteBodies(bodies);

    std::vector<std::shared_ptr<ChShaft>> shafts;
    shafts.push_back(m_axle[0]);
    shafts.push_back(m_axle[1]);
    database.WriteShafts(shafts);

    std::vector<std::shared_ptr<ChLink>> joints;
    joints.push_back(m_revolute[0]);
    joints.push_back(m_revolute[1]);
    database.WriteJoints(joints);

    std::vector<std::shared_ptr<ChLinkTSDA>> springs;
    springs.push_back(m_spring[0]);
    springs.push_back(m_spring[1]);
    springs.push_back(m_shock[0]);
    springs.push_back(m_shock[1]);
    database.WriteLinSprings(springs);
}

}  // end namespace vehicle
}  // end namespace chrono
