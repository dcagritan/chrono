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

#ifndef ROBO_SIMIAN_H
#define ROBO_SIMIAN_H

#include <string>
#include <unordered_map>
#include <vector>

#include "chrono/assets/ChColor.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/physics/ChLinkMotorRotationAngle.h"

namespace robosimian {

enum LimbID {
    FR = 0,  ///< front right
    RR = 1,  ///< rear right
    RL = 2,  ///< rear left
    FL = 3   ///< front left
};

enum class VisualizationType {
    NONE,        ///< no visualization
    PRIMITIVES,  ///< use primitve shapes
    MESH         ///< use meshes
};

enum class ActuationMode {
    ANGLE,  ///< prescribe time-series for joint angle
    SPEED,  ///< prescribe time-series for joint angular speed
    TORQUE  ///< prescribe time-series for joint torque
};

struct BoxShape {
    BoxShape(const chrono::ChVector<>& pos, const chrono::ChQuaternion<>& rot, const chrono::ChVector<>& dims)
        : m_pos(pos), m_rot(rot), m_dims(dims) {}
    chrono::ChVector<> m_pos;
    chrono::ChQuaternion<> m_rot;
    chrono::ChVector<> m_dims;
};

struct SphereShape {
    SphereShape(const chrono::ChVector<>& pos, double radius) : m_pos(pos), m_radius(radius) {}
    chrono::ChVector<> m_pos;
    double m_radius;
};

struct CylinderShape {
    CylinderShape(const chrono::ChVector<>& pos, const chrono::ChQuaternion<>& rot, double radius, double length)
        : m_pos(pos), m_rot(rot), m_radius(radius), m_length(length) {}
    chrono::ChVector<> m_pos;
    chrono::ChQuaternion<> m_rot;
    double m_radius;
    double m_length;
};

class Part {
  public:
    Part(const std::string& name, chrono::ChSystem* system);
    virtual ~Part() {}

    const std::string& GetName() const { return m_name; }
    void SetName(const std::string& name) { m_name = name; }
    void SetVisualizationType(VisualizationType vis);

  protected:
    void AddVisualizationAssets(VisualizationType vis);
    void AddCollisionShapes(int collision_family);

    std::string m_name;                            ///< subsystem name
    std::shared_ptr<chrono::ChBodyAuxRef> m_body;  ///< rigid body
    std::vector<BoxShape> m_boxes;                 ///< set of primitive boxes (collision and visualization)
    std::vector<SphereShape> m_spheres;            ///< set of primitive spheres (collision and visualization)
    std::vector<CylinderShape> m_cylinders;        ///< set of primitive cylinders (collision and visualization)
    std::string m_mesh_name;                       ///< visualization mesh name
    chrono::ChVector<> m_offset;                   ///< offset for visualization mesh
    chrono::ChColor m_color;                       ///< visualization asset color

    friend class RoboSimian;
    friend class Limb;
};

class Chassis : public Part {
  public:
    Chassis(const std::string& name, chrono::ChSystem* system, bool fixed);
    ~Chassis() {}

    void Initialize(const chrono::ChCoordsys<>& pos);

  private:
};

class WheelDD : public Part {
  public:
    WheelDD(const std::string& name, int id, chrono::ChSystem* system);
    ~WheelDD() {}

    void Initialize(std::shared_ptr<chrono::ChBodyAuxRef> chassis,  ///< chassis body
                    const chrono::ChVector<>& xyz,                  ///< location (relative to chassis)
                    const chrono::ChVector<>& rpy                   ///< roll-pitch-yaw (relative to chassis)
    );
};

class Link {
  public:
    Link(const std::string& mesh_name,
         const chrono::ChVector<>& offset,
         const chrono::ChColor& color,
         double mass,
         const chrono::ChVector<>& com,
         const chrono::ChVector<>& inertia_xx,
         const chrono::ChVector<>& inertia_xy,
         const std::vector<CylinderShape>& shapes)
        : m_mesh_name(mesh_name),
          m_offset(offset),
          m_color(color),
          m_mass(mass),
          m_com(com),
          m_inertia_xx(inertia_xx),
          m_inertia_xy(inertia_xy),
          m_shapes(shapes) {}

  private:
    std::string m_mesh_name;
    chrono::ChVector<> m_offset;
    chrono::ChColor m_color;
    double m_mass;
    chrono::ChVector<> m_com;
    chrono::ChVector<> m_inertia_xx;
    chrono::ChVector<> m_inertia_xy;
    std::vector<CylinderShape> m_shapes;

    friend class Limb;
};

struct LinkData {
    std::string name;
    Link link;
    bool include;
};

struct JointData {
    std::string name;
    std::string linkA;
    std::string linkB;
    bool fixed;
    chrono::ChVector<> xyz;
    chrono::ChVector<> rpy;
    chrono::ChVector<> axis;
};

class Limb {
  public:
    Limb(const std::string& name, LimbID id, const LinkData data[], chrono::ChSystem* system);
    ~Limb() {}

    void Initialize(std::shared_ptr<chrono::ChBodyAuxRef> chassis,  ///< chassis body
                    const chrono::ChVector<>& xyz,                  ///< location (relative to chassis)
                    const chrono::ChVector<>& rpy,                  ///< roll-pitch-yaw (relative to chassis)
                    ActuationMode mode                              ///< actuation mode (angle, speed, or torque)
    );

    void SetVisualizationType(VisualizationType vis);

    void Activate(const std::string& motor_name, double time, double val);

  private:
    LimbID m_id;
    std::string m_name;
    std::unordered_map<std::string, std::shared_ptr<Part>> m_links;
    std::unordered_map<std::string, std::shared_ptr<chrono::ChLink>> m_joints;
    std::unordered_map<std::string, std::shared_ptr<chrono::ChLinkMotorRotation>> m_motors;
};

class ContactManager;

class RoboSimian {
  public:
    RoboSimian(chrono::ChMaterialSurface::ContactMethod contact_method, bool fixed = false);
    RoboSimian(chrono::ChSystem* system, bool fixed = false);
    ~RoboSimian();

    chrono::ChSystem* GetSystem() { return m_system; }

    void SetActuationMode(ActuationMode mode) { m_mode = mode; }

    void SetVisualizationTypeChassis(VisualizationType vis);
    void SetVisualizationTypeLimbs(VisualizationType vis);
    void SetVisualizationTypeLimb(LimbID id, VisualizationType vis);
    void SetVisualizationTypeWheels(VisualizationType vis);

    void Initialize(const chrono::ChCoordsys<>& pos);

    void Activate(LimbID id, const std::string& motor_name, double time, double val);

    void ReportContacts();

  private:
    void Create(bool fixed);

    chrono::ChSystem* m_system;  ///< pointer to the Chrono system
    bool m_owns_system;          ///< true if system created at construction

    ActuationMode m_mode;

    std::shared_ptr<Chassis> m_chassis;          ///< robot chassis
    std::vector<std::shared_ptr<Limb>> m_limbs;  ///< robot limbs
    ////std::shared_ptr<WheelDD> m_wheel_left;       ///< left DD wheel
    ////std::shared_ptr<WheelDD> m_wheel_right;      ///< right DD wheel

    ContactManager* m_contacts;
};

}  // end namespace robosimian

#endif