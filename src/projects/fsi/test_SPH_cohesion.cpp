// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2022 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Deniz Tanyildiz, Radu Serban
// =============================================================================
//
// Object drop on SPH terrain
//
// =============================================================================

#include <cstdio>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <fstream>

#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/ChVisualizationFsi.h"

#include "chrono_vehicle/ChVehicleModelData.h"

#include "chrono_thirdparty/cxxopts/ChCLI.h"
#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::fsi;
using namespace chrono::vehicle;

using std::cout;
using std::cin;
using std::endl;

// ===================================================================================================================

#ifdef CHRONO_OPENGL

class SimStats : public opengl::ChOpenGLStats {
  public:
    SimStats(const ChSystemFsi& sysFSI) : ChOpenGLStats(), m_sys(sysFSI) {}
    virtual void GenerateStats(ChSystem& sys) override {
        char buffer[150];
        sprintf(buffer, "TIME:     %.4f s", m_sys.GetSimTime());
        text.Render(buffer, screen.LEFT, screen.TOP - 1 * screen.SPACING, screen.SX, screen.SY);
    }

    const ChSystemFsi& m_sys;
};

class ParticleSelector : public opengl::ChOpenGLParticleCB {
  public:
    ParticleSelector() {}
    virtual bool Render(const ChVector<>& pos) const override { return pos.y() > 0; }
};

#endif

// ===================================================================================================================

std::shared_ptr<chrono::ChBody> CreateSolids(chrono::ChSystemNSC& sys,
                                             chrono::fsi::ChSystemFsi& sysFSI,
                                             const ChVector<>& b_size) {
    // Set common material Properties
    auto mat = chrono_types::make_shared<ChMaterialSurfaceNSC>();
    mat->SetFriction(0.2f);
    mat->SetRestitution(0.05f);

    // Get particle spacing in the simulation
    auto init_spacing = sysFSI.GetInitialSpacing();

    // Create bottom plate
    double factor = 3.0;
    double gxDim = factor * b_size.x();
    double gyDim = factor * b_size.y();
    double gzDim = 0.1;
    double top_z = -3 * init_spacing;

    auto ground = chrono_types::make_shared<ChBodyEasyBox>(gxDim, gyDim, gzDim, 1000, true, true, mat);
    ground->SetPos(ChVector<>(0.0, 0.0, top_z - 0.5 * gzDim));
    ground->SetCollide(true);
    ground->SetBodyFixed(true);
    sys.AddBody(ground);

    // Add BCE markers
    sysFSI.AddBoxBCE(ground, ChFrame<>(), ChVector<>(gxDim, gyDim, gzDim), false);

    // Create a falling sphere
    double sphere_radius = 0.25;
    double volume = chrono::utils::CalcSphereVolume(sphere_radius);
    double density = 4000;
    double mass = density * volume;
    ChVector<> sphere_pos = ChVector<>(0.0, 0, b_size.z() + sphere_radius + 2 * init_spacing);
    ChVector<> sphere_vel = ChVector<>(0.0, 0.0, 0.0);
    ChVector<> gyration = chrono::utils::CalcSphereGyration(sphere_radius).diagonal();

    auto sphere = chrono_types::make_shared<ChBody>();
    sphere->SetPos(sphere_pos);
    sphere->SetPos_dt(sphere_vel);
    sphere->SetMass(mass);
    sphere->SetInertiaXX(mass * gyration);
    sphere->SetCollide(true);
    sphere->SetBodyFixed(false);
    sphere->GetCollisionModel()->ClearModel();
    chrono::utils::AddSphereGeometry(sphere.get(), mat, sphere_radius, VNULL, QUNIT);
    sphere->GetCollisionModel()->BuildModel();
    sys.AddBody(sphere);

    // Add this body to the FSI system and create BCEs
    sysFSI.AddFsiBody(sphere);
    sysFSI.AddSphereBCE(sphere, ChFrame<>(), sphere_radius, true);

    return sphere;
}

int main(int argc, char* argv[]) {
    double tend = 1.0;
    double step_size = 5e-4;
    double output_fps = 100;
    double render_fps = 50;

    bool render = true;
    bool output = true;

    // Create the Chrono systems
    ChSystemNSC sys;
    ChSystemFsi sysFSI(sys);

    const ChVector<> gravity(0, 0, -9.81);
    sysFSI.Set_G_acc(gravity);
    sys.Set_G_acc(gravity);

    // Soil material properties
    double rho = 1700.0;
    double friction = 0.1;
    double cohesion = 1.0e4;

    ChSystemFsi::ElasticMaterialProperties mat_props;
    mat_props.Young_modulus = 1e6;
    mat_props.Poisson_ratio = 0.3;
    mat_props.stress = 0;  // default
    mat_props.viscosity_alpha = 0.5;
    mat_props.viscosity_beta = 0.0;
    mat_props.mu_I0 = 0.03;
    mat_props.mu_fric_s = friction;
    mat_props.mu_fric_2 = friction;
    mat_props.average_diam = 0.005;
    mat_props.friction_angle = CH_C_PI / 10;  // default
    mat_props.dilation_angle = CH_C_PI / 10;  // default
    mat_props.cohesion_coeff = 0;             // default
    mat_props.kernel_threshold = 0.8;

    sysFSI.SetElasticSPH(mat_props);
    sysFSI.SetDensity(rho);
    sysFSI.SetCohesionForce(cohesion);

    double init_spacing = 0.02;
    double kernel_length = 0.03;
    sysFSI.SetInitialSpacing(init_spacing);
    sysFSI.SetKernelLength(kernel_length);

    // Dimension of soil block
    ChVector<> b_size(1.0, 1.0, 0.5);
    sysFSI.SetContainerDim(b_size);
    sysFSI.SetDiscreType(false, false);
    sysFSI.SetWallBC(BceVersion::ADAMI);
    sysFSI.SetRigidBodyBC(BceVersion::ADAMI);
    sysFSI.SetSPHMethod(FluidDynamics::WCSPH);
    sysFSI.SetOutputLength(0);
    sysFSI.SetVerbose(false);

    sysFSI.SetStepSize(step_size);
    sysFSI.SetMaxStepSize(step_size);

    // Set up the periodic boundary condition (if not, set relative larger values)
    sysFSI.SetBoundaries(-5.0 * b_size, +5.0 * b_size);

    // Initialize the SPH particles
    sysFSI.AddBoxSPH(ChVector<>(0.0, 0.0, b_size.z() / 2), b_size / 2);

    // Create bottom plate and dropping sphere with BCE markers
    auto sphere = CreateSolids(sys, sysFSI, b_size);

#ifdef CHRONO_OPENGL
    // Create run-time visualization
    opengl::ChVisualSystemOpenGL vis;
    ChVisualizationFsi visFSI(&sysFSI, &vis);
    visFSI.SetTitle("Object drop test");
    visFSI.SetSize(1280, 720);
    visFSI.SetCameraPosition(ChVector<>(0, -2 * b_size.y(), b_size.z()), ChVector<>(0, 0, b_size.z()));
    visFSI.SetCameraMoveScale(0.02f);
    visFSI.EnableFluidMarkers(true);
    visFSI.EnableRigidBodyMarkers(false);
    visFSI.EnableBoundaryMarkers(false);
    visFSI.SetRenderMode(ChVisualizationFsi::RenderMode::SOLID);

    auto stats = chrono_types::make_shared<SimStats>(sysFSI);
    vis.AttachStatsRenderer(stats);
    vis.EnableStats(true);

    auto sel = chrono_types::make_shared<ParticleSelector>();
    vis.AttachParticleSelector(sel);

    vis.AttachSystem(&sys);
    vis.Initialize();
#endif

    // Output directory
    std::string out_dir = GetChronoOutputPath() + "COHESION_SPH/";
    std::string vis_dir = out_dir + "f" + std::to_string(friction) + "_c" + std::to_string(cohesion) + "/";
    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        cout << "Error creating directory " << out_dir << endl;
        return 1;
    }
    if (!filesystem::create_directory(filesystem::path(vis_dir))) {
        cout << "Error creating directory " << vis_dir << endl;
        return 1;
    }

    // Simulation loop
    int output_steps = (int)std::round((1.0 / output_fps) / step_size);
    int render_steps = (int)std::round((1.0 / render_fps) / step_size);
    int vis_output_frame = 0;

    double t = 0;
    int frame = 0;
    sysFSI.Initialize();
#ifdef CHRONO_OPENGL
    visFSI.Initialize();
#endif

    while (t < tend) {
        // Visualization data output
        if (output && frame % output_steps == 0) {
            sysFSI.PrintParticleToFile(vis_dir);

            auto psph = sphere->GetPos();
            std::ofstream fsph;
            fsph.open(vis_dir + "sphere" + std::to_string(vis_output_frame) + ".csv");
            fsph << "x, y, z, |U|" << std::endl;
            fsph << psph.x() << ", " << psph.y() << ", " << psph.z() << ", " << sphere->GetPos_dt().Length()
                 << std::endl;
            fsph.close();

            vis_output_frame++;
        }

#ifdef CHRONO_OPENGL
        // Run-time visualization
        if (render && frame % render_steps == 0) {
            if (!visFSI.Render())
                break;
        }
#endif

        // Advance both FSI and embedded MBD systems
        sysFSI.DoStepDynamics_FSI();
        t += step_size;

        frame++;
    }

    return 0;
}
