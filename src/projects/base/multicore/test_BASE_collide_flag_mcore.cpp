#include <omp.h>
#include <cstdio>
#include <vector>
#include <cmath>

#include "chrono/ChConfig.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono_multicore/physics/ChSystemMulticore.h"

#include "chrono_opengl/ChVisualSystemOpenGL.h"

using namespace chrono;
using namespace chrono::collision;

int main(int argc, char* argv[]) {
    // BASIC SETUP
    double time_step = 1e-3;
    double time_end = 5;

    double out_fps = 50;

    double tolerance = 1e-3;
    unsigned int max_iteration = 100;

    // Create DEM system
    ChSystemMulticoreSMC my_sys;

    int num_threads = 1;
    my_sys.SetNumThreads(num_threads);

    my_sys.Set_G_acc(ChVector<double>(0, 0, -9.8));

    // Set solver parameters
    my_sys.GetSettings()->solver.max_iteration_bilateral = max_iteration;
    my_sys.GetSettings()->solver.tolerance = tolerance;

    my_sys.GetSettings()->collision.narrowphase_algorithm = ChNarrowphase::Algorithm::PRIMS;
    my_sys.GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

    my_sys.GetSettings()->solver.contact_force_model = ChSystemSMC::ContactForceModel::Hertz;
    my_sys.GetSettings()->solver.adhesion_force_model = ChSystemSMC::AdhesionForceModel::Constant;

    // Common material
    float Y = 2e6f;
    float mu = 0.4f;
    float cr = 0.4f;
    auto contact_mat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    contact_mat->SetYoungModulus(Y);
    contact_mat->SetFriction(mu);
    contact_mat->SetRestitution(cr);
    contact_mat->SetAdhesion(0);  // Magnitude of the adhesion in Constant adhesion model

    // Create the falling balls
    double mass = 1;
    double radius = 0.15;
    ChVector<> inertia = (2.0 / 5.0) * mass * radius * radius * ChVector<>(1, 1, 1);

    // Lower ball
    auto ball_lower = chrono_types::make_shared<ChBody>(ChCollisionSystemType::CHRONO);

    ball_lower->SetIdentifier(1);
    ball_lower->SetMass(mass);
    ball_lower->SetInertiaXX(inertia);
    ball_lower->SetPos(ChVector<>(0, 0, 10));
    ball_lower->SetRot(ChQuaternion<>(1, 0, 0, 0));
    ball_lower->SetBodyFixed(false);
    ball_lower->SetCollide(true);

    ball_lower->GetCollisionModel()->ClearModel();
    utils::AddSphereGeometry(ball_lower.get(), contact_mat, radius);
    ball_lower->GetCollisionModel()->BuildModel();

    my_sys.AddBody(ball_lower);

    // Upper ball
    auto ball_upper = chrono_types::make_shared<ChBody>(ChCollisionSystemType::CHRONO);

    ball_upper->SetIdentifier(2);
    ball_upper->SetMass(mass);
    ball_upper->SetInertiaXX(inertia);
    ball_upper->SetPos(ChVector<>(0, 0, 11));
    ball_upper->SetRot(ChQuaternion<>(1, 0, 0, 0));
    ball_upper->SetBodyFixed(false);
    ball_upper->SetCollide(true);

    ball_upper->GetCollisionModel()->ClearModel();
    utils::AddSphereGeometry(ball_upper.get(), contact_mat, radius);
    ball_upper->GetCollisionModel()->BuildModel();

    my_sys.AddBody(ball_upper);

    // Plate
    auto plate = chrono_types::make_shared<ChBody>(ChCollisionSystemType::CHRONO);

    plate->SetIdentifier(0);
    plate->SetPos(ChVector<>(0, 0, 8));
    plate->SetBodyFixed(true);
    plate->SetCollide(true);

    plate->GetCollisionModel()->ClearModel();
    utils::AddBoxGeometry(plate.get(), contact_mat, ChVector<>(4 * radius, 4 * radius, radius));
    plate->GetCollisionModel()->BuildModel();

    my_sys.AddBody(plate);

    // Create the visualization window
    opengl::ChVisualSystemOpenGL vis;
    vis.AttachSystem(&my_sys);
    vis.SetWindowTitle("Test");
    vis.SetWindowSize(1280, 720);
    vis.SetRenderMode(opengl::WIREFRAME);
    vis.Initialize();
    vis.AddCamera(ChVector<>(0, -4, 10), ChVector<>(0, 0, 10));
    vis.SetCameraVertical(CameraVerticalDir::Z);

    // Run simulation for specified time
    int out_steps = static_cast<int>(std::ceil((1 / time_step) / out_fps));
    double time = 0;
    int sim_frame = 0;
    while (my_sys.GetChTime() < time_end) {
        double z_lower = ball_lower->GetPos().z();
        double z_upper = ball_upper->GetPos().z();
        double vz_upper = ball_upper->GetPos_dt().z();

        if (sim_frame % out_steps == 0) {
            std::cout << "t: " << my_sys.GetChTime() << "  z_lower: " << z_lower << "  z_upper: " << z_upper
                      << std::endl;
        }

        my_sys.DoStepDynamics(time_step);
        sim_frame++;

        // Fix lower ball to ground
        if (sim_frame == 200) {
            std::cout << "------- Setting lower body to fixed" << std::endl;
            ball_lower->SetBodyFixed(true);
        }

        // After first interaction, disable contact on lower ball
        if (ball_lower->GetCollide() && vz_upper > 0 && z_upper > z_lower + 2 * radius) {
            std::cout << "------- Setting lower body to not collide" << std::endl;
            ball_lower->SetCollide(false);
        }

        if (vis.Run()) {
            vis.Render();
        } else {
            return 1;
        }
    }

    return 0;
}
