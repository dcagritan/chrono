#include <cmath>
#include <iostream>

#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_multicore/physics/ChSystemMulticore.h"

#include "chrono_distributed/collision/ChBoundary.h"
#include "chrono_distributed/collision/ChCollisionModelDistributed.h"

#include "chrono_opengl/ChVisualSystemOpenGL.h"

using namespace chrono;
using namespace chrono::collision;

// Granular material properties
float Y = 2e6f;
float mu = 0.4f;
float cr = 0.05f;
double gran_radius = 0.0025;  // 2.5mm radius
double rho = 4000;
double mass = 4.0 / 3.0 * CH_C_PI * gran_radius * gran_radius * gran_radius;
ChVector<> inertia = (2.0 / 5.0) * mass * gran_radius * gran_radius * ChVector<>(1, 1, 1);
double spacing = 4.0 * gran_radius;

// Dimensions
double hy = 20 * gran_radius;                // Half y dimension
double height = 50 * gran_radius;            // Height of the box
double slope_angle = CH_C_PI / 8.0;          // Angle of sloped wall from the horizontal
double dx = height / std::tan(slope_angle);  // x width of slope
double settling_gap = 0.0 * gran_radius;     // Width of opening of the hopper during settling phase
double pouring_gap = 4.0 * gran_radius;      // Witdth of opening of the hopper during pouring phase

// Simulation
double time_step = 1e-5;
unsigned int max_iteration = 100;
double tolerance = 1e-4;

std::shared_ptr<ChBoundary> AddContainer(ChSystemMulticore* sys) {
    int binId = -200;

    auto mat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    mat->SetYoungModulus(Y);
    mat->SetFriction(mu);
    mat->SetRestitution(cr);

    auto bin = chrono_types::make_shared<ChBody>(chrono_types::make_shared<ChCollisionModelDistributed>());
    bin->SetIdentifier(binId);
    bin->SetMass(1);
    bin->SetPos(ChVector<>(0, 0, 0));
    bin->SetCollide(true);
    bin->SetBodyFixed(true);
    sys->AddBody(bin);

    auto cb = std::shared_ptr<ChBoundary>(new ChBoundary(bin, mat));

    // Sloped Wall
    cb->AddPlane(ChFrame<>(ChVector<>(settling_gap + dx / 2, 0, height / 2), Q_from_AngY(-slope_angle)),
                 ChVector2<>(std::sqrt(dx * dx + height * height), 2 * hy));

    // Vertical wall
    cb->AddPlane(ChFrame<>(ChVector<>(0, 0, height / 2), Q_from_AngY(CH_C_PI_2)), ChVector2<>(height, 2 * hy));

    // Parallel vertical walls
    cb->AddPlane(ChFrame<>(ChVector<>((settling_gap + dx) / 2, -hy, height / 2), Q_from_AngX(-CH_C_PI_2)),
                 ChVector2<>(settling_gap + dx, height));
    cb->AddPlane(ChFrame<>(ChVector<>((settling_gap + dx) / 2, hy, height / 2), Q_from_AngX(CH_C_PI_2)),
                 ChVector2<>(settling_gap + dx, height));

    cb->AddVisualization(0, 3 * gran_radius);
    cb->AddVisualization(1, 3 * gran_radius);

    return cb;
}

size_t AddFallingBalls(ChSystemMulticore* sys) {
    ChVector<double> box_center((settling_gap + dx / 2) / 2, 0, 3 * height / 4);
    ChVector<double> h_dims = ChVector<>((settling_gap + dx / 2) / 2, hy, height / 4) - 3 * gran_radius;

    auto ballMat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    ballMat->SetYoungModulus(Y);
    ballMat->SetFriction(mu);
    ballMat->SetRestitution(cr);
    ballMat->SetAdhesion(0);

    utils::Generator gen(sys);
    std::shared_ptr<utils::MixtureIngredient> m1 = gen.AddMixtureIngredient(utils::MixtureType::SPHERE, 1.0);
    m1->setDefaultMaterial(ballMat);
    m1->setDefaultDensity(rho);
    m1->setDefaultSize(gran_radius);

    class MyFilter : public utils::Generator::CreateObjectsCallback {
      public:
        MyFilter(ChSystemMulticore* system, const ChVector<>& center) : m_system(system), m_center(center) {}

        unsigned int GetNumPoints() const { return m_num_points; }

        virtual void OnCreateObjects(const utils::PointVectorD& points, std::vector<bool>& flags) override {
            m_num_points = points.size();
            for (auto i = 0; i < points.size(); i++) {
                if (points[i].x() > m_center.x() && points[i].y() > m_center.y()) {
                    flags[i] = false;
                    m_num_points--;
                }
            }
        }

      private:
        ChSystemMulticore* m_system;
        ChVector<> m_center;
        size_t m_num_points;
    };

    auto filter = chrono_types::make_shared<MyFilter>(sys, box_center);
    gen.RegisterCreateObjectsCallback(filter);

    gen.setBodyIdentifier(0);

    utils::HCPSampler<> sampler(spacing);
    gen.CreateObjectsBox(sampler, box_center, h_dims);

    return gen.getTotalNumBodies();
}

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    int threads = 2;

    // Create system
    ChSystemMulticoreSMC my_sys;
    my_sys.Set_G_acc(ChVector<>(0, 0, -9.8));

    // Set number of threads.
    int max_threads = ChOMP::GetNumProcs();
    if (threads > max_threads)
        threads = max_threads;
   my_sys.SetNumThreads(threads);

    // Set solver parameters
    my_sys.GetSettings()->solver.max_iteration_bilateral = max_iteration;
    my_sys.GetSettings()->solver.tolerance = tolerance;

    my_sys.GetSettings()->solver.contact_force_model = ChSystemSMC::ContactForceModel::Hooke;
    my_sys.GetSettings()->solver.adhesion_force_model = ChSystemSMC::AdhesionForceModel::Constant;

    my_sys.GetSettings()->collision.narrowphase_algorithm = ChNarrowphase::Algorithm::MPR;
    my_sys.GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

    // Create objects
    auto cb = AddContainer(&my_sys);
    auto num_bodies = AddFallingBalls(&my_sys);
    std::cout << "Created " << num_bodies << " balls." << std::endl;

    // Perform the simulation
    opengl::ChVisualSystemOpenGL vis;
    vis.AttachSystem(&my_sys);
    vis.SetWindowTitle("Test");
    vis.SetWindowSize(1280, 720);
    vis.SetRenderMode(opengl::WIREFRAME);
    vis.Initialize();
    vis.AddCamera(ChVector<>(0, -100 * gran_radius, 0), ChVector<>(0, 0, 0));
    vis.SetCameraVertical(CameraVerticalDir::Z);

    bool moved = false;
    while (vis.Run()) {
        double time = my_sys.GetChTime();
        if (!moved && time > 0.25) {
            cb->UpdatePlane(0, ChFrame<>(ChVector<>(pouring_gap + dx / 2, 0, height / 2), Q_from_AngY(-slope_angle)));
            moved = true;
        }

        my_sys.DoStepDynamics(time_step);
        vis.Render();
    }

    return 0;
}
