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
// Author: Radu Serban
// =============================================================================
//
// Output data writer for Polaris on SPH terrain system
//
// =============================================================================

#pragma once

#include <array>
#include <fstream>

#include "chrono_vehicle/ChTerrain.h"
#include "chrono_vehicle/terrain/SCMTerrain.h"

#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"
#include "chrono/utils/ChFilters.h"

class DataWriter {
  public:
    virtual ~DataWriter();
    /// Initialize the data writer, specifying the output directory and output frequency parameters.
    void Initialize(const std::string& dir, double step_size);
    void Process(int sim_frame, double time, const std::array<chrono::vehicle::TerrainForce, 4>& tire_forces);
    std::array<chrono::vehicle::TerrainForce, 4> m_tire_forces;
  protected:
    /// Construct an output data writer for the specified system.
    DataWriter(chrono::ChSystem* sys);
    virtual int GetNumChannelsMBS() const = 0;
    /// Collect current values from all MBS output channels.
    virtual void CollectDataMBS() = 0;
    virtual void WriteDataMBS(const std::string& filename) = 0;
    std::vector<double> m_mbs_outputs;
  private:
    void Write();
    chrono::ChSystem* m_sys;
    std::string m_dir;
    int m_major_frame;
    std::ofstream m_mbs_stream;
    
    
};

// --------------------------------------------------------------------------------------------------------------------

class DataWriterVehicle : public DataWriter {
  public:
    // DataWriterVehicle(chrono::ChSystem* sysFSI, chrono::vehicle::ChVehicle* vehicle, chrono::vehicle::SCMTerrain& terrain);
    DataWriterVehicle(chrono::ChSystem* sys, chrono::vehicle::WheeledVehicle* vehicle, chrono::vehicle::ChTerrain& terrain);
    ~DataWriterVehicle() {}
     
  private:
    virtual int GetNumChannelsMBS() const override { return (7 + 6) + 4 * (7 + 6 + 3 + 3); }
    virtual void CollectDataMBS() override;
    virtual void WriteDataMBS(const std::string& filename) override;
    chrono::vehicle::WheeledVehicle* m_vehicle;
    std::array<std::shared_ptr<chrono::vehicle::ChWheel>, 4> m_wheels;
    chrono::vehicle::ChTerrain& m_terrain;
    

};

