/**
 * @file run_manifest.cuh
 * @brief Helper functions to persist reproducibility metadata for benchmark runs.
 */

#ifndef run_manifest_H
#define run_manifest_H

#include <fstream>
#include <iomanip>
#include <string>

namespace benchmarkManifest
{
    inline void writeRunManifest(
        const std::string &location,
        int seed,
        int numberOfParticles,
        int numberOfActiveParticles,
        int gridSize,
        int blockSize,
        var_type dt,
        int saveSteps,
        const std::string &deviceName)
    {
        std::ofstream out(location);
        if (!out.is_open())
        {
            return;
        }

        out << "run_type=reference_256_box_baseline\n";
        out << "seed=" << seed << "\n";
        out << "number_of_particles=" << numberOfParticles << "\n";
        out << "number_of_active_particles=" << numberOfActiveParticles << "\n";
        out << "grid_size=" << gridSize << "\n";
        out << "block_size=" << blockSize << "\n";
        out << std::setprecision(9) << "dt=" << dt << "\n";
        out << "save_steps=" << saveSteps << "\n";
        out << "device_name=" << deviceName << "\n";
        out << "build_date=" << __DATE__ << "\n";
        out << "build_time=" << __TIME__ << "\n";
        out.flush();
        out.close();
    }
}

#endif
