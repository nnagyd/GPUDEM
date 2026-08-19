/**
 * @file benchmark_compare.cuh
 * @brief Utilities to compare baseline and new benchmark outputs.
 */

#ifndef benchmark_compare_H
#define benchmark_compare_H

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace benchmarkCompare
{
    struct EnergyRow
    {
        int launch = 0;
        double time = 0.0;
        double kin = 0.0;
        double pot = 0.0;
        double total = 0.0;
    };

    struct TrackRow
    {
        int launch = 0;
        int particle = 0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct ComparisonSummary
    {
        double finalKinErrorPercent = 0.0;
        double finalPotErrorPercent = 0.0;
        double first100TrackErrorPercent = 0.0;

        bool finalKinPass = false;
        bool finalPotPass = false;
        bool first100TrackPass = false;
        bool comparisonComplete = false;

        std::string errorMessage;
    };

    inline double safeRelativePercent(double baseline, double candidate)
    {
        const double denominator = std::max(std::abs(baseline), 1e-18);
        return std::abs(candidate - baseline) * 100.0 / denominator;
    }

    inline bool readEnergyCsv(const std::string &location, std::vector<EnergyRow> &rows)
    {
        std::ifstream in(location);
        if (!in.is_open())
        {
            return false;
        }

        std::string line;
        std::getline(in, line); // header

        while (std::getline(in, line))
        {
            if (line.empty())
            {
                continue;
            }

            std::stringstream ss(line);
            std::string token;
            EnergyRow row;

            std::getline(ss, token, ',');
            row.launch = std::stoi(token);
            std::getline(ss, token, ',');
            row.time = std::stod(token);
            std::getline(ss, token, ',');
            row.kin = std::stod(token);
            std::getline(ss, token, ',');
            row.pot = std::stod(token);
            std::getline(ss, token, ',');
            row.total = std::stod(token);

            rows.push_back(row);
        }

        return !rows.empty();
    }

    inline bool readTracksCsv(const std::string &location, std::vector<TrackRow> &rows)
    {
        std::ifstream in(location);
        if (!in.is_open())
        {
            return false;
        }

        std::string line;
        std::getline(in, line); // header

        while (std::getline(in, line))
        {
            if (line.empty())
            {
                continue;
            }

            std::stringstream ss(line);
            std::string token;
            TrackRow row;

            std::getline(ss, token, ',');
            row.launch = std::stoi(token);
            std::getline(ss, token, ',');
            row.particle = std::stoi(token);
            std::getline(ss, token, ',');
            row.x = std::stod(token);
            std::getline(ss, token, ',');
            row.y = std::stod(token);
            std::getline(ss, token, ',');
            row.z = std::stod(token);

            rows.push_back(row);
        }

        return !rows.empty();
    }

    inline ComparisonSummary compareRuns(
        const std::string &baselineEnergy,
        const std::string &newEnergy,
        const std::string &baselineTracks,
        const std::string &newTracks,
        double energyTolerancePercent,
        double trackTolerancePercent)
    {
        ComparisonSummary summary;

        std::vector<EnergyRow> baselineEnergyRows;
        std::vector<EnergyRow> newEnergyRows;
        if (!readEnergyCsv(baselineEnergy, baselineEnergyRows) || !readEnergyCsv(newEnergy, newEnergyRows))
        {
            summary.errorMessage = "Unable to read baseline/new energy files.";
            return summary;
        }

        const EnergyRow &baselineFinal = baselineEnergyRows.back();
        const EnergyRow &newFinal = newEnergyRows.back();
        summary.finalKinErrorPercent = safeRelativePercent(baselineFinal.kin, newFinal.kin);
        summary.finalPotErrorPercent = safeRelativePercent(baselineFinal.pot, newFinal.pot);
        summary.finalKinPass = summary.finalKinErrorPercent <= energyTolerancePercent;
        summary.finalPotPass = summary.finalPotErrorPercent <= energyTolerancePercent;

        std::vector<TrackRow> baselineTrackRows;
        std::vector<TrackRow> newTrackRows;
        if (!readTracksCsv(baselineTracks, baselineTrackRows) || !readTracksCsv(newTracks, newTrackRows))
        {
            summary.errorMessage = "Unable to read baseline/new track files.";
            return summary;
        }

        const size_t trackCount = std::min(baselineTrackRows.size(), newTrackRows.size());
        if (trackCount == 0)
        {
            summary.errorMessage = "Track files were empty.";
            return summary;
        }

        double errAccum = 0.0;
        int errSamples = 0;
        for (size_t i = 0; i < trackCount; i++)
        {
            const TrackRow &a = baselineTrackRows[i];
            const TrackRow &b = newTrackRows[i];

            const double dx = b.x - a.x;
            const double dy = b.y - a.y;
            const double dz = b.z - a.z;
            const double diffNorm = std::sqrt(dx * dx + dy * dy + dz * dz);

            const double baseNorm = std::max(std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z), 1e-18);
            const double relPercent = 100.0 * diffNorm / baseNorm;

            errAccum += relPercent;
            errSamples++;
        }

        summary.first100TrackErrorPercent = errAccum / std::max(errSamples, 1);
        summary.first100TrackPass = summary.first100TrackErrorPercent <= trackTolerancePercent;
        summary.comparisonComplete = true;
        return summary;
    }

    inline bool writeSummary(const std::string &location, const ComparisonSummary &summary)
    {
        std::ofstream out(location);
        if (!out.is_open())
        {
            return false;
        }

        out << "comparison_complete=" << (summary.comparisonComplete ? 1 : 0) << "\n";
        out << "final_kin_error_percent=" << summary.finalKinErrorPercent << "\n";
        out << "final_pot_error_percent=" << summary.finalPotErrorPercent << "\n";
        out << "first100_track_error_percent=" << summary.first100TrackErrorPercent << "\n";

        out << "final_kin_pass=" << (summary.finalKinPass ? 1 : 0) << "\n";
        out << "final_pot_pass=" << (summary.finalPotPass ? 1 : 0) << "\n";
        out << "first100_track_pass=" << (summary.first100TrackPass ? 1 : 0) << "\n";

        out << "overall_pass="
            << ((summary.finalKinPass && summary.finalPotPass && summary.first100TrackPass) ? 1 : 0)
            << "\n";

        if (!summary.errorMessage.empty())
        {
            out << "error=" << summary.errorMessage << "\n";
        }

        out.flush();
        out.close();
        return true;
    }
}

#endif
