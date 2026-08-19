/**
 * @file config_parser.cuh
 * @brief Parser for simple key=value runtime configuration files.
 */

#ifndef config_parser_H
#define config_parser_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "config_runtime.cuh"

namespace runtimeConfig
{
    namespace detail
    {
        inline std::string trim(const std::string &value)
        {
            const std::string whitespace = " \t\n\r";
            const size_t start = value.find_first_not_of(whitespace);
            if (start == std::string::npos)
            {
                return "";
            }
            const size_t end = value.find_last_not_of(whitespace);
            return value.substr(start, end - start + 1);
        }

        inline bool parseBool(const std::string &text, bool &target)
        {
            std::string lower = text;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (lower == "1" || lower == "true" || lower == "yes" || lower == "on")
            {
                target = true;
                return true;
            }
            if (lower == "0" || lower == "false" || lower == "no" || lower == "off")
            {
                target = false;
                return true;
            }
            return false;
        }

        inline bool parseInt(const std::string &text, int &target)
        {
            std::istringstream stream(text);
            int value = 0;
            stream >> value;
            if (!stream.fail() && stream.eof())
            {
                target = value;
                return true;
            }
            return false;
        }

        inline bool parseFloat(const std::string &text, var_type &target)
        {
            std::istringstream stream(text);
            var_type value = 0.0f;
            stream >> value;
            if (!stream.fail() && stream.eof())
            {
                target = value;
                return true;
            }
            return false;
        }

        inline bool isFinite(var_type value)
        {
            return std::isfinite(static_cast<double>(value));
        }

        inline bool startsWith(const std::string &text, const std::string &prefix)
        {
            return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
        }

        inline bool parseIndexedKey(const std::string &key, const std::string &prefix, int &index, std::string &field)
        {
            const std::string marker = prefix + ".";
            if (!startsWith(key, marker))
            {
                return false;
            }

            const std::string tail = key.substr(marker.size());
            const size_t split = tail.find('.');
            if (split == std::string::npos)
            {
                return false;
            }

            const std::string indexText = tail.substr(0, split);
            if (indexText.empty())
            {
                return false;
            }

            int parsedIndex = -1;
            if (!parseInt(indexText, parsedIndex) || parsedIndex < 0)
            {
                return false;
            }

            index = parsedIndex;
            field = tail.substr(split + 1);
            return !field.empty();
        }

        inline bool parseBoundaryMode(const std::string &text, RuntimeBoundaryMode &mode)
        {
            std::string lower = text;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (lower == "fixed")
            {
                mode = RuntimeBoundaryMode::Fixed;
                return true;
            }
            if (lower == "forced" || lower == "forced_velocity" || lower == "forcedvelocity")
            {
                mode = RuntimeBoundaryMode::ForcedVelocity;
                return true;
            }
            if (lower == "free_body_z" || lower == "freebodyz" || lower == "free_body")
            {
                mode = RuntimeBoundaryMode::FreeBodyZ;
                return true;
            }
            return false;
        }

        inline bool parseParticleLayoutMode(const std::string &text, RuntimeParticleLayoutMode &mode)
        {
            std::string lower = text;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (lower == "auto")
            {
                mode = RuntimeParticleLayoutMode::Auto;
                return true;
            }
            if (lower == "file")
            {
                mode = RuntimeParticleLayoutMode::File;
                return true;
            }
            if (lower == "generated" || lower == "generate")
            {
                mode = RuntimeParticleLayoutMode::Generated;
                return true;
            }
            return false;
        }

        inline RuntimeMaterialEntry &ensureMaterialEntry(SimulationConfig &config, int index)
        {
            if (index >= static_cast<int>(config.materials.entries.size()))
            {
                config.materials.entries.resize(index + 1);
            }
            RuntimeMaterialEntry &entry = config.materials.entries[index];
            entry.isDefined = true;
            if (entry.id < 0)
            {
                entry.id = index;
            }
            return entry;
        }

        inline RuntimeBoundaryConfig &ensureBoundaryEntry(SimulationConfig &config, int index)
        {
            if (index >= static_cast<int>(config.boundaries.size()))
            {
                config.boundaries.resize(index + 1);
            }
            return config.boundaries[index];
        }

        inline bool applyMaterialsKey(SimulationConfig &config, const std::string &key, const std::string &value)
        {
            if (key == "materials.sigma")
            {
                if (parseFloat(value, config.materials.sigma))
                {
                    config.materials.hasSigma = true;
                    return true;
                }
                return false;
            }
            if (key == "materials.psi")
            {
                if (parseFloat(value, config.materials.psi))
                {
                    config.materials.hasPsi = true;
                    return true;
                }
                return false;
            }

            int index = -1;
            std::string field;
            if (!parseIndexedKey(key, "materials", index, field))
            {
                return false;
            }

            RuntimeMaterialEntry &entry = ensureMaterialEntry(config, index);
            if (field == "id")
            {
                return parseInt(value, entry.id);
            }
            if (field == "rho")
            {
                if (parseFloat(value, entry.rho))
                {
                    entry.hasRho = true;
                    return true;
                }
                return false;
            }
            if (field == "E" || field == "young")
            {
                if (parseFloat(value, entry.E))
                {
                    entry.hasE = true;
                    return true;
                }
                return false;
            }
            if (field == "nu")
            {
                if (parseFloat(value, entry.nu))
                {
                    entry.hasNu = true;
                    return true;
                }
                return false;
            }
            if (field == "e" || field == "restitution")
            {
                if (parseFloat(value, entry.e))
                {
                    entry.hasErest = true;
                    return true;
                }
                return false;
            }
            if (field == "mu")
            {
                if (parseFloat(value, entry.mu))
                {
                    entry.hasMu = true;
                    return true;
                }
                return false;
            }
            if (field == "mu0")
            {
                if (parseFloat(value, entry.mu0))
                {
                    entry.hasMu0 = true;
                    return true;
                }
                return false;
            }
            if (field == "mur")
            {
                if (parseFloat(value, entry.mur))
                {
                    entry.hasMur = true;
                    return true;
                }
                return false;
            }
            if (field == "theta")
            {
                if (parseFloat(value, entry.theta))
                {
                    entry.hasTheta = true;
                    return true;
                }
                return false;
            }

            return false;
        }

        inline bool applyParticleLayoutKey(SimulationConfig &config, const std::string &key, const std::string &value)
        {
            if (key == "particles.layout.mode")
            {
                if (parseParticleLayoutMode(value, config.particleLayout.mode))
                {
                    config.particleLayout.hasMode = true;
                    return true;
                }
                return false;
            }
            if (key == "particles.layout.file_path" || key == "particles.layout.file")
            {
                config.particleLayout.filePath = value;
                return true;
            }

            std::string generatedPrefix;
            if (startsWith(key, "particles.generated."))
            {
                generatedPrefix = "particles.generated.";
            }
            else if (startsWith(key, "particles.layout.generated."))
            {
                generatedPrefix = "particles.layout.generated.";
            }
            else
            {
                return false;
            }

            const std::string field = key.substr(generatedPrefix.size());
            if (field == "radius")
            {
                return parseFloat(value, config.particleLayout.generated.radius);
            }
            if (field == "minx")
            {
                return parseFloat(value, config.particleLayout.generated.minx);
            }
            if (field == "maxx")
            {
                return parseFloat(value, config.particleLayout.generated.maxx);
            }
            if (field == "miny")
            {
                return parseFloat(value, config.particleLayout.generated.miny);
            }
            if (field == "maxy")
            {
                return parseFloat(value, config.particleLayout.generated.maxy);
            }
            if (field == "minz")
            {
                return parseFloat(value, config.particleLayout.generated.minz);
            }
            if (field == "maxz")
            {
                return parseFloat(value, config.particleLayout.generated.maxz);
            }

            return false;
        }

        inline bool applyBoundaryKey(SimulationConfig &config, const std::string &key, const std::string &value)
        {
            int index = -1;
            std::string field;
            if (!parseIndexedKey(key, "boundaries", index, field))
            {
                return false;
            }

            RuntimeBoundaryConfig &boundary = ensureBoundaryEntry(config, index);

            if (field == "enabled")
            {
                if (parseBool(value, boundary.enabled))
                {
                    boundary.hasEnabled = true;
                    return true;
                }
                return false;
            }
            if (field == "name")
            {
                boundary.name = value;
                return true;
            }
            if (field == "stl_path")
            {
                boundary.stlPath = value;
                boundary.hasStlPath = true;
                return true;
            }
            if (field == "material")
            {
                if (parseInt(value, boundary.materialId))
                {
                    boundary.hasMaterial = true;
                    return true;
                }
                return false;
            }
            if (field == "mode")
            {
                if (parseBoundaryMode(value, boundary.mode))
                {
                    boundary.hasMode = true;
                    return true;
                }
                return false;
            }
            if (field == "velocity.x")
            {
                return parseFloat(value, boundary.velocityX);
            }
            if (field == "velocity.y")
            {
                return parseFloat(value, boundary.velocityY);
            }
            if (field == "velocity.z")
            {
                return parseFloat(value, boundary.velocityZ);
            }
            if (field == "mass")
            {
                return parseFloat(value, boundary.mass);
            }
            if (field == "initial.x")
            {
                return parseFloat(value, boundary.initialX);
            }
            if (field == "initial.y")
            {
                return parseFloat(value, boundary.initialY);
            }
            if (field == "initial.z")
            {
                return parseFloat(value, boundary.initialZ);
            }
            if (field == "initial_velocity_z")
            {
                return parseFloat(value, boundary.initialVelocityZ);
            }
            if (field == "start_time")
            {
                return parseFloat(value, boundary.startTime);
            }
            if (field == "end_time")
            {
                return parseFloat(value, boundary.endTime);
            }
            if (field == "motion_update_interval" || field == "motion.update_interval")
            {
                return parseInt(value, boundary.motion.updateIntervalLaunches);
            }
            if (field == "tracking.enabled")
            {
                return parseBool(value, boundary.tracking.enabled);
            }
            if (field == "tracking.csv_path")
            {
                boundary.tracking.csvPath = value;
                return true;
            }
            if (field == "tracking.write_interval")
            {
                return parseInt(value, boundary.tracking.writeIntervalLaunches);
            }

            return false;
        }

        inline bool applyKeyValue(SimulationConfig &config, const std::string &key, const std::string &value)
        {
            if (applyMaterialsKey(config, key, value))
            {
                return true;
            }
            if (applyParticleLayoutKey(config, key, value))
            {
                return true;
            }
            if (applyBoundaryKey(config, key, value))
            {
                return true;
            }

            if (key == "seed")
            {
                return parseInt(value, config.seed);
            }
            if (key == "time.start")
            {
                return parseFloat(value, config.time.start);
            }
            if (key == "time.end")
            {
                return parseFloat(value, config.time.end);
            }
            if (key == "time.dt")
            {
                return parseFloat(value, config.time.dt);
            }
            if (key == "time.save_steps")
            {
                return parseInt(value, config.time.saveSteps);
            }
            if (key == "gravity.x")
            {
                return parseFloat(value, config.gravity.x);
            }
            if (key == "gravity.y")
            {
                return parseFloat(value, config.gravity.y);
            }
            if (key == "gravity.z")
            {
                return parseFloat(value, config.gravity.z);
            }
            if (key == "mesh.minx")
            {
                if (parseFloat(value, config.mesh.minx))
                {
                    config.mesh.meshBoundsRuntimeApplied = true;
                    return true;
                }
                return false;
            }
            if (key == "mesh.miny")
            {
                if (parseFloat(value, config.mesh.miny))
                {
                    config.mesh.meshBoundsRuntimeApplied = true;
                    return true;
                }
                return false;
            }
            if (key == "mesh.minz")
            {
                if (parseFloat(value, config.mesh.minz))
                {
                    config.mesh.meshBoundsRuntimeApplied = true;
                    return true;
                }
                return false;
            }
            if (key == "mesh.maxx")
            {
                if (parseFloat(value, config.mesh.maxx))
                {
                    config.mesh.meshBoundsRuntimeApplied = true;
                    return true;
                }
                return false;
            }
            if (key == "mesh.maxy")
            {
                if (parseFloat(value, config.mesh.maxy))
                {
                    config.mesh.meshBoundsRuntimeApplied = true;
                    return true;
                }
                return false;
            }
            if (key == "mesh.maxz")
            {
                if (parseFloat(value, config.mesh.maxz))
                {
                    config.mesh.meshBoundsRuntimeApplied = true;
                    return true;
                }
                return false;
            }
            if (key == "mesh.nx")
            {
                if (parseInt(value, config.mesh.nx))
                {
                    config.mesh.meshCountsRuntimeApplied = true;
                    return true;
                }
                return false;
            }
            if (key == "mesh.ny")
            {
                if (parseInt(value, config.mesh.ny))
                {
                    config.mesh.meshCountsRuntimeApplied = true;
                    return true;
                }
                return false;
            }
            if (key == "mesh.nz")
            {
                if (parseInt(value, config.mesh.nz))
                {
                    config.mesh.meshCountsRuntimeApplied = true;
                    return true;
                }
                return false;
            }
            if (key == "output.save_velocity")
            {
                return parseBool(value, config.output.saveVelocity);
            }
            if (key == "output.save_angular_velocity")
            {
                return parseBool(value, config.output.saveAngularVelocity);
            }
            if (key == "output.save_force")
            {
                return parseBool(value, config.output.saveForce);
            }
            if (key == "output.save_torque")
            {
                return parseBool(value, config.output.saveTorque);
            }
            if (key == "output.save_id")
            {
                return parseBool(value, config.output.saveId);
            }
            if (key == "output.save_material")
            {
                return parseBool(value, config.output.saveMaterial);
            }
            if (key == "output.save_tracks_first100")
            {
                return parseBool(value, config.output.saveTracksFirst100);
            }

            return false;
        }

        inline bool validate(const SimulationConfig &config, std::string &error)
        {
            if (config.time.dt <= constant::ZERO)
            {
                error = "Invalid runtime config: time.dt must be positive.";
                return false;
            }
            if (config.time.end <= config.time.start)
            {
                error = "Invalid runtime config: time.end must be greater than time.start.";
                return false;
            }
            if (config.time.saveSteps <= 0)
            {
                error = "Invalid runtime config: time.save_steps must be positive.";
                return false;
            }
            if (config.mesh.maxx <= config.mesh.minx || config.mesh.maxy <= config.mesh.miny || config.mesh.maxz <= config.mesh.minz)
            {
                error = "Invalid runtime config: mesh max bounds must be greater than min bounds.";
                return false;
            }
            if (config.mesh.nx <= 0 || config.mesh.ny <= 0 || config.mesh.nz <= 0)
            {
                error = "Invalid runtime config: mesh cell counts must be positive.";
                return false;
            }
            if (config.mesh.nx > DecomposedDomainsConstants::Nx ||
                config.mesh.ny > DecomposedDomainsConstants::Ny ||
                config.mesh.nz > DecomposedDomainsConstants::Nz)
            {
                error = "Invalid runtime config: mesh cell counts exceed compile-time maximums.";
                return false;
            }
            if (config.mesh.nx * config.mesh.ny * config.mesh.nz > DecomposedDomainsConstants::Ncell)
            {
                error = "Invalid runtime config: mesh cell count exceeds compile-time linked-cell capacity.";
                return false;
            }

            if (!config.boundaries.empty())
            {
                if (!config.materials.hasSigma || !config.materials.hasPsi)
                {
                    error = "Invalid runtime config: materials.sigma and materials.psi are required for unified boundary mode.";
                    return false;
                }
                if (!config.particleLayout.hasMode)
                {
                    error = "Invalid runtime config: particles.layout.mode is required for unified boundary mode.";
                    return false;
                }

                if (config.particleLayout.mode == RuntimeParticleLayoutMode::File && config.particleLayout.filePath.empty())
                {
                    error = "Invalid runtime config: particles.layout.file_path is required when particles.layout.mode=file.";
                    return false;
                }
                if (config.particleLayout.mode == RuntimeParticleLayoutMode::Generated)
                {
                    const RuntimeParticleGeneratedLayoutConfig &g = config.particleLayout.generated;
                    if (!isFinite(g.radius) || !isFinite(g.minx) || !isFinite(g.maxx) || !isFinite(g.miny) || !isFinite(g.maxy) || !isFinite(g.minz) || !isFinite(g.maxz))
                    {
                        error = "Invalid runtime config: generated particle layout requires radius/min/max bounds in all axes.";
                        return false;
                    }
                    if (g.maxx <= g.minx || g.maxy <= g.miny || g.maxz <= g.minz)
                    {
                        error = "Invalid runtime config: generated particle layout max bounds must be greater than min bounds.";
                        return false;
                    }
                }

                std::set<int> definedMaterialIds;
                for (size_t i = 0; i < config.materials.entries.size(); ++i)
                {
                    const RuntimeMaterialEntry &entry = config.materials.entries[i];
                    if (!entry.isDefined)
                    {
                        continue;
                    }

                    const int id = (entry.id >= 0) ? entry.id : static_cast<int>(i);
                    if (id < 0 || id >= NumberOfMaterials)
                    {
                        error = "Invalid runtime config: material id out of compile-time range.";
                        return false;
                    }
                    if (!entry.hasRho || !entry.hasE || !entry.hasNu || !entry.hasErest || !entry.hasMu || !entry.hasMu0 || !entry.hasMur || !entry.hasTheta)
                    {
                        error = "Invalid runtime config: each defined materials.<i> entry requires rho,E,nu,e,mu,mu0,mur,theta.";
                        return false;
                    }
                    definedMaterialIds.insert(id);
                }

                if (definedMaterialIds.empty())
                {
                    error = "Invalid runtime config: unified boundary mode requires materials.<i> entries.";
                    return false;
                }

                bool anyEnabledBoundary = false;
                for (size_t i = 0; i < config.boundaries.size(); ++i)
                {
                    const RuntimeBoundaryConfig &boundary = config.boundaries[i];
                    if (!boundary.hasEnabled)
                    {
                        error = "Invalid runtime config: boundaries.<i>.enabled is mandatory for all boundary entries.";
                        return false;
                    }
                    if (!boundary.enabled)
                    {
                        continue;
                    }

                    anyEnabledBoundary = true;

                    if (!boundary.hasStlPath || boundary.stlPath.empty())
                    {
                        error = "Invalid runtime config: boundaries.<i>.stl_path is required for enabled boundaries.";
                        return false;
                    }
                    if (!boundary.hasMaterial)
                    {
                        error = "Invalid runtime config: boundaries.<i>.material is required for enabled boundaries.";
                        return false;
                    }
                    if (!boundary.hasMode)
                    {
                        error = "Invalid runtime config: boundaries.<i>.mode is required for enabled boundaries.";
                        return false;
                    }
                    if (boundary.materialId < 0 || boundary.materialId >= NumberOfMaterials)
                    {
                        error = "Invalid runtime config: boundaries.<i>.material is out of compile-time material range.";
                        return false;
                    }
                    if (definedMaterialIds.find(boundary.materialId) == definedMaterialIds.end())
                    {
                        error = "Invalid runtime config: boundary material reference has no matching materials.<i> definition.";
                        return false;
                    }
                    if (boundary.motion.updateIntervalLaunches <= 0)
                    {
                        error = "Invalid runtime config: boundaries.<i>.motion_update_interval must be positive.";
                        return false;
                    }
                    if (boundary.tracking.writeIntervalLaunches <= 0)
                    {
                        error = "Invalid runtime config: boundaries.<i>.tracking.write_interval must be positive.";
                        return false;
                    }

                    if (boundary.mode == RuntimeBoundaryMode::ForcedVelocity)
                    {
                        if (!isFinite(boundary.velocityX) || !isFinite(boundary.velocityY) || !isFinite(boundary.velocityZ))
                        {
                            error = "Invalid runtime config: forced boundary requires boundaries.<i>.velocity.(x|y|z).";
                            return false;
                        }
                    }
                    if (boundary.mode == RuntimeBoundaryMode::FreeBodyZ)
                    {
                        if (!isFinite(boundary.mass) || boundary.mass <= constant::ZERO)
                        {
                            error = "Invalid runtime config: free_body_z boundary requires positive boundaries.<i>.mass.";
                            return false;
                        }
                        if (!isFinite(boundary.initialX) || !isFinite(boundary.initialY) || !isFinite(boundary.initialZ))
                        {
                            error = "Invalid runtime config: free_body_z boundary requires boundaries.<i>.initial.(x|y|z).";
                            return false;
                        }
                    }
                }

                if (!anyEnabledBoundary)
                {
                    error = "Invalid runtime config: at least one boundaries.<i>.enabled=1 entry is required in unified boundary mode.";
                    return false;
                }
            }

            return true;
        }
    }

    inline bool loadFromFile(const std::string &location, SimulationConfig &config, std::string &error)
    {
        std::ifstream input(location);
        if (!input.is_open())
        {
            error = "Unable to open runtime config file: " + location;
            return false;
        }

        std::string line;
        int lineNumber = 0;
        while (std::getline(input, line))
        {
            lineNumber++;
            const std::string clean = detail::trim(line);
            if (clean.empty() || clean[0] == '#')
            {
                continue;
            }

            const size_t separator = clean.find('=');
            if (separator == std::string::npos)
            {
                error = "Invalid config line " + std::to_string(lineNumber) + ": expected key=value format.";
                return false;
            }

            const std::string key = detail::trim(clean.substr(0, separator));
            const std::string value = detail::trim(clean.substr(separator + 1));
            if (key.empty() || value.empty())
            {
                error = "Invalid config line " + std::to_string(lineNumber) + ": empty key or value.";
                return false;
            }

            if (!detail::applyKeyValue(config, key, value))
            {
                error = "Invalid or unsupported config key/value at line " + std::to_string(lineNumber) + ": " + key;
                return false;
            }
        }

        if (!detail::validate(config, error))
        {
            return false;
        }

        config.isLoadedFromFile = true;
        config.sourceFile = location;
        return true;
    }
}

#endif
