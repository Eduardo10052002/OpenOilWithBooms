# =============================================================================
# backend.py
#
# Description:
# This module serves as the simulation orchestrator for the Decision Support
# System (DSS). It acts as the bridge between the user interface (e.g., a
# Streamlit app) and the core scientific model (boom_model.py). Its primary
# responsibilities include:
#   1.  Configuring the simulation environment based on user inputs.
#   2.  Instantiating and running the OpenOilWithBooms model.
#   3.  Post-processing the raw NetCDF output to calculate key performance
#       metrics, mass balance timeseries, and aggregated boom statistics.
#   4.  Generating output artifacts such as videos, maps, and summary data files.
#
# Last Modified: 15-09-2025
# =============================================================================
import logging
import os
import traceback
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from boom_model import OpenOilWithBooms
from opendrift.models.openoil.openoil import (Density, KinematicViscosity,
                                              adios)
from opendrift.readers import reader_global_landmask, reader_netCDF_CF_generic
from opendrift.readers.reader_constant import Reader as ConstantReader

# Configure logging to provide informative output.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_available_oils():
    """
    Retrieves the list of available oil types from the integrated ADIOS database.
    
    Returns:
        list: A list of strings with the names of available oils.
              Returns a default list on failure.
    """
    try:
        o = OpenOilWithBooms()
        return o.oiltypes
    except Exception as e:
        logger.error(f"Error fetching oil list: {e}")
        return ["GENERIC MEDIUM CRUDE", "GENERIC LIGHT CRUDE", "GENERIC HEAVY CRUDE"]


def get_oil_properties(oil_name):
    """
    Fetches key physical properties (density and viscosity) for a specific
    oil type from the ADIOS database at a standard temperature (15°C).

    Args:
        oil_name (str): The name of the oil.

    Returns:
        dict: A dictionary with 'density' (kg/m^3) and 'viscosity' (cSt),
              or None if properties cannot be fetched.
    """
    try:
        oil = adios.find_full_oil_from_name(oil_name)
        if oil:
            # Standard temperature for property comparison is 288.15 K (15°C).
            density = Density(oil.oil).at_temp(288.15)
            viscosity_m2s = KinematicViscosity(oil.oil).at_temp(288.15)
            viscosity_cst = viscosity_m2s * 1e6  # Convert from m^2/s to cSt.
            return {'density': density, 'viscosity': viscosity_cst}
    except Exception as e:
        logger.warning(f"Could not fetch properties for {oil_name}: {e}")
    return None


def get_environmental_timeseries(o):
    """
    Processes the environmental data recorded by the model during the run
    to generate timeseries for plotting.

    Args:
        o (OpenOilWithBooms): The executed model instance.

    Returns:
        dict: A dictionary containing timeseries of key environmental variables.
    """
    env_ts = {}
    try:
        if hasattr(o, 'env_history') and o.env_history:
            df = pd.DataFrame(o.env_history)
            df['time_hours'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds() / 3600.0
            env_ts = {col: df[col].tolist() for col in df.columns if col != 'time'}
    except Exception as e:
        logger.error(f"Error processing environmental timeseries: {e}\n{traceback.format_exc()}")
    return env_ts


def get_final_particle_count_from_memory(o):
    """
    Calculates the final fate of particles (by count) by inspecting the in-memory
    o.result xarray Dataset after the simulation run.

    Args:
        o (OpenOilWithBooms): The executed model instance.

    Returns:
        dict: A dictionary summarizing the final count of particles in each state.
    """
    counts = defaultdict(int)
    status_map = {i: name for i, name in enumerate(o.status_categories)}

    if not hasattr(o, 'result') or 'status' not in o.result:
        logger.warning("`o.result` object not found or is missing 'status'. Cannot generate final particle counts.")
        return {fate: 0 for fate in ['Active On Surface', 'Active Submerged', 'Stranded', 'Absorbed', 'Contained']}

    try:
        # Extract the final state of all particles from the last time step.
        final_status_all = o.result['status'].squeeze().isel(time=-1).values
        final_z_all = o.result['z'].squeeze().isel(time=-1).values
        final_is_contained_all = o.result['is_contained'].squeeze().isel(time=-1).values
        particle_ids = o.result['trajectory'].values
    except Exception as e:
        logger.error(f"Could not extract final state from o.result: {e}. Aborting count.")
        return {fate: 0 for fate in ['Active On Surface', 'Active Submerged', 'Stranded', 'Absorbed', 'Contained']}

    counts['Contained'] = np.sum(final_is_contained_all == 1)

    for i in range(len(particle_ids)):
        if final_is_contained_all[i] == 1:
            continue

        status_code = final_status_all[i]

        # Handle cases where the final status is NaN by finding the last valid status.
        if np.isnan(status_code):
            particle_status_history = o.result['status'].sel(trajectory=particle_ids[i]).values
            valid_statuses = particle_status_history[~np.isnan(particle_status_history)]
            if valid_statuses.size > 0:
                status_code = valid_statuses[-1]
            else:
                continue

        status_name = status_map.get(int(status_code))

        if status_name == 'active':
            if final_z_all[i] >= 0:
                counts['Active On Surface'] += 1
            else:
                counts['Active Submerged'] += 1
        elif status_name:
            counts[status_name.title()] += 1

    all_fates = ['Active On Surface', 'Active Submerged', 'Stranded', 'Absorbed', 'Contained']
    final_counts = {fate: counts.get(fate, 0) for fate in all_fates}

    return final_counts


def process_netcdf_for_timeseries(o, netcdf_path):
    """
    Processes the simulation's NetCDF output file to generate timeseries data
    for mass balance charts and key performance metrics.

    This function is crucial for translating raw particle data into aggregated,
    interpretable results.

    Args:
        o (OpenOilWithBooms): The executed model instance.
        netcdf_path (str): The file path to the simulation output .nc file.

    Returns:
        dict: A dictionary containing timeseries data and key metrics.
    """
    stats = {
        'key_metrics': {},
        'mass_balance_timeseries': {},
        'timeseries': {}
    }
    if not os.path.exists(netcdf_path):
        return stats

    with xr.open_dataset(netcdf_path) as ds:
        # Load all necessary data matrices into memory.
        status_matrix = ds['status'].values
        mass_matrix = ds['mass_oil'].values
        time_vector_dt = ds['time'].values
        is_contained_matrix = ds.get('is_contained', xr.DataArray(np.zeros_like(status_matrix), dims=ds['status'].dims)).values

        start_time = pd.to_datetime(time_vector_dt[0])
        time_vector_hours = (pd.to_datetime(pd.Series(time_vector_dt)) - start_time).dt.total_seconds() / 3600.0
        stats['timeseries']['time_hours'] = time_vector_hours.tolist()

        total_initial_mass = np.nansum(mass_matrix[:, 0])
        status_codes = {name: o.status_categories.index(name) for name in o.status_categories}
        num_particles, num_timesteps = status_matrix.shape

        mass_ts = defaultdict(lambda: np.zeros(num_timesteps))

        # --- Calculate mass distribution over time for the mass balance plot ---
        stranded_mask = status_matrix == status_codes.get('stranded', -1)
        absorbed_mask = status_matrix == status_codes.get('absorbed', -1)
        contained_mask = is_contained_matrix == 1
        active_mask = status_matrix == status_codes['active']

        mass_ts['Contained'] = np.nansum(np.where(contained_mask, mass_matrix, 0), axis=0)
        mass_ts['Active'] = np.nansum(np.where(active_mask & ~contained_mask, mass_matrix, 0), axis=0)
        
        # For terminal states (Stranded, Absorbed), we calculate the cumulative mass
        # that has entered that state to avoid double counting.
        has_been_stranded = np.zeros(num_particles, dtype=bool)
        has_been_absorbed = np.zeros(num_particles, dtype=bool)

        for t in range(num_timesteps):
            if t > 0:
                mass_ts['Stranded'][t] = mass_ts['Stranded'][t-1]
                mass_ts['Absorbed'][t] = mass_ts['Absorbed'][t-1]

            newly_stranded_idx = np.where(stranded_mask[:, t] & ~has_been_stranded)[0]
            if newly_stranded_idx.size > 0:
                mass_flux = np.nansum(mass_matrix[newly_stranded_idx, t])
                mass_ts['Stranded'][t] += mass_flux
                has_been_stranded[newly_stranded_idx] = True

            newly_absorbed_idx = np.where(absorbed_mask[:, t] & ~has_been_absorbed)[0]
            if newly_absorbed_idx.size > 0:
                mass_flux = np.nansum(mass_matrix[newly_absorbed_idx, t])
                mass_ts['Absorbed'][t] += mass_flux
                has_been_absorbed[newly_absorbed_idx] = True

        # Weathered mass is the difference between initial mass and the sum of all
        # other physical components. Ensure it is monotonically increasing.
        total_physical_mass_ts = mass_ts['Active'] + mass_ts['Contained'] + mass_ts['Stranded'] + mass_ts['Absorbed']
        mass_ts['Lost Mass (Weathering)'] = np.maximum(0, total_initial_mass - total_physical_mass_ts)
        mass_ts['Lost Mass (Weathering)'] = np.maximum.accumulate(mass_ts['Lost Mass (Weathering)'])

        mass_balance = {'Hours': time_vector_hours.tolist()}
        for key, value in mass_ts.items():
            mass_balance[key] = value.tolist()

        stats['mass_balance_timeseries'] = mass_balance
        stats['timeseries']['cumulative_stranded_mass'] = mass_balance.get('Stranded', [0.0] * len(time_vector_hours))
        
        # --- Calculate final key metrics ---
        stats['key_metrics']['total_stranded_mass'] = mass_balance.get('Stranded', [0.0])[-1]
        if stats['key_metrics']['total_stranded_mass'] > 0:
            try:
                first_stranding_idx = np.where(np.array(mass_balance['Stranded']) > 0)[0][0]
                stats['key_metrics']['time_to_first_stranding_hours'] = time_vector_hours[first_stranding_idx]
            except IndexError:
                 stats['key_metrics']['time_to_first_stranding_hours'] = None
        else:
            stats['key_metrics']['time_to_first_stranding_hours'] = None
            
        stats['key_metrics']['total_leaked_mass'] = 0.0
        
    return stats


def run_simulation(simulation_params, video_filename, netcdf_filename, map_filename, booms_data=None, warning_placeholder=None, progress_bar=None):
    """
    The main orchestration function that sets up and runs a full simulation scenario.
    
    Args:
        simulation_params (dict): All parameters defining the spill scenario.
        video_filename (str): Filename for the output animation video.
        netcdf_filename (str): Filename for the raw simulation output data.
        map_filename (str): Filename for the final summary map image.
        booms_data (list, optional): A list of dictionaries, each defining a boom.
        warning_placeholder (streamlit.element, optional): Streamlit element to display warnings.
        progress_bar (streamlit.element, optional): Streamlit element to show progress.

    Returns:
        tuple: Contains paths to output files, statistics dictionary, and map bounds.
    """
    if progress_bar: progress_bar.progress(0, text="Initializing...")
    logger.info("Initiating simulation setup.")
    o = OpenOilWithBooms(loglevel=logging.INFO)
    stats = {'warnings': []}
    video_with_currents_path = None
    map_path = None
    
    # --- Step 1: Configure the model instance ---
    o.set_config('drift:max_speed', 5.0) # Set a maximum particle speed for numerical stability.
    
    if simulation_params.get('oil_source') == "Define Custom":
        o.set_config('seed:oil_type', 'GENERIC MEDIUM CRUDE')
    else:
        o.set_config('seed:oil_type', simulation_params['oil_type'])

    o.set_config('general:coastline_action', 'stranding')
    
    # --- Step 2: Load environmental data readers ---
    if progress_bar: progress_bar.progress(15, text="Loading environmental data readers...")
    reader_sources = {
        "ocean currents": simulation_params.get('reader_ocean'),
        "wind": simulation_params.get('reader_wind'),
        "waves": simulation_params.get('reader_wave')
    }
    loaded_readers = []
    for name, url in reader_sources.items():
        if url:
            try:
                loaded_readers.append(reader_netCDF_CF_generic.Reader(url))
            except Exception as e:
                warning_msg = f"Failed to load {name} data. Simulation will use fallback values (0)."
                stats['warnings'].append(warning_msg)
                if warning_placeholder:
                    warning_placeholder.warning(f"Could not load {name} data. Simulation will use default values (0).")

    loaded_readers.append(reader_global_landmask.Reader())
    o.add_reader(loaded_readers)

    # --- Step 3: Add booms to the simulation if they are defined ---
    if booms_data:
        for i, boom in enumerate(booms_data):
            o.add_boom_polyline(coords=boom['coordinates'], boom_type=boom['type'], name=f"Boom_{i+1}", **boom.get('overrides', {}))
    
    # --- Step 4: Seed the oil particles ---
    total_mass_kg = simulation_params.get('total_mass_kg', 1000)
    num_elements = simulation_params.get('number', 1000)
    mass_per_particle = total_mass_kg / num_elements

    seed_kwargs = {
        'lon': simulation_params['lon'], 'lat': simulation_params['lat'], 'radius': simulation_params['radius'], 
        'number': num_elements, 'time': simulation_params['start_time']
    }

    if progress_bar: progress_bar.progress(25, text="Seeding oil particles...")
    
    o.seed_elements(**seed_kwargs)
    # Manually assign mass as it's not handled directly by seed_elements for scheduled particles.
    o.elements_scheduled.mass_oil = np.full(num_elements, mass_per_particle)

    # Apply custom oil properties if defined by the user.
    if simulation_params.get('oil_source') == "Define Custom":
        custom_props = simulation_params.get('custom_oil_properties')
        if custom_props:
            o.elements.density = np.full(o.num_elements_total(), custom_props['density'])
            o.elements.viscosity = np.full(o.num_elements_total(), custom_props['viscosity'])

    # --- Step 5: Run the main simulation loop ---
    output_dir = simulation_params.get('output_dir', '.')
    video_path = os.path.join(output_dir, video_filename)
    netcdf_path = os.path.join(output_dir, netcdf_filename)
    
    try:
        if progress_bar: progress_bar.progress(30, text="Running main simulation...")
        o.run(duration=timedelta(hours=simulation_params['duration_hours']),
              time_step=timedelta(minutes=simulation_params.get('time_step', 30)),
              time_step_output=timedelta(minutes=simulation_params.get('time_step_output', 60)),
              outfile=netcdf_path)
    except Exception as e:
        logger.error(f"Simulation run failed: {e}\n{traceback.format_exc()}")
        stats['warnings'].append(f"Simulation run failed: {e}")
        return None, None, stats, None, None, None
    
    # --- Step 6: Generate output artifacts (animations and maps) ---
    if progress_bar: progress_bar.progress(85, text="Generating animations...")
    animation_corners = None
    try:
        # Calculate optimal map corners based on particle trajectories.
        lon, lat = o.result.lon.values.flatten(), o.result.lat.values.flatten()
        valid_lon, valid_lat = lon[np.isfinite(lon)], lat[np.isfinite(lat)]
        if len(valid_lon) > 0:
            lon_min, lon_max = np.min(valid_lon), np.max(valid_lon)
            lat_min, lat_max = np.min(valid_lat), np.max(valid_lat)
            margin = max((lon_max - lon_min), (lat_max-lat_min)) * 0.1
            animation_corners = [float(lon_min - margin), float(lon_max + margin), float(lat_min - margin), float(lat_max + margin)]
    except Exception:
        pass
    
    try:
        o.animation(filename=video_path, fps=5, corners=animation_corners, colorbar=False)
    except Exception as e:
        logger.error(f"CRITICAL ERROR generating animation: {e}\n{traceback.format_exc()}"); video_path = None
    
    # Generate a secondary animation with current vectors for the baseline scenario.
    if not booms_data:
        try:
            video_with_currents_path = os.path.join(output_dir, "initial_simulation_with_currents.mp4")
            if os.path.exists(video_with_currents_path): os.remove(video_with_currents_path)
            
            logger.info("Generating secondary animation with current vector plot...")
            o.animation(filename=video_with_currents_path, fps=5, corners=animation_corners, colorbar=True,
                        background=['x_sea_water_velocity', 'y_sea_water_velocity'], 
                        vector_kwargs={'scale': 150, 'width': 0.004})
        except Exception as e:
            logger.error(f"Could not generate animation with currents: {e}\n{traceback.format_exc()}")
            video_with_currents_path = None

    map_path = os.path.join(output_dir, map_filename)
    try:
        o.plot(filename=map_path, corners=animation_corners)
    except Exception as e:
        logger.error(f"CRITICAL ERROR generating summary map: {e}\n{traceback.format_exc()}")
        map_path = None

    map_bounds = [[animation_corners[2], animation_corners[0]], [animation_corners[3], animation_corners[1]]] if animation_corners else None
    
    # --- Step 7: Post-process results and calculate final statistics ---
    if progress_bar: progress_bar.progress(95, text="Calculating final statistics...")
    try:
        timeseries_stats = process_netcdf_for_timeseries(o, netcdf_path)
        stats.update(timeseries_stats)
        
        final_counts = get_final_particle_count_from_memory(o)
        stats['final_particle_fate'] = final_counts
        
        stats['start_time'] = o.start_time.isoformat()

        if booms_data:
            performance_barreiras = {}
            segment_data_storage = defaultdict(list)
            stats['leakage_events'] = []
            
            # Aggregate data from individual segments into parent boom structures.
            for boom_segment in o.booms: 
                segment_data_storage[boom_segment['main_boom_name']].append(boom_segment)
                # Log individual leakage events for detailed timeseries analysis.
                for leak_type, mass_list in boom_segment['leakage_mass_per_step'].items():
                    for i, mass in enumerate(mass_list):
                        if mass > 0:
                            stats['leakage_events'].append({
                                'time_hours': boom_segment['time_steps_hours'][i],
                                'leakage_type': leak_type.replace('_', ' ').title(),
                                'boom_name': boom_segment['main_boom_name'],
                                'mass': mass
                            })

            for main_boom_name, segments in segment_data_storage.items():
                
                # Store the configuration parameters used for this boom for reporting.
                first_segment_config = segments[0]
                config_to_store = {
                    key: first_segment_config.get(key) for key in [
                        'freeboard_height_m', 'skirt_depth_m', 'anchor_strength_N', 
                        'buoyancy_to_weight_ratio', 'absorption_rate', 'capacity_kg_per_m',
                        'critical_froude_number', 'entrainment_coefficient', 'drag_coeff',
                        'splashover_factor', 'accumulation_factor', 'wave_period_factor',
                        'accumulation_width_m'
                    ]
                }
                
                # Aggregate performance metrics from all segments of a boom.
                perf = {
                    "name": main_boom_name, 
                    "type": segments[0]['structural_type'], 
                    "structural_failure": any(s['structurally_failed'] for s in segments), 
                    "structural_failure_time": next((s['structural_failure_time'].isoformat() for s in segments if s['structural_failure_time']), None), 
                    "mass_contained": sum(s.get('mass_contained', 0) for s in segments),
                    "mass_absorbed": sum(s['mass_absorbed'] for s in segments), 
                    "max_anchor_force_N": segments[0].get('anchor_strength_N', 0), 
                    "breaking_force_N": max(s.get('breaking_force_N', 0) for s in segments), 
                    "leakage_details": defaultdict(float), 
                    "warnings": [], 
                    "total_capacity_kg": sum(s.get('capacity_kg_per_m', 0) * s.get('length_m', 0) for s in segments),
                    "total_length_m": sum(s.get('length_m', 0) for s in segments),
                    "configuration": config_to_store
                }
                
                for s in segments:
                    for k, v in s.items():
                        if k.startswith('mass_leaked_'): 
                            perf['leakage_details'][k.replace('mass_leaked_', '').replace('_', ' ').title()] += v
                    if s['rotationally_failed'] and "One or more segments may have failed due to planing/submergence." not in perf["warnings"]: 
                        perf["warnings"].append("One or more segments may have failed due to planing/submergence.")
                
                # Aggregate force timeseries by taking the maximum force experienced across all segments at each time step.
                all_force_ts = [s['force_timeseries_N'] for s in segments if s['force_timeseries_N']]
                if all_force_ts:
                    max_len = max(len(ts) for ts in all_force_ts)
                    padded_force_ts = [np.pad(ts, (0, max_len - len(ts))) for ts in all_force_ts]
                    max_force_timeseries = np.max(np.array(padded_force_ts), axis=0).tolist()
                    perf['force_timeseries_N'] = max_force_timeseries
                    perf['max_force_experienced_N'] = max(max_force_timeseries) if max_force_timeseries else 0
                else:
                    perf['force_timeseries_N'], perf['max_force_experienced_N'] = ([], 0)

                # Calculate overall boom efficiency.
                perf['total_mass_leaked'] = sum(perf['leakage_details'].values())
                mass_retained = perf['mass_contained'] + perf['mass_absorbed']
                total_interacted_mass = mass_retained + perf['total_mass_leaked']
                perf['efficiency'] = (mass_retained / total_interacted_mass) * 100 if total_interacted_mass > 1e-9 else 100.0
                
                performance_barreiras[main_boom_name] = perf
            
            stats['boom_performance'] = list(performance_barreiras.values())
            
            # Update the total leaked mass metric with the sum from all booms.
            total_leaked_from_booms = sum(p.get('total_mass_leaked', 0) for p in stats['boom_performance'])
            stats['key_metrics']['total_leaked_mass'] = total_leaked_from_booms
        
        stats['environmental_timeseries'] = get_environmental_timeseries(o)

    except Exception as e:
        logger.error(f"Error calculating stats: {e}\n{traceback.format_exc()}")

    # --- Step 8: Return all results and artifacts ---
    return video_path, netcdf_path, stats, map_bounds, video_with_currents_path, map_path