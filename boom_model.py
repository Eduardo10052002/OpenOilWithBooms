# =============================================================================
# boom_model.py
#
# Description:
# This module provides the core scientific model for simulating the interaction
# between oil spills and containment booms. It was developed as the central
# artifact for the Master's thesis "Sistema de Apoio à Decisão para Otimização
# de Contenção de Derrames de Petróleo".
#
# The model extends the OpenDrift framework, specifically its OpenOil module,
# to incorporate the complex physics of boom-oil interaction. This is achieved
# through the OpenOilWithBooms class, which simulates hydrodynamic and
# structural failure modes of containment booms, treating each boom segment as
# an independent entity.
#
# Key Features:
# - Extends OpenDrift's Lagrangian particle tracking model (OpenOil).
# - Implements multiple, parameterizable boom typologies (e.g., Curtain, Fence).
# - Models key hydrodynamic failure modes: Entrainment, Drainage, Splash-over,
#   and Critical Accumulation.
# - Simulates mechanical failures: Structural (anchor) failure and rotational
#   failure (Planing/Submergence).
# - Records detailed interaction data (containment status, leakage cause) for
#   each oil particle, enabling detailed post-simulation diagnostics.
# - Designed to be integrated into a Decision Support System (DSS).
#
# Last Modified: 15-09-2025
# =============================================================================

import logging
import re
from datetime import datetime

import cartopy.crs as ccrs
import numpy as np
import pyproj
from opendrift.config import CONFIG_LEVEL_BASIC
from opendrift.models.openoil import OpenOil
from shapely.geometry import LineString, Point

# Set up a logger for this module to provide informative output during simulation.
logger = logging.getLogger(__name__)


class BoomElementType(OpenOil.ElementType):
    """
    Extends the standard OpenOil.ElementType to include custom variables
    for tracking the state of each oil particle as it interacts with booms.

    This custom data structure is essential for post-simulation analysis,
    allowing for a detailed diagnosis of boom performance and failure causes.
    """
    # Inherit all variables from the parent OpenOil.ElementType.
    variables = OpenOil.ElementType.variables.copy()

    # Add new, boom-specific state variables to each particle.
    variables.update({
        'leakage_status': {
            'dtype': np.float32,
            'default': 0,
            'units': '1',  # Dimensionless code
            'long_name': 'Leakage Mechanism Code'
        },
        'is_contained': {
            'dtype': np.int8,
            'default': 0,
            'units': '1',  # Boolean flag (0 or 1)
            'long_name': 'Containment Status Flag'
        },
        'contained_by_boom_id': {
            'dtype': np.int16,
            'default': -1,  # -1 indicates not contained
            'units': '1',
            'long_name': 'ID of Containing Boom Segment'
        },
    })


class OpenOilWithBooms(OpenOil):
    """
    An extension of the OpenOil model that simulates the interaction of oil
    particles with physical containment booms, treating each segment independently.

    This class encapsulates the core logic of the thesis, including the
    definition of boom properties, calculation of environmental forces, and
    the evaluation of various failure modes. It overrides the standard `update`
    method of OpenDrift to inject the boom interaction logic at each
    simulation time step.
    """
    # Use the custom ElementType to ensure each particle has the required
    # state variables for boom interaction.
    ElementType = BoomElementType

    def __init__(self, *args, **kwargs):
        """
        Initializes the OpenOilWithBooms model, setting up boom-specific
        properties, data structures, and configurations.
        """
        super().__init__(*args, **kwargs)

        self.rng = None
        logger.info("Extending OpenOil model with independent boom segment functionality.")

        # --- Data structures for managing simulation state ---
        self.booms = []
        self._boom_projection = None
        self.boom_previous_lon = {}
        self.boom_previous_lat = {}
        self.env_history = []
        # Counter to generate a unique ID for each individual boom segment.
        self.next_segment_id = 1

        # A dictionary defining the default physical and operational parameters
        # for different types of containment booms. These values are based on
        # technical literature and manufacturer specifications, serving as a
        # realistic baseline that can be overridden by the user.
        self.boom_types = {
            'curtain': {
                'description': 'Standard curtain boom',
                'structural_type': 'curtain',
                'freeboard_height_m': 0.4,                  # Height above water [m]
                'skirt_depth_m': 0.6,                       # Skirt depth below water [m]
                'anchor_strength_N': 50000,                 # Maximum anchor force before failure [N]
                'critical_froude_number': 0.50,             # Dimensionless Froude number for drainage failure [-]
                'entrainment_coefficient': 0.001,           # Coefficient for entrainment probability [-]
                'entrainment_critical_velocity_ms': 0.3,    # Critical velocity for entrainment onset [m/s]
                'viscosity_damping_factor': 50,             # Viscosity damping factor for entrainment [-]
                'splashover_factor': 1.0,                   # Multiplier for effective wave height in splashover [-]
                'buoyancy_to_weight_ratio': 10.0,           # Buoyancy-to-weight ratio for stability [-]
                'drag_coeff': 1.2,                          # Hydrodynamic and wind drag coefficient [-]
                'wave_force_coeff': 100.0,                  # Empirical coefficient for wave force [-]
                'critical_accumulation_viscosity_cst': 20000,# Kinematic viscosity threshold for accumulation failure [cSt]
                'critical_accumulation_froude': 0.15,       # Dimensionless Froude number for accumulation failure [-]
                'absorption_rate': 0.0,                     # Absorption rate for sorbent booms (0 to 1) [-]
                'capacity_kg_per_m': 0.0,                   # Absorption capacity [kg/m]
                'accumulation_factor': 1.0,                 # Factor for increased effective oil thickness [-]
                'wave_period_factor': 2.0,                  # Factor for wave period influence on splashover [-]
                'accumulation_width_m': 1.0,                # Width of accumulated oil slick for drag calculation [m]
                'restoring_force_per_m': 500                # Restoring force per meter for rotational moment [N/m]
            },
            'fence': {
                'description': 'Rigid fence boom',
                'structural_type': 'fence',
                'freeboard_height_m': 0.5,                  # [m]
                'skirt_depth_m': 0.5,                       # [m]
                'anchor_strength_N': 40000,                 # [N]
                'critical_froude_number': 0.50,             # [-]
                'entrainment_coefficient': 0.0015,          # [-]
                'entrainment_critical_velocity_ms': 0.3,    # [m/s]
                'viscosity_damping_factor': 50,             # [-]
                'splashover_factor': 1.0,                   # [-]
                'buoyancy_to_weight_ratio': 3.0,            # [-]
                'drag_coeff': 1.5,                          # [-]
                'wave_force_coeff': 100.0,                  # [-]
                'critical_accumulation_viscosity_cst': 20000,# [cSt]
                'critical_accumulation_froude': 0.15,       # [-]
                'absorption_rate': 0.0,                     # [-]
                'capacity_kg_per_m': 0.0,                   # [kg/m]
                'accumulation_factor': 1.0,                 # [-]
                'wave_period_factor': 2.0,                  # [-]
                'accumulation_width_m': 1.0,                # [m]
                'restoring_force_per_m': 500                # [N/m]
            },
            'sorbent': {
                'description': 'Sorbent boom',
                'structural_type': 'sorbent',
                'freeboard_height_m': 0.2,                  # [m]
                'skirt_depth_m': 0.3,                       # [m]
                'anchor_strength_N': 20000,                 # [N]
                'critical_froude_number': 0.45,             # [-]
                'entrainment_coefficient': 0.003,           # [-]
                'entrainment_critical_velocity_ms': 0.3,    # [m/s]
                'viscosity_damping_factor': 50,             # [-]
                'splashover_factor': 1.0,                   # [-]
                'buoyancy_to_weight_ratio': 4.0,            # [-]
                'drag_coeff': 1.1,                          # [-]
                'wave_force_coeff': 100.0,                  # [-]
                'critical_accumulation_viscosity_cst': 20000,# [cSt]
                'critical_accumulation_froude': 0.15,       # [-]
                'absorption_rate': 0.2,                     # [-]
                'capacity_kg_per_m': 10.0,                  # [kg/m]
                'accumulation_factor': 1.0,                 # [-]
                'wave_period_factor': 2.0,                  # [-]
                'accumulation_width_m': 1.0,                # [m]
                'restoring_force_per_m': 500                # [N/m]
            },
            'shore_sealing': {
                'description': 'Shore-sealing boom',
                'structural_type': 'shore_sealing',
                'freeboard_height_m': 0.5,                  # [m]
                'skirt_depth_m': 0.7,                       # [m]
                'anchor_strength_N': 40000,                 # [N]
                'critical_froude_number': 0.60,             # [-]
                'entrainment_coefficient': 0.002,           # [-]
                'entrainment_critical_velocity_ms': 0.3,    # [m/s]
                'viscosity_damping_factor': 50,             # [-]
                'splashover_factor': 1.0,                   # [-]
                'buoyancy_to_weight_ratio': 10.0,           # [-]
                'drag_coeff': 1.3,                          # [-]
                'wave_force_coeff': 100.0,                  # [-]
                'critical_accumulation_viscosity_cst': 20000,# [cSt]
                'critical_accumulation_froude': 0.15,       # [-]
                'absorption_rate': 0.0,                     # [-]
                'capacity_kg_per_m': 0.0,                   # [kg/m]
                'accumulation_factor': 1.0,                 # [-]
                'wave_period_factor': 2.0,                  # [-]
                'accumulation_width_m': 1.0,                # [m]
                'restoring_force_per_m': 500                # [N/m]
            }
        }
        
        # --- Configure visualization and status tracking ---
        for boom_type, color in [('curtain', 'green'), ('fence', '#FF7F00'), ('sorbent', '#9467BD'), ('shore_sealing', '#8C564B')]:
            self.boom_types[boom_type]['plot_props'] = {'color': color}

        if 'absorbed' not in self.status_categories:
            self.status_categories.append('absorbed')
        
        self.status_colors.update({
            'active': '#000000', 'absorbed': '#e377c2', 'contained': '#00FFFF',
            'leaked_entrainment': '#ff7f0e', 'leaked_drainage': '#8c564b', 'leaked_splashover': '#1f77b4',
            'leaked_structural': '#d62728', 'leaked_planing': '#9467bd', 'leaked_submergence': '#7f7f7f',
            'leaked_accumulation': '#bcbd22'
        })
        
        self.leakage_types = {'none': 0, 'entrainment': 1, 'drainage': 2, 'splashover': 3, 'structural': 4, 'planing': 5, 'submergence': 6, 'accumulation': 7}
        self.leakage_type_names = {v: k for k, v in self.leakage_types.items()}
        
        # Add a new configuration setting to OpenDrift, allowing the user
        # to easily enable or disable the boom interaction process.
        self._add_config({
            'processes:booms_interaction': {
                'type': 'bool', 'default': True,
                'description': 'Enables interaction of oil particles with containment booms.',
                'level': CONFIG_LEVEL_BASIC
            }
        })

    def add_boom_polyline(self, coords, boom_type='curtain', name='Boom', **overrides):
        """
        Adds a boom to the simulation defined by a polyline (a list of lon/lat points).
        The polyline is discretized into individual linear segments, each of which
        is treated as an independent physical entity in the simulation.

        Args:
            coords (list of tuples): A list of (longitude, latitude) coordinates
                                     defining the boom's geometry.
            boom_type (str): The type of boom (e.g., 'curtain', 'fence'). Must
                             be a key in `self.boom_types`.
            name (str): A user-defined name for the overall boom structure.
            **overrides: Keyword arguments to override default parameters for
                         this specific boom (e.g., anchor_strength_N=100000).
        """
        if len(coords) < 2:
            logger.warning(f"Boom '{name}' has fewer than 2 points and will be ignored.")
            return

        # Internally, a long polyline is treated as a series of connected segments.
        for i in range(len(coords) - 1):
            start_lon, start_lat = coords[i]
            end_lon, end_lat = coords[i+1]
            segment_name = f"{name}_segment_{i+1}"
            self._add_boom_segment(start_lon, start_lat, end_lon, end_lat, boom_type, segment_name, main_boom_name=name, **overrides)
        
        logger.info(f"Added full boom '{name}' of type '{boom_type}' with {len(coords)-1} independent segments.")

    def _add_boom_segment(self, start_lon, start_lat, end_lon, end_lat, boom_type, name, main_boom_name, **overrides):
        """
        Internal helper method to create and store the data structure for a
        single, independent boom segment.

        Args:
            start_lon, start_lat (float): Coordinates of the segment's start point.
            end_lon, end_lat (float): Coordinates of the segment's end point.
            boom_type (str): The type of boom (e.g., 'curtain').
            name (str): The unique name of this specific segment.
            main_boom_name (str): The name of the parent boom structure.
            **overrides: Custom parameters to apply over the defaults.
        """
        if boom_type not in self.boom_types:
            raise ValueError(f"Unknown boom type: '{boom_type}'. Available types: {list(self.boom_types.keys())}")
        
        base_properties = self.boom_types[boom_type].copy()
        base_properties.update(overrides)
        
        # Calculate the geodetic length of the segment.
        geod = pyproj.Geod(ellps='WGS84')
        _, _, length_m = geod.inv(start_lon, start_lat, end_lon, end_lat)
        
        # Create the dictionary that holds all static and dynamic properties of the segment.
        boom_data = {
            'lon': np.array([start_lon, end_lon]),
            'lat': np.array([start_lat, end_lat]),
            'name': name,
            'main_boom_name': main_boom_name,
            'segment_id': self.next_segment_id, # Assign a unique ID.
            'length_m': length_m,
            **base_properties,
            # --- State variables initialized at the start of the simulation ---
            'structurally_failed': False,
            'rotationally_failed': False,
            'breaking_force_N': 0.0,
            'max_force_experienced_N': 0.0,
            'force_timeseries_N': [],
            'structural_failure_time': None,
            'failure_processed': False,
            'leakage_events': [],
            'mass_contained': 0.0,
            'mass_absorbed': 0.0,
            'leakage_mass_per_step': {leak_type: [] for leak_type in self.leakage_type_names.values() if leak_type != 'none'},
            'time_steps_hours': []
        }
        # Initialize leakage mass counters.
        for leak_type in self.leakage_type_names.values():
            if leak_type != 'none':
                boom_data[f'mass_leaked_{leak_type}'] = 0.0
        
        self.booms.append(boom_data)
        self.next_segment_id += 1  # Increment for the next segment.
        
    def _get_boom_projection(self):
        """
        Creates and caches a local stereographic projection.
        This is necessary for accurate and efficient geometric calculations
        (e.g., intersections) in a projected Cartesian space, avoiding the
        complexities of geodetic calculations on a sphere.
        The projection is centered on the mean location of the particles/booms.
        """
        if self._boom_projection:
            return self._boom_projection
        
        if self.num_elements_active() > 0:
            mean_lon, mean_lat = self.elements.lon.mean(), self.elements.lat.mean()
        elif self.booms:
            all_lons = np.concatenate([b['lon'] for b in self.booms])
            all_lats = np.concatenate([b['lat'] for b in self.booms])
            mean_lon, mean_lat = np.mean(all_lons), np.mean(all_lats)
        else:
            mean_lon, mean_lat = 0, 0
            
        proj_string = f'+proj=stere +lat_0={mean_lat} +lon_0={mean_lon} +datum=WGS84'
        self._boom_projection = pyproj.Proj(proj_string)
        return self._boom_projection

    def _get_intersection_point(self, p1_lon, p1_lat, p2_lon, p2_lat, boom_linestring_proj):
        """
        Checks if a particle's path (from its previous to current position)
        intersects with a given boom segment.

        Args:
            p1_lon, p1_lat (float): Particle's previous longitude and latitude.
            p2_lon, p2_lat (float): Particle's current longitude and latitude.
            boom_linestring_proj (shapely.LineString): The boom segment,
                                                      already in projected coordinates.
        Returns:
            A tuple (lon, lat) of the intersection point, or None if no intersection.
        """
        proj = self._get_boom_projection()
        try:
            p1_x, p1_y = proj(p1_lon, p1_lat)
            p2_x, p2_y = proj(p2_lon, p2_lat)
        except Exception:
            # Can fail if coordinates are invalid.
            return None

        particle_path_proj = LineString([(p1_x, p1_y), (p2_x, p2_y)])
        
        if particle_path_proj.intersects(boom_linestring_proj):
            intersection_xy = particle_path_proj.intersection(boom_linestring_proj)
            if isinstance(intersection_xy, Point):
                lon, lat = proj(intersection_xy.x, intersection_xy.y, inverse=True)
                return lon, lat
        return None

    def _update_boom_forces(self, boom, env, mean_oil_viscosity_m2s):
        """
        Calculates the total environmental force acting on a boom segment.
        This is a parameterized representation of complex fluid-structure
        interaction, suitable for an operational-scale model. The total force is
        the sum of forces from current, wind, waves, and the drag from contained oil.

        The general hydrodynamic drag force is modeled as:
        $F = 0.5 * \rho * U^2 * A * C_d$
        where:
            \rho: fluid density (kg/m^3)
            U: perpendicular fluid velocity (m/s)
            A: projected area (m^2)
            C_d: dimensionless drag coefficient

        Args:
            boom (dict): The data dictionary for the boom segment.
            env (dict): A dictionary of environmental conditions at the boom's location.
            mean_oil_viscosity_m2s (float): Mean viscosity of contained oil (m^2/s).
        
        Returns:
            (dict): A dictionary containing the individual and total forces in Newtons (N).
        """
        # Extract local environmental conditions.
        local_x_vel = env['x_sea_water_velocity']
        local_y_vel = env['y_sea_water_velocity']
        local_x_wind = env['x_wind']
        local_y_wind = env['y_wind']
        local_wave_height = env['sea_surface_wave_significant_height']
        local_wave_period = env.get('sea_surface_wave_period_at_variance_spectral_density_maximum', 5.0)
         
        # Calculate the vector normal to the boom segment to find perpendicular velocities.
        b1_x, b1_y = boom['projected_linestring'].coords[0]
        b2_x, b2_y = boom['projected_linestring'].coords[1]
        normal_dx, normal_dy = -(b2_y - b1_y), (b2_x - b1_x)
        norm_normal = np.sqrt(normal_dx**2 + normal_dy**2)
    
        perp_current_speed, perp_wind_speed = 0, 0
        if norm_normal > 1e-9:
            normal_vec = np.array([normal_dx, normal_dy]) / norm_normal
            perp_current_speed = np.abs(np.dot(np.array([local_x_vel, local_y_vel]), normal_vec))
            perp_wind_speed = np.abs(np.dot(np.array([local_x_wind, local_y_wind]), normal_vec))

        # --- Calculate individual force components ---
        # Force from sea current on the submerged skirt.
        force_current = 0.5 * boom['length_m'] * 1025 * (perp_current_speed**2) * boom['skirt_depth_m'] * boom['drag_coeff']
        # Force from wind on the exposed freeboard.
        force_wind = 0.5 * boom['length_m'] * 1.225 * (perp_wind_speed**2) * boom['freeboard_height_m'] * boom['drag_coeff']
        # Parameterized force from waves.
        force_wave = boom['wave_force_coeff'] * boom['length_m'] * (local_wave_height**2) * (1 + 1 / (local_wave_period + 1.0))
        
        # Additional drag force from the accumulated oil slick itself.
        force_oil_drag = 0
        if boom['mass_contained'] > 1.0:
            oil_layer_thickness = boom['oil_thickness']
            U_oil_water = perp_current_speed
            Re_oil = (U_oil_water * oil_layer_thickness) / (mean_oil_viscosity_m2s + 1e-9)
            Cf_oil = 0.074 / (Re_oil**0.2) if Re_oil > 100 else 1.328 / np.sqrt(Re_oil + 1e-9)
            force_oil_drag = 0.5 * 1025 * (U_oil_water**2) * Cf_oil * boom['length_m'] * boom['accumulation_width_m']

        total_force = force_current + force_wind + force_wave + force_oil_drag
        
        # Update boom state with force data.
        boom['max_force_experienced_N'] = max(boom['max_force_experienced_N'], total_force)
        boom['force_timeseries_N'].append(total_force)
        
        return {'current': force_current, 'wind': force_wind, 'wave': force_wave, 'total': total_force}

    def _check_structural_failure(self, boom, total_force):
        """
        Checks if the total force on the boom exceeds its structural strength.
        If it does, the boom segment is marked as failed.

        Args:
            boom (dict): The boom segment's data dictionary.
            total_force (float): The total environmental force in Newtons (N).
        """
        if not boom['structurally_failed'] and total_force > boom['anchor_strength_N']:
            boom['structurally_failed'] = True
            boom['breaking_force_N'] = total_force
            boom['structural_failure_time'] = self.time
            logger.warning(f"STRUCTURAL FAILURE on {boom['name']}. Force: {total_force:.0f}N > Strength: {boom['anchor_strength_N']:.0f}N.")

    def _check_rotational_failure(self, boom, forces):
        """
        Checks for rotational failure modes (planing or submergence).
        This is modeled as a balance between the overturning moments from wind and
        current, and the restoring moment from the boom's buoyancy.

        Args:
            boom (dict): The boom segment's data dictionary.
            forces (dict): A dictionary of the force components.
        """
        if boom['structurally_failed'] or boom['rotationally_failed']:
            return

        moment_current = forces['current'] * (boom['skirt_depth_m'] / 2.0)
        moment_wind = forces['wind'] * (boom['freeboard_height_m'] / 2.0)
        total_overturning_moment = moment_current + moment_wind
        
        # The restoring moment is proportional to the boom's excess buoyancy.
        restoring_moment = (boom['buoyancy_to_weight_ratio'] - 1.0) * boom['length_m'] * boom['restoring_force_per_m']
        
        if total_overturning_moment > max(restoring_moment, 1.0):
            boom['rotationally_failed'] = True
            logger.warning(f"ROTATIONAL FAILURE (Planing/Submergence) on {boom['name']}.")
            
    def _get_leakage_cause(self, boom, element_index, env_props):
        """
        Determines if an oil particle escapes containment and identifies the
        physical mechanism responsible. This is the core of the boom failure physics.

        Args:
            boom (dict): The boom segment the particle is interacting with.
            element_index (int): The index of the particle in `self.elements`.
            env_props (dict): Environmental conditions at the particle's location.

        Returns:
            (str): The name of the leakage mechanism ('none' if contained).
        """
        if self.rng is None:
            # Initialize the Random Number Generator if not already done.
            try:
                seed = self.get_config('seed:random_seed')
            except ValueError:
                seed = 0
            self.rng = np.random.default_rng(seed)

        g = 9.81  # Acceleration due to gravity (m/s^2)
        
        # --- Extract local environmental properties for the particle ---
        element_x_vel = env_props['x_sea_water_velocity']
        element_y_vel = env_props['y_sea_water_velocity']
        element_wh = env_props['sea_surface_wave_significant_height']
        element_swd = env_props['sea_water_density']
        element_sfd = env_props['sea_floor_depth_below_sea_level']
        element_wp = env_props['sea_surface_wave_period_at_variance_spectral_density_maximum']

        # Adjust boom geometry based on environmental conditions.
        effective_freeboard = boom['freeboard_height_m'] * (1 - (element_wh / boom['buoyancy_to_weight_ratio']) * 0.05)
        effective_skirt_depth = boom['skirt_depth_m'] - (boom.get('oil_thickness', 0.0) * boom['accumulation_factor'])
        effective_skirt_depth = max(0.1, effective_skirt_depth)

        # Specific logic for shore-sealing booms.
        if boom['structural_type'] == 'shore_sealing' and element_sfd < effective_skirt_depth:
            return 'none'  # Boom is sealed to the seabed.
        
        # --- Evaluate Hydrodynamic Failure Modes ---
        b1_x, b1_y = boom['projected_linestring'].coords[0]
        b2_x, b2_y = boom['projected_linestring'].coords[1]
        normal_vec_m = np.array([-(b2_y - b1_y), b2_x - b1_x])
        norm_m = np.linalg.norm(normal_vec_m)
        U_perp = 0 # Perpendicular current velocity
        if norm_m > 1e-9:
            normal_vec_m /= norm_m
            U_perp = np.abs(np.dot(np.array([element_x_vel, element_y_vel]), normal_vec_m))

        # 1. Drainage Failure: Governed by the Densiometric Froude Number.
        # Occurs when current inertia overcomes the gravitational stability of the
        # oil layer. Formula: Fr = U_perp / sqrt(g' * D_eff), where g' is reduced gravity.
        delta_rho_frac = (element_swd - self.elements.density[element_index]) / (element_swd + 1e-9)
        if delta_rho_frac <= 0: delta_rho_frac = 1e-6
        froude_number = U_perp / (np.sqrt(g * effective_skirt_depth * delta_rho_frac) + 1e-9)
        if froude_number > boom['critical_froude_number']:
            return 'drainage'

        # 2. Entrainment Failure: Stochastic model of droplet tear-off.
        # Represents Kelvin-Helmholtz instabilities at the oil-water interface.
        # The probability of leakage increases with velocity and decreases with oil viscosity.
        if U_perp > boom['entrainment_critical_velocity_ms']:
            dynamic_entrainment_coeff = boom['entrainment_coefficient'] / (1 + boom['viscosity_damping_factor'] * self.elements.viscosity[element_index])
            entrainment_prob = (1 - np.exp(-(dynamic_entrainment_coeff * (U_perp**2)) * self.time_step.total_seconds()))
            if self.rng.random() < entrainment_prob:
                return 'entrainment'
        
        # 3. Splash-over Failure: Probabilistic model of wave overtopping.
        effective_wh_splashover = element_wh * (1 + boom['wave_period_factor'] / (element_wp + 1e-6))
        splashover_threshold = effective_freeboard * boom['splashover_factor']
        if effective_wh_splashover > splashover_threshold:
            splashover_prob = (effective_wh_splashover - splashover_threshold) / (effective_wh_splashover + 1e-9)
            if self.rng.random() < splashover_prob:
                return 'splashover'

        # 4. Critical Accumulation Failure: For highly viscous oils.
        if self.elements.viscosity[element_index] * 1e6 > boom['critical_accumulation_viscosity_cst'] and froude_number > boom['critical_accumulation_froude']:
            return 'accumulation'

        # If no failure conditions are met, the particle is contained.
        return 'none'

    def interact_with_booms(self):
        """
        Main loop for processing boom-oil interactions at each time step.
        It first updates the state of all booms (checking for structural failures)
        and then iterates through active oil particles to check for collisions
        and determine containment or leakage.
        """
        if not self.booms or self.num_elements_active() == 0:
            return

        if self.rng is None:
            try:
                seed = self.get_config('seed:random_seed')
            except ValueError:
                seed = 0
            self.rng = np.random.default_rng(seed)
            
        # --- Pre-computation for efficiency ---
        active_status_code = self.status_categories.index('active')
        active_indices_array = np.where(self.elements.status == active_status_code)[0]
        active_indices_map = {full_idx: active_idx for active_idx, full_idx in enumerate(active_indices_array)}

        env = self.environment
        default_0 = np.zeros(self.num_elements_active())
        default_inf = np.full(self.num_elements_active(), np.inf)
        default_rho = np.full(self.num_elements_active(), 1025.0)
        default_wp = np.full(self.num_elements_active(), 5.0)
        env_x_vel = getattr(env, 'x_sea_water_velocity', default_0)
        env_y_vel = getattr(env, 'y_sea_water_velocity', default_0)
        env_x_wind = getattr(env, 'x_wind', default_0)
        env_y_wind = getattr(env, 'y_wind', default_0)
        env_wh = getattr(env, 'sea_surface_wave_significant_height', default_0)
        env_wd = getattr(env, 'sea_water_density', default_rho)
        env_sfd = getattr(env, 'sea_floor_depth_below_sea_level', default_inf)
        env_wp = getattr(env, 'sea_surface_wave_period_at_variance_spectral_density_maximum', default_wp)
        
        pre_step_leakage = {b['name']: {f'mass_leaked_{k}': b[f'mass_leaked_{k}'] for k in self.leakage_type_names.values() if k != 'none'} for b in self.booms}
        
        # --- Phase 1: Update the state of each independent boom segment ---
        for boom in self.booms:
            if 'projected_linestring' not in boom:
                proj = self._get_boom_projection()
                b1_x, b1_y = proj(boom['lon'][0], boom['lat'][0])
                b2_x, b2_y = proj(boom['lon'][1], boom['lat'][1])
                boom['projected_linestring'] = LineString([(b1_x, b1_y), (b2_x, b2_y)])
            
            # Create a mask for particles contained ONLY by this specific segment.
            boom_specific_mask = self.elements.contained_by_boom_id == boom['segment_id']
            boom['mass_contained'] = np.nansum(self.elements.mass_oil[boom_specific_mask])
            
            mean_oil_density = np.mean(self.elements.density[boom_specific_mask]) if np.any(boom_specific_mask) else 900.0
            denominator = boom['length_m'] * boom['accumulation_width_m'] * mean_oil_density
            boom['oil_thickness'] = boom['mass_contained'] / (denominator + 1e-9)
            
            mean_viscosity = np.mean(self.elements.viscosity[boom_specific_mask]) if np.any(boom_specific_mask) else 1e-5

            # Determine environment at segment: use contained particles or fallback to global mean.
            sanitized_boom_env = {}
            contained_indices_in_full_array = np.where(boom_specific_mask)[0]
            contained_active_indices = [active_indices_map[i] for i in contained_indices_in_full_array if i in active_indices_map]

            if contained_active_indices:
                sanitized_boom_env = {
                    'x_sea_water_velocity': np.mean(env_x_vel[contained_active_indices]), 'y_sea_water_velocity': np.mean(env_y_vel[contained_active_indices]),
                    'x_wind': np.mean(env_x_wind[contained_active_indices]), 'y_wind': np.mean(env_y_wind[contained_active_indices]),
                    'sea_surface_wave_significant_height': np.mean(env_wh[contained_active_indices]),
                    'sea_surface_wave_period_at_variance_spectral_density_maximum': np.mean(env_wp[contained_active_indices])
                }
            elif self.num_elements_active() > 0: # Fallback to global average if segment is empty
                sanitized_boom_env = {
                    'x_sea_water_velocity': np.mean(env_x_vel), 'y_sea_water_velocity': np.mean(env_y_vel),
                    'x_wind': np.mean(env_x_wind), 'y_wind': np.mean(env_y_wind),
                    'sea_surface_wave_significant_height': np.mean(env_wh),
                    'sea_surface_wave_period_at_variance_spectral_density_maximum': np.mean(env_wp)
                }
            else: # Fallback to zeros if no particles are active
                sanitized_boom_env = {'x_sea_water_velocity': 0.0, 'y_sea_water_velocity': 0.0, 'x_wind': 0.0, 'y_wind': 0.0,
                                      'sea_surface_wave_significant_height': 0.0, 'sea_surface_wave_period_at_variance_spectral_density_maximum': 5.0}
            
            # Calculate forces and check for mechanical failures for this segment.
            forces = self._update_boom_forces(boom, sanitized_boom_env, mean_viscosity)
            self._check_structural_failure(boom, forces['total'])
            self._check_rotational_failure(boom, forces)

            # If segment has failed, release ONLY the particles it contains.
            if (boom['structurally_failed'] or boom['rotationally_failed']) and not boom.get('failure_processed', False):
                fail_type = 'structural' if boom['structurally_failed'] else 'planing'
                contained_indices = np.where(self.elements.contained_by_boom_id == boom['segment_id'])[0]
                if contained_indices.size > 0:
                    self.elements.leakage_status[contained_indices] = self.leakage_types[fail_type]
                    self.elements.is_contained[contained_indices] = 0
                    self.elements.contained_by_boom_id[contained_indices] = -1
                    leaked_mass = np.sum(self.elements.mass_oil[contained_indices])
                    boom[f'mass_leaked_{fail_type}'] += leaked_mass
                boom['failure_processed'] = True # Ensure this logic runs only once.
        
        # --- Phase 2: Process particle interactions with non-failed booms ---
        indices_to_process = np.where(np.isin(self.elements.ID, list(self.boom_previous_lon.keys())))[0]

        for element_index in indices_to_process:
            active_idx = active_indices_map.get(element_index)
            if active_idx is None: continue

            p_id = self.elements.ID[element_index]
            current_segment_id = self.elements.contained_by_boom_id[element_index]
            
            for boom in self.booms:
                if boom['structurally_failed'] or boom['rotationally_failed']:
                    continue # Skip interaction with failed segments.
        
                is_contained_by_this_segment = (current_segment_id == boom['segment_id'])
                
                # A particle can only collide if it is not already contained by ANY segment.
                is_colliding = False
                if current_segment_id == -1:
                    intersection_point = self._get_intersection_point(
                        self.boom_previous_lon[p_id], self.boom_previous_lat[p_id],
                        self.elements.lon[element_index], self.elements.lat[element_index],
                        boom['projected_linestring'])
                    if intersection_point:
                        is_colliding = True

                if not (is_contained_by_this_segment or is_colliding):
                    continue
                
                # Get local environmental conditions at the particle's location.
                env_props_particle = {
                    'x_sea_water_velocity': env_x_vel[active_idx], 'y_sea_water_velocity': env_y_vel[active_idx],
                    'sea_surface_wave_significant_height': env_wh[active_idx], 'sea_water_density': env_wd[active_idx],
                    'sea_floor_depth_below_sea_level': env_sfd[active_idx],
                    'sea_surface_wave_period_at_variance_spectral_density_maximum': env_wp[active_idx]
                }
                
                leakage_cause = self._get_leakage_cause(boom, element_index, env_props_particle)

                if leakage_cause != 'none':
                    # Leakage occurred: update particle state and allow it to move.
                    self.elements.is_contained[element_index] = 0
                    self.elements.contained_by_boom_id[element_index] = -1
                    self.elements.leakage_status[element_index] = self.leakage_types[leakage_cause]
                    leaked_mass = self.elements.mass_oil[element_index]
                    boom[f'mass_leaked_{leakage_cause}'] += leaked_mass
                else:
                    # Containment: update particle state and revert its position.
                    self.elements.is_contained[element_index] = 1
                    self.elements.contained_by_boom_id[element_index] = boom['segment_id']
                    # Revert particle position, adding a small random "jitter" to avoid
                    # numerical sticking. This acts as a proxy for micro-turbulence.
                    jitter_m = 0.1
                    prev_lat = self.boom_previous_lat[p_id]
                    lon_jitter_deg = (self.rng.uniform(-jitter_m, jitter_m) / (111320 * np.cos(np.radians(prev_lat))))
                    lat_jitter_deg = self.rng.uniform(-jitter_m, jitter_m) / 110574
                    self.elements.lon[element_index] = self.boom_previous_lon[p_id] + lon_jitter_deg
                    self.elements.lat[element_index] = prev_lat + lat_jitter_deg
                    
                    # Handle specific logic for sorbent booms (mass removal).
                    if boom['structural_type'] == 'sorbent':
                        capacity_kg = boom['capacity_kg_per_m'] * boom['length_m']
                        if boom['mass_absorbed'] < capacity_kg:
                            absorbable_mass = self.elements.mass_oil[element_index] * boom['absorption_rate']
                            actual_absorbed = min(absorbable_mass, capacity_kg - boom['mass_absorbed'])
                            self.elements.mass_oil[element_index] -= actual_absorbed
                            boom['mass_absorbed'] += actual_absorbed
                            if self.elements.mass_oil[element_index] < 1e-6:
                                self.elements.status[element_index] = self.status_categories.index('absorbed')
                                self.elements.is_contained[element_index] = 0
                                self.elements.contained_by_boom_id[element_index] = -1
                break # Particle interacts with only one segment per time step.

        # --- Finalize time step by recording summary data ---
        current_hours = (self.time - self.start_time).total_seconds() / 3600.0
        for boom in self.booms:
            boom['time_steps_hours'].append(current_hours)
            for leak_type in boom['leakage_mass_per_step'].keys():
                key = f'mass_leaked_{leak_type}'
                delta_mass = boom.get(key, 0) - pre_step_leakage.get(boom['name'], {}).get(key, 0)
                boom['leakage_mass_per_step'][leak_type].append(delta_mass)

    def _plot_booms(self, ax):
        """
        Custom plotting method to draw booms on the output map.
        It is called by the `set_up_map` hook.
        Failed booms are styled with a dashed line for clear visual feedback.
        """
        if not self.booms or ax is None:
            return
        
        handled_labels = set()
        for boom in self.booms:
            props = boom['plot_props'].copy()
            props.update({'transform': ccrs.Geodetic(), 'linewidth': 2.5})
            
            # Logic to ensure each overall boom structure only gets one label in the legend.
            if boom['main_boom_name'] not in handled_labels:
                props['label'] = boom['main_boom_name']
                try:
                    boom_number = re.search(r'\d+', boom['main_boom_name']).group()
                    label = f"B{boom_number}"
                    ax.text(np.mean(boom['lon']), np.mean(boom['lat']), label, 
                            transform=ccrs.Geodetic(), fontsize=9, fontweight='bold', color='white',
                            bbox=dict(facecolor=props['color'], alpha=0.8, pad=1, edgecolor='none'))
                except AttributeError:
                    pass
                handled_labels.add(boom['main_boom_name'])
            
            if boom['structurally_failed'] or boom['rotationally_failed']:
                props['linestyle'] = '--' # Visual indicator for failed segments.
            
            ax.plot(boom['lon'], boom['lat'], **props)
    
    def get_element_colors(self, variables=None):
        """
        Overrides the default particle coloring method to provide custom colors
        based on boom interaction states (contained, type of leakage).
        """
        colors = np.array(super().get_element_colors(variables))
        
        contained_mask = self.elements.is_contained == 1
        colors[contained_mask] = self.status_colors['contained']

        # Color particles that have leaked based on the cause of leakage.
        active_status_code = self.status_categories.index('active')
        active_mask = self.elements.status == active_status_code
        leakage_mask = self.elements.leakage_status != 0
        leaked_and_active = active_mask & leakage_mask
        
        if np.any(leaked_and_active):
            indices = np.where(leaked_and_active)[0]
            for i in indices:
                if not contained_mask[i]:
                    leakage_code = self.elements.leakage_status[i]
                    leakage_name = 'leaked_' + self.leakage_type_names.get(int(leakage_code), 'unknown')
                    colors[i] = self.status_colors.get(leakage_name, '#FFFFFF')

        return colors.tolist()

    def update(self):
        """
        Overrides the main update method of the OpenDrift model.
        
        This function is called at every time step. It first stores the current
        particle positions, then calls the parent's update method (for standard
        advection and weathering), and finally calls `interact_with_booms`
        to apply the custom containment physics.
        """
        if self.num_elements_active() > 0:
            # Store particle positions before movement for collision detection.
            self.boom_previous_lon = {ID: lon for ID, lon in zip(self.elements.ID, self.elements.lon)}
            self.boom_previous_lat = {ID: lat for ID, lat in zip(self.elements.ID, self.elements.lat)}
            
            try:
                # Log mean environmental conditions for post-simulation analysis.
                env = self.environment
                mean_speed = np.sqrt(np.nanmean(getattr(env, 'x_sea_water_velocity', 0.0))**2 + np.nanmean(getattr(env, 'y_sea_water_velocity', 0.0))**2)
                mean_wave_height = np.nanmean(getattr(env, 'sea_surface_wave_significant_height', 0.0))
                mean_wind_speed = np.sqrt(np.nanmean(getattr(env, 'x_wind', 0.0))**2 + np.nanmean(getattr(env, 'y_wind', 0.0))**2)

                self.env_history.append({
                    'time': self.time,
                    'current_speed_mps': float(np.nan_to_num(mean_speed)),
                    'wave_height_m': float(np.nan_to_num(mean_wave_height)),
                    'wind_speed_ms': float(np.nan_to_num(mean_wind_speed))
                })
            except Exception as e:
                logger.warning(f"Could not retrieve full environmental data for history: {e}")

        # Call the standard OpenOil update method to move particles and apply weathering.
        super().update()
        
        # If the boom interaction process is enabled, execute it now.
        if self.get_config('processes:booms_interaction'):
            self.interact_with_booms()

    def set_up_map(self, *args, **kwargs):
        """
        Hooks into the OpenDrift plotting process to ensure the custom `_plot_booms`
        method is called when a map or animation is generated.
        """
        fig, ax, crs, x, y, index_of_first, index_of_last = super().set_up_map(*args, **kwargs)
        self._plot_booms(ax)
        return fig, ax, crs, x, y, index_of_first, index_of_last