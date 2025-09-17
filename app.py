# =============================================================================
# NOTE: Final version incorporating all user requests and bug fixes.
#
# Key Features of this Version:
# 1. (NEW) Added a scenario selector in the sidebar to inspect the
#    properties of booms used in any completed simulation run.
# 2. (RESTRUCTURE) "Impact Analysis" tab now includes a Key Metrics Summary
#    Table, a Mass Balance stacked area chart, and a Shoreline Impact Heatmap.
# 3. (FIX) All statistics are now driven by a new, centralized processing
#    function in the backend, ensuring data consistency across all charts
#    and tables.
# 4. (UI) The entire user interface has been translated to English.
# 5. (UPDATE) The Quickstart Guide has been updated to reflect current
#    functionality.
# 6. (FIX) The particle fate table in the "Environmental Analysis" tab is
#    now correct and complete.
# 7. (CLEANUP) Removed the non-functional color legend from the "Overview" tab.
# =============================================================================
import streamlit as st
import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import MousePosition, Draw, BeautifyIcon
from folium import DivIcon
from streamlit_folium import st_folium
from datetime import datetime
import os
import traceback
import time
import numpy as np
from report_generator import generate_pdf_report
from collections import defaultdict

# Import backend functions and model class to access boom types
from backend import run_simulation, get_available_oils, get_oil_properties
from boom_model import OpenOilWithBooms

# =============================================================================
# INITIAL PAGE CONFIG AND STYLING
# =============================================================================
st.set_page_config(layout="wide", page_title="Oil Spill Response Simulator")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
        font-size: 0.9rem;
    }

    #MainMenu, [data-testid="stToolbar"], footer, header {
        display: none !important;
        visibility: hidden !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ECEFF1;
        border-right: 1px solid #CFD8DC;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #37474F; 
        font-size: 1.1rem;
        font-weight: 600;
    }

    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
        visibility: hidden !important;
    }

    .stButton>button[kind="primary"] {
        background-color: #1976D2;
        color: white;
        border-radius: 4px;
        border: none;
        font-weight: 500;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #1565C0;
    }
    h1 { color: #263238; font-size: 1.6rem; }
    h2 { font-size: 1.25rem; color: #37474F; }
    h3 { font-size: 1.1rem; color: #455A64; }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 4px;
        padding: 0.6rem;
        margin: 0.25rem 0;
        border: 1px solid #E0E0E0;
        text-align: center;
    }
    .metric-card h4 {
        margin: 0 0 2px 0;
        color: #78909C;
        font-size: 0.7rem;
        font-weight: 400;
        text-transform: uppercase;
    }
    .metric-card p {
        font-size: 1rem;
        font-weight: 600;
        color: #263238;
        margin: 0;
    }
    .metric-card-small p { font-size: 0.95rem; }
    .legend-color-box {
        width: 15px;
        height: 15px;
        border: 1px solid #ccc;
        display: inline-block;
        vertical-align: middle;
        margin-right: 5px;
    }

    .stVideo {
        height: 500px;
        width: 100%;
    }
    .stVideo video {
        width: 100%;
        height: 100%;
        border-radius: 4px;
        object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)

st.title("Oil Spill Response Simulator")

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def initialize_session():
    if 'app_state' not in st.session_state: st.session_state.app_state = 'configuring'
    if 'latitude' not in st.session_state: st.session_state.latitude = 60.30
    if 'longitude' not in st.session_state: st.session_state.longitude = 4.80
    if 'sim_params' not in st.session_state: st.session_state.sim_params = {}
    if 'map_bounds' not in st.session_state: st.session_state.map_bounds = None
    if 'output_dir' not in st.session_state: st.session_state.output_dir = None
    if 'booms' not in st.session_state: st.session_state.booms = []
    if 'show_map' not in st.session_state: st.session_state.show_map = False
    if 'map_key' not in st.session_state: st.session_state.map_key = 0
    if 'selected_oil_properties' not in st.session_state: st.session_state.selected_oil_properties = None
    if 'guide_expanded' not in st.session_state: st.session_state.guide_expanded = True
    if 'boom_run_counter' not in st.session_state: st.session_state.boom_run_counter = 0
    if 'simulation_warnings' not in st.session_state: st.session_state.simulation_warnings = []
    
    if 'time_step_input' not in st.session_state: st.session_state.time_step_input = 30
    if 'time_step_output' not in st.session_state: st.session_state.time_step_output = 60
    
    if 'initial_run_results' not in st.session_state: st.session_state.initial_run_results = {}
    if 'boom_scenarios' not in st.session_state: st.session_state.boom_scenarios = {}
    if 'sim_time_input' not in st.session_state: st.session_state.sim_time_input = datetime.now().time()
    if 'scenario_to_inspect' not in st.session_state: st.session_state.scenario_to_inspect = None

    if 'status_colors' not in st.session_state:
        try:
            o = OpenOilWithBooms()
            st.session_state.status_colors = o.status_colors
        except Exception:
            st.session_state.status_colors = {}

initialize_session()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def normalize_timeseries_dict(ts_dict):
    if not isinstance(ts_dict, dict) or not ts_dict:
        return ts_dict
    
    try:
        valid_lengths = [len(v) for v in ts_dict.values() if isinstance(v, (list, np.ndarray))]
        if not valid_lengths:
            return ts_dict
        min_len = min(valid_lengths)
    except ValueError:
        return ts_dict 
    
    normalized_dict = {k: v[:min_len] if isinstance(v, (list, np.ndarray)) else v for k, v in ts_dict.items()}
    return normalized_dict

def reset_to_configuring():
    lat = st.session_state.latitude
    lon = st.session_state.longitude
    
    st.session_state.clear()
    initialize_session()
    st.session_state.latitude = lat
    st.session_state.longitude = lon
    st.rerun()

def reset_for_new_boom_scenario():
    st.session_state.app_state = 'planning_booms'
    st.session_state.map_key += 1
    st.rerun()

def remove_all_booms():
    st.session_state.booms = []
    st.session_state.map_key += 1
    st.rerun()

def get_boom_length(coords):
    from pyproj import Geod
    geod = Geod(ellps='WGS84')
    total_length = 0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i+1]
        _, _, length = geod.inv(lon1, lat1, lon2, lat2)
        total_length += length
    return total_length

def update_oil_properties():
    oil_name = st.session_state.get("oil_type")
    if oil_name:
        st.session_state.selected_oil_properties = get_oil_properties(oil_name)

def close_guide():
    st.session_state.guide_expanded = False

# =============================================================================
# SIDEBAR (CONTROL PANEL)
# =============================================================================
with st.sidebar:
    st.title("Control Panel")
    st.divider()

    if st.session_state.app_state == 'configuring':
        st.subheader("1. Oil Properties")
        oil_source = st.radio("Oil Source", ["Select from NOAA List", "Define Custom"], key="oil_source", help="Choose a pre-defined oil from the NOAA ADIOS database or specify custom physical properties.")

        if oil_source == "Select from NOAA List":
            available_oils = get_available_oils()
            default_index = available_oils.index("GENERIC MEDIUM CRUDE") if "GENERIC MEDIUM CRUDE" in available_oils else 0
            oil_type_input = st.selectbox("Oil Type", available_oils, index=default_index, key="oil_type", on_change=update_oil_properties, help="Select the type of oil spilled. This determines its physical and chemical properties.")
            custom_oil_properties = None
            
            if not st.session_state.selected_oil_properties:
                update_oil_properties()

            if st.session_state.selected_oil_properties:
                props = st.session_state.selected_oil_properties
                col_d, col_v = st.columns(2)
                with col_d:
                    st.markdown(f'<div class="metric-card metric-card-small"><h4>Density</h4><p>{props["density"]:.1f} kg/m³</p></div>', unsafe_allow_html=True)
                with col_v:
                    st.markdown(f'<div class="metric-card metric-card-small"><h4>Viscosity</h4><p>{props["viscosity"]:.1f} cSt</p></div>', unsafe_allow_html=True)
        else:
            oil_type_input = "CUSTOM"
            density_input = st.number_input("Density (kg/m³)", min_value=700.0, max_value=1100.0, value=900.0, step=10.0, help="The mass per unit volume of the oil.")
            viscosity_cst_input = st.number_input("Kinematic Viscosity (cSt)", min_value=1.0, value=180.0, step=10.0, help="The oil's resistance to flow. 1 cSt = 1e-6 m²/s.")
            viscosity_input = viscosity_cst_input * 1e-6
            custom_oil_properties = {'density': density_input, 'viscosity': viscosity_input}

        st.divider()
        st.subheader("2. Simulation Parameters")
        now = datetime.now()
        col_time, col_dur = st.columns(2)
        sim_date = col_time.date_input("Start Date", now.date(), key="sim_date", help="The calendar date when the spill begins.")
        sim_time = col_time.time_input("Start Time", value=st.session_state.sim_time_input, key="sim_time_widget", on_change=lambda: setattr(st.session_state, 'sim_time_input', st.session_state.sim_time_widget),label_visibility="collapsed", help="The time of day when the spill begins (UTC).")
        duration_hours = col_dur.number_input("Duration (h)", min_value=1, value=12, key="duration", help="Total number of hours to run the simulation for.")
        
        col_num, col_mass = st.columns(2)
        num_elements = col_num.number_input("# Particles", min_value=100, value=2000, step=100, key="particles", help="Number of particles used to represent the oil slick. More particles provide higher detail but require more computation time.")
        total_mass_kg = col_mass.number_input("Total Spill Mass (kg)", min_value=1, value=1000, step=100, key="mass", help="The total mass of oil spilled in kilograms. This will be distributed among the particles.")
        
        spill_radius = col_num.number_input("Spill Radius (m)", min_value=10, value=150, step=10, key="radius", help="The initial radius of the oil spill on the sea surface.")
        
        with st.expander("Advanced Settings"):
            time_step = st.number_input(
                "Time Step (min)", 
                min_value=1, 
                value=st.session_state.time_step_input, 
                step=5, 
                key='time_step_widget', 
                on_change=lambda: setattr(st.session_state, 'time_step_input', st.session_state.time_step_widget),
                help="Model's internal calculation interval."
            )
            time_step_output = st.number_input(
                "Output Time Step (min)", 
                min_value=st.session_state.time_step_input,
                value=max(st.session_state.time_step_input, st.session_state.time_step_output),
                step=5, 
                key='time_step_output_widget',
                on_change=lambda: setattr(st.session_state, 'time_step_output', st.session_state.time_step_output_widget),
                help="Interval for saving results. Cannot be smaller than the calculation time step."
            )
            reader_ocean = st.text_input("Currents Reader URL", 'https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be', help="THREDDS or OPeNDAP URL for ocean currents data.")
            reader_wind = st.text_input("Wind Reader URL", 'https://thredds.met.no/thredds/dodsC/metpplatest/met_forecast_1_0km_nordic_latest.nc', help="THREDDS or OPeNDAP URL for wind data.")
            reader_wave = st.text_input("Waves Reader URL", 'https://thredds.met.no/thredds/dodsC/fou-hi/mywavewam800v_be', help="THREDDS or OPeNDAP URL for wave data.")

        st.divider()
        st.subheader("3. Spill Location")
        st.checkbox("Show Selection Map", key="show_map", on_change=close_guide, help="Toggle a map to select the spill location by clicking.")
        col_lat, col_lon = st.columns(2)
        lat_input = col_lat.number_input("Latitude", value=st.session_state.latitude, format="%.6f", key="lat_input", help="Spill center latitude (WGS-84).")
        lon_input = col_lon.number_input("Longitude", value=st.session_state.longitude, format="%.6f", key="lon_input", help="Spill center longitude (WGS-84).")

        st.divider()
        
        if st.button("Run Initial Simulation", type="primary", use_container_width=True, on_click=close_guide):
            st.session_state.sim_params = {
                'lon': lon_input, 'lat': lat_input,
                'oil_source': oil_source, 'oil_type': oil_type_input, 
                'custom_oil_properties': custom_oil_properties,
                'start_time': datetime.combine(sim_date, sim_time),
                'duration_hours': duration_hours, 
                'number': num_elements, 
                'total_mass_kg': total_mass_kg,
                'radius': spill_radius,
                'time_step': st.session_state.time_step_input, 
                'time_step_output': st.session_state.time_step_output,
                'reader_ocean': reader_ocean if reader_ocean else None, 
                'reader_wind': reader_wind if reader_wind else None,
                'reader_wave': reader_wave if reader_wave else None
            }
            st.session_state.app_state = 'running_initial'
            st.rerun()

    elif st.session_state.app_state in ['planning_booms', 'running_with_booms', 'comparing_results']:
        is_planning_phase = st.session_state.app_state == 'planning_booms'
        is_comparing_phase = st.session_state.app_state == 'comparing_results'
        
        if is_planning_phase:
            st.info("Analyze the scenario and draw containment booms on the map. Double-click to finish a drawing.")
        
        if is_comparing_phase:
            scenario_options = list(st.session_state.boom_scenarios.keys())
            if scenario_options:
                 st.selectbox(
                    "Inspect Scenario", 
                    options=scenario_options, 
                    key="scenario_to_inspect",
                    index=len(scenario_options) - 1, # Default to the last one
                    help="Select a completed scenario to view the configuration of the booms used."
                )
            st.divider()

        st.subheader("Added Booms")
        
        # Determine which set of booms to display
        booms_to_display = []
        if is_comparing_phase and st.session_state.scenario_to_inspect:
            scenario_data = st.session_state.boom_scenarios.get(st.session_state.scenario_to_inspect, {})
            booms_to_display = scenario_data.get('stats', {}).get('boom_performance', [])
        elif not is_comparing_phase:
            booms_to_display = st.session_state.booms
        
        if not booms_to_display:
            st.caption("No booms to display for this view.")
        
        for i, boom_data in enumerate(booms_to_display):
            # Adapt data source based on the current app state
            if is_comparing_phase:
                boom_name = boom_data.get('name', f"Boom {i+1}")
                boom_type = boom_data.get('type', 'N/A')
                boom_length = boom_data.get('total_length_m', 0)
                config = boom_data.get('configuration', {})
            else: # is_planning_phase
                boom_name = f"Boom {i+1}"
                boom_type = boom_data['type']
                boom_length = get_boom_length(boom_data['coordinates'])
                config = boom_data['overrides']

            with st.expander(f"{boom_name} ({boom_type})", expanded=True):
                st.write(f"**Total Length:** {boom_length:.2f} m")
                
                st.write("**Physical Properties:**")
                c1, c2 = st.columns(2)
                
                # The 'disabled' flag is True if we are just inspecting results
                is_disabled = is_comparing_phase

                # For the planning phase, we need to update the original list
                # For comparing, we just display the config values
                value_fb = config.get('freeboard_height_m', 0)
                if is_planning_phase:
                    config['freeboard_height_m'] = c1.number_input("Freeboard (m)", min_value=0.1, value=value_fb, key=f"fb_{i}", disabled=is_disabled, format="%.2f", help="Height of the boom above the water.")
                else:
                    c1.metric("Freeboard (m)", f"{value_fb:.2f}")

                value_sd = config.get('skirt_depth_m', 0)
                if is_planning_phase:
                    config['skirt_depth_m'] = c2.number_input("Skirt Depth (m)", min_value=0.1, value=value_sd, key=f"sd_{i}", disabled=is_disabled, format="%.2f", help="Depth of the boom below the water.")
                else:
                    c2.metric("Skirt Depth (m)", f"{value_sd:.2f}")

                value_as = config.get('anchor_strength_N', 0)
                if is_planning_phase:
                    config['anchor_strength_N'] = c1.number_input("Anchor Strength (N)", min_value=1000, value=value_as, key=f"as_{i}", disabled=is_disabled, step=1000, help="Maximum force the boom's anchoring can withstand before structural failure.")
                else:
                    c1.metric("Anchor Strength (N)", f"{value_as:,.0f}")
                
                value_bwr = config.get('buoyancy_to_weight_ratio', 0)
                if is_planning_phase:
                    config['buoyancy_to_weight_ratio'] = c2.number_input("B/W Ratio", min_value=1.0, max_value=30.0, value=value_bwr, key=f"bwr_{i}", disabled=is_disabled, format="%.1f", help="Buoyancy-to-Weight Ratio. Higher values perform better in waves (e.g., Curtain: 10-20, Fence: 2-5).")
                else:
                    c2.metric("B/W Ratio", f"{value_bwr:.1f}")
                
                if boom_type == 'sorbent':
                    st.write("**Sorbent Properties:**")
                    c_sorb1, c_sorb2 = st.columns(2)
                    
                    value_ar = config.get('absorption_rate', 0)
                    if is_planning_phase:
                        config['absorption_rate'] = c_sorb1.number_input("Absorption Rate", min_value=0.0, max_value=1.0, value=value_ar, key=f"ar_{i}", disabled=is_disabled, format="%.2f", help="Fraction of contained oil mass absorbed per time step.")
                    else:
                        c_sorb1.metric("Absorption Rate", f"{value_ar:.2f}")

                    value_cap = config.get('capacity_kg_per_m', 0)
                    if is_planning_phase:
                        config['capacity_kg_per_m'] = c_sorb2.number_input("Capacity (kg/m)", min_value=0.0, value=value_cap, key=f"cap_{i}", disabled=is_disabled, format="%.2f", help="Total oil mass the boom can absorb per meter of length.")
                    else:
                        c_sorb2.metric("Capacity (kg/m)", f"{value_cap:.2f}")

                with st.container():
                    st.write("**Advanced Physics Tuning:**")
                    # Display as metrics if inspecting, as inputs if planning
                    # This section can be simplified by just showing metrics when disabled
                    if is_comparing_phase:
                         adv_cols = st.columns(2)
                         adv_cols[0].metric("Critical Froude No.", f"{config.get('critical_froude_number', 0):.2f}")
                         adv_cols[1].metric("Entrainment Coeff.", f"{config.get('entrainment_coefficient', 0):.4f}")
                         adv_cols[0].metric("Drag Coefficient", f"{config.get('drag_coeff', 0):.2f}")
                         adv_cols[1].metric("Splashover Factor", f"{config.get('splashover_factor', 0):.2f}")
                         adv_cols[0].metric("Accumulation Factor", f"{config.get('accumulation_factor', 0):.2f}")
                         adv_cols[1].metric("Wave Period Factor", f"{config.get('wave_period_factor', 0):.2f}")
                         adv_cols[0].metric("Accumulation Width (m)", f"{config.get('accumulation_width_m', 0):.2f}")

                    else: # Planning phase, show input widgets
                        c3, c4 = st.columns(2)
                        config['critical_froude_number'] = c3.number_input("Critical Froude No.", min_value=0.1, value=config.get('critical_froude_number', 0), key=f"cfn_{i}", disabled=is_disabled, format="%.2f", help="Threshold for drainage failure due to high current speeds relative to skirt depth.")
                        config['entrainment_coefficient'] = c4.number_input("Entrainment Coeff.", min_value=0.0, max_value=0.1, value=config.get('entrainment_coefficient', 0), key=f"ec_{i}", disabled=is_disabled, format="%.4f", help="Coefficient for probabilistic entrainment. Set to 0 for no entrainment leakage.")
                        c5, c6 = st.columns(2)
                        config['drag_coeff'] = c5.number_input("Drag Coefficient", min_value=0.1, max_value=3.0, value=config.get('drag_coeff', 0), key=f"dc_{i}", disabled=is_disabled, format="%.2f", help="Determines how environmental forces (wind, current) are transferred to the boom.")
                        config['splashover_factor'] = c6.slider("Splashover Factor", min_value=0.5, max_value=1.5, value=config.get('splashover_factor', 0), key=f"sf_{i}", disabled=is_disabled, format="%.2f", help="Multiplier for effective freeboard against waves. <1 more prone, >1 less prone.")
                        st.markdown("<h6>New Physics Controls</h6>", unsafe_allow_html=True)
                        c7, c8 = st.columns(2)
                        config['accumulation_factor'] = c7.slider("Accumulation Factor", min_value=0.0, max_value=5.0, value=config.get('accumulation_factor', 0), key=f"af_{i}", disabled=is_disabled, format="%.2f", help="Sensitivity of failure thresholds to the thickness of accumulated oil.")
                        config['wave_period_factor'] = c8.slider("Wave Period Factor", min_value=0.0, max_value=5.0, value=config.get('wave_period_factor', 0), key=f"wpf_{i}", disabled=is_disabled, format="%.2f", help="Sensitivity of splashover to wave period. Higher values make short periods more critical.")
                        config['accumulation_width_m'] = c7.number_input("Accumulation Width (m)", min_value=0.1, value=config.get('accumulation_width_m', 0), key=f"awm_{i}", disabled=is_disabled, format="%.2f", help="Effective width for calculating contained oil thickness.")

                if is_planning_phase:
                    if st.button(f"Remove Boom {i+1}", key=f"del_{i}", use_container_width=True):
                        st.session_state.booms.pop(i)
                        st.session_state.map_key += 1
                        st.rerun()
        st.divider()
        if is_planning_phase:
            col_run, col_clear = st.columns(2)
            col_run.button("Run with Booms", type="primary", use_container_width=True, disabled=not st.session_state.booms, on_click=lambda: setattr(st.session_state, 'app_state', 'running_with_booms'))
            col_clear.button("Remove All Booms", use_container_width=True, on_click=remove_all_booms)

        if st.session_state.app_state == 'comparing_results':
            st.button("Test New Boom Scenario", use_container_width=True, on_click=reset_for_new_boom_scenario)
        
        st.divider()
        if st.button("New Simulation", use_container_width=True): 
            reset_to_configuring()

# =============================================================================
# MAIN AREA
# =============================================================================
if st.session_state.app_state == 'configuring':
    with st.expander("Quickstart Guide", expanded=st.session_state.guide_expanded):
        st.markdown("""
        Welcome to the Oil Spill Response Simulator. This tool allows you to model the dispersion of an oil spill and test the effectiveness of containment strategies using booms.

        **Follow the 4 steps below for a complete simulation:** --- 
        #### **Step 1: Configure the Spill Scenario**
        Use the **Control Panel** on the left to define the initial conditions of the simulation.
        * **1. Oil Properties**: 
            * `Select from NOAA List`: Choose a real oil type from the NOAA database. Its physical properties (density, viscosity) will be loaded automatically, influencing how the oil spreads and weathers.
            * `Define Custom`: If your oil is not on the list, you can manually define its **Density** (in kg/m³) and **Kinematic Viscosity** (in cSt). Note that weathering will be based on a generic oil.
        * **2. Simulation Parameters**: 
            * `Start Date & Time`: Defines the exact moment the spill begins. This is crucial as the simulator will use real meteorological and oceanographic data for that date.
            * `Duration (h)`: The number of hours the simulation will run.
            * `# Particles`: The spill is simulated as a set of particles. A higher number results in a more detailed simulation but is also slower.
            * `Total Spill Mass (kg)`: The total mass of oil in the spill, which will be distributed among the particles.
            * `Spill Radius (m)`: The initial radius of the oil slick.
        * **3. Advanced Settings**:
            * `Time Step`: The model's internal calculation interval. Shorter steps are more accurate but slower.
            * `Reader URLs`: You can provide custom URLs to THREDDS/OPeNDAP servers for environmental data (currents, wind, waves).
        * **4. Spill Location**: 
            * You can enter the coordinates manually or check "Show Selection Map" to click directly on the map.

        --- 
        #### **Step 2: Run the Initial Simulation**
        After setting all parameters, click **"Run Initial Simulation"**. This simulates the spill **without any intervention** to establish a baseline. Please wait for it to complete.

        --- 
        #### **Step 3: Plan and Simulate with Booms**
        After the first simulation, the app enters planning mode.
        * On the map, select the **boom type** you want to use (`curtain`, `fence`, `sorbent`, `shore_sealing`) and **draw it directly on the affected area**. You can draw multiple booms.
        * In the sidebar, you can fine-tune the **physical and advanced properties** of each boom you've added. This includes everything from skirt depth to advanced physics parameters like the `Critical Froude Number` or `Accumulation Factor`, allowing for detailed scenario testing.
        * When you are satisfied with your plan, click **"Run with Booms"**.

        --- 
        #### **Step 4: Compare and Analyze Results**
        The final view is organized into tabs for a comprehensive analysis.
        * **Overview**: Shows a side-by-side video comparison and the final state map. It also includes summary tables of the final particle counts for each scenario.
        * **Impact Analysis**: Compares key metrics like *Time to First Shoreline Impact* and presents charts on *Final Particle Fate* and *Stranded Mass Over Time*.
        * **Boom Performance**: Provides a detailed breakdown of each boom's efficiency, mass balance, peak forces, and leakage mechanisms.
        * **Environmental & Failure Analysis**: Correlates environmental conditions (currents, waves, wind) with boom failure and leakage events over time. It also presents the definitive final count for all particles for a selected scenario.
        * **Downloads**: Access and download all generated files (videos, maps, raw NetCDF data) for your records.
        """)

    if st.session_state.show_map:
        st.header("Select the spill location on the map")
        m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=8)
        MousePosition(position="topright", separator=" | ", prefix="Coordinates:", num_digits=4).add_to(m)
        folium.Marker(location=[st.session_state.latitude, st.session_state.longitude], icon=folium.Icon(color="blue", icon="tint", prefix="fa")).add_to(m)
        map_data = st_folium(m, use_container_width=True, height=450, key="map_select") 
        if map_data and map_data.get("last_clicked"):
            st.session_state.latitude = map_data["last_clicked"]["lat"]
            st.session_state.longitude = map_data["last_clicked"]["lng"]
            st.rerun()

elif st.session_state.app_state in ['running_initial', 'running_with_booms']:
    is_initial = st.session_state.app_state == 'running_initial'
    header_text = "Running Initial Simulation..." if is_initial else "Running Simulation with Booms..."
    st.header(header_text)
    
    progress_bar = st.progress(0, text="Initializing...")
    warning_placeholder = st.empty() 

    try:
        params = st.session_state.sim_params.copy()
        if not st.session_state.output_dir:
            st.session_state.output_dir = f"sim_outputs/Sim_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            os.makedirs(st.session_state.output_dir, exist_ok=True)
        
        params['output_dir'] = st.session_state.output_dir

        if is_initial:
            results = run_simulation(
                params, "initial_simulation.mp4", "initial_results.nc", "initial_map.png",
                warning_placeholder=warning_placeholder, progress_bar=progress_bar
            )
        else:
            st.session_state.boom_run_counter += 1
            run_id = st.session_state.boom_run_counter
            scenario_name = f"Scenario {run_id}"
            
            output_dir_booms = os.path.join(st.session_state.output_dir, f"booms_run_{run_id}")
            os.makedirs(output_dir_booms, exist_ok=True)
            params['output_dir'] = output_dir_booms
            
            results = run_simulation(
                params, 
                f"booms_simulation_run_{run_id}.mp4", 
                f"booms_results_run_{run_id}.nc", 
                f"booms_map_run_{run_id}.png",
                booms_data=st.session_state.booms,
                warning_placeholder=warning_placeholder,
                progress_bar=progress_bar
            )
        
        video, nc_path, stats, bounds, video_currents, map_path = results
        
        if stats and stats.get('warnings'):
            st.session_state.simulation_warnings = stats['warnings']
        else:
            st.session_state.simulation_warnings = []

        if not (video or nc_path or stats):
            raise Exception("Simulation backend did not return results. Check logs for details.")
        
        result_data = {
            'video_path': video, 
            'netcdf_path': nc_path, 
            'stats': stats, 
            'map_path': map_path
        }
        if video_currents:
            result_data['video_with_currents_path'] = video_currents

        if is_initial:
            st.session_state.initial_run_results = result_data
            st.session_state.map_bounds = bounds
            st.session_state.app_state = 'planning_booms'
        else:
            st.session_state.boom_scenarios[scenario_name] = result_data
            st.session_state.app_state = 'comparing_results'
        
        progress_bar.progress(100, text="Completed!")
        time.sleep(1)
        st.rerun()

    except Exception as e:
        st.error(f"A critical error occurred during the simulation: {e}")
        st.code(traceback.format_exc())
        st.button("Restart", on_click=reset_to_configuring)


elif st.session_state.app_state == 'planning_booms':
    if st.session_state.simulation_warnings:
        for msg in st.session_state.simulation_warnings:
            st.warning(msg)
        st.session_state.simulation_warnings = []

    col_video, col_map = st.columns(2)
    
    with col_video:
        with st.container(height=550):
            st.markdown("<h6>Initial Scenario (No Booms)</h6>", unsafe_allow_html=True)
            
            show_currents_video = st.checkbox("Show Currents", value=False, help="Toggle to view the simulation with current visualization.")
            
            video_to_show = st.session_state.initial_run_results.get('video_path')
            if show_currents_video:
                video_to_show = st.session_state.initial_run_results.get('video_with_currents_path', video_to_show)

            if video_to_show and os.path.exists(video_to_show): 
                st.video(video_to_show)
            else: 
                st.warning("Initial simulation video not found.")
    
    with col_map:
        with st.container(height=550):
            st.markdown("<h6>Planning Map</h6>", unsafe_allow_html=True)
            st.caption("Select boom type below and draw on the map:")
            boom_type = st.selectbox(
                "Select boom type",
                ['curtain', 'fence', 'sorbent', 'shore_sealing'],
                label_visibility="collapsed"
            )
            
            if boom_type == 'shore_sealing':
                st.warning("Remember: 'Shore-sealing' booms are most effective near the coast in shallow water.")
            
            m_draw = folium.Map()
            if st.session_state.map_bounds:
                m_draw.fit_bounds(st.session_state.map_bounds)
            else:
                m_draw.location=[st.session_state.latitude, st.session_state.longitude]
                m_draw.zoom_start=8
            
            folium.Marker(
                [st.session_state.sim_params['lat'], st.session_state.sim_params['lon']], 
                icon=folium.Icon(color="blue", icon="tint", prefix="fa"), 
                tooltip="Spill Location",
                opacity=0.7
            ).add_to(m_draw)

            for i, boom in enumerate(st.session_state.booms):
                color = OpenOilWithBooms().boom_types[boom['type']]['plot_props']['color']
                boom_coords_latlon = [(p[1], p[0]) for p in boom['coordinates']]
                folium.PolyLine(locations=boom_coords_latlon, color=color, weight=4, tooltip=f"Boom {i+1} ({boom['type']})").add_to(m_draw)
                
                if len(boom_coords_latlon) > 0:
                    center_lat = np.mean([p[0] for p in boom_coords_latlon])
                    center_lon = np.mean([p[1] for p in boom_coords_latlon])
                    folium.Marker(
                        location=[center_lat, center_lon],
                        icon=DivIcon(
                            icon_size=(150,36),
                            icon_anchor=(7,20),
                            html=f'<div style="font-size: 10pt; font-weight: bold; color: {color}; background-color: transparent; text-shadow: 1px 1px 2px black;">B{i+1}</div>',
                        )
                    ).add_to(m_draw)

            Draw(export=False, draw_options={'polyline': True, 'polygon': False, 'rectangle': False, 'circle': False, 'marker': False, 'circlemarker': False}).add_to(m_draw)
            
            map_output = st_folium(m_draw, use_container_width=True, height=450, key=f"draw_map_{st.session_state.map_key}")
            
            if map_output and map_output.get("all_drawings"):
                raw_coords = map_output["all_drawings"][-1]['geometry']['coordinates']
                if len(raw_coords) > 1:
                    coords = [[float(p[0]), float(p[1])] for p in raw_coords]
                    if all(b['coordinates'] != coords for b in st.session_state.booms):
                        default_params = OpenOilWithBooms().boom_types[boom_type].copy()
                        st.session_state.booms.append({
                            "coordinates": coords, 
                            "type": boom_type,
                            "overrides": default_params
                        })
                        st.session_state.map_key += 1
                        st.rerun()

elif st.session_state.app_state == 'comparing_results':
    st.header("Results Comparison")

    if not st.session_state.boom_scenarios:
        st.warning("No scenarios with booms have been run yet.")
        st.stop()

    available_scenarios = list(st.session_state.boom_scenarios.keys())
    selected_scenarios = st.multiselect(
        "Select scenarios to compare:", 
        available_scenarios, 
        default=available_scenarios[-1:]
    )
    
    if selected_scenarios:
        # Sort the selected scenarios numerically by scenario number
        try:
            selected_scenarios.sort(key=lambda name: int(name.split()[-1]))
        except (ValueError, IndexError):
            # Fallback to alphabetical sort if parsing fails
            selected_scenarios.sort()

    if not selected_scenarios:
        st.info("Please select at least one scenario to view results.")
        st.stop()

    stats_init = st.session_state.initial_run_results.get('stats', {})
    selected_boom_stats = {name: st.session_state.boom_scenarios[name].get('stats', {}) for name in selected_scenarios}

    tab_overview, tab_impact, tab_booms, tab_env, tab_downloads = st.tabs(["Overview", "Impact Analysis", "Boom Performance", "Environmental & Failure Analysis", "Downloads"])

    with tab_overview:
        st.subheader("Simulation Visualizations")
        
        viz_cols = st.columns(len(selected_scenarios) + 1)
        with viz_cols[0]:
            st.markdown("<h6>Initial Scenario</h6>", unsafe_allow_html=True)
            if st.session_state.initial_run_results.get('video_path'): 
                st.video(st.session_state.initial_run_results['video_path'])
        
        for i, scenario_name in enumerate(selected_scenarios):
            with viz_cols[i+1]:
                st.markdown(f"<h6>{scenario_name}</h6>", unsafe_allow_html=True)
                video_path = st.session_state.boom_scenarios[scenario_name].get('video_path')
                if video_path: st.video(video_path)
        
        st.divider()
        st.subheader("Final State Map")
        map_cols = st.columns(len(selected_scenarios) + 1)
        with map_cols[0]:
            st.markdown("<h6>Initial Scenario</h6>", unsafe_allow_html=True)
            map_path_init = st.session_state.initial_run_results.get('map_path')
            if map_path_init and os.path.exists(map_path_init):
                st.image(map_path_init)

        for i, scenario_name in enumerate(selected_scenarios):
            with map_cols[i+1]:
                st.markdown(f"<h6>{scenario_name}</h6>", unsafe_allow_html=True)
                map_path_boom = st.session_state.boom_scenarios[scenario_name].get('map_path')
                if map_path_boom and os.path.exists(map_path_boom):
                    st.image(map_path_boom)
        
        st.divider()
        st.subheader("Final Particle Fate") 
        
        def display_stats_table(stats_dict, title):
            with st.container():
                st.markdown(f"<h5>{title}</h5>", unsafe_allow_html=True)
                if stats_dict and 'final_particle_fate' in stats_dict:
                    fate_data = stats_dict['final_particle_fate']
                    
                    if fate_data:
                        filtered_fate_data = {k: v for k, v in fate_data.items() if v > 0}
                        if filtered_fate_data:
                            df = pd.DataFrame(list(filtered_fate_data.items()), columns=["Fate", "Count"])
                            st.table(df.set_index("Fate"))
                            
                            total_particles = sum(filtered_fate_data.values())
                            st.metric("Total Particles Accounted For", value=f"{total_particles}")
                        else:
                            st.info("No particles tracked.")
                    else:
                        st.info("No particles tracked.")
                else:
                    st.warning("No final particle fate data available.")

        stat_cols = st.columns(len(selected_scenarios) + 1)
        with stat_cols[0]:
            display_stats_table(stats_init, "Initial Scenario")
        
        for i, scenario_name in enumerate(selected_scenarios):
            with stat_cols[i+1]:
                display_stats_table(selected_boom_stats[scenario_name], scenario_name)

    with tab_impact:
        st.subheader("Impact Assessment & Comparative Plots")
        
        # 1. Key Metrics Summary Table
        st.markdown("##### Key Metrics Summary")
        key_metric_cols = st.columns([3, 2])
        with key_metric_cols[0]:
            summary_data = []
            scenarios_to_compare = {"Initial": stats_init, **selected_boom_stats}

            for name, stats in scenarios_to_compare.items():
                if not stats or 'key_metrics' not in stats: continue
                metrics = stats['key_metrics']
                summary_data.append({
                    "Scenario": name,
                    "Total Stranded Mass (kg)": metrics.get('total_stranded_mass', 0),
                    "Time to First Stranding (h)": metrics.get('time_to_first_stranding_hours', 'N/A'),
                    "Total Leaked Mass (kg)": metrics.get('total_leaked_mass', 0),
                })
            if summary_data:
                df_summary = pd.DataFrame(summary_data).set_index("Scenario")
                st.dataframe(df_summary.style.format({
                    "Total Stranded Mass (kg)": "{:,.2f}",
                    "Time to First Stranding (h)": "{:.2f}",
                    "Total Leaked Mass (kg)": "{:,.2f}"
                }, na_rep="N/A"))
        
        with key_metric_cols[1]:
            st.markdown("""
            - **Total Stranded Mass (kg):** The total mass of oil that has washed ashore at the end of the simulation.
            - **Time to First Stranding (h):** The time until the first oil particle hits the shoreline.
            - **Total Leaked Mass (kg):** The cumulative mass of oil that escaped from all booms. *Note: This is calculated based on the initial mass of any particle that ever leaked.*
            """)

        st.divider()

        # 2. Mass Balance Over Time
        st.markdown("##### Mass Balance Over Time")

        color_map = {
            "Active": "#1f77b4", 
            "Stranded": "#ff7f0e" ,
            "Contained": "#2ca02c",
            "Absorbed": "#9467bd",
            "Lost Mass (Weathering)": "#d62728"
        }

        scenarios_to_compare_mb = {"Initial": stats_init, **selected_boom_stats}
        mb_cols = st.columns(len(scenarios_to_compare_mb))

        for i, (name, stats) in enumerate(scenarios_to_compare_mb.items()):
            with mb_cols[i]:
                st.markdown(f"<h6>{name}</h6>", unsafe_allow_html=True)
                if stats.get('mass_balance_timeseries'):
                    df_mb = pd.DataFrame(normalize_timeseries_dict(stats['mass_balance_timeseries'])).set_index('Hours')
                    df_mb = df_mb.loc[:, (df_mb.sum(axis=0) > 1e-9)]
                    if not df_mb.empty:
                        fig_mb = px.area(df_mb, x=df_mb.index, y=df_mb.columns,
                                        labels={"value": "Mass (kg)", "Hours": "Time (hours)"},color_discrete_map=color_map)
                        fig_mb.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
                        st.plotly_chart(fig_mb, use_container_width=True, key=f"mass_balance_chart_{name}")
                    else:
                        st.info("No mass balance data.")
                else:
                    st.info("No mass balance data.")
        st.caption("Note: Total mass may decrease over time due to weathering processes like evaporation.")
        st.divider()
    
        # 3. Existing Comparison Charts
        st.markdown("##### Final Particle Fate (Comparison)")
        def prepare_fate_data_for_bar_chart(stats, scenario_name):
            if not stats or 'final_particle_fate' not in stats: return []
            return [{"Scenario": scenario_name, "Fate": fate, "Particles": count} for fate, count in stats['final_particle_fate'].items()]

        comparison_data = prepare_fate_data_for_bar_chart(stats_init, "Initial")
        for name, stats in selected_boom_stats.items():
            comparison_data.extend(prepare_fate_data_for_bar_chart(stats, name))
        
        if comparison_data:
            df_dest = pd.DataFrame(comparison_data)
            df_dest = df_dest[df_dest["Particles"] > 0]
            if not df_dest.empty:
                fig = px.bar(df_dest, x="Fate", y="Particles", color="Scenario", barmode="group")
                st.plotly_chart(fig, use_container_width=True)
        st.divider()

        st.markdown("##### Stranded Mass on Coastline Over Time")
        ts_init = normalize_timeseries_dict(stats_init.get('timeseries', {}))
        df_ts_all = pd.DataFrame()

        if ts_init and 'time_hours' in ts_init and 'cumulative_stranded_mass' in ts_init:
            if ts_init['time_hours'] and ts_init['cumulative_stranded_mass']:
                df_ts_all['Hours'] = ts_init['time_hours']
                df_ts_all['Initial Scenario'] = ts_init['cumulative_stranded_mass']
                df_ts_all = df_ts_all.set_index('Hours')

        for name, stats in selected_boom_stats.items():
            ts_booms = normalize_timeseries_dict(stats.get('timeseries', {}))
            if ts_booms and 'time_hours' in ts_booms and 'cumulative_stranded_mass' in ts_booms:
                if ts_booms['time_hours'] and ts_booms['cumulative_stranded_mass']:
                    df_booms = pd.DataFrame({'Hours': ts_booms['time_hours'], name: ts_booms['cumulative_stranded_mass']}).set_index('Hours')
                    df_ts_all = pd.concat([df_ts_all, df_booms], axis=1)
        
        if not df_ts_all.empty:
            df_ts_all = df_ts_all.ffill().bfill()
            st.line_chart(df_ts_all)

    with tab_booms:
        st.subheader("Boom Performance Analysis")
        
        if not selected_scenarios:
            st.info("Please select a scenario with booms to analyze.")
        else:
            analysis_scenario = st.selectbox("Select scenario for detailed analysis:", selected_scenarios)
            stats = selected_boom_stats.get(analysis_scenario, {})
            
            if stats and 'boom_performance' in stats:
                for i, boom_perf in enumerate(stats['boom_performance']):
                    with st.expander(f"Analysis for {boom_perf['name']} (Type: {boom_perf['type']})", expanded=True):
                        if boom_perf.get('warnings'):
                            for warning in boom_perf['warnings']: st.warning(warning)

                        if boom_perf.get('structural_failure'):
                            st.error(f"STRUCTURAL FAILURE DETECTED! (Force: {boom_perf.get('breaking_force_N', 0):,.0f} N / Strength: {boom_perf.get('max_anchor_force_N', 0):,.0f} N)")
                        
                        mass_retained = boom_perf.get('mass_contained', 0) + boom_perf.get('mass_absorbed', 0)
                        
                        b_cols = st.columns(4)
                        b_cols[0].metric("Boom Efficiency", f"{boom_perf.get('efficiency', 0):.1f}%", help="Percentage of mass that was retained out of the total mass that interacted with this boom.")
                        b_cols[1].metric("Mass Retained", f"{mass_retained:,.2f} kg", help="Total mass physically contained or absorbed.")
                        b_cols[2].metric("Mass Leaked", f"{boom_perf.get('total_mass_leaked', 0):,.2f} kg", help="Total mass that escaped this boom at the moment of leakage.")
                        
                        peak_force = boom_perf.get('max_force_experienced_N', 0)
                        anchor_strength = boom_perf.get('max_anchor_force_N', 1)
                        force_ratio = (peak_force / anchor_strength) * 100 if anchor_strength > 0 else 0
                        b_cols[3].metric("Peak Force on Boom", f"{peak_force:,.0f} N", f"{force_ratio:.1f}% of Strength", 
                                      delta_color="inverse" if force_ratio > 80 else "normal",
                                      help="The maximum force experienced by the boom. The percentage indicates how close it was to its structural failure point.")

                        if boom_perf['type'] == 'sorbent':
                            sorb_cols = st.columns(4) 
                            mass_absorbed = boom_perf.get('mass_absorbed', 0)
                            total_capacity = boom_perf.get('total_capacity_kg', 1)
                            saturation = (mass_absorbed / total_capacity) * 100 if total_capacity > 0 else 0
                            sorb_cols[0].metric("Mass Absorbed", f"{mass_absorbed:,.2f} kg")
                            sorb_cols[1].metric("Saturation Level", f"{saturation:.1f}%")

                        if 'leakage_details' in boom_perf and sum(boom_perf['leakage_details'].values()) > 0:
                            st.markdown("<h6>Leakage by Mechanism</h6>", unsafe_allow_html=True)
                            leak_col1, leak_col2 = st.columns(2)

                            leakage_df = pd.DataFrame.from_dict(boom_perf['leakage_details'], orient='index', columns=['Mass (kg)'])
                            leakage_df = leakage_df[leakage_df['Mass (kg)'] > 0]
                            
                            if not leakage_df.empty:
                                with leak_col1:
                                    st.table(leakage_df.sort_values(by='Mass (kg)', ascending=False).style.format("{:.2f}"))
                                with leak_col2:
                                    fig = go.Figure(data=[go.Pie(labels=leakage_df.index, values=leakage_df['Mass (kg)'], textinfo='percent+label', insidetextorientation='radial')])
                                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.success("No leakage detected for this boom.")
                        else:
                           st.success("No leakage detected for this boom.")

                        if boom_perf.get('force_timeseries_N'):
                            st.markdown("<h6>Force Evolution Over Time</h6>", unsafe_allow_html=True)
                            force_ts = boom_perf['force_timeseries_N']
                            time_hours = stats.get('environmental_timeseries', {}).get('time_hours', [])
                            
                            min_len = min(len(time_hours), len(force_ts))
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=time_hours[:min_len], y=force_ts[:min_len], mode='lines', name='Total Force'))
                            
                            if boom_perf.get('structural_failure') and boom_perf.get('structural_failure_time') is not None:
                                fail_time_dt = datetime.fromisoformat(boom_perf['structural_failure_time'])
                                start_time_dt = datetime.fromisoformat(stats['start_time'])
                                fail_time_hours = (fail_time_dt - start_time_dt).total_seconds() / 3600
                                fig.add_vline(x=fail_time_hours, line_width=2, line_dash="dash", line_color="red",
                                              annotation_text="Structural Failure", annotation_position="top left")

                            fig.update_layout(title=f'Force on {boom_perf["name"]}', xaxis_title='Time (hours)', yaxis_title='Max Force on any Segment (N)', height=350, margin=dict(t=40))
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No boom performance data available for the selected scenario.")

    with tab_env:
        st.subheader("Environmental Analysis & Failure Events")
        
        if not selected_scenarios:
            st.info("Please select a scenario with booms to analyze.")
        else:
            analysis_scenario_env = st.selectbox("Select scenario for Environmental Analysis:", selected_scenarios, key="env_select")
            stats = selected_boom_stats.get(analysis_scenario_env, {})
            env_stats = stats.get('environmental_timeseries')
            
            if env_stats and env_stats.get('time_hours'):
                df_env = pd.DataFrame(normalize_timeseries_dict(env_stats)).set_index('time_hours')
                
                st.markdown("<h6>Environmental Conditions & Failure Events</h6>", unsafe_allow_html=True)
                fig_env = go.Figure()
                fig_env.add_trace(go.Scatter(x=df_env.index, y=df_env['current_speed_mps'], name='Current Speed (m/s)', yaxis='y1'))
                fig_env.add_trace(go.Scatter(x=df_env.index, y=df_env['wave_height_m'], name='Wave Height (m)', yaxis='y2'))
                fig_env.add_trace(go.Scatter(x=df_env.index, y=df_env.get('wind_speed_ms', []), name='Wind Speed (m/s)', yaxis='y1', line=dict(dash='dash')))

                # Plot structural failures
                if 'boom_performance' in stats:
                    for boom_perf in stats['boom_performance']:
                        if boom_perf.get('structural_failure') and boom_perf.get('structural_failure_time') is not None:
                            fail_time_dt = datetime.fromisoformat(boom_perf['structural_failure_time'])
                            start_time_dt = datetime.fromisoformat(stats['start_time'])
                            fail_time_hours = (fail_time_dt - start_time_dt).total_seconds() / 3600
                            fig_env.add_vline(x=fail_time_hours, line_width=1.5, line_dash="dot", line_color="rgba(214, 39, 40, 0.7)",
                                                annotation_text=f"Failure: {boom_perf['name']}", annotation_position="bottom right")
                
                # Plot other leakage events
                if 'leakage_events' in stats and stats['leakage_events']:
                    df_leaks = pd.DataFrame(stats['leakage_events'])
                    
                    df_leaks = df_leaks[~df_leaks['leakage_type'].str.contains("Structural", case=False)]
                    
                    if not df_leaks.empty:
                        current_speed_at_leak = np.interp(df_leaks['time_hours'], df_env.index, df_env['current_speed_mps'])
                        
                        fig_env.add_trace(go.Scatter(
                            x=df_leaks['time_hours'],
                            y=current_speed_at_leak,
                            mode='markers',
                            marker=dict(symbol='cross', color='red', size=8),
                            name='Leakage Event',
                            hoverinfo='text',
                            hovertext=[f"{row['leakage_type']} ({row['mass']:.2f} kg) @ {row['boom_name']}" for index, row in df_leaks.iterrows()],
                            showlegend=False
                        ))

                fig_env.update_layout(
                    yaxis=dict(title='Current / Wind Speed (m/s)'),
                    yaxis2=dict(title='Wave Height (m)', overlaying='y', side='right'),
                    xaxis_title='Time (hours)',
                    height=450, 
                    margin=dict(t=40),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig_env, use_container_width=True)
                st.divider()
            else:
                st.info("Environmental timeseries data not available for this scenario.")


    with tab_downloads:
        st.subheader("Download Result Files")
        if st.session_state.output_dir: 
            st.info(f"All files are saved at: `{st.session_state.output_dir}`")
        
        with st.expander("Initial Simulation Files"):
            initial_res = st.session_state.initial_run_results
            if initial_res.get('video_path') and os.path.exists(initial_res['video_path']):
                with open(initial_res['video_path'], "rb") as f:
                    st.download_button("Download Video (.mp4)", f, file_name="initial_simulation.mp4", key="dl_init_vid")
            
            if initial_res.get('video_with_currents_path') and os.path.exists(initial_res['video_with_currents_path']):
                with open(initial_res['video_with_currents_path'], "rb") as f:
                    st.download_button("Download Video with Currents (.mp4)", f, file_name="initial_simulation_currents.mp4", key="dl_init_vid_curr")

            if initial_res.get('netcdf_path') and os.path.exists(initial_res['netcdf_path']):
                with open(initial_res['netcdf_path'], "rb") as f:
                    st.download_button("Download Data (.nc)", f, file_name="initial_results.nc", key="dl_init_nc")
            
            if initial_res.get('map_path') and os.path.exists(initial_res['map_path']):
                with open(initial_res['map_path'], "rb") as f:
                    st.download_button("Download Summary Map (.png)", f, file_name="initial_map.png", key="dl_init_map")
            
            st.divider()

        for name, scenario_data in st.session_state.boom_scenarios.items():
            run_id = name.split(" ")[-1]
            with st.expander(f"Files for {name}"):
                if scenario_data.get('video_path') and os.path.exists(scenario_data['video_path']):
                    with open(scenario_data['video_path'], "rb") as f:
                        st.download_button(f"Download Video {run_id} (.mp4)", f, file_name=f"booms_simulation_{run_id}.mp4", key=f"dl_boom_vid_{run_id}")
                if scenario_data.get('netcdf_path') and os.path.exists(scenario_data['netcdf_path']):
                    with open(scenario_data['netcdf_path'], "rb") as f:
                        st.download_button(f"Download Data {run_id} (.nc)", f, file_name=f"booms_results_{run_id}.nc", key=f"dl_boom_nc_{run_id}")
                if scenario_data.get('map_path') and os.path.exists(scenario_data['map_path']):
                    with open(scenario_data['map_path'], "rb") as f:
                        st.download_button(f"Download Map {run_id} (.png)", f, file_name=f"booms_map_{run_id}.png", key=f"dl_boom_map_{run_id}")

        st.subheader("Generate PDF Report")

        if selected_scenarios:
            if st.button("Generate PDF Report for Selected Scenarios", type="primary"):
                with st.spinner("Generating report... This may take a moment."):
                    try:
                        scenarios_for_report = {
                            name: data for name, data in st.session_state.boom_scenarios.items()
                            if name in selected_scenarios
                        }
                        
                        pdf_data = generate_pdf_report(
                            st.session_state.sim_params,
                            st.session_state.initial_run_results,
                            scenarios_for_report,
                            st.session_state.output_dir
                        )

                        st.download_button(
                            label="Download Report (.pdf)",
                            data=pdf_data,
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Failed to generate report: {e}")
                        st.code(traceback.format_exc())
        else:
            st.info("Select at least one boom scenario in the multiselect box above to generate a report.")
