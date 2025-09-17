# =============================================================================
# NOTE: Final Definitive Version
# - (NEW) Added a "Boom Configuration Details" table to the report,
#   detailing the physical parameters of each boom used in a scenario.
# - The "Flowable too large" error is fixed by changing the layout
#   structure for the maps section. Instead of nesting complex objects
#   (KeepTogether) inside a table cell, a simpler "table-in-table"
#   approach is used, which is more stable for the ReportLab layout engine.
# - The Pillow-based image validation remains as a robust safeguard.
# - ADDED: Environmental analysis chart, correlating environmental
#   conditions with failure events, is now included in the report.
# =============================================================================

import io
import os
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.units import cm
from reportlab.lib import colors
from PIL import Image as PilImage # Import Pillow
import numpy as np # Needed for interpolation

# --- Global Styles ---
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleStyle', fontSize=22, alignment=TA_CENTER, spaceAfter=16, textColor=colors.HexColor('#263238'), fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='Header1', fontSize=16, alignment=TA_LEFT, spaceAfter=12, textColor=colors.HexColor('#37474F'), fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='Header2', fontSize=12, alignment=TA_LEFT, spaceAfter=6, textColor=colors.HexColor('#455A64'), fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='Body', fontSize=9, alignment=TA_JUSTIFY, leading=14))
styles.add(ParagraphStyle(name='Footer', fontSize=8, alignment=TA_CENTER, textColor=colors.grey))
styles.add(ParagraphStyle(name='TableTitle', fontSize=10, alignment=TA_LEFT, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor('#455A64'), fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='ImageCaption', fontSize=8, alignment=TA_CENTER, spaceAfter=6, textColor=colors.grey))
styles.add(ParagraphStyle(name='BodyRight', parent=styles['Body'], alignment=TA_RIGHT))


# --- Helper Functions ---

def _header_footer(canvas, doc):
    canvas.saveState()
    header = Paragraph("Oil Spill Response Simulation Report", styles['Footer'])
    w, h = header.wrap(doc.width, doc.topMargin)
    header.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin + 0.5*cm)
    footer = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    |    Page {doc.page}", styles['Footer'])
    w, h = footer.wrap(doc.width, doc.bottomMargin)
    footer.drawOn(canvas, doc.leftMargin, 0.5 * cm)
    canvas.restoreState()

def _is_valid_image(filepath):
    """
    Checks if a file is a valid, non-corrupted image that can be opened by Pillow.
    """
    if not filepath or not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return False
    try:
        with PilImage.open(filepath) as img:
            img.verify()  # Verifies the integrity of the image data
        # Re-open after verify, as verify() can corrupt the file handle for some formats
        with PilImage.open(filepath) as img:
            if img.width > 0 and img.height > 0:
                return True
    except Exception:
        # This will catch errors from PIL like file format errors, etc.
        return False
    return False

def _create_plotly_image_file(fig, output_dir):
    """ Saves a Plotly figure to a temporary PNG file and returns the path. """
    fig.update_layout(template="plotly_white", font=dict(size=10))
    temp_filename = os.path.join(output_dir, f"temp_chart_{uuid.uuid4()}.png")
    fig.write_image(temp_filename, scale=3)
    return temp_filename

def _build_summary_table(sim_params):
    data = [
        [Paragraph('<b>Simulation Parameter</b>', styles['Body']), Paragraph('<b>Value</b>', styles['Body'])],
        ['Oil Type:', sim_params.get('oil_type', 'N/A')],
        ['Total Spill Mass (kg):', f"{sim_params.get('total_mass_kg', 0):,}"],
        ['Number of Particles:', f"{sim_params.get('number', 0):,}"],
        ['Simulation Start (UTC):', sim_params.get('start_time').strftime('%Y-%m-%d %H:%M')],
        ['Duration (hours):', sim_params.get('duration_hours', 'N/A')],
        ['Location (Lat, Lon):', f"{sim_params.get('lat', 0):.4f}, {sim_params.get('lon', 0):.4f}"]
    ]
    table = Table(data, colWidths=[5*cm, 12*cm], hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8EAF6')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1A237E')), ('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0, 0), (-1, -1), 5), ('TOPPADDING', (0, 0), (-1, -1), 5), ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#C5CAE9'))]))
    return table

def _build_comparative_metrics_table(stats_init, selected_boom_stats):
    scenario_names = list(selected_boom_stats.keys())
    header = [Paragraph('<b>Impact Metric</b>', styles['Body']), Paragraph('<b>Initial Scenario</b>', styles['Body'])] + \
             [Paragraph(f'<b>{name}</b>', styles['Body']) for name in scenario_names]
    
    init_stranded = stats_init.get('key_metrics', {}).get('total_stranded_mass', 0)
    
    rows_data = {"Total Stranded Mass (kg)": [f"{init_stranded:,.2f}"], "Total Leaked Mass (kg)": ["0.00"]}
    for name, scenario_data in selected_boom_stats.items():
        stats = scenario_data.get('stats', {})
        metrics = stats.get('key_metrics', {})
        stranded = metrics.get('total_stranded_mass', 0)
        leaked = metrics.get('total_leaked_mass', 0)
        
        reduction_str = ""
        if init_stranded > 0.01:
            reduction = ((init_stranded - stranded) / init_stranded * 100)
            color = 'green' if reduction > 0 else 'red'
            reduction_str = f"<font size='8' color='{color}'>({reduction:+.1f}%)</font>"
        
        rows_data["Total Stranded Mass (kg)"].append(Paragraph(f"{stranded:,.2f} {reduction_str}", styles['Body']))
        rows_data["Total Leaked Mass (kg)"].append(f"{leaked:,.2f}")
    final_table_data = [header] + [[key] + val for key, val in rows_data.items()]
    table = Table(final_table_data, colWidths=[4*cm] + [4.25*cm] * (1 + len(scenario_names)), hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8EAF6')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1A237E')), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#C5CAE9'))]))
    return table

def _build_boom_properties_table(boom_performance_data):
    """ Creates a table detailing the configuration of the booms used. """
    header = [
        Paragraph('<b>Boom Name</b>', styles['Body']),
        Paragraph('<b>Type</b>', styles['Body']),
        Paragraph('<b>Length (m)</b>', styles['Body']),
        Paragraph('<b>Freeboard (m)</b>', styles['Body']),
        Paragraph('<b>Skirt Depth (m)</b>', styles['Body']),
        Paragraph('<b>Anchor Strength (N)</b>', styles['Body']),
    ]
    
    table_data = [header]
    for boom_perf in boom_performance_data:
        config = boom_perf.get('configuration', {})
        row = [
            boom_perf.get('name', 'N/A'),
            boom_perf.get('type', 'N/A'),
            f"{boom_perf.get('total_length_m', 0):.2f}",
            f"{config.get('freeboard_height_m', 0):.2f}",
            f"{config.get('skirt_depth_m', 0):.2f}",
            f"{config.get('anchor_strength_N', 0):,}"
        ]
        table_data.append(row)
        
    table = Table(table_data, colWidths=[3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm, 3.5*cm], hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8EAF6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1A237E')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#C5CAE9')),
        ('FONTSIZE', (0, 1), (-1, -1), 8) # Smaller font for data rows
    ]))
    return table

def _create_environmental_analysis_chart(stats, scenario_name, output_dir):
    """ Creates and saves the environmental conditions vs. failure events chart. """
    env_stats = stats.get('environmental_timeseries')
    if not env_stats or not env_stats.get('time_hours'):
        return None

    try:
        df_env = pd.DataFrame(env_stats).set_index('time_hours')
        fig = go.Figure()

        # Add environmental traces
        fig.add_trace(go.Scatter(x=df_env.index, y=df_env['current_speed_mps'], name='Current Speed (m/s)', yaxis='y1'))
        fig.add_trace(go.Scatter(x=df_env.index, y=df_env['wave_height_m'], name='Wave Height (m)', yaxis='y2'))
        if 'wind_speed_ms' in df_env.columns:
             fig.add_trace(go.Scatter(x=df_env.index, y=df_env['wind_speed_ms'], name='Wind Speed (m/s)', yaxis='y1', line=dict(dash='dash')))

        # Add structural failure lines
        if 'boom_performance' in stats:
            for boom_perf in stats['boom_performance']:
                if boom_perf.get('structural_failure') and boom_perf.get('structural_failure_time'):
                    fail_time_dt = datetime.fromisoformat(boom_perf['structural_failure_time'])
                    start_time_dt = datetime.fromisoformat(stats['start_time'])
                    fail_time_hours = (fail_time_dt - start_time_dt).total_seconds() / 3600
                    fig.add_vline(x=fail_time_hours, line_width=1.5, line_dash="dot", line_color="rgba(214, 39, 40, 0.7)",
                                  annotation_text=f"Failure: {boom_perf['name']}", annotation_position="bottom right",
                                  annotation_font_size=9)

        # Add other leakage event markers
        if 'leakage_events' in stats and stats['leakage_events']:
            df_leaks = pd.DataFrame(stats['leakage_events'])
            df_leaks = df_leaks[~df_leaks['leakage_type'].str.contains("Structural", case=False)]
            if not df_leaks.empty:
                current_speed_at_leak = np.interp(df_leaks['time_hours'], df_env.index, df_env['current_speed_mps'])
                fig.add_trace(go.Scatter(
                    x=df_leaks['time_hours'], y=current_speed_at_leak,
                    mode='markers', marker=dict(symbol='cross', color='red', size=8),
                    name='Leakage Event', showlegend=False,
                    hoverinfo='text',
                    hovertext=[f"{row['leakage_type']} ({row['mass']:.2f} kg) @ {row['boom_name']}" for _, row in df_leaks.iterrows()]
                ))

        fig.update_layout(
            title=f"Environmental Conditions vs. Failure Events - {scenario_name}",
            yaxis=dict(title='Current / Wind Speed (m/s)'),
            yaxis2=dict(title='Wave Height (m)', overlaying='y', side='right'),
            xaxis_title='Time (hours)',
            height=350, margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return _create_plotly_image_file(fig, output_dir)
    except Exception:
        # Silently fail if chart generation has an issue, to not break the whole report
        return None

# --- Main Function ---

def generate_pdf_report(sim_params, initial_run_results, boom_scenarios, output_dir):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=3*cm, bottomMargin=2.5*cm)
    story = []
    temp_files = []

    try:
        # --- TITLE PAGE ---
        story.append(Paragraph("Oil Spill Response Simulation Report", styles['TitleStyle']))
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("This document presents the results of an oil spill simulation and a comparative analysis of one or more boom response scenarios. The objective is to evaluate the effectiveness of containment strategies and quantify the reduction in environmental impact.", styles['Body']))
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph("Simulation Summary", styles['Header2']))
        story.append(Paragraph("The simulation was configured with the following key parameters:", styles['Body']))
        story.append(Spacer(1, 0.2*cm))
        story.append(_build_summary_table(sim_params))
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph("Comparative Impact Metrics Summary", styles['Header2']))
        story.append(Paragraph("The table below summarizes the most important environmental impact metrics. 'Total Stranded Mass' represents the amount of oil that reached the coastline, with its reduction being the primary indicator of a successful response.", styles['Body']))
        story.append(Spacer(1, 0.2*cm))
        story.append(_build_comparative_metrics_table(initial_run_results['stats'], boom_scenarios))
        
        # --- DETAILED SCENARIO ANALYSIS ---
        stats_init = initial_run_results.get('stats', {})
        map_path_init = initial_run_results.get('map_path')
        
        for scenario_name, scenario_data in boom_scenarios.items():
            story.append(PageBreak())
            stats_boom = scenario_data.get('stats', {})
            map_path_boom = scenario_data.get('map_path')

            story.append(Paragraph(f"Detailed Analysis: {scenario_name}", styles['Header1']))
            story.append(Paragraph(f"This section provides a detailed, side-by-side analysis of the Initial Scenario (no intervention) versus {scenario_name}.", styles['Body']))
            story.append(Spacer(1, 1*cm))

            # --- NEW BOOM CONFIGURATION TABLE ---
            story.append(Paragraph("Boom Configuration Details", styles['Header2']))
            story.append(Paragraph("The table below details the physical properties of each boom deployed in this scenario.", styles['Body']))
            if stats_boom and 'boom_performance' in stats_boom:
                story.append(_build_boom_properties_table(stats_boom['boom_performance']))
            story.append(Spacer(1, 1*cm))

            story.append(Paragraph("Final State Maps", styles['Header2']))
            story.append(Paragraph("The maps below show the final distribution of oil particles. Red particles indicate oil stranded on the coastline; black particles represent active oil on the water's surface; green indicates the initial spill location.", styles['Body']))
            
            # REVISED: Use a "table-in-table" structure to avoid layout bugs.
            map_elements = []
            
            # Create a small, stable table for the initial scenario map
            if _is_valid_image(map_path_init):
                img_init = Image(map_path_init, width=8.5*cm, height=8.5*cm)
                img_init.hAlign = 'CENTER'
                caption_init = Paragraph("Initial Scenario", styles['ImageCaption'])
                map_table_init = Table([[img_init], [caption_init]], rowHeights=[8.7*cm, 0.5*cm])
                map_table_init.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 0), ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 0),]))
                map_elements.append(map_table_init)

            # Create a small, stable table for the boom scenario map
            if _is_valid_image(map_path_boom):
                img_boom = Image(map_path_boom, width=8.5*cm, height=8.5*cm)
                img_boom.hAlign = 'CENTER'
                caption_boom = Paragraph(scenario_name, styles['ImageCaption'])
                map_table_boom = Table([[img_boom], [caption_boom]], rowHeights=[8.7*cm, 0.5*cm])
                map_table_boom.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 0), ('TOPPADDING', (0, 0), (-1, -1), 0), ('BOTTOMPADDING', (0, 0), (-1, -1), 0),]))
                map_elements.append(map_table_boom)
            
            if map_elements:
                outer_table = Table([map_elements], colWidths=[9*cm] * len(map_elements), hAlign='CENTER')
                outer_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
                story.append(outer_table)

            story.append(Spacer(1, 0.5*cm))

            story.append(Paragraph("Mass Balance Over Time", styles['Header2']))
            story.append(Paragraph("The following charts illustrate the fate of the oil mass over time. 'Active' is oil on the water surface, 'Stranded' is the cumulative mass on the coastline, and 'Lost Mass' represents evaporation.", styles['Body']))
            
            color_map = { "Active": "#1f77b4", "Contained": "#ff7f0e", "Stranded": "#2ca02c", "Absorbed": "#9467bd", "Lost Mass (Weathering)": "#d62728" }
            if stats_init and stats_init.get('mass_balance_timeseries'):
                df_mb_init = pd.DataFrame(stats_init['mass_balance_timeseries']).set_index('Hours')
                df_mb_init = df_mb_init.loc[:, (df_mb_init.sum(axis=0) > 1e-9)]
                if not df_mb_init.empty:
                    fig = px.area(df_mb_init, title="Mass Balance - Initial Scenario", color_discrete_map=color_map)
                    fig.update_layout(height=250, margin=dict(l=10,r=10,t=40,b=10), legend_title_text='', yaxis_title="Mass (kg)", xaxis_title="Time (hours)")
                    chart_path = _create_plotly_image_file(fig, output_dir); temp_files.append(chart_path)
                    if _is_valid_image(chart_path):
                        story.append(Image(chart_path, width=17*cm, height=7*cm))

            if stats_boom and stats_boom.get('mass_balance_timeseries'):
                df_mb_boom = pd.DataFrame(stats_boom['mass_balance_timeseries']).set_index('Hours')
                df_mb_boom = df_mb_boom.loc[:, (df_mb_boom.sum(axis=0) > 1e-9)]
                if not df_mb_boom.empty:
                    fig = px.area(df_mb_boom, title=f"Mass Balance - {scenario_name}", color_discrete_map=color_map)
                    fig.update_layout(height=250, margin=dict(l=10,r=10,t=40,b=10), legend_title_text='', yaxis_title="Mass (kg)", xaxis_title="Time (hours)")
                    chart_path = _create_plotly_image_file(fig, output_dir); temp_files.append(chart_path)
                    if _is_valid_image(chart_path):
                        story.append(Image(chart_path, width=17*cm, height=7*cm))
            
            story.append(Spacer(1, 1*cm))
            story.append(Paragraph("Boom Performance Analysis", styles['Header2']))
            story.append(Paragraph(f"Detailed performance analysis for each boom deployed in {scenario_name}.", styles['Body']))
            
            if stats_boom and 'boom_performance' in stats_boom:
                for boom_perf in stats_boom['boom_performance']:
                    boom_section_content = []
                    boom_section_content.append(Paragraph(f"<b>{boom_perf['name']}</b> (Type: {boom_perf['type']})", styles['TableTitle']))
                    
                    mass_retained = boom_perf.get('mass_contained', 0) + boom_perf.get('mass_absorbed', 0)
                    peak_force = boom_perf.get('max_force_experienced_N', 0)
                    anchor_strength = boom_perf.get('max_anchor_force_N', 1)
                    force_ratio = (peak_force / anchor_strength) * 100 if anchor_strength > 0 else 0
                    
                    perf_data = [
                        [Paragraph('<b>Metric</b>', styles['Body']), Paragraph('<b>Value</b>', styles['BodyRight'])],
                        ['Efficiency', f"{boom_perf.get('efficiency', 0):.1f} %"],
                        ['Mass Retained', f"{mass_retained:,.2f} kg"],
                        ['Mass Leaked', f"{boom_perf.get('total_mass_leaked', 0):,.2f} kg"],
                        ['Peak Force', f"{peak_force:,.0f} N ({force_ratio:.1f}% of capacity)"]
                    ]
                    perf_table = Table(perf_data, colWidths=[5*cm, 5*cm], hAlign='LEFT')
                    perf_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (1,1), (1,-1), 'RIGHT')]))
                    boom_section_content.append(perf_table)
                    
                    leak_df = pd.DataFrame.from_dict(boom_perf.get('leakage_details',{}), orient='index', columns=['Mass (kg)'])
                    leak_df = leak_df[leak_df['Mass (kg)'] > 1e-3].sort_values(by='Mass (kg)', ascending=False)
                    
                    if not leak_df.empty:
                        boom_section_content.append(Spacer(1, 0.5*cm))
                        boom_section_content.append(Paragraph("Leakage by Mechanism", styles['TableTitle']))
                        pie = px.pie(leak_df, values='Mass (kg)', names=leak_df.index)
                        pie.update_layout(height=200, width=300, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
                        pie.update_traces(textinfo='percent+label', textfont_size=10)
                        chart_path = _create_plotly_image_file(pie, output_dir); temp_files.append(chart_path)
                        if _is_valid_image(chart_path):
                            boom_section_content.append(Image(chart_path, width=8*cm, height=6*cm))
                    
                    forces_chart_path = None
                    if boom_perf and boom_perf.get('force_timeseries_N'):
                        force_ts = boom_perf['force_timeseries_N']
                        time_hours = stats_boom.get('environmental_timeseries', {}).get('time_hours', [])
                        min_len = min(len(time_hours), len(force_ts))
                        if min_len > 0:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=time_hours[:min_len], y=force_ts[:min_len], mode='lines', name='Total Force'))
                            fig.update_layout(title=f'Force Evolution on {boom_perf["name"]}', xaxis_title='Time (hours)', yaxis_title='Force (N)', height=250, margin=dict(l=10,r=10,t=40,b=10))
                            forces_chart_path = _create_plotly_image_file(fig, output_dir); temp_files.append(forces_chart_path)

                    if _is_valid_image(forces_chart_path):
                        boom_section_content.append(Spacer(1, 0.5*cm))
                        boom_section_content.append(Paragraph("Force Evolution on Boom", styles['TableTitle']))
                        boom_section_content.append(Image(forces_chart_path, width=17*cm, height=7*cm))
                    
                    story.append(KeepTogether(boom_section_content))
                    story.append(Spacer(1, 0.5*cm))

            story.append(PageBreak())
            story.append(Paragraph("Environmental Analysis and Failure Events", styles['Header1']))
            story.append(Paragraph(
                "The chart below correlates the primary environmental drivers (currents, waves, wind) with critical boom events. "
                "Vertical dashed lines indicate the exact moment of a structural failure. Red crosses mark other leakage events, "
                "allowing for a visual diagnosis of why and when containment was compromised.",
                styles['Body']
            ))
            env_chart_path = _create_environmental_analysis_chart(stats_boom, scenario_name, output_dir)
            if _is_valid_image(env_chart_path):
                temp_files.append(env_chart_path)
                story.append(Spacer(1, 0.5 * cm))
                story.append(Image(env_chart_path, width=18*cm, height=9*cm))


        doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
        
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data
    finally:
        # Clean up temporary image files
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception as e:
                # Using a simple print to avoid dependency on a logger
                print(f"Warning: Could not remove temporary report file {f}: {e}")