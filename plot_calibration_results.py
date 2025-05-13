import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np

# --- Configuration ---
INPUT_CSV = "calibration_results_n200_empirical_norm_v2_median_subtracted.csv" # Updated input file for N=200
OUTPUT_DIR = "calibration_plots_n200_empirical_norm_v2"      # Updated output directory for N=200
PARETO_PLOT_3D_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_3d_interactive_empirical_n200.html")
PARETO_PLOT_3D_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_3d_static_empirical_n200.png")
PARETO_PLOT_HS_PS_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_ps_empirical_n200.html")
PARETO_PLOT_HS_PS_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_ps_empirical_n200.png")
PARETO_PLOT_HS_MD_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_md_empirical_n200.html")
PARETO_PLOT_HS_MD_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_md_empirical_n200.png")
PARETO_PLOT_PS_MD_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_ps_vs_md_empirical_n200.html")
PARETO_PLOT_PS_MD_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_ps_vs_md_empirical_n200.png")


# Objectives to minimize
# OBJECTIVES = ['norm_hs', 'norm_ps', 'norm_md'] # Old version - uses theoretical norms
OBJECTIVES = ['emp_norm_hs', 'emp_norm_ps', 'emp_norm_md'] # Use empirical objectives for Pareto analysis and primary plotting
THEORETICAL_OBJECTIVES = ['norm_hs', 'norm_ps', 'norm_md'] # Theoretical norms, for informational display
RAW_OBJECTIVES = ['raw_hs', 'raw_ps', 'raw_md'] # Raw values, for informational display
# --- End Configuration ---

# def is_pareto_efficient(costs, return_mask=True):
#     """
#     Find the Pareto-efficient points.
#     :param costs: An (n_points, n_costs) array
#     :param return_mask: True to return a boolean mask, False to return integer indices
#     :return: An array of indices of Pareto-efficient points.
#              If return_mask is True, this will be a boolean array of length n_points.
#     """
#     is_efficient = np.ones(costs.shape[0], dtype=bool)
#     for i, c in enumerate(costs):
#         if is_efficient[i]:
#             # Keep any point with a strictly dominating cost
#             is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
#             # And keep self
#             is_efficient[i] = True
#     if return_mask:
#         return is_efficient
#     else:
#         return np.where(is_efficient)[0]

def is_pareto_efficient_corrected(costs, return_mask=True):
    """
    Find the Pareto-efficient points. Assumes lower values are better.
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a boolean mask, False to return integer indices
    :return: An array of indices of Pareto-efficient points or a boolean mask.
    """
    num_points = costs.shape[0]
    is_efficient_mask = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        if not is_efficient_mask[i]:  # If already marked as dominated, skip
            continue
        for j in range(num_points):
            if i == j:
                continue
            # Check if point j dominates point i
            # Domination: all objectives of j are <= objectives of i AND at least one objective of j is < objective of i
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                is_efficient_mask[i] = False  # Point i is dominated by point j
                break
    if return_mask:
        return is_efficient_mask
    else:
        return np.where(is_efficient_mask)[0]

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load data
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV}' not found. Please run the calibration script first.")
        exit()

    df.dropna(subset=OBJECTIVES, inplace=True) # Remove rows with NaN in objective values
    if df.empty:
        print(f"Error: No valid data found in '{INPUT_CSV}' after dropping NaNs.")
        exit()

    # Identify Pareto-efficient points using the EMPIRICAL OBJECTIVES
    # We want to minimize all objectives in the OBJECTIVES list
    pareto_mask = is_pareto_efficient_corrected(df[OBJECTIVES].values)
    df['is_pareto'] = pareto_mask
    pareto_df = df[df['is_pareto']].copy()

    print(f"Loaded {len(df)} data points.")
    print(f"Found {len(pareto_df)} Pareto-efficient points (using empirical values: {', '.join(OBJECTIVES)}).")

    # --- 3D Interactive Pareto Plot ---
    fig3d = go.Figure()

    # Define customdata for hover
    custom_data_cols = ['alpha', 'beta', 'empirical_score'] + RAW_OBJECTIVES + THEORETICAL_OBJECTIVES + ['total_norm_score']

    # Non-Pareto points
    fig3d.add_trace(go.Scatter3d(
        x=df[~df['is_pareto']][OBJECTIVES[0]],
        y=df[~df['is_pareto']][OBJECTIVES[1]],
        z=df[~df['is_pareto']][OBJECTIVES[2]],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.5),
        name='Dominated Solutions',
        customdata=df[~df['is_pareto']][custom_data_cols],
        hovertemplate=(
            '<b>Dominated</b><br>' +
            f'{OBJECTIVES[0]}: %{{x:.4f}}<br>' +
            f'{OBJECTIVES[1]}: %{{y:.4f}}<br>' +
            f'{OBJECTIVES[2]}: %{{z:.4f}}<br>' +
            'Alpha: %{customdata[0]:.2f}<br>' +
            'Beta: %{customdata[1]:.2f}<br>' +
            'Z_empirical: %{customdata[2]:.4f}<br>' +
            'Raw HS: %{customdata[3]:.0f}, PS: %{customdata[4]:.0f}, MD: %{customdata[5]:.0f}<br>' +
            'Theoretical Norm HS: %{customdata[6]:.4f}, PS: %{customdata[7]:.4f}, MD: %{customdata[8]:.4f}<br>' +
            'Total Theoretical Score: %{customdata[9]:.4f}<extra></extra>'
        )
    ))

    # Pareto points
    fig3d.add_trace(go.Scatter3d(
        x=pareto_df[OBJECTIVES[0]],
        y=pareto_df[OBJECTIVES[1]],
        z=pareto_df[OBJECTIVES[2]],
        mode='markers',
        marker=dict(size=7, color='red', symbol='diamond'),
        name='Pareto Frontier',
        customdata=pareto_df[custom_data_cols],
        hovertemplate=(
            '<b>Pareto Optimal</b><br>' +
            f'{OBJECTIVES[0]}: %{{x:.4f}}<br>' +
            f'{OBJECTIVES[1]}: %{{y:.4f}}<br>' +
            f'{OBJECTIVES[2]}: %{{z:.4f}}<br>' +
            'Alpha: %{customdata[0]:.2f}<br>' +
            'Beta: %{customdata[1]:.2f}<br>' +
            'Z_empirical: %{customdata[2]:.4f}<br>' +
            'Raw HS: %{customdata[3]:.0f}, PS: %{customdata[4]:.0f}, MD: %{customdata[5]:.0f}<br>' +
            'Theoretical Norm HS: %{customdata[6]:.4f}, PS: %{customdata[7]:.4f}, MD: %{customdata[8]:.4f}<br>' +
            'Total Theoretical Score: %{customdata[9]:.4f}<extra></extra>'
        )
    ))

    fig3d.update_layout(
        title=f'3D Pareto Frontier for ({OBJECTIVES[0]}, {OBJECTIVES[1]}, {OBJECTIVES[2]})',
        scene=dict(
            xaxis_title=f'Empirical Norm HomeStrength ({OBJECTIVES[0]})',
            yaxis_title=f'Empirical Norm PenaltySequence ({OBJECTIVES[1]})',
            zaxis_title=f'Empirical Norm MaxDeviation ({OBJECTIVES[2]})'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig3d.write_html(PARETO_PLOT_3D_FILE_HTML)
    print(f"Saved 3D interactive plot to: {PARETO_PLOT_3D_FILE_HTML}")
    try:
        fig3d.write_image(PARETO_PLOT_3D_FILE_PNG, scale=2) # Higher scale for better quality
        print(f"Saved 3D static plot to: {PARETO_PLOT_3D_FILE_PNG}")
    except Exception as e:
        print(f"Could not save 3D static plot: {e}. Ensure kaleido is installed.")

    # --- 2D Projections ---
    def create_2d_plot(df_all, df_pareto, x_col, y_col, title, filename_html, filename_png):
        fig2d = go.Figure()
        
        custom_data_cols_2d = ['alpha', 'beta', 'empirical_score'] + RAW_OBJECTIVES + THEORETICAL_OBJECTIVES + ['total_norm_score']
        # Determine which of the OBJECTIVES are not x_col and y_col to display the third one
        # other_emp_obj = [obj for obj in OBJECTIVES if obj not in [x_col, y_col]][0] # This was commented out, but not needed for hovertemplate

        # Non-Pareto points
        fig2d.add_trace(go.Scatter(
            x=df_all[~df_all['is_pareto']][x_col],
            y=df_all[~df_all['is_pareto']][y_col],
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.5),
            name='Dominated Solutions',
            customdata=df_all[~df_all['is_pareto']][custom_data_cols_2d],
            hovertemplate=(
                f'<b>Dominated</b><br>{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<br>' +
                'Alpha: %{customdata[0]:.2f}<br>Beta: %{customdata[1]:.2f}<br>' +
                'Z_empirical: %{customdata[2]:.4f}<br>' +
                f'Raw HS: %{{customdata[3]:.0f}}, PS: %{{customdata[4]:.0f}}, MD: %{{customdata[5]:.0f}}<br>' +
                f'Theoretical Norm HS: %{{customdata[6]:.4f}}, PS: %{{customdata[7]:.4f}}, MD: %{{customdata[8]:.4f}}<br>' +
                f'Total Theoretical Score: %{{customdata[9]:.4f}}<extra></extra>'
            )
        ))
        # Pareto points
        fig2d.add_trace(go.Scatter(
            x=df_pareto[x_col],
            y=df_pareto[y_col],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Pareto Frontier Points',
            customdata=df_pareto[custom_data_cols_2d],
            hovertemplate=(
                f'<b>Pareto Optimal</b><br>{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<br>' +
                'Alpha: %{customdata[0]:.2f}<br>Beta: %{customdata[1]:.2f}<br>' +
                'Z_empirical: %{customdata[2]:.4f}<br>' +
                f'Raw HS: %{{customdata[3]:.0f}}, PS: %{{customdata[4]:.0f}}, MD: %{{customdata[5]:.0f}}<br>' +
                f'Theoretical Norm HS: %{{customdata[6]:.4f}}, PS: %{{customdata[7]:.4f}}, MD: %{{customdata[8]:.4f}}<br>' +
                f'Total Theoretical Score: %{{customdata[9]:.4f}}<extra></extra>'

            )
        ))
        fig2d.update_layout(
            title=title,
            xaxis_title=f'Empirical Norm ({x_col})',
            yaxis_title=f'Empirical Norm ({y_col})',
            legend_title_text='Solution Type'
        )
        fig2d.write_html(filename_html)
        print(f"Saved 2D plot (empirical) to: {filename_html}")
        try:
            fig2d.write_image(filename_png, scale=2) # Higher scale for better quality
            print(f"Saved 2D static plot (empirical) to: {filename_png}")
        except Exception as e:
            print(f"Could not save 2D static plot (empirical) ({filename_png}): {e}. Ensure kaleido is installed.")


    # HS vs PS
    create_2d_plot(df, pareto_df, OBJECTIVES[0], OBJECTIVES[1],
                   f'2D Pareto Projection: {OBJECTIVES[0]} vs {OBJECTIVES[1]}',
                   PARETO_PLOT_HS_PS_FILE_HTML, PARETO_PLOT_HS_PS_FILE_PNG)

    # HS vs MD
    create_2d_plot(df, pareto_df, OBJECTIVES[0], OBJECTIVES[2],
                   f'2D Pareto Projection: {OBJECTIVES[0]} vs {OBJECTIVES[2]}',
                   PARETO_PLOT_HS_MD_FILE_HTML, PARETO_PLOT_HS_MD_FILE_PNG)

    # PS vs MD
    create_2d_plot(df, pareto_df, OBJECTIVES[1], OBJECTIVES[2],
                   f'2D Pareto Projection: {OBJECTIVES[1]} vs {OBJECTIVES[2]}',
                   PARETO_PLOT_PS_MD_FILE_HTML, PARETO_PLOT_PS_MD_FILE_PNG)

    print("All plots generated.")

    # Suggest some Pareto optimal (alpha, beta) combinations
    if not pareto_df.empty:
        print("\\n--- Suggested Pareto Optimal (alpha, beta) combinations (based on empirical objectives) ---")
        # Sort by sum of empirical normalized objectives (lower is better)
        pareto_df['sum_emp_norm_objectIVES'] = pareto_df[OBJECTIVES].sum(axis=1) # Use empirical objectives
        suggestions = pareto_df.sort_values(by='sum_emp_norm_objectIVES').head(5)
        for _, row in suggestions.iterrows():
            print(f"Alpha: {row['alpha']:.2f}, Beta: {row['beta']:.2f} -> "
                  f"{OBJECTIVES[0]}: {row[OBJECTIVES[0]]:.4f}, {OBJECTIVES[1]}: {row[OBJECTIVES[1]]:.4f}, {OBJECTIVES[2]}: {row[OBJECTIVES[2]]:.4f} "
                  f"(Sum Emp Norm: {row['sum_emp_norm_objectIVES']:.4f})")
            print(f"    Raw values -> HS: {row[RAW_OBJECTIVES[0]]:.0f}, PS: {row[RAW_OBJECTIVES[1]]:.0f}, MD: {row[RAW_OBJECTIVES[2]]:.0f}")
            print(f"    Theoretical norms -> HS: {row[THEORETICAL_OBJECTIVES[0]]:.4f}, PS: {row[THEORETICAL_OBJECTIVES[1]]:.4f}, MD: {row[THEORETICAL_OBJECTIVES[2]]:.4f}")
    else:
        print("\\nNo Pareto optimal points found to suggest combinations.")

    # Remove the call to plot_pareto_frontier if it exists and is now redundant
    # The main plotting logic is now updated to use empirical norms.
    # The plot_pareto_frontier function definition can remain if it's used elsewhere.
    # However, the main block now generates the primary plots using empirical data.

    # --- Additional Function for Empirical Norms ---
    def plot_pareto_frontier(df_results, title_suffix):
        # Use empirically normalized metrics for Pareto calculation and plotting
        costs = df_results[['emp_norm_hs', 'emp_norm_ps', 'emp_norm_md']].values
        pareto_indices = is_pareto_efficient_corrected(costs)
        pareto_points = df_results[pareto_indices]

        print(f"Number of Pareto-efficient points for {title_suffix}: {len(pareto_points)}")

        # --- 3D Interactive Pareto Plot ---
        fig3d_emp = go.Figure()

        # Non-Pareto points
        fig3d_emp.add_trace(go.Scatter3d(
            x=df_results[~pareto_indices]['emp_norm_hs'],
            y=df_results[~pareto_indices]['emp_norm_ps'],
            z=df_results[~pareto_indices]['emp_norm_md'],
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.5),
            name='Dominated Solutions',
            customdata=df_results[~pareto_indices][['alpha', 'beta', 'total_norm_score']],
            hovertemplate='<b>Dominated</b><br>' +
                          'EmpHS: %{x:.4f}<br>' +
                          'EmpPS: %{y:.4f}<br>' +
                          'EmpMD: %{z:.4f}<br>' +
                          'Alpha: %{customdata[0]:.2f}<br>' +
                          'Beta: %{customdata[1]:.2f}<br>' +
                          'Total Score: %{customdata[2]:.4f}<extra></extra>'
        ))

        # Pareto points
        fig3d_emp.add_trace(go.Scatter3d(
            x=pareto_points['emp_norm_hs'],
            y=pareto_points['emp_norm_ps'],
            z=pareto_points['emp_norm_md'],
            mode='markers',
            marker=dict(size=7, color='red', symbol='diamond'),
            name='Pareto Frontier',
            customdata=pareto_points[['alpha', 'beta', 'total_norm_score']],
            hovertemplate='<b>Pareto Optimal</b><br>' +
                          'EmpHS: %{x:.4f}<br>' +
                          'EmpPS: %{y:.4f}<br>' +
                          'EmpMD: %{z:.4f}<br>' +
                          'Alpha: %{customdata[0]:.2f}<br>' +
                          'Beta: %{customdata[1]:.2f}<br>' +
                          'Total Score: %{customdata[2]:.4f}<extra></extra>'
        ))

        fig3d_emp.update_layout(
            title=f'3D Pareto Frontier for (emp_norm_hs, emp_norm_ps, emp_norm_md) {title_suffix}',
            scene=dict(
                xaxis_title='Empirical HomeStrength (HS_raw / sigma_HS)',
                yaxis_title='Empirical PenaltySequence (PS_raw / sigma_PS)',
                zaxis_title='Empirical MaxDeviation (MD_raw / sigma_MD)'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig3d_emp.write_html(os.path.join(OUTPUT_DIR, f"pareto_3d_interactive_empirical{title_suffix}.html"))
        print(f"Saved 3D interactive plot (empirical) to: {os.path.join(OUTPUT_DIR, f'pareto_3d_interactive_empirical{title_suffix}.html')}")
        try:
            fig3d_emp.write_image(os.path.join(OUTPUT_DIR, f"pareto_3d_static_empirical{title_suffix}.png"), scale=2) # Higher scale for better quality
            print(f"Saved 3D static plot (empirical) to: {os.path.join(OUTPUT_DIR, f'pareto_3d_static_empirical{title_suffix}.png')}")
        except Exception as e:
            print(f"Could not save 3D static plot (empirical): {e}. Ensure kaleido is installed.")

        # --- 2D Projections ---
        def create_2d_plot_emp(df_all, df_pareto, x_col, y_col, title, filename_html, filename_png):
            fig2d_emp = go.Figure()
            # Non-Pareto points
            fig2d_emp.add_trace(go.Scatter(
                x=df_all[~df_all['is_pareto']][x_col],
                y=df_all[~df_all['is_pareto']][y_col],
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.5),
                name='Dominated Solutions',
                customdata=df_all[~df_all['is_pareto']][['alpha', 'beta', 'total_norm_score'] + OBJECTIVES],
                hovertemplate=f'<b>Dominated</b><br>{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<br>' +
                              'Alpha: %{customdata[0]:.2f}<br>Beta: %{customdata[1]:.2f}<br>' +
                              'Total Score: %{customdata[2]:.4f}<br>' +
                              'HS: %{customdata[3]:.4f}, PS: %{customdata[4]:.4f}, MD: %{customdata[5]:.4f}<extra></extra>'
            ))
            # Pareto points
            fig2d_emp.add_trace(go.Scatter(
                x=df_pareto[x_col],
                y=df_pareto[y_col],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond'),
                name='Pareto Frontier Points',
                customdata=df_pareto[['alpha', 'beta', 'total_norm_score'] + OBJECTIVES],
                hovertemplate=f'<b>Pareto Optimal</b><br>{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<br>' +
                              'Alpha: %{customdata[0]:.2f}<br>Beta: %{customdata[1]:.2f}<br>' +
                              'Total Score: %{customdata[2]:.4f}<br>' +
                              'HS: %{customdata[3]:.4f}, PS: %{customdata[4]:.4f}, MD: %{customdata[5]:.4f}<extra></extra>'
            ))
            fig2d_emp.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
                legend_title_text='Solution Type'
            )
            fig2d_emp.write_html(filename_html)
            print(f"Saved 2D plot (empirical) to: {filename_html}")
            try:
                fig2d_emp.write_image(filename_png, scale=2) # Higher scale for better quality
                print(f"Saved 2D static plot (empirical) to: {filename_png}")
            except Exception as e:
                print(f"Could not save 2D static plot (empirical) ({filename_png}): {e}. Ensure kaleido is installed.")


        # HS vs PS
        create_2d_plot_emp(df_results, pareto_points, 'emp_norm_hs', 'emp_norm_ps',
                           '2D Pareto Projection: Empirical HS_norm vs PS_norm',
                           os.path.join(OUTPUT_DIR, f"pareto_2d_hs_vs_ps_empirical{title_suffix}.html"),
                           os.path.join(OUTPUT_DIR, f"pareto_2d_hs_vs_ps_empirical{title_suffix}.png"))

        # HS vs MD
        create_2d_plot_emp(df_results, pareto_points, 'emp_norm_hs', 'emp_norm_md',
                           '2D Pareto Projection: Empirical HS_norm vs MD_norm',
                           os.path.join(OUTPUT_DIR, f"pareto_2d_hs_vs_md_empirical{title_suffix}.html"),
                           os.path.join(OUTPUT_DIR, f"pareto_2d_hs_vs_md_empirical{title_suffix}.png"))

        # PS vs MD
        create_2d_plot_emp(df_results, pareto_points, 'emp_norm_ps', 'emp_norm_md',
                           '2D Pareto Projection: Empirical PS_norm vs MD_norm',
                           os.path.join(OUTPUT_DIR, f"pareto_2d_ps_vs_md_empirical{title_suffix}.html"),
                           os.path.join(OUTPUT_DIR, f"pareto_2d_ps_vs_md_empirical{title_suffix}.png"))

        print("All empirical plots generated.")

        # Print suggested Pareto-optimal (alpha, beta) combinations based on empirical norms
        print("\nSuggested Pareto-optimal (alpha, beta) combinations based on empirical norms:")
        for index, row in pareto_points.iterrows():
            print(f"Alpha: {row['alpha']:.2f}, Beta: {row['beta']:.2f} -> EmpHS: {row['emp_norm_hs']:.4f}, EmpPS: {row['emp_norm_ps']:.4f}, EmpMD: {row['emp_norm_md']:.4f} (Raw HS: {row['raw_hs']:.0f}, Raw PS: {row['raw_ps']:.0f}, Raw MD: {row['raw_md']:.0f})")
