import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np

# --- Configuration ---
INPUT_CSV = "calibration_results_n200_empirical_norm_v2_median_subtracted.csv" # Updated input file
OUTPUT_DIR = "calibration_plots_n200_empirical_norm_v2_median_subtracted"      # Updated output directory
PARETO_PLOT_3D_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_3d_interactive_n200_emp_norm_v2.html")
PARETO_PLOT_3D_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_3d_static_n200_emp_norm_v2.png")
PARETO_PLOT_HS_PS_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_ps_n200_emp_norm_v2.html")
PARETO_PLOT_HS_PS_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_ps_n200_emp_norm_v2.png")
PARETO_PLOT_HS_MD_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_md_n200_emp_norm_v2.html")
PARETO_PLOT_HS_MD_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_md_n200_emp_norm_v2.png")
PARETO_PLOT_PS_MD_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_ps_vs_md_n200_emp_norm_v2.html")
PARETO_PLOT_PS_MD_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_ps_vs_md_n200_emp_norm_v2.png")


# Objectives to minimize (using the median-subtracted empirical normalized scores)
OBJECTIVES = ['emp_norm_hs', 'emp_norm_ps', 'emp_norm_md']
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

    # Identify Pareto-efficient points
    # We want to minimize all objectives
    # pareto_mask = is_pareto_efficient(df[OBJECTIVES].values) # Old version
    pareto_mask = is_pareto_efficient_corrected(df[OBJECTIVES].values) # Corrected version
    df['is_pareto'] = pareto_mask
    pareto_df = df[df['is_pareto']].copy()

    print(f"Loaded {len(df)} data points.")
    print(f"Found {len(pareto_df)} Pareto-efficient points.")

    # --- 3D Interactive Pareto Plot ---
    fig3d = go.Figure()

    # Non-Pareto points
    fig3d.add_trace(go.Scatter3d(
        x=df[~df['is_pareto']]['emp_norm_hs'],
        y=df[~df['is_pareto']]['emp_norm_ps'],
        z=df[~df['is_pareto']]['emp_norm_md'],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.5),
        name='Dominated Solutions',
        customdata=df[~df['is_pareto']][['alpha', 'beta', 'total_norm_score', 'raw_hs', 'raw_ps', 'raw_md']],
        hovertemplate='<b>Dominated</b><br>' +
                      'EmpNorm HS: %{x:.4f}<br>' +
                      'EmpNorm PS: %{y:.4f}<br>' +
                      'EmpNorm MD: %{z:.4f}<br>' +
                      'Alpha: %{customdata[0]:.2f}<br>' +
                      'Beta: %{customdata[1]:.2f}<br>' +
                      'Total Score (SA obj): %{customdata[2]:.4f}<br>' +
                      'Raw HS: %{customdata[3]:.0f}<br>' +
                      'Raw PS: %{customdata[4]:.0f}<br>' +
                      'Raw MD: %{customdata[5]:.2f}<extra></extra>'
    ))

    # Pareto points
    fig3d.add_trace(go.Scatter3d(
        x=pareto_df['emp_norm_hs'],
        y=pareto_df['emp_norm_ps'],
        z=pareto_df['emp_norm_md'],
        mode='markers',
        marker=dict(size=7, color='red', symbol='diamond'),
        name='Pareto Frontier',
        customdata=pareto_df[['alpha', 'beta', 'total_norm_score', 'raw_hs', 'raw_ps', 'raw_md']],
        hovertemplate='<b>Pareto Optimal</b><br>' +
                      'EmpNorm HS: %{x:.4f}<br>' +
                      'EmpNorm PS: %{y:.4f}<br>' +
                      'EmpNorm MD: %{z:.4f}<br>' +
                      'Alpha: %{customdata[0]:.2f}<br>' +
                      'Beta: %{customdata[1]:.2f}<br>' +
                      'Total Score (SA obj): %{customdata[2]:.4f}<br>' +
                      'Raw HS: %{customdata[3]:.0f}<br>' +
                      'Raw PS: %{customdata[4]:.0f}<br>' +
                      'Raw MD: %{customdata[5]:.2f}<extra></extra>'
    ))

    fig3d.update_layout(
        title='3D Pareto Frontier for Empirically Normalized Metrics (Median Subtracted)',
        scene=dict(
            xaxis_title='Emp. Norm HS ((raw-med)/σ)',
            yaxis_title='Emp. Norm PS ((raw-med)/σ)',
            zaxis_title='Emp. Norm MD ((raw-med)/σ)'
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
        # Non-Pareto points
        fig2d.add_trace(go.Scatter(
            x=df_all[~df_all['is_pareto']][x_col],
            y=df_all[~df_all['is_pareto']][y_col],
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.5),
            name='Dominated Solutions',
            customdata=df_all[~df_all['is_pareto']][['alpha', 'beta', 'total_norm_score'] + OBJECTIVES + ['raw_hs', 'raw_ps', 'raw_md']],
            hovertemplate=f'<b>Dominated</b><br>{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<br>' +
                          'Alpha: %{customdata[0]:.2f}<br>Beta: %{customdata[1]:.2f}<br>' +
                          'Total Score (SA obj): %{customdata[2]:.4f}<br>' +
                          f'{OBJECTIVES[0]}: %{{customdata[3]:.4f}}<br>{OBJECTIVES[1]}: %{{customdata[4]:.4f}}<br>{OBJECTIVES[2]}: %{{customdata[5]:.4f}}<br>' +
                          'Raw HS: %{customdata[6]:.0f}<br>Raw PS: %{customdata[7]:.0f}<br>Raw MD: %{customdata[8]:.2f}<extra></extra>'
        ))
        # Pareto points
        fig2d.add_trace(go.Scatter(
            x=df_pareto[x_col],
            y=df_pareto[y_col],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Pareto Frontier Points',
            customdata=df_pareto[['alpha', 'beta', 'total_norm_score'] + OBJECTIVES + ['raw_hs', 'raw_ps', 'raw_md']],
            hovertemplate=f'<b>Pareto Optimal</b><br>{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<br>' +
                          'Alpha: %{customdata[0]:.2f}<br>Beta: %{customdata[1]:.2f}<br>' +
                          'Total Score (SA obj): %{customdata[2]:.4f}<br>' +
                          f'{OBJECTIVES[0]}: %{{customdata[3]:.4f}}<br>{OBJECTIVES[1]}: %{{customdata[4]:.4f}}<br>{OBJECTIVES[2]}: %{{customdata[5]:.4f}}<br>' +
                          'Raw HS: %{customdata[6]:.0f}<br>Raw PS: %{customdata[7]:.0f}<br>Raw MD: %{customdata[8]:.2f}<extra></extra>'
        ))
        fig2d.update_layout(
            title=title,
            xaxis_title=f'{x_col} ((raw-med)/σ)',
            yaxis_title=f'{y_col} ((raw-med)/σ)',
            legend_title_text='Solution Type'
        )
        fig2d.write_html(filename_html)
        print(f"Saved 2D plot to: {filename_html}")
        try:
            fig2d.write_image(filename_png, scale=2) # Higher scale for better quality
            print(f"Saved 2D static plot to: {filename_png}")
        except Exception as e:
            print(f"Could not save 2D static plot (filename_png): {e}. Ensure kaleido is installed.")


    # HS vs PS
    create_2d_plot(df, pareto_df, 'emp_norm_hs', 'emp_norm_ps',
                   '2D Pareto: Emp. Norm HS vs PS (Median Subtracted)',
                   PARETO_PLOT_HS_PS_FILE_HTML, PARETO_PLOT_HS_PS_FILE_PNG)

    # HS vs MD
    create_2d_plot(df, pareto_df, 'emp_norm_hs', 'emp_norm_md',
                   '2D Pareto: Emp. Norm HS vs MD (Median Subtracted)',
                   PARETO_PLOT_HS_MD_FILE_HTML, PARETO_PLOT_HS_MD_FILE_PNG)

    # PS vs MD
    create_2d_plot(df, pareto_df, 'emp_norm_ps', 'emp_norm_md',
                   '2D Pareto: Emp. Norm PS vs MD (Median Subtracted)',
                   PARETO_PLOT_PS_MD_FILE_HTML, PARETO_PLOT_PS_MD_FILE_PNG)

    print("All plots generated.")

    # Suggest some Pareto optimal (alpha, beta) combinations
    if not pareto_df.empty:
        print("\\n--- Suggested Pareto Optimal (alpha, beta) combinations (using Empirically Normalized Metrics) ---")
        # Sort by a composite score or individual objectives to give varied suggestions
        # Example: sort by sum of normalized objectives (lower is better)
        pareto_df['sum_emp_norm_objectives'] = pareto_df[OBJECTIVES].sum(axis=1) # Use updated OBJECTIVES
        suggestions = pareto_df.sort_values(by='sum_emp_norm_objectives').head(5)
        for _, row in suggestions.iterrows():
            print(f"Alpha: {row['alpha']:.2f}, Beta: {row['beta']:.2f} -> "
                  f"EmpNorm_HS: {row['emp_norm_hs']:.4f}, EmpNorm_PS: {row['emp_norm_ps']:.4f}, EmpNorm_MD: {row['emp_norm_md']:.4f} "
                  f"(Sum EmpNorm: {row['sum_emp_norm_objectives']:.4f}) | "
                  f"Raw HS: {row['raw_hs']:.0f}, Raw PS: {row['raw_ps']:.0f}, Raw MD: {row['raw_md']:.2f}")
    else:
        print("\\nNo Pareto optimal points found to suggest combinations.")

    # --- Additional Function for Empirical Norms ---
    # This function is now effectively integrated into the main script logic above.
    # It can be removed to avoid confusion.
    # def plot_pareto_frontier(df_results, title_suffix):
    #     ... (rest of the function)
