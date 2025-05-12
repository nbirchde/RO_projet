import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np

# --- Configuration ---
INPUT_CSV = "calibration_results_n6_dense.csv" # Updated input file
OUTPUT_DIR = "calibration_plots_dense"      # Updated output directory
PARETO_PLOT_3D_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_3d_interactive.html")
PARETO_PLOT_3D_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_3d_static.png")
PARETO_PLOT_HS_PS_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_ps.html")
PARETO_PLOT_HS_PS_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_ps.png")
PARETO_PLOT_HS_MD_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_md.html")
PARETO_PLOT_HS_MD_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_md.png")
PARETO_PLOT_PS_MD_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_ps_vs_md.html")
PARETO_PLOT_PS_MD_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_ps_vs_md.png")


# Objectives to minimize
OBJECTIVES = ['norm_hs', 'norm_ps', 'norm_md']
# --- End Configuration ---

def is_pareto_efficient(costs, return_mask=True):
    """
    Find the Pareto-efficient points.
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a boolean mask, False to return integer indices
    :return: An array of indices of Pareto-efficient points.
             If return_mask is True, this will be a boolean array of length n_points.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a strictly dominating cost
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            # And keep self
            is_efficient[i] = True
    if return_mask:
        return is_efficient
    else:
        return np.where(is_efficient)[0]

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
    pareto_mask = is_pareto_efficient(df[OBJECTIVES].values)
    df['is_pareto'] = pareto_mask
    pareto_df = df[df['is_pareto']].copy()

    print(f"Loaded {len(df)} data points.")
    print(f"Found {len(pareto_df)} Pareto-efficient points.")

    # --- 3D Interactive Pareto Plot ---
    fig3d = go.Figure()

    # Non-Pareto points
    fig3d.add_trace(go.Scatter3d(
        x=df[~df['is_pareto']]['norm_hs'],
        y=df[~df['is_pareto']]['norm_ps'],
        z=df[~df['is_pareto']]['norm_md'],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.5),
        name='Dominated Solutions',
        customdata=df[~df['is_pareto']][['alpha', 'beta', 'total_norm_score']],
        hovertemplate='<b>Dominated</b><br>' +
                      'HS_norm: %{x:.4f}<br>' +
                      'PS_norm: %{y:.4f}<br>' +
                      'MD_norm: %{z:.4f}<br>' +
                      'Alpha: %{customdata[0]:.2f}<br>' +
                      'Beta: %{customdata[1]:.2f}<br>' +
                      'Total Score: %{customdata[2]:.4f}<extra></extra>'
    ))

    # Pareto points
    fig3d.add_trace(go.Scatter3d(
        x=pareto_df['norm_hs'],
        y=pareto_df['norm_ps'],
        z=pareto_df['norm_md'],
        mode='markers',
        marker=dict(size=7, color='red', symbol='diamond'),
        name='Pareto Frontier',
        customdata=pareto_df[['alpha', 'beta', 'total_norm_score']],
        hovertemplate='<b>Pareto Optimal</b><br>' +
                      'HS_norm: %{x:.4f}<br>' +
                      'PS_norm: %{y:.4f}<br>' +
                      'MD_norm: %{z:.4f}<br>' +
                      'Alpha: %{customdata[0]:.2f}<br>' +
                      'Beta: %{customdata[1]:.2f}<br>' +
                      'Total Score: %{customdata[2]:.4f}<extra></extra>'
    ))

    fig3d.update_layout(
        title='3D Pareto Frontier for (norm_hs, norm_ps, norm_md)',
        scene=dict(
            xaxis_title='Normalized HomeStrength (HS_norm)',
            yaxis_title='Normalized PenaltySequence (PS_norm)',
            zaxis_title='Normalized MaxDeviation (MD_norm)'
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
            customdata=df_all[~df_all['is_pareto']][['alpha', 'beta', 'total_norm_score'] + OBJECTIVES],
            hovertemplate=f'<b>Dominated</b><br>{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<br>' +
                          'Alpha: %{customdata[0]:.2f}<br>Beta: %{customdata[1]:.2f}<br>' +
                          'Total Score: %{customdata[2]:.4f}<br>' +
                          'HS: %{customdata[3]:.4f}, PS: %{customdata[4]:.4f}, MD: %{customdata[5]:.4f}<extra></extra>'
        ))
        # Pareto points
        fig2d.add_trace(go.Scatter(
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
        fig2d.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend_title_text='Solution Type'
        )
        fig2d.write_html(filename_html)
        print(f"Saved 2D plot to: {filename_html}")
        try:
            fig2d.write_image(filename_png, scale=2) # Higher scale for better quality
            print(f"Saved 2D static plot to: {filename_png}")
        except Exception as e:
            print(f"Could not save 2D static plot ({filename_png}): {e}. Ensure kaleido is installed.")


    # HS vs PS
    create_2d_plot(df, pareto_df, 'norm_hs', 'norm_ps',
                   '2D Pareto Projection: HS_norm vs PS_norm',
                   PARETO_PLOT_HS_PS_FILE_HTML, PARETO_PLOT_HS_PS_FILE_PNG)

    # HS vs MD
    create_2d_plot(df, pareto_df, 'norm_hs', 'norm_md',
                   '2D Pareto Projection: HS_norm vs MD_norm',
                   PARETO_PLOT_HS_MD_FILE_HTML, PARETO_PLOT_HS_MD_FILE_PNG)

    # PS vs MD
    create_2d_plot(df, pareto_df, 'norm_ps', 'norm_md',
                   '2D Pareto Projection: PS_norm vs MD_norm',
                   PARETO_PLOT_PS_MD_FILE_HTML, PARETO_PLOT_PS_MD_FILE_PNG)

    print("All plots generated.")

    # Suggest some Pareto optimal (alpha, beta) combinations
    if not pareto_df.empty:
        print("\n--- Suggested Pareto Optimal (alpha, beta) combinations ---")
        # Sort by a composite score or individual objectives to give varied suggestions
        # Example: sort by sum of normalized objectives (lower is better)
        pareto_df['sum_norm_objectives'] = pareto_df[OBJECTIVES].sum(axis=1)
        suggestions = pareto_df.sort_values(by='sum_norm_objectives').head(5)
        for _, row in suggestions.iterrows():
            print(f"Alpha: {row['alpha']:.2f}, Beta: {row['beta']:.2f} -> "
                  f"HS_norm: {row['norm_hs']:.4f}, PS_norm: {row['norm_ps']:.4f}, MD_norm: {row['norm_md']:.4f} "
                  f"(Sum: {row['sum_norm_objectives']:.4f})")
    else:
        print("\nNo Pareto optimal points found to suggest combinations.")
