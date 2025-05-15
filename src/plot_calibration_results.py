import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np

INPUT_CSV = "calibration_results_n500_analytical_norm.csv"
OUTPUT_DIR = "calibration_plots_n500_analytical_norm"
PARETO_PLOT_3D_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_3d_interactive_n500_anal_norm.html")
PARETO_PLOT_3D_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_3d_static_n500_anal_norm.png")
PARETO_PLOT_HS_PS_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_ps_n500_anal_norm.html")
PARETO_PLOT_HS_PS_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_ps_n500_anal_norm.png")
PARETO_PLOT_HS_MD_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_md_n500_anal_norm.html")
PARETO_PLOT_HS_MD_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_hs_vs_md_n500_anal_norm.png")
PARETO_PLOT_PS_MD_FILE_HTML = os.path.join(OUTPUT_DIR, "pareto_2d_ps_vs_md_n500_anal_norm.html")
PARETO_PLOT_PS_MD_FILE_PNG = os.path.join(OUTPUT_DIR, "pareto_2d_ps_vs_md_n500_anal_norm.png")

OBJECTIVES = ['scaled_hs', 'scaled_ps', 'scaled_md']


def is_pareto_efficient_corrected(costs, return_mask=True):
    """
    Find the Pareto-efficient points (lower values are better).
    """
    num_points = costs.shape[0]
    is_efficient_mask = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        if not is_efficient_mask[i]:
            continue
        for j in range(num_points):
            if i == j:
                continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                is_efficient_mask[i] = False
                break
    if return_mask:
        return is_efficient_mask
    else:
        return np.where(is_efficient_mask)[0]

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV}' not found. Please run the calibration script first.")
        exit()

    df.dropna(subset=OBJECTIVES, inplace=True)
    if df.empty:
        print(f"Error: No valid data found in '{INPUT_CSV}' after dropping NaNs.")
        exit()

    pareto_mask = is_pareto_efficient_corrected(df[OBJECTIVES].values)
    df['is_pareto'] = pareto_mask
    pareto_df = df[df['is_pareto']].copy()

    print(f"Loaded {len(df)} data points.")
    print(f"Found {len(pareto_df)} Pareto-efficient points.")

    fig3d = go.Figure()

    fig3d.add_trace(go.Scatter3d(
        x=df[~df['is_pareto']]['scaled_hs'],
        y=df[~df['is_pareto']]['scaled_ps'],
        z=df[~df['is_pareto']]['scaled_md'],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.5),
        name='Dominated Solutions',
        customdata=df[~df['is_pareto']][['alpha', 'beta', 'total_scaled_score', 'anal_norm_hs', 'anal_norm_ps', 'anal_norm_md', 'raw_hs', 'raw_ps', 'raw_md']],
        hovertemplate='<b>Dominated</b><br>' +
                      'Scaled HS: %{x:.4f}<br>' +
                      'Scaled PS: %{y:.4f}<br>' +
                      'Scaled MD: %{z:.4f}<br>' +
                      'Alpha: %{customdata[0]:.2f}<br>' +
                      'Beta: %{customdata[1]:.2f}<br>' +
                      'Total Scaled Score: %{customdata[2]:.4f}<br>' +
                      'Anal Norm HS: %{customdata[3]:.4f}<br>' +
                      'Anal Norm PS: %{customdata[4]:.4f}<br>' +
                      'Anal Norm MD: %{customdata[5]:.4f}<br>' +
                      'Raw HS: %{customdata[6]:.0f}<br>' +
                      'Raw PS: %{customdata[7]:.0f}<br>' +
                      'Raw MD: %{customdata[8]:.2f}<extra></extra>'
    ))

    fig3d.add_trace(go.Scatter3d(
        x=pareto_df['scaled_hs'],
        y=pareto_df['scaled_ps'],
        z=pareto_df['scaled_md'],
        mode='markers',
        marker=dict(size=7, color='red', symbol='diamond'),
        name='Pareto Frontier',
        customdata=pareto_df[['alpha', 'beta', 'total_scaled_score', 'anal_norm_hs', 'anal_norm_ps', 'anal_norm_md', 'raw_hs', 'raw_ps', 'raw_md']],
        hovertemplate='<b>Pareto Optimal</b><br>' +
                      'Scaled HS: %{x:.4f}<br>' +
                      'Scaled PS: %{y:.4f}<br>' +
                      'Scaled MD: %{z:.4f}<br>' +
                      'Alpha: %{customdata[0]:.2f}<br>' +
                      'Beta: %{customdata[1]:.2f}<br>' +
                      'Total Scaled Score: %{customdata[2]:.4f}<br>' +
                      'Anal Norm HS: %{customdata[3]:.4f}<br>' +
                      'Anal Norm PS: %{customdata[4]:.4f}<br>' +
                      'Anal Norm MD: %{customdata[5]:.4f}<br>' +
                      'Raw HS: %{customdata[6]:.0f}<br>' +
                      'Raw PS: %{customdata[7]:.0f}<br>' +
                      'Raw MD: %{customdata[8]:.2f}<extra></extra>'
    ))

    fig3d.update_layout(
        title='3D Pareto Frontier for Scaled Metrics ([0,1])',
        scene=dict(
            xaxis_title='Scaled HS',
            yaxis_title='Scaled PS',
            zaxis_title='Scaled MD'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig3d.write_html(PARETO_PLOT_3D_FILE_HTML)
    print(f"Saved 3D interactive plot to: {PARETO_PLOT_3D_FILE_HTML}")
    try:
        fig3d.write_image(PARETO_PLOT_3D_FILE_PNG, scale=2)
        print(f"Saved 3D static plot to: {PARETO_PLOT_3D_FILE_PNG}")
    except Exception as e:
        print(f"Could not save 3D static plot: {e}. Ensure kaleido is installed.")

    def create_2d_plot(df_all, df_pareto, x_col, y_col, title, filename_html, filename_png):
        fig2d = go.Figure()
        fig2d.add_trace(go.Scatter(
            x=df_all[~df_all['is_pareto']][x_col],
            y=df_all[~df_all['is_pareto']][y_col],
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.5),
            name='Dominated Solutions',
            customdata=df_all[~df_all['is_pareto']][['alpha', 'beta', 'total_scaled_score', 'scaled_hs', 'scaled_ps', 'scaled_md', 'anal_norm_hs', 'anal_norm_ps', 'anal_norm_md', 'raw_hs', 'raw_ps', 'raw_md']],
            hovertemplate=f'<b>Dominated</b><br>{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<br>' +
                          'Alpha: %{customdata[0]:.2f}<br>' +
                          'Beta: %{customdata[1]:.2f}<br>' +
                          'Total Scaled Score: %{customdata[2]:.4f}<br>' +
                          f'Scaled HS: %{{customdata[3]:.4f}}<br>Scaled PS: %{{customdata[4]:.4f}}<br>Scaled MD: %{{customdata[5]:.4f}}<br>' +
                          'Anal Norm HS: %{customdata[6]:.4f}<br>' +
                          'Anal Norm PS: %{customdata[7]:.4f}<br>' +
                          'Anal Norm MD: %{customdata[8]:.4f}<br>' +
                          'Raw HS: %{customdata[9]:.0f}<br>Raw PS: %{customdata[10]:.0f}<br>Raw MD: %{customdata[11]:.2f}<extra></extra>'
        ))
        fig2d.add_trace(go.Scatter(
            x=df_pareto[x_col],
            y=df_pareto[y_col],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Pareto Frontier Points',
            customdata=df_pareto[['alpha', 'beta', 'total_scaled_score', 'scaled_hs', 'scaled_ps', 'scaled_md', 'anal_norm_hs', 'anal_norm_ps', 'anal_norm_md', 'raw_hs', 'raw_ps', 'raw_md']],
            hovertemplate=f'<b>Pareto Optimal</b><br>{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<br>' +
                          'Alpha: %{customdata[0]:.2f}<br>Beta: %{customdata[1]:.2f}<br>' +
                          'Total Scaled Score: %{customdata[2]:.4f}<br>' +
                          f'Scaled HS: %{{customdata[3]:.4f}}<br>Scaled PS: %{{customdata[4]:.4f}}<br>Scaled MD: %{{customdata[5]:.4f}}<br>' +
                          'Anal Norm HS: %{customdata[6]:.4f}<br>' +
                          'Anal Norm PS: %{customdata[7]:.4f}<br>' +
                          'Anal Norm MD: %{customdata[8]:.4f}<br>' +
                          'Raw HS: %{customdata[9]:.0f}<br>Raw PS: %{customdata[10]:.0f}<br>Raw MD: %{customdata[11]:.2f}<extra></extra>'
        ))
        fig2d.update_layout(
            title=title,
            xaxis_title=f'{x_col}',
            yaxis_title=f'{y_col}',
            legend_title_text='Solution Type'
        )
        fig2d.write_html(filename_html)
        print(f"Saved 2D plot to: {filename_html}")
        try:
            fig2d.write_image(filename_png, scale=2)
            print(f"Saved 2D static plot to: {filename_png}")
        except Exception as e:
            print(f"Could not save 2D static plot (filename_png): {e}. Ensure kaleido is installed.")

    create_2d_plot(df, pareto_df, 'scaled_hs', 'scaled_ps',
                   '2D Pareto: Scaled HS vs PS ([0,1])',
                   PARETO_PLOT_HS_PS_FILE_HTML, PARETO_PLOT_HS_PS_FILE_PNG)

    create_2d_plot(df, pareto_df, 'scaled_hs', 'scaled_md',
                   '2D Pareto: Scaled HS vs MD ([0,1])',
                   PARETO_PLOT_HS_MD_FILE_HTML, PARETO_PLOT_HS_MD_FILE_PNG)

    create_2d_plot(df, pareto_df, 'scaled_ps', 'scaled_md',
                   '2D Pareto: Scaled PS vs MD ([0,1])',
                   PARETO_PLOT_PS_MD_FILE_HTML, PARETO_PLOT_PS_MD_FILE_PNG)

    print("All plots generated.")

    if not pareto_df.empty:
        print("\n--- Suggested Pareto Optimal (alpha, beta) combinations (using Scaled Metrics) ---")
        pareto_df['sum_scaled_objectives'] = pareto_df[OBJECTIVES].sum(axis=1)
        suggestions = pareto_df.sort_values(by='sum_scaled_objectives').head(5)
        for _, row in suggestions.iterrows():
            print(f"Alpha: {row['alpha']:.2f}, Beta: {row['beta']:.2f} -> "
                  f"Scaled HS: {row['scaled_hs']:.4f}, Scaled PS: {row['scaled_ps']:.4f}, Scaled MD: {row['scaled_md']:.4f} "
                  f"(Sum Scaled: {row['sum_scaled_objectives']:.4f}) | "
                  f"Raw HS: {row['raw_hs']:.0f}, Raw PS: {row['raw_ps']:.0f}, Raw MD: {row['raw_md']:.2f}")
    else:
        print("\nNo Pareto optimal points found to suggest combinations.")
