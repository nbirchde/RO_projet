#!/usr/bin/env python3
"""
Plots Pareto frontier analysis using 2D subplots from calibration results.
Shows only unique non-dominated points in the metric space.

Reads a CSV file containing alpha, beta, and normalized metric values,
then generates a figure with three 2D scatter subplots:
1. PS_norm vs. HS_norm
2. MD_norm vs. HS_norm
3. MD_norm vs. PS_norm
Points are colored by z_norm_calculated. Allows highlighting a specific (alpha, beta) point.
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

def get_non_dominated_set(df, metrics_to_consider):
    """
    Identifies the set of non-dominated solutions from a DataFrame.
    Assumes lower values are better for all metrics_to_consider.

    Args:
        df (pd.DataFrame): DataFrame containing the solutions.
        metrics_to_consider (list of str): List of column names for the metrics.

    Returns:
        pd.DataFrame: A DataFrame containing only the non-dominated solutions.
    """
    is_dominated = np.zeros(len(df), dtype=bool)
    values = df[metrics_to_consider].values

    for i in range(len(df)):
        if is_dominated[i]: 
            continue
        for j in range(len(df)):
            if i == j:
                continue
            if is_dominated[j]: 
                continue

            all_j_le_i = np.all(values[j] <= values[i])
            any_j_lt_i = np.any(values[j] < values[i])

            if all_j_le_i and any_j_lt_i:
                is_dominated[i] = True
                break 
    
    return df[~is_dominated].copy()

def plot_pareto_frontier_2d_subplots(csv_path, output_image_path, n_teams, alpha_highlight=None, beta_highlight=None):
    """
    Generates and saves a figure with 2D subplots of the unique non-dominated Pareto frontier.
    """
    try:
        df_orig = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    df = df_orig[df_orig['status'] == 'Optimal'].copy()
    
    metric_cols = ['hs_norm', 'ps_norm', 'md_norm', 'z_norm_calculated', 'alpha', 'beta']
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=metric_cols, inplace=True)
    df = df[~((np.isclose(df['alpha'], 0)) & (np.isclose(df['beta'], 0)))]

    if df.empty:
        print("No optimal solutions with valid metric data (after initial filtering) found in the CSV.")
        return

    chosen_point_data_for_highlight = None
    if alpha_highlight is not None and beta_highlight is not None:
        chosen_point_df_temp = df[
            np.isclose(df['alpha'], alpha_highlight) & 
            np.isclose(df['beta'], beta_highlight)
        ]
        if not chosen_point_df_temp.empty:
            chosen_point_data_for_highlight = chosen_point_df_temp.iloc[0].copy()
            print(f"Point to highlight (α={alpha_highlight}, β={beta_highlight}) found in data:")
            print(f"  Metrics: HSn={chosen_point_data_for_highlight['hs_norm']:.3f}, PSn={chosen_point_data_for_highlight['ps_norm']:.3f}, MDn={chosen_point_data_for_highlight['md_norm']:.3f}")
        else:
            print(f"Warning: Point for α={alpha_highlight}, β={beta_highlight} (to be highlighted) not found in the initial filtered data.")

    metrics_for_pareto = ['hs_norm', 'ps_norm', 'md_norm']
    non_dominated_df_all_occurrences = get_non_dominated_set(df, metrics_for_pareto)
    
    if non_dominated_df_all_occurrences.empty:
        print("No non-dominated solutions found after filtering.")
        if df.empty: return
        non_dominated_df_unique_metrics = df.drop_duplicates(subset=metrics_for_pareto, keep='first').copy()
        plot_title_suffix = "(All Optimal, α=0,β=0 excluded - No non-dominated found, showing unique metrics)"
    else:
        # Drop duplicates based on metric values to plot only unique points in metric space
        non_dominated_df_unique_metrics = non_dominated_df_all_occurrences.drop_duplicates(subset=metrics_for_pareto, keep='first').copy()
        plot_title_suffix = "(Unique Non-Dominated in Metric Space, α=0,β=0 excluded)"
        print(f"Total non-dominated solution occurrences (from different alpha/beta): {len(non_dominated_df_all_occurrences)}")


    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    fig.suptitle(f'Pareto Frontier Analysis for n={n_teams} {plot_title_suffix}\nLower values are better for all axes', fontsize=18, y=1.04)

    # Adjust scatter size and alpha based on number of unique points
    num_unique_points = len(non_dominated_df_unique_metrics)
    if num_unique_points > 50:
        scatter_size = 40
        scatter_alpha = 0.6
    elif num_unique_points > 20:
        scatter_size = 60
        scatter_alpha = 0.7
    else:
        scatter_size = 80
        scatter_alpha = 0.75
        
    cmap = 'viridis_r'

    ax1 = axes[0]
    sc1 = ax1.scatter(non_dominated_df_unique_metrics['hs_norm'], non_dominated_df_unique_metrics['ps_norm'], 
                      c=non_dominated_df_unique_metrics['z_norm_calculated'], cmap=cmap, 
                      s=scatter_size, alpha=scatter_alpha, label='Non-dominated points')
    ax1.set_xlabel('Normalized Home Strength (HS_norm)', fontsize=13)
    ax1.set_ylabel('Normalized Penalty Sequence (PS_norm)', fontsize=13)
    ax1.set_title('PS_norm vs HS_norm', fontsize=15)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = axes[1]
    sc2 = ax2.scatter(non_dominated_df_unique_metrics['hs_norm'], non_dominated_df_unique_metrics['md_norm'], 
                      c=non_dominated_df_unique_metrics['z_norm_calculated'], cmap=cmap, 
                      s=scatter_size, alpha=scatter_alpha)
    ax2.set_xlabel('Normalized Home Strength (HS_norm)', fontsize=13)
    ax2.set_ylabel('Normalized Max Deviation (MD_norm)', fontsize=13)
    ax2.set_title('MD_norm vs HS_norm', fontsize=15)
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3 = axes[2]
    sc3 = ax3.scatter(non_dominated_df_unique_metrics['ps_norm'], non_dominated_df_unique_metrics['md_norm'], 
                      c=non_dominated_df_unique_metrics['z_norm_calculated'], cmap=cmap, 
                      s=scatter_size, alpha=scatter_alpha)
    ax3.set_xlabel('Normalized Penalty Sequence (PS_norm)', fontsize=13)
    ax3.set_ylabel('Normalized Max Deviation (MD_norm)', fontsize=13)
    ax3.set_title('MD_norm vs PS_norm', fontsize=15)
    ax3.grid(True, linestyle='--', alpha=0.6)

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label='Unique Non-dominated points',
                                  markerfacecolor='grey', markersize=10, alpha=scatter_alpha)]

    if chosen_point_data_for_highlight is not None:
        highlight_color = 'red'
        highlight_size = max(scatter_size * 2, 150) # Ensure highlight is visible
        highlight_marker = '*'
        highlight_edgecolor = 'black'
        
        ax1.scatter(chosen_point_data_for_highlight['hs_norm'], chosen_point_data_for_highlight['ps_norm'], 
                    c=highlight_color, s=highlight_size, marker=highlight_marker, edgecolor=highlight_edgecolor, 
                    label=f"Chosen (α={alpha_highlight:.1f}, β={beta_highlight:.1f})", zorder=10)
        ax2.scatter(chosen_point_data_for_highlight['hs_norm'], chosen_point_data_for_highlight['md_norm'], 
                    c=highlight_color, s=highlight_size, marker=highlight_marker, edgecolor=highlight_edgecolor, zorder=10)
        ax3.scatter(chosen_point_data_for_highlight['ps_norm'], chosen_point_data_for_highlight['md_norm'], 
                    c=highlight_color, s=highlight_size, marker=highlight_marker, edgecolor=highlight_edgecolor, zorder=10)
        
        legend_handles.append(plt.Line2D([0], [0], marker=highlight_marker, color='w', 
                                          label=f"Ref. Point (α={alpha_highlight:.1f}, β={beta_highlight:.1f})",
                                          markerfacecolor=highlight_color, markeredgecolor=highlight_edgecolor, markersize=12))

    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=12)
    fig.subplots_adjust(right=0.89, bottom=0.12, top=0.9)
    cbar_ax = fig.add_axes([0.91, 0.12, 0.015, 0.78]) 
    cbar = fig.colorbar(sc1, cax=cbar_ax) 
    cbar.set_label('Z_norm_calculated (Overall Objective)', fontsize=13)

    print(f"Plotting {len(non_dominated_df_unique_metrics)} unique non-dominated solutions (in metric space) for n={n_teams}.")
    if chosen_point_data_for_highlight is not None:
        is_highlighted_in_unique_non_dominated = non_dominated_df_unique_metrics.apply(lambda row: 
            np.isclose(row['hs_norm'], chosen_point_data_for_highlight['hs_norm']) and \
            np.isclose(row['ps_norm'], chosen_point_data_for_highlight['ps_norm']) and \
            np.isclose(row['md_norm'], chosen_point_data_for_highlight['md_norm']), axis=1).any()
        if is_highlighted_in_unique_non_dominated:
            print(f"The highlighted reference point's metrics ARE present in the unique non-dominated set.")
        else:
            # This could happen if the highlighted point itself was non-dominated but another (alpha,beta) yielded the same metrics and was kept by 'keep=first'
            print(f"The highlighted reference point's metrics ARE NOT strictly identical to any single entry kept in the unique non-dominated set (could be due to float precision or drop_duplicates 'keep' behavior if multiple alpha/beta yield same metrics).")


    try:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"Pareto frontier 2D subplots (unique non-dominated) saved to {output_image_path}")
    except Exception as e:
        print(f"Error saving 2D subplots: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot unique non-dominated Pareto frontier using 2D subplots from calibration CSV.")
    parser.add_argument("--input_csv", type=str, required=True, 
                        help="Path to the input CSV file from calibration.")
    parser.add_argument("--output_image", type=str, required=True, 
                        help="Path to save the output PNG image.")
    parser.add_argument("--n_teams", type=int, required=True,
                        help="Number of teams (e.g., 6 for n=6).")
    parser.add_argument("--alpha_highlight", type=float, default=None,
                        help="Alpha value of the specific point to highlight.")
    parser.add_argument("--beta_highlight", type=float, default=None,
                        help="Beta value of the specific point to highlight.")
    
    args = parser.parse_args()
    plot_pareto_frontier_2d_subplots(args.input_csv, args.output_image, args.n_teams, args.alpha_highlight, args.beta_highlight)
