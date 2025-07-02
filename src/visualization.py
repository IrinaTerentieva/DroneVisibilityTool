import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def plot_visibility_results(results: List[Dict],
                            show_staging_ids: Optional[List[int]] = None,
                            color_by_cluster: bool = True,
                            save_path: Optional[str] = None):
    """
    Create visualization of visibility analysis results.

    Args:
        results: List of analysis results
        show_staging_ids: Optional list of IDs to display
        color_by_cluster: Whether to color by cluster
        save_path: Optional path to save figure
    """
    if not results:
        logger.warning("No results to plot")
        return

    # Filter results if needed
    if show_staging_ids:
        filtered_results = [r for r in results if r['staging_id'] in show_staging_ids]
    else:
        filtered_results = results

    if not filtered_results:
        logger.warning("No results match the filter criteria")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))

    if color_by_cluster:
        # Get unique clusters
        clusters = [r['cluster'] for r in filtered_results if r['cluster'] is not None]
        unique_clusters = sorted(list(set(clusters))) if clusters else []

        if unique_clusters:
            # Create color map
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

            # Plot by cluster
            for result in filtered_results:
                staging_id = result['staging_id']
                staging_x, staging_y = result['staging_coords']
                polygon = result['visibility_polygon']
                cluster = result['cluster']

                if polygon and cluster in cluster_colors:
                    x, y = polygon.exterior.xy
                    color = cluster_colors[cluster]

                    # Only label first occurrence of each cluster
                    label = f'Cluster {cluster}' if cluster not in [r['cluster'] for r in filtered_results[
                                                                                          :filtered_results.index(
                                                                                              result)]] else ""

                    # Fill polygon
                    ax.fill(x, y, alpha=0.3, color=color, label=label)
                    # Outline
                    ax.plot(x, y, color=color, linewidth=2)

                    # Plot staging point
                    ax.plot(staging_x, staging_y, 'o', color=color, markersize=8,
                            markeredgecolor='black', markeredgewidth=1)

                    # Add label
                    ax.annotate(f'ID {staging_id}\nC{cluster}', (staging_x, staging_y),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            color_by_cluster = False

    if not color_by_cluster:
        # Color by staging ID
        colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_results)))

        for i, result in enumerate(filtered_results):
            staging_id = result['staging_id']
            staging_x, staging_y = result['staging_coords']
            polygon = result['visibility_polygon']
            cluster = result['cluster']

            if polygon:
                x, y = polygon.exterior.xy
                color = colors[i]

                # Fill polygon
                label = f'Staging {staging_id}'
                if cluster:
                    label += f' (C{cluster})'
                ax.fill(x, y, alpha=0.3, color=color, label=label)
                # Outline
                ax.plot(x, y, color=color, linewidth=2)

                # Plot staging point
                ax.plot(staging_x, staging_y, 'o', color=color, markersize=8,
                        markeredgecolor='black', markeredgewidth=1)

                # Add label
                ax.annotate(f'ID {staging_id}', (staging_x, staging_y),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Formatting
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

    # title = f'Drone Visibility Analysis - {filtered_results[0]["elevation_angle"]}° Elevation'
    # if color_by_cluster and unique_clusters:
    #     title += ' (Colored by Cluster)'
    # ax.set_title(title)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {save_path}")
    else:
        plt.show()

    logger.info(f"Displayed {len(filtered_results)} visibility zones")


def create_summary_plot(summary: Dict, save_path: Optional[str] = None):
    """
    Create summary visualization of analysis results.

    Args:
        summary: Summary statistics dictionary
        save_path: Optional path to save figure
    """
    if not summary or 'cluster_statistics' not in summary:
        logger.warning("No summary data to plot")
        return

    cluster_stats = summary['cluster_statistics']

    # Replace None with 'Unassigned' for cluster labels
    clusters = [str(c) if c is not None else 'Unassigned' for c in cluster_stats.keys()]
    total_areas = [stats['total_area'] for stats in cluster_stats.values()]
    avg_areas = [stats['average_area'] for stats in cluster_stats.values()]
    counts = [stats['count'] for stats in cluster_stats.values()]

    x = np.arange(len(clusters))
    width = 0.35

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Area by cluster
    ax1.bar(x - width / 2, total_areas, width, label='Total Area', alpha=0.8)
    ax1.bar(x + width / 2, avg_areas, width, label='Average Area', alpha=0.8)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Area (hectares)')
    ax1.set_title('Visibility Area by Cluster')
    ax1.set_xticks(x)
    ax1.set_xticklabels(clusters, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Point count by cluster
    ax2.bar(x, counts, alpha=0.8, color='green')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Staging Points')
    ax2.set_title('Staging Points per Cluster')
    ax2.set_xticks(x)
    ax2.set_xticklabels(clusters, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, count in enumerate(counts):
        ax2.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved summary plot to: {save_path}")
    else:
        plt.show()
