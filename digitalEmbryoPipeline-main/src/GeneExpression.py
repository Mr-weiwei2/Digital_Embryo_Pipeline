import numpy as np
import pandas as pd
from collections import defaultdict
import scanpy as sc
from PIL import Image
from cellpose.io import imread,imsave
import stereo as st
import warnings
from scipy.ndimage import center_of_mass
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')
import anndata as ad
from anndata import AnnData
import logging
import os
# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# 设置logging配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import cv2
import tifffile as tiff
from cellpose import models

def evaluate_segmentation(raw_image, current_segmentation, spot2gene, genes):
    """
    评估当前分割结果，并计算与Cellpose分割结果的基因表达相关性。

    参数：
    raw_image: ndarray - 原始图像
    current_segmentation: dict - 当前分割结果，每个细胞包含像素点坐标
    spot2gene: dict - 每个像素点对应基因表达的映射
    genes: list - 基因列表

    返回：
    corr_current: list - 当前分割与基因表达的Pearson相关性
    corr_cellpose: list - Cellpose分割与基因表达的Pearson相关性
    stats_current: dict - 当前分割相关性统计指标
    stats_cellpose: dict - Cellpose分割相关性统计指标
    """
    # 使用Cellpose进行分割
    model = models.Cellpose(model_type='cyto')
    masks, flows, styles, diams = model.eval(raw_image, diameter=None, channels=[0, 0])

    # 构建Cellpose分割结果
    cellpose_segmentation = {}
    for i in range(1, np.max(masks) + 1):
        coords = np.argwhere(masks == i)
        cellpose_segmentation[str(i)] = [f"{x[0]}:{x[1]}" for x in coords]

    # 计算相关性
    corr_current = []
    corr_cellpose = []
    
    def calc_expression(coords):
        """计算给定坐标的基因表达量"""
        exp = np.zeros(len(genes))
        for coord in coords:
            if coord in spot2gene:
                for gene, count in spot2gene[coord]:
                    exp[genes.index(gene)] += count
        return exp

    def find_best_match(current_coords, cellpose_segmentation):
        """找到最匹配的Cellpose细胞"""
        best_match = None
        max_intersection = 0
        for cp_id, cp_coords in cellpose_segmentation.items():
            cp_set = set(cp_coords)
            intersection_size = len(current_coords.intersection(cp_set))
            if intersection_size > max_intersection:
                max_intersection = intersection_size
                best_match = cp_id
        return best_match

    for cell_id, current_coords in current_segmentation.items():
        current_coords = set(current_coords)
        best_match = find_best_match(current_coords, cellpose_segmentation)
        if not best_match:
            continue
        cellpose_coords = set(cellpose_segmentation[best_match])
        intersection = current_coords.intersection(cellpose_coords)
        diff_current = current_coords.difference(intersection)
        diff_cellpose = cellpose_coords.difference(intersection)

        # 计算基因表达量
        exp_int = calc_expression(intersection)
        exp_curr = calc_expression(diff_current)
        exp_cellpose = calc_expression(diff_cellpose)

        # 计算Pearson相关性
        if np.sum(exp_int) >= 50 and np.sum(exp_curr) >= 50 and np.sum(exp_cellpose) >= 50:
            r_curr, _ = pearsonr(exp_int, exp_curr)
            r_cellpose, _ = pearsonr(exp_int, exp_cellpose)
            corr_current.append(r_curr if not np.isnan(r_curr) else 0)
            corr_cellpose.append(r_cellpose if not np.isnan(r_cellpose) else 0)

    # 计算统计指标
    def calc_stats(corr_data):
        return {
            'mean': np.mean(corr_data),
            'std': np.std(corr_data),
            'min': np.min(corr_data),
            'max': np.max(corr_data)
        }

    stats_current = calc_stats(corr_current)
    stats_cellpose = calc_stats(corr_cellpose)

    return corr_current, corr_cellpose, stats_current, stats_cellpose


def plot_images(raw_image, current_mask, cellpose_mask):
    """
    绘制原始图像、当前分割和Cellpose分割的三张图像。

    参数：
    raw_image: ndarray - 原始图像
    current_mask: ndarray - 当前分割结果
    cellpose_mask: ndarray - Cellpose分割结果

    返回：
    None
    """
    # 创建subplot显示三个图像
    fig = sp.make_subplots(rows=1, cols=3, subplot_titles=("Raw Image", "Current Segmentation", "Cellpose Segmentation"))

    # 添加原始图像
    fig.add_trace(go.Image(z=raw_image, name="Raw Image"), row=1, col=1)

    # 添加当前分割结果
    fig.add_trace(go.Image(z=current_mask, name="Current Segmentation", colorscale='Jet'), row=1, col=2)

    # 添加Cellpose分割结果
    fig.add_trace(go.Image(z=cellpose_mask, name="Cellpose Segmentation", colorscale='Nipy_spectral'), row=1, col=3)

    # 更新布局
    fig.update_layout(
        title="Segmentation Comparison",
        showlegend=False
    )

    fig.show()


def plot_correlation_boxplot(corr_current, corr_cellpose):
    """
    绘制分割结果的Pearson相关性箱线图。

    参数：
    corr_current: list - 当前分割与基因表达的Pearson相关性
    corr_cellpose: list - Cellpose分割与基因表达的Pearson相关性

    返回：
    None
    """
    # 使用plotly绘制箱线图
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=corr_current,
        name="Current Segmentation",
        boxmean="sd",  # 显示均值和标准差
        marker=dict(color='blue')
    ))

    fig.add_trace(go.Box(
        y=corr_cellpose,
        name="Cellpose Segmentation",
        boxmean="sd",  # 显示均值和标准差
        marker=dict(color='red')
    ))

    fig.update_layout(
        title="Segmentation Comparison",
        yaxis_title="Pearson Correlation",
        showlegend=True
    )

    fig.show()

def average_gene_occurrence_in_spots(spot2gene, target_gene):
    """
    统计包含目标基因的所有像素点中，该基因的平均出现次数。
    
    参数:
    - spot2gene (dict): 字典，键是像素点坐标（如 '1597:1540'），值是该像素点的基因信息。
    - target_gene (str): 要统计的基因名称。
    
    返回:
    - average_count (float): 包含目标基因的像素点中，该基因的平均出现次数。
    """
    total_count = 0  # 所有包含目标基因的像素点的基因总计数
    num_pixels_with_gene = 0  # 包含目标基因的像素点的数量
    
    # 遍历所有像素点，查找目标基因
    for spot, genes_info in spot2gene.items():
        # 获取目标基因在该像素点的计数
        count = sum(gene[1] for gene in genes_info if gene[0] == target_gene)
        
        # 如果目标基因存在，则累加计数
        if count > 0:
            total_count += count
            num_pixels_with_gene += 1
    
    # 计算平均值
    average_count = total_count / num_pixels_with_gene if num_pixels_with_gene > 0 else 0
    
    return average_count
# def evaluate_segmentation(raw_image, current_segmentation, gene_expression, genes, spot2gene):
#     """
#     评估当前分割结果，并与基于Cellpose算法的分割结果进行对比。

#     参数：
#     raw_image: ndarray - 原始图像
#     current_segmentation: dict - 当前分割结果，每个细胞包含像素点坐标
#     gene_expression: dict - 每个像素点表达的基因及其表达量
#     genes: list - 基因列表
#     spot2gene: dict - 每个像素点对应基因表达的映射

#     返回：
#     None
#     """
#     # 检查 current_segmentation 类型
#     if isinstance(current_segmentation, list):
#         current_segmentation = {str(i): current_segmentation[i] for i in range(len(current_segmentation))}
#     elif not isinstance(current_segmentation, dict):
#         raise TypeError("current_segmentation 必须是字典或列表")

#     # 使用Cellpose进行分割
#     model = models.Cellpose(model_type='cyto')
#     masks, flows, styles, diams = model.eval(raw_image, diameter=None, channels=[0, 0])

#     # 绘制原图
#     fig = go.Figure()

#     fig.add_trace(go.Image(z=raw_image, name="Original Image", colormodel='rgb'))

#     # 绘制当前分割结果
#     gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY) if len(raw_image.shape) == 3 else raw_image
#     current_mask = np.zeros_like(gray)
#     for cell_id, coords in current_segmentation.items():
#         for coord in coords:
#             x, y = map(int, coord.split(':'))
#             current_mask[x, y] = 255
#     fig.add_trace(go.Image(z=current_mask, name="Current Segmentation", colormodel='rgb'))

#     # 绘制Cellpose分割结果
#     fig.add_trace(go.Image(z=masks, name="Cellpose Segmentation", colormodel='rgb'))

#     fig.update_layout(title="Segmentation Comparison", showlegend=True)
#     fig.show()

#     # 构建Cellpose分割结果
#     cellpose_segmentation = {}
#     for i in range(1, np.max(masks) + 1):
#         coords = np.argwhere(masks == i)
#         cellpose_segmentation[str(i)] = [f"{x[0]}:{x[1]}" for x in coords]

#     # 确保细胞匹配，通过像素点交集找到最匹配的细胞
#     def find_best_match(current_coords, cellpose_segmentation):
#         best_match = None
#         max_intersection = 0
#         for cp_id, cp_coords in cellpose_segmentation.items():
#             cp_set = set(cp_coords)
#             intersection_size = len(current_coords.intersection(cp_set))
#             if intersection_size > max_intersection:
#                 max_intersection = intersection_size
#                 best_match = cp_id
#         return best_match

#     # 计算相关性
#     corr_current = []
#     corr_cellpose = []
#     for cell_id in current_segmentation:
#         current_coords = set(current_segmentation[cell_id])
#         best_match = find_best_match(current_coords, cellpose_segmentation)
#         if not best_match:
#             continue
#         cellpose_coords = set(cellpose_segmentation[best_match])
#         intersection = current_coords.intersection(cellpose_coords)
#         diff_current = current_coords.difference(intersection)
#         diff_cellpose = cellpose_coords.difference(intersection)

#         # 计算基因表达量
#         def calc_expression(coords):
#             exp = np.zeros(len(genes))
#             for coord in coords:
#                 if coord in spot2gene:
#                     for gene, count in spot2gene[coord]:
#                         exp[genes.index(gene)] += count
#             return exp

#         exp_int = calc_expression(intersection)
#         exp_curr = calc_expression(diff_current)
#         exp_cellpose = calc_expression(diff_cellpose)

#         # 计算Pearson相关性，降低过滤条件
#         if np.sum(exp_int) >= 50 and np.sum(exp_curr) >= 50 and np.sum(exp_cellpose) >= 50:
#             r_curr, _ = pearsonr(exp_int, exp_curr)
#             r_cellpose, _ = pearsonr(exp_int, exp_cellpose)
#             corr_current.append(r_curr if not np.isnan(r_curr) else 0)
#             corr_cellpose.append(r_cellpose if not np.isnan(r_cellpose) else 0)

#     # 绘制箱线图比较相关性
#     data = [corr_current, corr_cellpose]
#     plt.boxplot(data, labels=['Current', 'Cellpose'])
#     plt.title('Segmentation Comparison')
#     plt.ylabel('Pearson Correlation')
#     plt.show()

#     # 输出相关性统计
#     print(f'Mean Correlation (Current): {np.mean(corr_current):.4f}')
#     print(f'Mean Correlation (Cellpose): {np.mean(corr_cellpose):.4f}')
#     print(f'Median Correlation (Current): {np.median(corr_current):.4f}')
#     print(f'Median Correlation (Cellpose): {np.median(corr_cellpose):.4f}')

#     return data
def count_genes_per_pixel(data, fig=True):
    """
    Count the number of genes expressed per pixel and optionally plot the distribution.

    Parameters:
        data (dict): Keys are coordinates ("x:y"), values are lists of [geneID, MIDCounts].
        fig (bool): Whether to render the plot (True) or return the data (False).

    Returns:
        dict or None: If fig=False, returns a dictionary with coordinates and gene counts.
    """
    pixel_gene_counts = defaultdict(int)
    
    for coord, gene_list in data.items():
        pixel_gene_counts[coord] = len(gene_list)
    
    # If fig=False, return the raw data
    if not fig:
        return pixel_gene_counts
    
    # Prepare data for plotting
    gene_count_values = list(pixel_gene_counts.values())
    fig = px.histogram(
        x=gene_count_values,
        nbins=20,
        labels={'x': 'Number of Genes per Pixel', 'y': 'Frequency'},
        title='Distribution of Gene Counts per Pixel',
        color_discrete_sequence=['skyblue']
    )
    fig.show()
    return pixel_gene_counts
def plot_pixel_count_distribution(cell2spot, fig=True):
    """
    Plot or return the distribution of the number of pixels per cell.

    Parameters:
        cell2spot (dict): Maps cell IDs to lists of pixel coordinates.
        fig (bool): Whether to render the plot (True) or return the data (False).

    Returns:
        list or None: List of pixel counts per cell if fig=False.
    """
    cell_pixel_counts = [len(spot_list) for spot_list in cell2spot.values()]
    
    if not fig:
        return cell_pixel_counts
    
    fig = px.histogram(
        x=cell_pixel_counts,
        nbins=20,
        labels={'x': 'Number of Pixels per Cell', 'y': 'Frequency'},
        title='Distribution of Pixel Counts per Cell',
        color_discrete_sequence=['skyblue']
    )
    fig.show()
    return cell_pixel_counts

def plot_valid_pixel_count_distribution(cell2spot, spot2gene, fig=True):
    """
    Plot or return the distribution of valid pixels per cell.

    Parameters:
        cell2spot (dict): Maps cell IDs to lists of pixel coordinates.
        spot2gene (dict): Maps pixel coordinates to gene information.
        fig (bool): Whether to render the plot (True) or return the data (False).

    Returns:
        list or None: List of valid pixel counts per cell if fig=False.
    """
    valid_pixel_counts = []
    for spot_list in cell2spot.values():
        valid_pixels = [spot for spot in spot_list if spot in spot2gene]
        valid_pixel_counts.append(len(valid_pixels))
    
    if not fig:
        return valid_pixel_counts
    
    fig = px.histogram(
        x=valid_pixel_counts,
        nbins=20,
        labels={'x': 'Valid Pixels per Cell', 'y': 'Frequency'},
        title='Distribution of Valid Pixels per Cell',
        color_discrete_sequence=['skyblue']
    )
    fig.show()
    return valid_pixel_counts
def plot_valid_pixel_ratio_distribution(cell2spot, spot2gene, fig=True):
    """
    Plot or return the distribution of valid pixel ratios per cell.

    Parameters:
        cell2spot (dict): Maps cell IDs to lists of pixel coordinates.
        spot2gene (dict): Maps pixel coordinates to gene information.
        fig (bool): Whether to render the plot (True) or return the data (False).

    Returns:
        list or None: List of valid pixel ratios per cell if fig=False.
    """
    valid_pixel_ratios = []
    for spot_list in cell2spot.values():
        total_pixels = len(spot_list)
        valid_pixels = [spot for spot in spot_list if spot in spot2gene]
        ratio = len(valid_pixels) / total_pixels if total_pixels > 0 else 0
        valid_pixel_ratios.append(ratio)
    
    if not fig:
        return valid_pixel_ratios
    
    fig = px.histogram(
        x=valid_pixel_ratios,
        nbins=20,
        labels={'x': 'Valid Pixel Ratio per Cell', 'y': 'Frequency'},
        title='Distribution of Valid Pixel Ratios per Cell',
        color_discrete_sequence=['skyblue']
    )
    fig.show()
    return valid_pixel_ratios

def plot_gene_occurrence_histogram(spot2gene, target_gene, fig=True):
    """
    Plot or return the distribution of occurrences of a specific gene across pixels.

    Parameters:
        spot2gene (dict): Maps pixel coordinates to gene information.
        target_gene (str): The target gene to analyze.
        fig (bool): Whether to render the plot (True) or return the data (False).

    Returns:
        list or None: List of occurrences of the target gene if fig=False.
    """
    gene_counts = []
    for genes_info in spot2gene.values():
        count = sum(gene[1] for gene in genes_info if gene[0] == target_gene)
        if count > 0:
            gene_counts.append(count)
    
    if not fig:
        return gene_counts
    
    fig = px.histogram(
        x=gene_counts,
        nbins=20,
        labels={'x': f'{target_gene} Occurrences per Pixel', 'y': 'Frequency'},
        title=f'Histogram of {target_gene} Occurrence Across Pixels',
        color_discrete_sequence=['skyblue']
    )
    fig.show()
    return gene_counts

def generate_and_visualize_heatmaps(genes, spot2gene, height, width, fig=True):
    """
    Generate heatmaps for gene expression data, including total gene counts and gene type counts.
    Optionally visualize the data using Plotly or return the processed data.

    Parameters:
    - genes (list): List of genes.
    - spot2gene (dict): Dictionary mapping each pixel to its genes and counts.
    - height (int): Height of the image.
    - width (int): Width of the image.
    - fig (bool): If True, display the heatmap; otherwise, return the data for further processing.

    Returns:
    - dict: If fig=False, returns a dictionary containing the data:
        - 'gene_count_matrix': Matrix of total gene counts for each pixel.
        - 'gene_type_matrix': Matrix of gene type counts for each pixel.
    """
    # Create empty matrices for gene count and gene type
    gene_count_matrix = np.zeros((height, width))  # Total gene count per pixel
    gene_type_matrix = np.zeros((height, width))  # Gene type count per pixel

    # Process each spot in the input data
    for spot, genes_info in spot2gene.items():
        x, y = map(int, spot.split(":"))  # Parse coordinates
        if x < width and y < height:  # Ensure coordinates are within bounds
            total_count = sum(gene[1] for gene in genes_info)  # Total gene count
            gene_count_matrix[x, y] = total_count
            gene_type_matrix[x, y] = len(genes_info)  # Number of gene types

    # Return the data if visualization is not required
    if not fig:
        return {
            'gene_count_matrix': gene_count_matrix,
            'gene_type_matrix': gene_type_matrix
        }

    # Visualization using Plotly
    # Create subplots with shared axes
    fig = sp.make_subplots(
        rows=1, cols=2,  # 1 row, 2 columns
        subplot_titles=("Gene Total Count Heatmap", "Gene Type Count Heatmap"),
        horizontal_spacing=0.1  # Adjust spacing between plots
    )
    
    # Plot for Gene Total Count
    heatmap1 = go.Heatmap(
        z=gene_count_matrix,
        colorscale='YlGnBu',  # Color scheme for total count
        colorbar=dict(title="Total Gene Count")  # Color bar label
    )
    fig.add_trace(heatmap1, row=1, col=1)  # Add to first subplot
    
    # Plot for Gene Type Count
    heatmap2 = go.Heatmap(
        z=gene_type_matrix,
        colorscale='YlOrRd',  # Color scheme for type count
        colorbar=dict(title="Gene Type Count")  # Color bar label
    )
    fig.add_trace(heatmap2, row=1, col=2)  # Add to second subplot

    # Update layout
    fig.update_layout(
        height=600, width=1200,  # Set plot size
        title_text="Gene Expression Heatmaps",  # Overall title
        showlegend=False
    )
    
    # Display the plot
    fig.show()

    # Return the data in case it is needed later
    return {
        'gene_count_matrix': gene_count_matrix,
        'gene_type_matrix': gene_type_matrix
    }
def plot_sorted_gene_expression_bar_chart(sorted_genes, fig=True):
    """
    Plot an interactive bar chart of gene expression levels (MIDCounts) sorted in descending order.
    Optionally, return the data for plotting without rendering the chart.
    
    Parameters:
        sorted_genes (list): A list of tuples with gene IDs and expression counts, 
                             e.g., [('GeneID', expression_count), ...]
        fig (bool): Whether to render the plot (True) or just return the data (False). Default is True.
    
    Returns:
        dict: A dictionary containing 'genes' and 'expression_counts' for further processing 
              if fig=False, otherwise None.
    """
    # Sort genes by expression count in descending order
    sorted_genes = sorted(sorted_genes, key=lambda x: x[1], reverse=True)
    
    # Extract gene names and expression counts
    genes = [gene for gene, _ in sorted_genes]
    expression_counts = [count for _, count in sorted_genes]
    
    # If fig=False, return the data for external processing
    if not fig:
        return {'genes': genes, 'expression_counts': expression_counts}
    
    # Create an interactive Plotly bar chart
    fig = px.bar(
        x=genes, 
        y=expression_counts, 
        labels={'x': 'Gene ID', 'y': 'Gene Expression (MIDCounts)'},  # Axis labels
        title='Gene Expression (MIDCounts)',  # Title
        color=expression_counts,  # Color based on expression counts
        color_continuous_scale='Blues'  # Color scale
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_tickangle=-45,  # Rotate x-axis labels
        xaxis_title='Gene ID',
        yaxis_title='Gene Expression (MIDCounts)',
        template='plotly',  # Use Plotly's default theme
        height=600, 
        width=1000
    )
    
    # Show the interactive chart
    fig.show()

    # Return the data in case it's needed later
    return {'genes': genes, 'expression_counts': expression_counts}
def show_three_genes(genes, data, gene_params):
    """
    Function to process gene expression data and assign RGB colors to coordinates.

    Parameters:
        genes (list): List of all unique gene IDs.
        data (dict): Dictionary where keys are coordinates (x:y) and values are lists of [geneID, MIDCounts].
        gene_params (dict): Contains the following keys:
            - genes: List of three genes to visualize.
            - low_th: List of lower thresholds for each gene.
            - high_th: List of upper thresholds for each gene.
            - main_bi_color: String indicating the main color channel ("Red", "Green", "Blue").

    Returns:
        dict: A dictionary mapping each coordinate to an RGB color tuple.
    """
    # Extract parameters
    selected_genes = gene_params['genes']
    low_th = gene_params['low_th']
    high_th = gene_params['high_th']
    main_bi_color = gene_params['main_bi_color']

    # Check if selected genes exist in the genes list
    for gene in selected_genes:
        if gene not in genes:
            return {"error": f"'{gene}' not found in genes list"}

    # Extract gene expression data for selected genes from data
    coord_expression = {}
    for coord, gene_list in data.items():
        expression = {}
        for gene_id, count in gene_list:
            if gene_id in selected_genes:
                expression[gene_id] = count
        coord_expression[coord] = expression

    # Prepare expression arrays
    gene1_values = []
    gene2_values = []
    gene3_values = []
    coords = []

    for coord, expression in coord_expression.items():
        if all(gene in expression for gene in selected_genes):
            gene1_values.append(expression[selected_genes[0]])
            gene2_values.append(expression[selected_genes[1]])
            gene3_values.append(expression[selected_genes[2]])
            coords.append(coord)

    if not gene1_values or not gene2_values or not gene3_values:
        return {"error": "No valid data points found for the selected genes."}

    gene1_values = np.array(gene1_values)
    gene2_values = np.array(gene2_values)
    gene3_values = np.array(gene3_values)

    # Compute thresholds
    min_vals = [
        np.percentile(gene1_values, low_th[0]),
        np.percentile(gene2_values, low_th[1]),
        np.percentile(gene3_values, low_th[2])
    ]
    max_vals = [
        np.percentile(gene1_values, high_th[0]),
        np.percentile(gene2_values, high_th[1]),
        np.percentile(gene3_values, high_th[2])
    ]

    # Normalize and threshold gene expressions
    norm = lambda values, min_val, max_val: (values - min_val) / (max_val - min_val)
    norm_gene1 = norm(gene1_values, min_vals[0], max_vals[0])
    norm_gene2 = norm(gene2_values, min_vals[1], max_vals[1])
    norm_gene3 = norm(gene3_values, min_vals[2], max_vals[2])

    norm_gene1 = np.clip(norm_gene1, 0, 1)
    norm_gene2 = np.clip(norm_gene2, 0, 1)
    norm_gene3 = np.clip(norm_gene3, 0, 1)

    # Build the RGB array
    final_C = np.zeros((len(coords), 3))

    # Determine the channel to which each gene is assigned based on main_bi_color
    on_channel = (np.array(["Red", "Green", "Blue"]) != main_bi_color).astype(int)
    on_channel[2] = 2  # Ensure all three channels are mapped

    # Assign normalized values to the respective color channels
    final_C[:, on_channel[0]] = norm_gene1
    final_C[:, on_channel[1]] = norm_gene2
    final_C[:, on_channel[2]] = norm_gene3

    # Map colors back to coordinates
    coord_colors = {coord: final_C[idx] for idx, coord in enumerate(coords)}

    return coord_colors
def visualize_gene_expression(color_mapping, image_shape=(100, 100), background_color=(1, 1, 1)):
    """
    Visualize gene expression by updating only specific pixels in the image based on coordinates and colors.

    Parameters:
        color_mapping (dict): Dictionary where keys are coordinates (x:y) and values are RGB color tuples.
        image_shape (tuple): Shape of the image (height, width). Default is (100, 100).
        background_color (tuple): RGB tuple for the background color. Default is white (1, 1, 1).

    Displays:
        A modified image with only selected pixels colored.
    """
    # Create an empty image (background color)
    img = np.full((image_shape[0], image_shape[1], 3), background_color, dtype=np.float32)
    
    # Iterate over the coordinates in the color mapping and update the image
    for coord, color in color_mapping.items():
        # Parse the coordinate (assuming it's in "x:y" format)
        x, y = map(int, coord.split(':'))

        # Check if the coordinates are within bounds
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            # Update the pixel color at (y, x) (note that image is [y, x] in shape)
            # img[y, x] = color
            img[x,y] = color

    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')  # Hide the axis
    plt.show()

def overlay_gene_expression_on_mask(cells, cell_data, gene_data, gene_params, img_width, img_height):
    """
    将细胞掩膜图像和基因表达图像叠加，并确保图像尺寸足够大。

    参数:
        cells (list): 细胞的ID列表。
        cell_data (dict): 每个细胞对应的坐标数据，字典格式，键为细胞ID，值为坐标列表。
        gene_data (dict): 基因表达数据，格式为：坐标 -> [(基因ID, MIDCounts), ...]。
        gene_params (dict): 基因表达的参数（如基因选择、阈值、主颜色通道等）。
        img_width (int): 图像的宽度。
        img_height (int): 图像的高度。

    返回:
        None: 展示叠加图像。
    """
    mask_image = convert_spot_to_image(cells, cell_data)

    coord_colors = show_three_genes(gene_data.keys(), gene_data, gene_params)

    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)  # 创建指定大小的空图像

    for coord, color in coord_colors.items():
        try:
            x, y = map(int, coord.split(':'))
            if x < 0 or y < 0:  # 坐标不能为负数
                print(f"Skipping coordinate with negative value: {coord}")
                continue
        except ValueError:
            print(f"Skipping invalid coordinate format: {coord}")
            continue

        # 确保坐标在图像范围内
        if 0 <= x < img_width and 0 <= y < img_height:
            print(f"Applying color {color} at coordinate ({x}, {y})")
            img[y, x] = color  # 将基因表达的RGB颜色覆盖到掩膜图像上
        else:
            print(f"Skipping coordinate out of image bounds: ({x}, {y})")

    # 7. 显示最终图像
    plt.imshow(img)
    plt.axis('off')
    plt.title('Overlayed Cell Mask and Gene Expression')
    plt.show()

def get_image_dimensions_from_file(file_path):
    """
    从文本文件中提取图像的宽度和高度
    假设文本文件格式是每行一个坐标 (x:y) 和对应的值。
    """
    max_x = 0
    max_y = 0
    
    with open(file_path, 'r') as f:
        for line in f.readlines():
            # 解析每行数据
            spt = line.strip().split('\t')  # 使用制表符分割
            coords = spt[0].split(':')  # x:y 的坐标部分
            
            x = int(coords[0])
            y = int(coords[1])
            
            # 更新最大坐标值
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    
    # 图像的宽度和高度分别是最大x坐标和最大y坐标加1
    width = max_x + 1
    height = max_y + 1
    
    return width, height

def get_spot_gene(file_path):
    """
        解析基因坐标文件
    """
    print('---parsing gene coordinates---')
    data = {}
    genes = set()
    df = pd.read_csv(file_path, sep='\t')
    for x, y, geneID, MIDCounts in df[['x', 'y', 'geneID', 'MIDCounts']].values:
        key = str(x)+":"+str(y)
        genes.add(geneID)
        if key in data:
            data[key].append([geneID, int(MIDCounts)])
        else:
            data[key] = [[geneID, int(MIDCounts)]]
    return list(genes), data


def convert_tif_to_cell_spot(tif_file_path):
    """
    将TIF格式的mask文件转换为get_cell_spot可以使用的格式
    """
    print('---Converting TIF mask to cell spot format---')
    
    # 打开TIF文件并读取数据
    img = Image.open(tif_file_path)
    img_data = np.array(img)  # 转换为NumPy数组
    
    # 获取文件名并生成 slice_marker（去掉扩展名）
    file_name = os.path.basename(tif_file_path)
    slice_marker = file_name[:file_name.find('.tif')]
    
    # 创建一个字典来保存细胞信息
    data = {}
    cells = set()  # 用来存储所有细胞ID
    
    # 遍历图像中的每个像素
    for y in range(img_data.shape[0]):  # 遍历图像的每一行（y坐标）
        for x in range(img_data.shape[1]):  # 遍历图像的每一列（x坐标）
            cell_id = img_data[y, x]  # 获取该像素点的细胞ID（可能是一个整数）
            
            if cell_id == 0:  
                continue
            
            # 生成细胞ID字符串，格式为 'slice_marker_cellID'
            cell = f"{slice_marker}_{int(cell_id)}"
            cells.add(cell)  # 将细胞ID加入集合
            
            # 将(x, y)坐标转换为字符串并添加到字典中
            if cell not in data:
                data[cell] = []
            data[cell].append(f"{x},{y}")
    
    # 返回细胞ID列表和数据字典
    return list(cells), data
def convert_spot_to_image(cells, data):
    """
    根据从get_cell_spot获得的细胞数据生成掩膜图像并展示
    """
    print('---Converting cell spots to image---')
    
    # 计算最大坐标值
    max_x, max_y = 0, 0
    for cell_id in cells:
        for coord in data[cell_id]:
            x, y = map(int, coord.split(':'))  # 解析坐标
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    # 初始化一个空的图像矩阵
    mask_image = np.zeros((max_y + 1, max_x + 1), dtype=np.int32)  # 图像大小：最大坐标 + 1

    # 为每个细胞分配一个唯一的ID（从1开始递增）
    cell_to_unique_id = {}
    current_id = 1

    # 填充图像
    for cell_id in cells:
        cell_number =int(cell_id.split('_')[-1])  # 获取细胞ID的数字部分

        # 为每个唯一数字部分分配一个从1开始的唯一数字
        if cell_number not in cell_to_unique_id:
            cell_to_unique_id[cell_number] = current_id
            current_id += 1  # 避免数字冲突，递增

        # 获取映射后的ID
        mapped_id = cell_to_unique_id[cell_number]

        # 填充图像
        for coord in data[cell_id]:
            x, y = map(int, coord.split(':'))  # 解析坐标
            mask_image[y, x] = mapped_id  # 使用唯一ID填充图像像素

    # 展示图像

    return mask_image
def get_cell_spot(file_path):
    """
        解析分割结果
    """
    print('---parsing cell mask---')
    file_name = os.path.basename(file_path)
    slice_marker = file_name[:file_name.find('.tif')]
    data = {}
    cells = set()
    with open(file_path, 'r') as f:
        for line in f.readlines():
            spt = line.split()
            cell = slice_marker+"_"+str(int(float(spt[1])))
            cells.add(cell)
            if cell in data:
                data[cell].append(spt[0])
            else:
                data[cell] = [spt[0]]
    return list(cells), data
def filter_spot_gene_with_genes(genes, gene_data, x_min, x_max, y_min, y_max):
    """
    根据指定坐标范围筛选get_spot_gene返回的数据，并同步更新genes
    """
    filtered_gene_data = {}
    filtered_genes = set()
    
    for key, gene_data_list in gene_data.items():
        x, y = map(int, key.split(':'))
        if x_min <= x <= x_max and y_min <= y <= y_max:
            filtered_gene_data[key] = gene_data_list
            for gene in gene_data_list:
                filtered_genes.add(gene[0])  # gene[0] 是 geneID
    
    return list(filtered_genes), filtered_gene_data


def filter_cell_spot_with_cells(cells, cell_data, x_min, x_max, y_min, y_max):
    """
    根据指定坐标范围筛选get_cell_spot返回的数据，并同步更新cells
    """
    filtered_cell_data = {}
    filtered_cells = set()
    
    for cell_id, positions in cell_data.items():
        filtered_positions = [
            pos for pos in positions
            if x_min <= int(pos.split(':')[0]) <= x_max and y_min <= int(pos.split(':')[1]) <= y_max
        ]
        if filtered_positions:
            filtered_cell_data[cell_id] = filtered_positions
            filtered_cells.add(cell_id)
    
    return list(filtered_cells), filtered_cell_data
def get_heatmaps(genes, spot2gene,height,width):
    """
    从 spot2gene 中提取每个像素的基因信息，返回每个像素的基因总计数和基因种类数。
    
    参数:
    - genes (list): 基因的列表
    - spot2gene (dict): 字典，包含每个像素点及其基因和计数信息
    
    返回:
    - coord_colors (tuple): 包含两个矩阵的元组：
        - 第一个矩阵为基因总计数
        - 第二个矩阵为基因种类数
    """
    # 假设 manager.height 和 manager.width 是图像的尺寸

    
    # 创建空矩阵用于存储基因总计数和基因种类数
    gene_count_matrix = np.zeros((height, width))  # 每个像素点的基因总计数
    gene_type_matrix = np.zeros((height, width))  # 每个像素点的基因种类数

    # 遍历每个像素点的基因数据
    for spot, genes_info in spot2gene.items():
        x, y = map(int, spot.split(":"))  # 解析坐标
        if x < width and y < height:  # 确保坐标在图像范围内
            total_count = sum(gene[1] for gene in genes_info)  # 计算该点的基因总计数
            gene_count_matrix[x, y] = total_count
            gene_type_matrix[x, y] = len(genes_info)  # 计算该点的基因种类数

    return gene_count_matrix, gene_type_matrix
# def visualize_heatmaps(coord_colors, shape):
#     """
#     根据 coord_colors（包含基因总计数和基因种类数）生成热图。
    
#     参数:
#     - coord_colors (tuple): 一个元组，包含两个矩阵
#         - 第一个矩阵为基因总计数
#         - 第二个矩阵为基因种类数
#     - shape (tuple): 图像的尺寸 (height, width)
#     - color_map (tuple): 一个 RGB 元组，定义热图的颜色方案，默认是 (1, 1, 1) 作为默认色
#     """
#     gene_count_matrix, gene_type_matrix = coord_colors
    
#     # 设置画布大小
#     fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
#     # 基因总计数热图
#     sns.heatmap(gene_count_matrix, cmap="YlGnBu", ax=axes[0], cbar_kws={'label': 'Total Gene Count'})
#     axes[0].set_title("Gene Total Count Heatmap")
#     axes[0].set_xlabel("X Coordinate")
#     axes[0].set_ylabel("Y Coordinate")
    
#     # 基因种类数热图
#     sns.heatmap(gene_type_matrix, cmap="YlOrRd", ax=axes[1], cbar_kws={'label': 'Gene Type Count'})
#     axes[1].set_title("Gene Type Count Heatmap")
#     axes[1].set_xlabel("X Coordinate")
#     axes[1].set_ylabel("Y Coordinate")

#     plt.tight_layout()
#     plt.show()

def visualize_heatmaps(coord_colors, shape):
    """
    Generate interactive heatmaps using Plotly for gene total counts and gene type counts.

    Parameters:
    - coord_colors (tuple): A tuple containing two matrices
        - The first matrix represents total gene counts.
        - The second matrix represents gene type counts.
    - shape (tuple): The dimensions of the image (height, width).
    """
    # Unpack the input matrices
    gene_count_matrix, gene_type_matrix = coord_colors
    
    # Create subplots with shared axes
    fig = sp.make_subplots(
        rows=1, cols=2,  # 1 row, 2 columns
        subplot_titles=("Gene Total Count Heatmap", "Gene Type Count Heatmap"),
        horizontal_spacing=0.1  # Adjust spacing between plots
    )
    
    # Plot for Gene Total Count
    heatmap1 = go.Heatmap(
        z=gene_count_matrix,
        colorscale='YlGnBu',  # Color scheme for total count
        colorbar=dict(title="Total Gene Count")  # Color bar label
    )
    fig.add_trace(heatmap1, row=1, col=1)  # Add to first subplot
    
    # Plot for Gene Type Count
    heatmap2 = go.Heatmap(
        z=gene_type_matrix,
        colorscale='YlOrRd',  # Color scheme for type count
        colorbar=dict(title="Gene Type Count")  # Color bar label
    )
    fig.add_trace(heatmap2, row=1, col=2)  # Add to second subplot

    # Update layout
    fig.update_layout(
        height=600, width=1200,  # Set plot size
        title_text="Gene Expression Heatmaps",  # Overall title
        showlegend=False
    )
    # Display the plot
    fig.show()
class GeneExpressionManager:
    def __init__(self, mask_path, expression_path):

        self.mask_path = mask_path
        self.expression_path = expression_path
        
        logging.info(f"Loading mask image from {mask_path}")
        # self.mask = imread(mask_path)
        logging.info(f"Loading gene expression data from {expression_path}")
        self.genes, self.spot2gene = get_spot_gene(expression_path)
        file_extension = os.path.splitext(mask_path)[1].lower()  # 获取文件扩展名并转换为小写
        if file_extension == '.tif':
            self.cells, self.cell2spot = convert_tif_to_cell_spot(mask_path)
        elif file_extension == '.txt':
            self.cells, self.cell2spot = get_cell_spot(mask_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        self.height, self.width=get_image_dimensions_from_file(mask_path)
        # self.height, self.width = self.mask.shape
        logging.info(f"Mask image loaded with dimensions: height={self.height}, width={self.width}")


    def get_region(self, x_start, y_start, x_end, y_end):
        """获取指定区域的mask和表达数据"""
        logging.info(f"Fetching region: x_start={x_start}, y_start={y_start}, x_end={x_end}, y_end={y_end}")
        
        # 确保范围在图像内
        x_start = max(x_start, 0)
        y_start = max(y_start, 0)
        x_end = min(x_end, self.width)
        y_end = min(y_end, self.height)
        
        # 获取当前区域的mask
        # mask_region = self.mask[y_start:y_end, x_start:x_end]
        
        filtered_genes, filtered_gene_data = filter_spot_gene_with_genes(self.genes, self.spot2gene, x_start, x_end, y_start,y_end)
        filtered_cells, filtered_cell_data = filter_cell_spot_with_cells(self.cells, self.cell2spot,  x_start, x_end, y_start,y_end)
        return filtered_genes, filtered_gene_data, filtered_cells, filtered_cell_data
        
    def process_region(self,cells,cell2spot,genes,spot2gene):
        """
        处理指定区域
            """
            # 生成细胞空间坐标
        x_all = []
        y_all = []
        z_all = []
        for cell in cells:
            z = cell[cell.find('z')+1:cell.rfind('_')]
            z_all.append(z)
            x_sum = 0
            y_sum = 0
            for spot in cell2spot[cell]:
                spt = spot.split(":")
                y_sum+=int(spot.split(":")[0])
                x_sum+=int(spot.split(":")[1])
            x = x_sum/len(cell2spot[cell])
            y = y_sum/len(cell2spot[cell])
            x_all.append(x)
            y_all.append(y)
        counts = np.zeros((len(cells), len(genes)), dtype=np.float32)
        adata = ad.AnnData(counts)
        adata.obs_names = cells
        adata.var_names = genes
        adata.obs['x'] = x_all
        adata.obs['y'] = y_all
        adata.obs['z'] = z_all

        adata.obsm['X_spatial_registered'] = np.column_stack((x_all, y_all, z_all)).astype(np.float32)

        for cell in cell2spot:
            cell_id = adata.obs_names.get_loc(cell)
            for spot in cell2spot[cell]:
                if spot not in spot2gene:
                    continue
                for geneUnit in spot2gene[spot]:
                    geneName = geneUnit[0]
                    geneCnt = int(geneUnit[1])
                    gene_id = adata.var_names.get_loc(geneName)
                    adata.X[cell_id][gene_id]+=geneCnt
        adata.X = csr_matrix(adata.X)
        return adata

#         num_genes = exp_data.exp_matrix.shape[1]

#         cell_ids = np.unique(mask)

#         cell_ids = cell_ids[cell_ids > 0]
#         new_cell_ids = {old_id: new_id + 1 for new_id, old_id in enumerate(cell_ids)}

#         cell_ids = np.vectorize(lambda x: new_cell_ids.get(x, 0))(mask)
#         cell_expr = []
#         cell_positions = []

#         all_genes = set()

#         for cell_id in cell_ids:

#             cell_pixels = np.where(mask == cell_id)
            
#             pixel_indices = np.array(cell_pixels).T
            
#             cell_center = center_of_mass(mask == cell_id)
            
#             cell_gene_expr = np.zeros(num_genes)
#             for idx in pixel_indices:
#                 # position_idx = idx[0]  # 
#                 position_idx = idx
#                 pixel_expr = exp_data.exp_matrix[position_idx, :].toarray().flatten()
#                 cell_gene_expr += pixel_expr
#             cell_expr.append(cell_gene_expr)
#             cell_positions.append(cell_center)
        
#         # 将累加后的表达矩阵转换为 np.array
#         cell_expr = np.array(cell_expr)
#         logging.info(f"Aggregated gene expression matrix shape: {cell_expr.shape}")
        
#         # 将基因索引转换为列表，并排序
#         # all_genes = sorted(list(all_genes))
#         gene_names = exp_data.gene_names
#         logging.info(f"length of gene_names: {len(gene_names)}")
#         return cell_expr,all_genes,gene_names,cell_positions
        


    def process_custom_region(self,x_start=None, y_start=None, x_end=None, y_end=None):
        """
        指定感兴趣的坐标范围处理数据，生成 AnnData 对象。
        
        参数:
        - x_start (int): 区域起始的 x 坐标
        - y_start (int): 区域起始的 y 坐标
        - x_end (int): 区域结束的 x 坐标
        - y_end (int): 区域结束的 y 坐标
        
        返回:
        - adata (AnnData): 生成的 AnnData 对象
        """
        if x_start is None:
            x_start = 0
        if y_start is None:
            y_start = 0
        if x_end is None:
            x_end = self.width
        if y_end is None:
            y_end = self.height
        logging.info(f"Starting processing for custom region from ({x_start}, {y_start}) to ({x_end}, {y_end})")
        
        try:
            # 获取指定区域的 mask 和基因表达
            filtered_genes, filtered_gene_data, filtered_cells, filtered_cell_data = self.get_region(x_start, y_start, x_end, y_end)
            result = self.process_region(filtered_cells,filtered_cell_data,filtered_genes,filtered_gene_data)
            self.cells, self.cell2spot,self.genes, self.spot2gene =  filtered_cells, filtered_cell_data, filtered_genes, filtered_gene_data
            
            return result
        except Exception as e:
            logging.error(f"Error during processing: {e}")
            raise

def generate_h5ad_all(mask_paths, gene_paths):
    print('---generate h5ad all---')
    cell2spot_dict = {}
    cells_all = []

    for mp in mask_paths:
        cells, cell2spot = get_cell_spot(mp)
        cells_all+=cells
        file = os.path.basename(mp)
        z = file[file.find('z')+1:file.find('.tif')]
        cell2spot_dict[z] = cell2spot

    x_all = []
    y_all = []
    z_all = []

    for cell in cells_all:
        z = cell[cell.find('z')+1:cell.rfind('_')]
        z_all.append(int(z))
        x_sum = 0
        y_sum = 0
        length = len(cell2spot_dict[z][cell])
        for spot in cell2spot_dict[z][cell]:
            spt = spot.split(":")
            y_sum+=int(spot.split(":")[0])
            x_sum+=int(spot.split(":")[1])
        x = x_sum/length
        y = y_sum/length
        x_all.append(x)
        y_all.append(y)

    genes_all = set()
    spot2gene_dict = {}

    for gp in gene_paths:
        genes, spot2gene = get_spot_gene(gp)
        genes_all = genes_all|set(genes)
        file = os.path.basename(gp)
        z = file[file.find('z')+1:file.find('.tif')]
        spot2gene_dict[z] = spot2gene

    genes_all = list(genes_all)    
    slices = [cell[:cell.rfind('_')] for cell in cells_all]
    counts = np.zeros((len(cells_all), len(genes_all)), dtype=np.float32)
    adata = ad.AnnData(counts)
    adata.obs_names = cells_all
    adata.var_names = genes_all
    adata.obs['x'] = x_all
    adata.obs['y'] = y_all
    adata.obs['z'] = z_all
    adata.obs['dataset'] = pd.Categorical(slices)
    adata.obs['orig.ident'] = pd.Categorical(slices)
    adata.obsm['X_spatial'] = np.column_stack((x_all, y_all)).astype(np.float32)
    adata.obsm['X_spatial_registered'] = np.column_stack((x_all, y_all, z_all)).astype(np.float32)

    for cell in cells_all:
        cell_id = adata.obs_names.get_loc(cell)
        z = str(adata.obs['z'][cell_id])
        for spot in cell2spot_dict[z][cell]:
            if spot not in spot2gene_dict[z]:
                continue
            for geneUnit in spot2gene_dict[z][spot]:
                geneName = geneUnit[0]
                geneCnt = int(geneUnit[1])
                gene_id = adata.var_names.get_loc(geneName)
                adata.X[cell_id][gene_id]+=geneCnt
    adata.X = csr_matrix(adata.X)
    adata.var['feature_name'] = adata.var_names
    return adata

# class MerfishExpData:
#     """
#     A class to encapsulate gene expression data from a CSV file.
    
#     The CSV is expected to have the following columns:
#     - geneID: Identifier for the gene.
#     - x: X-coordinate of the spatial position.
#     - y: Y-coordinate of the spatial position.
#     - MIDcount: Expression count for the gene at the given position.
#     """
    
#     def __init__(self, expr_df: pd.DataFrame):
#         self.expr_df = expr_df
#         self.gene_names = expr_df['geneID'].unique().tolist()
#         self.position = expr_df[['x', 'y']].drop_duplicates().reset_index(drop=True).values
#         # Create a pivot table with positions as rows and genes as columns
#         self.exp_matrix = expr_df.pivot_table(index=['x', 'y'], columns='geneID', values='MIDCount', fill_value=0).values
#         logging.info("MerfishExpData initialized with gene expression matrix shape: "
#                      f"{self.exp_matrix.shape}")

# class GeneExpressionManager_Merfish:
#     def __init__(self, mask_path: str, expression_path: str):
#         """
#         Initializes the GeneExpressionManager with paths to the mask image and gene expression CSV.

#         Parameters:
#         - mask_path (str): Path to the mask image file.
#         - expression_path (str): Path to the gene expression CSV file.
#         """
#         # Initialize paths
#         self.mask_path = mask_path
#         self.expression_path = expression_path

#         logging.info(f"Loading mask image from {mask_path}")
#         self.mask = imread(mask_path)
#         logging.info(f"Mask image loaded with shape: {self.mask.shape}")

#         logging.info(f"Loading gene expression data from {expression_path}")
#         try:
#             self.expr_df = pd.read_csv(expression_path)
#             # Validate the required columns
#             required_columns = {'geneID', 'x', 'y', 'MIDCount'}
#             if not required_columns.issubset(self.expr_df.columns):
#                 missing = required_columns - set(self.expr_df.columns)
#                 raise ValueError(f"Missing columns in expression CSV: {missing}")
#             logging.info("Gene expression data loaded successfully")
#         except Exception as e:
#             logging.error(f"Failed to load gene expression data: {e}")
#             raise

#         self.height, self.width = self.mask.shape
#         logging.info(f"Mask image dimensions: height={self.height}, width={self.width}")

#     def get_region(self, x_start: int, y_start: int, x_end: int, y_end: int) -> pd.DataFrame:
#         """
#         Fetches the gene expression data for a specified spatial region.

#         Parameters:
#         - x_start (int): Starting x-coordinate.
#         - y_start (int): Starting y-coordinate.
#         - x_end (int): Ending x-coordinate.
#         - y_end (int): Ending y-coordinate.

#         Returns:
#         - expr_region_df (pd.DataFrame): Filtered gene expression data within the specified region.
#         """
#         x_start = max(x_start, 0)
#         y_start = max(y_start, 0)
#         x_end = min(x_end, self.width)
#         y_end = min(y_end, self.height)
#         logging.info(f"Fetching gene expression data for region: "
#                      f"x_start={x_start}, y_start={y_start}, x_end={x_end}, y_end={y_end}")

#         # Ensure the region is within image bounds
#         x_start = max(x_start, 0)
#         y_start = max(y_start, 0)
#         x_end = min(x_end, self.width)
#         y_end = min(y_end, self.height)

#         # Filter the DataFrame for the specified region
#         expr_region_df = self.expr_df[
#             (self.expr_df['x'] >= x_start) & (self.expr_df['x'] < x_end) &
#             (self.expr_df['y'] >= y_start) & (self.expr_df['y'] < y_end)
#         ].copy()

#         logging.info(f"Fetched {expr_region_df.shape[0]} gene expression entries in the specified region")
#         return expr_region_df

#     def process_region(self, mask: np.ndarray, expr_region_df: pd.DataFrame) -> tuple:
#         """
#         Processes the specified region to compute cell centers and aggregate gene expressions.

#         Parameters:
#         - mask (np.ndarray): Mask for the specified region.
#         - expr_region_df (pd.DataFrame): Gene expression data within the region.

#         Returns:
#         - cell_expr (np.ndarray): Aggregated gene expression matrix (cells x genes).
#         - gene_names (list): List of gene names.
#         - cell_positions (list): List of cell center coordinates.
#         """
#         logging.info("Processing gene expression data for the specified region")

#         # Initialize MerfishExpData with the filtered DataFrame
#         exp_data = MerfishExpData(expr_region_df)

#         num_genes = len(exp_data.gene_names)
#         # Get unique cell IDs excluding background (assuming background is labeled as 0)
#         cell_ids = np.unique(mask)
#         cell_ids = cell_ids[cell_ids > 0]
#         logging.info(f"Processing {len(cell_ids)} cell IDs")

#         # Prepare storage for aggregated expressions and cell positions
#         cell_expr = []
#         cell_positions = []

#         for cell_id in cell_ids:
#             # Get pixel positions for the current cell
#             cell_pixels = np.where(mask == cell_id)
#             pixel_coords = list(zip(cell_pixels[1], cell_pixels[0]))  # (x, y) tuples

#             # Compute center of mass for the cell
#             cell_center = center_of_mass(mask == cell_id)
#             cell_positions.append((cell_center[1], cell_center[0]))  # (x, y)

#             # Aggregate gene expression for the cell
#             # Filter expression data for the cell's pixels
#             cell_expr_df = expr_region_df[
#                 (expr_region_df['x'].isin([coord[0] for coord in pixel_coords])) &
#                 (expr_region_df['y'].isin([coord[1] for coord in pixel_coords]))
#             ]

#             # Pivot the data to create a matrix (genes x counts) and sum counts per gene
#             gene_counts = cell_expr_df.groupby('geneID')['MIDCount'].sum()
#             # Ensure all genes are represented
#             gene_counts = gene_counts.reindex(exp_data.gene_names, fill_value=0).values
#             cell_expr.append(gene_counts)

#         # Convert to numpy array
#         cell_expr = np.array(cell_expr)
#         logging.info(f"Aggregated gene expression matrix shape: {cell_expr.shape}")

#         gene_names = exp_data.gene_names
#         logging.info(f"Number of genes: {len(gene_names)}")

#         return cell_expr, gene_names, cell_positions

#     def process_custom_region(self, x_start=None, y_start=None, x_end=None, y_end=None) -> AnnData:
#         """
#         Processes a custom region and generates an AnnData object.

#         Parameters:
#         - x_start (int): Starting x-coordinate.
#         - y_start (int): Starting y-coordinate.
#         - x_end (int): Ending x-coordinate.
#         - y_end (int): Ending y-coordinate.

#         Returns:
#         - adata (AnnData): The resulting AnnData object.
#         """
#         if x_start is None:
#             x_start = 0
#         if y_start is None:
#             y_start = 0
#         if x_end is None:
#             x_end = self.width
#         if y_end is None:
#             y_end = self.height
#         logging.info(f"Starting processing for region: "
#                      f"({x_start}, {y_start}) to ({x_end}, {y_end})")

#         try:
#             # Extract the mask region
#             mask_region = self.mask[y_start:y_end, x_start:x_end]
#             logging.info(f"Extracted mask region with shape: {mask_region.shape}")

#             # Fetch the gene expression data for the region
#             expr_region_df = self.get_region(x_start, y_start, x_end, y_end)

#             # Process the region to get aggregated expressions and cell positions
#             cell_expr, gene_names, cell_positions = self.process_region(mask_region, expr_region_df)

#             # Create AnnData object
#             adata = AnnData(X=cell_expr)
#             adata.var.index = gene_names
#             logging.info(f"Created AnnData object with {adata.n_obs} cells and {adata.n_vars} genes")

#             # Add spatial coordinates to the AnnData object
#             adata.obs['x'] = [pos[0] for pos in cell_positions]  # x-coordinates
#             adata.obs['y'] = [pos[1] for pos in cell_positions]  # y-coordinates
#             adata.obsm['X_spatial'] = np.array(cell_positions)

#             return adata

#         except Exception as e:
#             logging.error(f"Error during processing: {e}")
#             raise
# def process_and_merge_multi_slice(csv_path, output_h5ad, platform='stereo'):
#     """从CSV文件读取信息并处理多个区域，整合成一个 h5ad 文件"""
#     # 读取CSV文件
#     df = pd.read_csv(csv_path)
    
#     # 创建一个空的AnnData列表
#     all_adata = []
    
#     for _, row in df.iterrows():
#         expression_path = row['expression_path']
#         mask_path = row['mask_path']
#         x_start, y_start, x_end, y_end = row['x_start'], row['y_start'], row['x_end'], row['y_end']
#         z_index = row['z_index']
        
#         # 根据平台选择对应的处理类
#         if platform == 'stereo':
#             manager = GeneExpressionManager_stereoseq(mask_path, expression_path)
#         elif platform == 'merfish':
#             manager = GeneExpressionManager_Merfish(mask_path, expression_path)
#         else:
#             raise ValueError(f"Unsupported platform: {platform}")
        
#         # 处理当前区域
#         adata = manager.process_custom_region(x_start, y_start, x_end, y_end)
        
#         # 为该区域赋值z索引
#         adata.obs['z'] = [z_index] * adata.n_obs
        
#         # 将处理后的AnnData对象添加到列表中
#         all_adata.append(adata)
    
#     # 合并所有AnnData对象
#     merged_adata = all_adata[0].concatenate(all_adata[1:], join='outer', batch_key='batch')
    
#     # 保存最终的AnnData对象到h5ad文件
#     logging.info(f"Saving merged AnnData to {output_h5ad}")
#     merged_adata.write(output_h5ad)