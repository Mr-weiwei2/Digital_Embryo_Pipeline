import numpy as np
import pandas as pd
import tifffile
from collections import defaultdict
import scanpy as sc
from cellpose.io import imread,imsave
import stereo as st
import warnings
from scipy.ndimage import center_of_mass
warnings.filterwarnings('ignore')
from anndata import AnnData
import logging

# 设置logging配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeneExpressionManager_stereoseq:
    def __init__(self, mask_path, expression_path):

        self.mask_path = mask_path
        self.expression_path = expression_path
        
        logging.info(f"Loading mask image from {mask_path}")
        self.mask = imread(mask_path)
        logging.info(f"Loading gene expression data from {expression_path}")
        self.expr_df =st.io.read_gem(
        file_path=expression_path,
        bin_type='bins',
        bin_size=50,
        is_sparse=True,
        )

        
        self.height, self.width = self.mask.shape
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
        mask_region = self.mask[y_start:y_end, x_start:x_end]
        
        # 获取在该区域范围内的表达数据
        positions = self.expr_df.position
        in_region = (positions[:, 0] >= x_start) & (positions[:, 0] < x_end) & \
                    (positions[:, 1] >= y_start) & (positions[:, 1] < y_end)
        
        expr_region = self.expr_df.sub_by_index(in_region)
        logging.info(f"Fetched {expr_region.shape[0]} gene expression entries in the specified region")
        return mask_region, expr_region
        
    def process_region(self, mask, exp_data):
        """
        处理指定区域，计算细胞中心和基因表达汇总。
        """

        num_genes = exp_data.exp_matrix.shape[1]

        cell_ids = np.unique(mask)

        cell_ids = cell_ids[cell_ids > 0]

        cell_expr = []
        cell_positions = []

        all_genes = set()

        for cell_id in cell_ids:

            cell_pixels = np.where(mask == cell_id)
            
            pixel_indices = np.array(cell_pixels).T
            
            cell_center = center_of_mass(mask == cell_id)
            
            cell_gene_expr = np.zeros(num_genes)
            for idx in pixel_indices:
                position_idx = idx[0]  # 
                pixel_expr = exp_data.exp_matrix[position_idx, :].toarray().flatten()
                cell_gene_expr += pixel_expr
            cell_expr.append(cell_gene_expr)
            cell_positions.append(cell_center)
        
        # 将累加后的表达矩阵转换为 np.array
        cell_expr = np.array(cell_expr)
        logging.info(f"Aggregated gene expression matrix shape: {cell_expr.shape}")
        
        # 将基因索引转换为列表，并排序
        # all_genes = sorted(list(all_genes))
        gene_names = exp_data.gene_names
        logging.info(f"length of gene_names: {len(gene_names)}")
        return cell_expr,all_genes,gene_names,cell_positions
        


    def process_custom_region(self, x_start, y_start, x_end, y_end):
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
            mask_region, expr_region = self.get_region(x_start, y_start, x_end, y_end)
            # 处理区域数据
            cell_expr,all_genes,gene_names,cell_positions = self.process_region(mask_region, expr_region)
                        # 将结果整合为 AnnData
            adata = anndata.AnnData(X=cell_expr)
            adata.var.index = gene_names# 保存为anndata文件
            logging.info(f"Processing completed for custom region. AnnData object created with {adata.n_obs} cells and {adata.n_vars} genes")
            reversed_cell_positions = [(pos[1], pos[0]) for pos in cell_positions]
            adata.obs['x'] = [pos[0] for pos in reversed_cell_positions]  # x坐标
            adata.obs['y'] = [pos[1] for pos in reversed_cell_positions]  # y坐标
            # cell_positions = [(pos[0], pos[1]) for pos in cell_positions]
            # adata.obs['x'] = [pos[0] for pos in cell_positions]
            # adata.obs['y'] = [pos[1] for pos in cell_positions] 
            adata.obsm['X_spatial'] = np.array(reversed_cell_positions)
            # 将细胞位置作为obs中的信息，并添加predicted.id字段
            
            
            return adata
        except Exception as e:
            logging.error(f"Error during processing: {e}")
            raise



class MerfishExpData:
    """
    A class to encapsulate gene expression data from a CSV file.
    
    The CSV is expected to have the following columns:
    - geneID: Identifier for the gene.
    - x: X-coordinate of the spatial position.
    - y: Y-coordinate of the spatial position.
    - MIDcount: Expression count for the gene at the given position.
    """
    
    def __init__(self, expr_df: pd.DataFrame):
        self.expr_df = expr_df
        self.gene_names = expr_df['geneID'].unique().tolist()
        self.position = expr_df[['x', 'y']].drop_duplicates().reset_index(drop=True).values
        # Create a pivot table with positions as rows and genes as columns
        self.exp_matrix = expr_df.pivot_table(index=['x', 'y'], columns='geneID', values='MIDCount', fill_value=0).values
        logging.info("MerfishExpData initialized with gene expression matrix shape: "
                     f"{self.exp_matrix.shape}")

class GeneExpressionManager_Merfish:
    def __init__(self, mask_path: str, expression_path: str):
        """
        Initializes the GeneExpressionManager with paths to the mask image and gene expression CSV.

        Parameters:
        - mask_path (str): Path to the mask image file.
        - expression_path (str): Path to the gene expression CSV file.
        """
        # Initialize paths
        self.mask_path = mask_path
        self.expression_path = expression_path

        logging.info(f"Loading mask image from {mask_path}")
        self.mask = imread(mask_path)
        logging.info(f"Mask image loaded with shape: {self.mask.shape}")

        logging.info(f"Loading gene expression data from {expression_path}")
        try:
            self.expr_df = pd.read_csv(expression_path)
            # Validate the required columns
            required_columns = {'geneID', 'x', 'y', 'MIDCount'}
            if not required_columns.issubset(self.expr_df.columns):
                missing = required_columns - set(self.expr_df.columns)
                raise ValueError(f"Missing columns in expression CSV: {missing}")
            logging.info("Gene expression data loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load gene expression data: {e}")
            raise

        self.height, self.width = self.mask.shape
        logging.info(f"Mask image dimensions: height={self.height}, width={self.width}")

    def get_region(self, x_start: int, y_start: int, x_end: int, y_end: int) -> pd.DataFrame:
        """
        Fetches the gene expression data for a specified spatial region.

        Parameters:
        - x_start (int): Starting x-coordinate.
        - y_start (int): Starting y-coordinate.
        - x_end (int): Ending x-coordinate.
        - y_end (int): Ending y-coordinate.

        Returns:
        - expr_region_df (pd.DataFrame): Filtered gene expression data within the specified region.
        """
        x_start = max(x_start, 0)
        y_start = max(y_start, 0)
        x_end = min(x_end, self.width)
        y_end = min(y_end, self.height)
        logging.info(f"Fetching gene expression data for region: "
                     f"x_start={x_start}, y_start={y_start}, x_end={x_end}, y_end={y_end}")

        # Ensure the region is within image bounds
        x_start = max(x_start, 0)
        y_start = max(y_start, 0)
        x_end = min(x_end, self.width)
        y_end = min(y_end, self.height)

        # Filter the DataFrame for the specified region
        expr_region_df = self.expr_df[
            (self.expr_df['x'] >= x_start) & (self.expr_df['x'] < x_end) &
            (self.expr_df['y'] >= y_start) & (self.expr_df['y'] < y_end)
        ].copy()

        logging.info(f"Fetched {expr_region_df.shape[0]} gene expression entries in the specified region")
        return expr_region_df

    def process_region(self, mask: np.ndarray, expr_region_df: pd.DataFrame) -> tuple:
        """
        Processes the specified region to compute cell centers and aggregate gene expressions.

        Parameters:
        - mask (np.ndarray): Mask for the specified region.
        - expr_region_df (pd.DataFrame): Gene expression data within the region.

        Returns:
        - cell_expr (np.ndarray): Aggregated gene expression matrix (cells x genes).
        - gene_names (list): List of gene names.
        - cell_positions (list): List of cell center coordinates.
        """
        logging.info("Processing gene expression data for the specified region")

        # Initialize MerfishExpData with the filtered DataFrame
        exp_data = MerfishExpData(expr_region_df)

        num_genes = len(exp_data.gene_names)
        # Get unique cell IDs excluding background (assuming background is labeled as 0)
        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids > 0]
        logging.info(f"Processing {len(cell_ids)} cell IDs")

        # Prepare storage for aggregated expressions and cell positions
        cell_expr = []
        cell_positions = []

        for cell_id in cell_ids:
            # Get pixel positions for the current cell
            cell_pixels = np.where(mask == cell_id)
            pixel_coords = list(zip(cell_pixels[1], cell_pixels[0]))  # (x, y) tuples

            # Compute center of mass for the cell
            cell_center = center_of_mass(mask == cell_id)
            cell_positions.append((cell_center[1], cell_center[0]))  # (x, y)

            # Aggregate gene expression for the cell
            # Filter expression data for the cell's pixels
            cell_expr_df = expr_region_df[
                (expr_region_df['x'].isin([coord[0] for coord in pixel_coords])) &
                (expr_region_df['y'].isin([coord[1] for coord in pixel_coords]))
            ]

            # Pivot the data to create a matrix (genes x counts) and sum counts per gene
            gene_counts = cell_expr_df.groupby('geneID')['MIDCount'].sum()
            # Ensure all genes are represented
            gene_counts = gene_counts.reindex(exp_data.gene_names, fill_value=0).values
            cell_expr.append(gene_counts)

        # Convert to numpy array
        cell_expr = np.array(cell_expr)
        logging.info(f"Aggregated gene expression matrix shape: {cell_expr.shape}")

        gene_names = exp_data.gene_names
        logging.info(f"Number of genes: {len(gene_names)}")

        return cell_expr, gene_names, cell_positions

    def process_custom_region(self, x_start=None, y_start=None, x_end=None, y_end=None) -> AnnData:
        """
        Processes a custom region and generates an AnnData object.

        Parameters:
        - x_start (int): Starting x-coordinate.
        - y_start (int): Starting y-coordinate.
        - x_end (int): Ending x-coordinate.
        - y_end (int): Ending y-coordinate.

        Returns:
        - adata (AnnData): The resulting AnnData object.
        """
        if x_start is None:
            x_start = 0
        if y_start is None:
            y_start = 0
        if x_end is None:
            x_end = self.width
        if y_end is None:
            y_end = self.height
        logging.info(f"Starting processing for region: "
                     f"({x_start}, {y_start}) to ({x_end}, {y_end})")

        try:
            # Extract the mask region
            mask_region = self.mask[y_start:y_end, x_start:x_end]
            logging.info(f"Extracted mask region with shape: {mask_region.shape}")

            # Fetch the gene expression data for the region
            expr_region_df = self.get_region(x_start, y_start, x_end, y_end)

            # Process the region to get aggregated expressions and cell positions
            cell_expr, gene_names, cell_positions = self.process_region(mask_region, expr_region_df)

            # Create AnnData object
            adata = AnnData(X=cell_expr)
            adata.var.index = gene_names
            logging.info(f"Created AnnData object with {adata.n_obs} cells and {adata.n_vars} genes")

            # Add spatial coordinates to the AnnData object
            adata.obs['x'] = [pos[0] for pos in cell_positions]  # x-coordinates
            adata.obs['y'] = [pos[1] for pos in cell_positions]  # y-coordinates
            adata.obsm['X_spatial'] = np.array(cell_positions)

            return adata

        except Exception as e:
            logging.error(f"Error during processing: {e}")
            raise
