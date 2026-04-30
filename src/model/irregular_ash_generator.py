"""
不规则火山灰云生成器
支持扰动、拉伸、分形边界等效果，使火山灰云形状更接近真实世界
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import Tuple, Optional, Dict
import cv2
from .perlin_noise import PerlinNoise, SimplexNoise


class IrregularAshGenerator:
    """生成不规则火山灰云的工具类"""

    def __init__(self, seed: Optional[int] = None):
        """
        初始化生成器

        Args:
            seed: 随机种子，用于可重复的结果
        """
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed or np.random.randint(0, 10000)
        self.perlin = PerlinNoise(self.seed)
        self.simplex = SimplexNoise(self.seed)

    def generate_perlin_noise(self,
                             shape: Tuple[int, int],
                             scale: float = 100.0,
                             octaves: int = 6,
                             persistence: float = 0.5,
                             lacunarity: float = 2.0) -> np.ndarray:
        """
        生成Perlin噪声场

        Args:
            shape: 输出形状 (height, width)
            scale: 噪声比例，越大越平滑
            octaves: 八度数，越多细节越丰富
            persistence: 持续性，控制振幅衰减
            lacunarity: 间隙度，控制频率增长

        Returns:
            归一化的噪声场 [0, 1]
        """
        return self.perlin.generate_2d_noise(
            shape=shape,
            scale=scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity
        )

    def generate_simplex_noise(self,
                               shape: Tuple[int, int],
                               scale: float = 100.0,
                               octaves: int = 4) -> np.ndarray:
        """
        生成Simplex噪声场（比Perlin更快，更自然）

        Args:
            shape: 输出形状 (height, width)
            scale: 噪声比例
            octaves: 八度数

        Returns:
            归一化的噪声场 [0, 1]
        """
        return self.simplex.generate_2d_noise(
            shape=shape,
            scale=scale,
            octaves=octaves
        )

    def apply_turbulence(self,
                        field: np.ndarray,
                        turbulence_scale: float = 0.1,
                        noise_scale: float = 50.0,
                        octaves: int = 4) -> np.ndarray:
        """
        对浓度场应用湍流扰动

        Args:
            field: 输入浓度场
            turbulence_scale: 扰动强度 [0, 1]
            noise_scale: 噪声比例
            octaves: 噪声细节层次

        Returns:
            扰动后的浓度场
        """
        height, width = field.shape

        # 生成两个方向的噪声场作为位移场
        noise_x = self.generate_perlin_noise(
            (height, width),
            scale=noise_scale,
            octaves=octaves
        )
        noise_y = self.generate_perlin_noise(
            (height, width),
            scale=noise_scale,
            octaves=octaves
        )

        # 将噪声转换为位移（-turbulence_scale 到 +turbulence_scale 的像素位移）
        max_displacement = min(width, height) * turbulence_scale
        displacement_x = (noise_x - 0.5) * 2 * max_displacement
        displacement_y = (noise_y - 0.5) * 2 * max_displacement

        # 创建扰动后的坐标网格
        y_coords, x_coords = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing='ij'
        )

        # 应用位移
        new_x = x_coords + displacement_x
        new_y = y_coords + displacement_y

        # 使用样条插值进行重采样
        turbulent_field = map_coordinates(
            field,
            [new_y, new_x],
            order=3,
            mode='nearest'
        )

        return turbulent_field

    def apply_wind_stretching(self,
                             field: np.ndarray,
                             wind_direction: float = 0.0,
                             wind_strength: float = 0.3,
                             turbulent_wind: bool = True) -> np.ndarray:
        """
        应用风场拉伸效果

        Args:
            field: 输入浓度场
            wind_direction: 风向（角度，0为向右，逆时针）
            wind_strength: 风力强度 [0, 1]
            turbulent_wind: 是否添加湍流风场

        Returns:
            拉伸后的浓度场
        """
        height, width = field.shape

        # 转换风向为弧度
        wind_rad = np.radians(wind_direction)
        wind_dx = np.cos(wind_rad) * wind_strength
        wind_dy = np.sin(wind_rad) * wind_strength

        # 创建基础风场
        y_coords, x_coords = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing='ij'
        )

        # 计算每个点到中心的距离，距离越远风的影响越大
        center_y, center_x = height / 2, width / 2
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        distance_factor = distance / max_distance

        # 如果使用湍流风场，添加噪声扰动
        if turbulent_wind:
            noise_field = self.generate_perlin_noise((height, width), scale=80.0, octaves=3)
            wind_turbulence = (noise_field - 0.5) * 0.5  # [-0.25, 0.25]
            distance_factor = distance_factor * (1 + wind_turbulence)

        # 应用风场位移
        max_displacement = min(width, height) * 0.5
        displacement_x = wind_dx * distance_factor * max_displacement
        displacement_y = wind_dy * distance_factor * max_displacement

        new_x = x_coords + displacement_x
        new_y = y_coords + displacement_y

        # 重采样
        stretched_field = map_coordinates(
            field,
            [new_y, new_x],
            order=3,
            mode='constant',
            cval=0.0
        )

        return stretched_field

    def create_fractal_boundary(self,
                               field: np.ndarray,
                               threshold: float = 0.3,
                               fractal_dimension: float = 1.5,
                               iterations: int = 3) -> np.ndarray:
        """
        创建分形边界，使边缘更不规则

        Args:
            field: 输入浓度场
            threshold: 边界阈值
            fractal_dimension: 分形维度 [1.0, 2.0]，越大越粗糙
            iterations: 迭代次数

        Returns:
            具有分形边界的浓度场
        """
        result = field.copy()

        for _ in range(iterations):
            # 生成噪声
            noise = self.generate_perlin_noise(
                field.shape,
                scale=30.0 / (fractal_dimension),
                octaves=int(fractal_dimension * 3)
            )

            # 在边界附近混合噪声
            boundary_mask = (result > threshold * 0.5) & (result < threshold * 1.5)
            noise_contribution = (noise - 0.5) * 0.3
            result[boundary_mask] += noise_contribution[boundary_mask]

            # 平滑并裁剪
            result = gaussian_filter(result, sigma=1.0)
            result = np.clip(result, 0, 1)

        return result

    def add_filaments(self,
                     field: np.ndarray,
                     num_filaments: int = 5,
                     filament_width: float = 3.0,
                     filament_strength: float = 0.3) -> np.ndarray:
        """
        添加细丝状结构（类似火山灰云的分支）

        Args:
            field: 输入浓度场
            num_filaments: 细丝数量
            filament_width: 细丝宽度
            filament_strength: 细丝强度

        Returns:
            添加细丝后的浓度场
        """
        height, width = field.shape
        result = field.copy()

        # 找到浓度较高的区域作为细丝起点
        high_conc_mask = field > 0.5
        high_conc_coords = np.argwhere(high_conc_mask)

        if len(high_conc_coords) == 0:
            return result

        for _ in range(num_filaments):
            # 随机选择起点
            start_idx = np.random.randint(0, len(high_conc_coords))
            y, x = high_conc_coords[start_idx]

            # 生成随机方向和长度
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.uniform(min(width, height) * 0.1, min(width, height) * 0.3)

            # 生成细丝路径（贝塞尔曲线或随机游走）
            num_points = int(length)
            points = []
            current_angle = angle

            for i in range(num_points):
                # 添加角度扰动
                current_angle += np.random.normal(0, 0.2)

                x += np.cos(current_angle)
                y += np.sin(current_angle)

                if 0 <= int(x) < width and 0 <= int(y) < height:
                    points.append((int(y), int(x)))

            # 在路径上绘制细丝
            for py, px in points:
                # 使用高斯分布创建细丝
                for dy in range(-int(filament_width * 2), int(filament_width * 2) + 1):
                    for dx in range(-int(filament_width * 2), int(filament_width * 2) + 1):
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            distance = np.sqrt(dx**2 + dy**2)
                            if distance <= filament_width * 2:
                                contribution = filament_strength * np.exp(-distance**2 / (2 * filament_width**2))
                                result[ny, nx] = min(1.0, result[ny, nx] + contribution)

        return result

    def generate_irregular_ash_cloud(self,
                                    base_field: np.ndarray,
                                    config: Optional[Dict] = None) -> np.ndarray:
        """
        生成不规则火山灰云

        Args:
            base_field: 基础浓度场（来自GMM等）
            config: 配置参数字典，包含：
                - turbulence_scale: 湍流强度
                - wind_direction: 风向
                - wind_strength: 风力
                - add_fractal: 是否添加分形边界
                - add_filaments: 是否添加细丝
                - num_filaments: 细丝数量

        Returns:
            不规则的火山灰云浓度场
        """
        if config is None:
            config = {}

        # 获取配置参数
        turbulence_scale = config.get('turbulence_scale', 0.15)
        wind_direction = config.get('wind_direction', 45.0)
        wind_strength = config.get('wind_strength', 0.3)
        add_fractal = config.get('add_fractal', True)
        add_filaments_flag = config.get('add_filaments', True)
        num_filaments = config.get('num_filaments', 5)
        fractal_dimension = config.get('fractal_dimension', 1.5)

        result = base_field.copy()

        # 1. 应用湍流扰动
        if turbulence_scale > 0:
            result = self.apply_turbulence(
                result,
                turbulence_scale=turbulence_scale,
                noise_scale=60.0
            )

        # 2. 应用风场拉伸
        if wind_strength > 0:
            result = self.apply_wind_stretching(
                result,
                wind_direction=wind_direction,
                wind_strength=wind_strength,
                turbulent_wind=True
            )

        # 3. 添加分形边界
        if add_fractal:
            result = self.create_fractal_boundary(
                result,
                threshold=0.3,
                fractal_dimension=fractal_dimension,
                iterations=2
            )

        # 4. 添加细丝结构
        if add_filaments_flag and num_filaments > 0:
            result = self.add_filaments(
                result,
                num_filaments=num_filaments,
                filament_width=3.0,
                filament_strength=0.25
            )

        # 5. 最后平滑和归一化
        result = gaussian_filter(result, sigma=1.5)
        result = np.clip(result, 0, 1)

        return result


def create_diverse_ash_clouds(base_field: np.ndarray,
                              num_variations: int = 5,
                              seed: Optional[int] = None) -> list:
    """
    创建多样化的火山灰云变体

    Args:
        base_field: 基础浓度场
        num_variations: 变体数量
        seed: 随机种子

    Returns:
        火山灰云变体列表
    """
    variations = []

    # 预定义多种配置
    configs = [
        {  # 轻度扰动，向东拉伸
            'turbulence_scale': 0.1,
            'wind_direction': 0,
            'wind_strength': 0.25,
            'add_fractal': True,
            'add_filaments': True,
            'num_filaments': 3,
            'fractal_dimension': 1.3
        },
        {  # 强扰动，向东北拉伸
            'turbulence_scale': 0.2,
            'wind_direction': 45,
            'wind_strength': 0.4,
            'add_fractal': True,
            'add_filaments': True,
            'num_filaments': 7,
            'fractal_dimension': 1.6
        },
        {  # 中等扰动，向南拉伸
            'turbulence_scale': 0.15,
            'wind_direction': 270,
            'wind_strength': 0.3,
            'add_fractal': True,
            'add_filaments': True,
            'num_filaments': 5,
            'fractal_dimension': 1.5
        },
        {  # 复杂分形，多细丝
            'turbulence_scale': 0.18,
            'wind_direction': 135,
            'wind_strength': 0.35,
            'add_fractal': True,
            'add_filaments': True,
            'num_filaments': 10,
            'fractal_dimension': 1.8
        },
        {  # 轻微拉伸，自然扩散
            'turbulence_scale': 0.12,
            'wind_direction': 315,
            'wind_strength': 0.2,
            'add_fractal': True,
            'add_filaments': True,
            'num_filaments': 4,
            'fractal_dimension': 1.4
        }
    ]

    for i in range(min(num_variations, len(configs))):
        gen_seed = (seed + i * 1000) if seed is not None else None
        generator = IrregularAshGenerator(seed=gen_seed)

        irregular_cloud = generator.generate_irregular_ash_cloud(
            base_field,
            config=configs[i]
        )
        variations.append(irregular_cloud)

    return variations
