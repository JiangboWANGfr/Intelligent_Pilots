"""
纯Python实现的Perlin噪声生成器
无需C++编译依赖
"""
import numpy as np
from typing import Tuple


class PerlinNoise:
    """Perlin噪声生成器"""

    def __init__(self, seed: int = 0):
        """
        初始化Perlin噪声生成器

        Args:
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)

        # 生成置换表
        self.p = np.arange(256, dtype=int)
        np.random.shuffle(self.p)
        self.p = np.concatenate([self.p, self.p])  # 复制一份避免溢出

    def fade(self, t):
        """淡化函数：6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(self, t, a, b):
        """线性插值"""
        return a + t * (b - a)

    def grad(self, hash_val, x, y):
        """梯度函数"""
        # 取hash值的最后4位
        h = hash_val & 15

        # 将前3位转换为梯度向量
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else 0)

        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def noise(self, x, y):
        """
        生成2D Perlin噪声

        Args:
            x, y: 坐标

        Returns:
            噪声值 [-1, 1]
        """
        # 找到单位格子
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255

        # 计算相对位置
        x -= np.floor(x)
        y -= np.floor(y)

        # 计算淡化曲线
        u = self.fade(x)
        v = self.fade(y)

        # Hash坐标
        aa = self.p[self.p[X] + Y]
        ab = self.p[self.p[X] + Y + 1]
        ba = self.p[self.p[X + 1] + Y]
        bb = self.p[self.p[X + 1] + Y + 1]

        # 计算并插值梯度
        res = self.lerp(
            v,
            self.lerp(u, self.grad(aa, x, y), self.grad(ba, x - 1, y)),
            self.lerp(u, self.grad(ab, x, y - 1), self.grad(bb, x - 1, y - 1))
        )

        return res

    def octave_noise(self, x, y, octaves=6, persistence=0.5, lacunarity=2.0):
        """
        生成多八度Perlin噪声

        Args:
            x, y: 坐标
            octaves: 八度数
            persistence: 持续性（振幅衰减）
            lacunarity: 间隙度（频率增长）

        Returns:
            噪声值
        """
        total = 0
        frequency = 1
        amplitude = 1
        max_value = 0

        for _ in range(octaves):
            total += self.noise(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        return total / max_value

    def generate_2d_noise(self,
                         shape: Tuple[int, int],
                         scale: float = 100.0,
                         octaves: int = 6,
                         persistence: float = 0.5,
                         lacunarity: float = 2.0) -> np.ndarray:
        """
        生成2D噪声场

        Args:
            shape: 输出形状 (height, width)
            scale: 噪声比例，越大越平滑
            octaves: 八度数
            persistence: 持续性
            lacunarity: 间隙度

        Returns:
            噪声场 [0, 1]
        """
        height, width = shape
        noise_map = np.zeros(shape)

        for y in range(height):
            for x in range(width):
                noise_map[y, x] = self.octave_noise(
                    x / scale,
                    y / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity
                )

        # 归一化到 [0, 1]
        noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min() + 1e-8)
        return noise_map


class SimplexNoise:
    """简化的Simplex噪声生成器（基于Perlin）"""

    def __init__(self, seed: int = 0):
        self.perlin = PerlinNoise(seed)

    def noise(self, x, y):
        """生成Simplex风格的噪声（使用Perlin作为基础）"""
        # 简化实现：使用Perlin噪声加上一些变换
        return self.perlin.noise(x * 0.866, y * 0.866 + x * 0.5)

    def octave_noise(self, x, y, octaves=4, persistence=0.5, lacunarity=2.0):
        total = 0
        frequency = 1
        amplitude = 1
        max_value = 0

        for _ in range(octaves):
            total += self.noise(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        return total / max_value

    def generate_2d_noise(self,
                         shape: Tuple[int, int],
                         scale: float = 100.0,
                         octaves: int = 4) -> np.ndarray:
        """生成2D Simplex风格噪声场"""
        height, width = shape
        noise_map = np.zeros(shape)

        for y in range(height):
            for x in range(width):
                noise_map[y, x] = self.octave_noise(
                    x / scale,
                    y / scale,
                    octaves=octaves
                )

        # 归一化到 [0, 1]
        noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min() + 1e-8)
        return noise_map
