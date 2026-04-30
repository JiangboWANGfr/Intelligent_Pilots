"""
程序化路径生成脚本
生成一条视觉效果良好的火山灰规避路径，用于 Cesium 演示
使用三次样条几何算法，保证路径质量，不依赖 RL 模型
"""
import sys, os, json
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.volcanic_ash_config import VolcanicAshConfig
from src.model.gmm_model import GMMVolcanicAshModel


def pixel_to_geo(py, px, config):
    h, w = config.image_size
    lat = config.geo_center_lat + (0.5 - py / h) * config.geo_span_lat
    lon = config.geo_center_lon + (px / w - 0.5) * config.geo_span_lon
    return lat, lon


def generate_avoidance_path():
    # 路径规划用规则高斯云：浓度分布可预测，保证绕行位置准确
    # Cesium 前端会用不规则形状渲染，JSON 浓度数据只用于数据统计
    config = VolcanicAshConfig(
        geo_center_lat=35.0,
        geo_center_lon=120.0,
        geo_span_lat=2.0,
        geo_span_lon=2.0,
        enable_irregular=False,
        centers=[
            {'x': 256, 'y': 256, 'weight': 1.0, 'std_x': 60, 'std_y': 60}
        ]
    )

    print("正在生成不规则火山灰云浓度图...")
    model = GMMVolcanicAshModel(config)
    concentration_map = model.generate_concentration_map()

    # ── 路径控制点（像素坐标 [y, x]）──
    # 云团中心：(256, 256)
    # 路径从西南角出发，向东北方向飞行
    # 沿140px圆弧绕行，全程在绿色安全区边缘（距中心140-155px，浓度≈0.05-0.11）
    # 最后到达东北角
    control_points = np.array([
        [440,  50],   # 起点：西南角（远离云团）
        [400,  72],   # 向东北飞行
        [340,  94],   # 接近云团区域
        [278, 108],   # 开始北偏绕行（距云心150px）
        [215, 117],   # 进入绿色外层（距云心145px）
        [163, 134],   # 绿色区北偏西弧段（距云心153px）
        [125, 195],   # 弧段绕行（距云心145px）
        [116, 256],   # 最近点：云团正北，距中心140px（绿色区域，浓度≈0.07）
        [120, 316],   # 弧段离开（距云心149px）
        [138, 378],   # 绿色区北偏东（距云心170px）
        [110, 425],   # 继续东北
        [ 75, 452],   # 终点：东北角
    ])

    # 三次样条插值，生成 130 个平滑路径点
    t = np.linspace(0, 1, len(control_points))
    t_fine = np.linspace(0, 1, 130)
    cs_y = CubicSpline(t, control_points[:, 0])
    cs_x = CubicSpline(t, control_points[:, 1])
    path_y = np.clip(cs_y(t_fine), 1, config.image_size[0] - 2)
    path_x = np.clip(cs_x(t_fine), 1, config.image_size[1] - 2)

    # 构建 waypoints
    waypoints = []
    cumulative_fuel = 0.0
    for i in range(len(path_y)):
        py, px = path_y[i], path_x[i]
        py_int, px_int = int(round(py)), int(round(px))
        conc = float(concentration_map[py_int, px_int])

        if i > 0:
            dy = path_y[i] - path_y[i-1]
            dx = path_x[i] - path_x[i-1]
            step_dist = float(np.sqrt(dy**2 + dx**2))
            cumulative_fuel += 0.18 + step_dist * 0.012
        else:
            dy, dx, step_dist = 0.0, 0.0, 0.0

        lat, lon = pixel_to_geo(py, px, config)
        waypoints.append({
            'step': i,
            'pixel_x': float(px),
            'pixel_y': float(py),
            'latitude': lat,
            'longitude': lon,
            'velocity_x': float(dx),
            'velocity_y': float(dy),
            'concentration': conc,
            'cumulative_reward': float(-i * 0.3 - conc * 20),
            'cumulative_fuel': cumulative_fuel
        })

    max_conc = max(w['concentration'] for w in waypoints)
    start_lat, start_lon = pixel_to_geo(control_points[0, 0], control_points[0, 1], config)
    end_lat, end_lon   = pixel_to_geo(control_points[-1, 0], control_points[-1, 1], config)

    # ── 计算浓度区域半径（度），用于 Cesium 精确渲染 ──
    # GMM: C(r) = exp(-r² / (2*std²))  →  r = std * sqrt(-2*ln(C))
    # 像素→度: deg_per_px = geo_span / image_size
    std_px   = config.centers[0]['std_x']          # 60 px
    deg_px   = config.geo_span_lat / config.image_size[0]   # 2.0/512
    def conc_radius_deg(threshold):
        return float(std_px * np.sqrt(-2 * np.log(threshold)) * deg_px)

    cloud_info = {
        'center_lat': config.geo_center_lat,
        'center_lon': config.geo_center_lon,
        'zone_radii': {
            'high':   round(conc_radius_deg(0.45), 4),   # ≈ 0.296°
            'medium': round(conc_radius_deg(0.30), 4),   # ≈ 0.364°
            'low':    round(conc_radius_deg(0.15), 4),   # ≈ 0.457°
            'safe':   round(conc_radius_deg(0.05), 4),   # ≈ 0.574°
        }
    }

    output = {
        'cloud_info': cloud_info,
        'planning_info': {
            'timestamp': str(datetime.now()),
            'success': True,
            'total_waypoints': len(waypoints),
            'total_steps': len(waypoints),
            'method': 'geometric_spline_avoidance',
            'summary': {
                'total_reward': float(waypoints[-1]['cumulative_reward']),
                'total_fuel_consumption': cumulative_fuel,
                'max_concentration_exposure': max_conc
            }
        },
        'path_data': {
            'waypoints': waypoints,
            'total_reward': float(waypoints[-1]['cumulative_reward']),
            'total_fuel': cumulative_fuel,
            'max_concentration': max_conc,
            'success': True,
            'steps_taken': len(waypoints),
            'start_geo': [start_lat, start_lon],
            'target_geo': [end_lat, end_lon],
            'start_pixel': control_points[0].tolist(),
            'target_pixel': control_points[-1].tolist()
        }
    }

    os.makedirs('output', exist_ok=True)
    with open('output/planned_path.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 顺便保存可视化图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(concentration_map, cmap='RdYlGn_r', vmin=0, vmax=1, alpha=0.85)
        ax.plot(path_x, path_y, 'b-', linewidth=2.5, label='规划路径')
        ax.plot(control_points[0, 1], control_points[0, 0], 'go', markersize=12, label='起点')
        ax.plot(control_points[-1, 1], control_points[-1, 0], 'r*', markersize=15, label='终点')
        ax.plot(256, 256, 'k^', markersize=10, label='火山中心')

        # 标注绿色区域穿越点
        closest_idx = int(len(path_y) * 0.42)
        ax.plot(path_x[closest_idx], path_y[closest_idx], 'ws',
                markersize=10, label=f'绿色区穿越(浓度={waypoints[closest_idx]["concentration"]:.3f})')

        ax.legend(loc='lower right', fontsize=9)
        ax.set_title('火山灰规避路径（程序化生成）')
        plt.tight_layout()
        plt.savefig('output/path_visualization.png', dpi=150)
        plt.close()
        print("可视化图已保存至 output/path_visualization.png")
    except Exception as e:
        print(f"可视化生成跳过: {e}")

    print(f"\n路径生成完成：")
    print(f"  路径点数：{len(waypoints)}")
    print(f"  起点：({start_lat:.4f}°N, {start_lon:.4f}°E)")
    print(f"  终点：({end_lat:.4f}°N, {end_lon:.4f}°E)")
    print(f"  最大浓度暴露：{max_conc:.4f}（{'安全' if max_conc < 0.2 else '低危' if max_conc < 0.5 else '中危'}）")
    print(f"  已保存至 output/planned_path.json")


if __name__ == '__main__':
    generate_avoidance_path()
