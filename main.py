import sys
import os
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.volcanic_ash_config import VolcanicAshConfig, get_preset_configs, get_training_scene_configs
from src.model.gmm_model import GMMVolcanicAshModel
from src.generation.image_generator import StaticImageGenerator, DynamicSimulation
from src.rl_env.volcanic_ash_env import VolcanicAshEnv
from src.rl_training.trainer import Trainer
from src.rl_training.ddpg_agent import DDPGAgent, create_agent, infer_checkpoint_algorithm
from src.path_planning.planner import PathPlanner
from src.path_planning.multi_constraint import MultiConstraintPlanner
from src.path_planning.validation_pipeline import ValidationPipeline
from src.analysis.data_analyzer import DataAnalyzer


def build_agent_for_config(config: VolcanicAshConfig,
                           model_path: str,
                           allow_missing_model: bool = False):
    env = VolcanicAshEnv(config)
    state_dim = len(DDPGAgent.flatten_state(env.reset()[0]))
    algorithm = infer_checkpoint_algorithm(model_path) if os.path.exists(model_path) else 'td3'
    agent = create_agent(algorithm, state_dim=state_dim, action_dim=2)
    if not os.path.exists(model_path):
        if allow_missing_model:
            return None
        raise FileNotFoundError(f'模型文件不存在: {model_path}')
    agent.load_model(model_path)
    return agent


def run_config_demo():
    print("=" * 60)
    print("1. 火山灰云模型参数配置演示")
    print("=" * 60)
    
    presets = get_preset_configs()
    print(f"\n可用预设模型: {list(presets.keys())}")
    
    for name, config in presets.items():
        print(f"\n【{name}】")
        print(f"  模型类型: {config.model_type}")
        print(f"  中心数: {config.num_centers}")
        print(f"  云团尺寸: {config.cloud_size}")
        print(f"  浓度阈值: {config.concentration_threshold}")
        
        config.save(f'output/config_{name}.json')
        print(f"  配置已保存至: output/config_{name}.json")


def run_generation_demo():
    print("\n" + "=" * 60)
    print("2. 火山灰云图像生成演示")
    print("=" * 60)
    
    config = VolcanicAshConfig(
        model_type="double_center",
        num_centers=2,
        cloud_size=100.0,
        concentration_threshold=0.3,
        centers=[
            {'x': 200, 'y': 220, 'weight': 0.6, 'std_x': 50, 'std_y': 45},
            {'x': 320, 'y': 290, 'weight': 0.4, 'std_x': 55, 'std_y': 50}
        ]
    )
    
    print("\n生成静态图像...")
    generator = StaticImageGenerator(config)
    static_results = generator.generate_static_images(output_dir='output/static',
                                                      num_images=5)
    print(f"成功生成: {len(static_results['generated_images'])} 张图像")
    print(f"无效图像: {static_results['failed_count']} 张")
    
    print("\n生成动态仿真序列...")
    simulator = DynamicSimulation(GMMVolcanicAshModel(config))
    dynamic_results = simulator.generate_dynamic_sequence(num_frames=20,
                                                         output_dir='output/dynamic')
    print(f"生成帧数: {len(dynamic_results['frames'])}")


def run_training_demo():
    print("\n" + "=" * 60)
    print("3. 强化学习训练演示")
    print("=" * 60)
    
    scene_configs = get_training_scene_configs()
    config = VolcanicAshConfig.from_dict(scene_configs[0].to_dict())
    config.training_scene_names = [scene.scene_name for scene in scene_configs]
    
    trainer = Trainer(
        config=config,
        num_episodes=300,
        max_steps_per_episode=300,
        learning_rate=1e-4,
        save_dir='models',
        scene_configs=scene_configs
    )
    
    print(f"\n开始训练 ( episodes={trainer.num_episodes} )...")
    agent, history = trainer.train(log_interval=50)
    
    return agent


def run_planning_demo(agent):
    print("\n" + "=" * 60)
    print("4. 路径规划演示")
    print("=" * 60)
    
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

    planner = PathPlanner(config, agent)

    # 起终点对应像素约 (410,102) → (102,410)，对角穿越，云在中央
    start_geo = (34.4, 119.4)
    target_geo = (35.6, 120.6)

    print(f"\n起点坐标: {start_geo}")
    print(f"终点坐标: {target_geo}")

    path_result = planner.plan_path_geo_with_fallback(start_geo, target_geo, max_steps=500)

    print(f"\n规划结果:")
    print(f"  是否成功: {path_result['success']}")
    print(f"  总步数: {path_result['steps_taken']}")
    print(f"  总奖励: {path_result['total_reward']:.2f}")
    print(f"  油料消耗: {path_result['total_fuel']:.2f}")
    print(f"  最大浓度暴露: {path_result.get('max_concentration', 0):.4f}")
    
    output_path = 'output/planned_path.json'
    planner.export_path_json(path_result, output_path)
    print(f"\n路径数据已导出至: {output_path}")
    
    viz_path = 'output/path_visualization.png'
    planner.visualize_path(path_result, save_path=viz_path)
    
    return path_result


def run_multi_constraint_demo(agent):
    print("\n" + "=" * 60)
    print("5. 多约束路径方案演示")
    print("=" * 60)
    
    config = VolcanicAshConfig(
        geo_center_lat=35.0,
        geo_center_lon=120.0,
        centers=[{'x': 256, 'y': 256, 'weight': 1.0, 'std_x': 60, 'std_y': 60}]
    )
    
    multi_planner = MultiConstraintPlanner(config, agent)
    
    start_geo = (34.8, 119.8)
    target_geo = (35.2, 120.2)
    
    print(f"\n生成多约束路径方案...")
    solutions = multi_planner.generate_multiple_solutions(
        start_geo=start_geo,
        target_geo=target_geo,
        risk_tolerance_levels=['low', 'medium', 'high'],
        fuel_constraints=[80.0, 120.0],
        max_steps=350
    )
    
    report = multi_planner.generate_comparison_report(solutions)
    print(report)
    
    multi_planner.export_solutions_json(solutions, 'output/multi_constraint_solutions.json')
    
    return solutions


def run_analysis_demo():
    print("\n" + "=" * 60)
    print("6. 数据分析演示")
    print("=" * 60)
    
    analyzer = DataAnalyzer()
    
    sample_flight_data = {
        'success': True,
        'total_reward': 85.5,
        'total_fuel': 65.3,
        'max_concentration': 0.25,
        'steps_taken': 280,
        'waypoints': [
            {
                'step': i,
                'pixel_x': 100 + i * 0.5,
                'pixel_y': 150 + np.sin(i * 0.05) * 30,
                'velocity_x': 2.0 + np.random.uniform(-0.5, 0.5),
                'velocity_y': np.random.uniform(-1, 1),
                'concentration': max(0, 0.2 - abs(i - 140) / 200 + np.random.uniform(-0.05, 0.05)),
                'cumulative_reward': i * 0.3,
                'cumulative_fuel': i * 0.23
            }
            for i in range(250)
        ]
    }
    
    analyzer.add_simulation_result(sample_flight_data)
    
    analysis = analyzer.analyze_flight_data(sample_flight_data)
    print("\n单次飞行数据分析:")
    for key, value in analysis.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
    
    report = analyzer.generate_comprehensive_report()
    
    text_report = analyzer.format_text_report(report)
    print(text_report)
    
    analyzer.export_report_json(report, 'output/analysis_report.json')
    
    with open('output/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(text_report)


def run_image_validation_demo(image_path: str):
    print("\n" + "=" * 60)
    print("7. 图像验证路径规划演示")
    print("=" * 60)
    
    config = VolcanicAshConfig.load('output/current_config.json') if os.path.exists('output/current_config.json') else VolcanicAshConfig()
    pipeline = ValidationPipeline(config=config, model_path='models/final_model.pth')
    animation_output_dir = 'output/validation_animation'
    animation_gif_path = os.path.join(animation_output_dir, 'validated_path.gif')
    animation_video_path = os.path.join(animation_output_dir, 'validated_path.mp4')
    
    start_geo = (config.geo_center_lat - config.geo_span_lat * 0.35,
                 config.geo_center_lon - config.geo_span_lon * 0.35)
    target_geo = (config.geo_center_lat + config.geo_span_lat * 0.35,
                 config.geo_center_lon + config.geo_span_lon * 0.35)
    
    result = pipeline.validate_image(
        image_source=image_path,
        start_geo=start_geo,
        target_geo=target_geo,
        output_json_path='output/validated_path.json',
        output_plot_path='output/validated_path.png',
        scene_name='cli_image_validation',
        animation_output_dir=animation_output_dir,
        animation_gif_path=animation_gif_path,
        animation_video_path=animation_video_path,
        animation_fps=12,
        animation_max_frames=180
    )
    
    manifest = pipeline.build_animation_export_manifest(result)
    animation_export = result.get('validation_info', {}).get('animation_export', {})
    print(f"  规划方式: {result.get('planning_method')}")
    print(f"  是否成功: {result.get('success')}")
    print(f"  步数: {result.get('steps_taken')}")
    print(f"  最大浓度: {result.get('max_concentration', 0):.4f}")
    print(f"  帧清单数量: {manifest['frame_count']}")
    print(f"  GIF 动画: {animation_export.get('gif_path')}")
    print(f"  视频动画: {animation_export.get('video_path')}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='火山灰云规避决策系统')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['config', 'generation', 'training',
                                'planning', 'multi_constraint', 'analysis', 'validate_image', 'full'],
                       help='运行模式')
    parser.add_argument('--episodes', type=int, default=300,
                       help='训练轮数（仅training/full模式）')
    parser.add_argument('--image-path', type=str, default='',
                       help='图像验证模式使用的输入图像路径')
    
    args = parser.parse_args()
    
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if args.mode == 'full':
        run_config_demo()
        run_generation_demo()
        agent = run_training_demo()
        run_planning_demo(agent)
        run_multi_constraint_demo(agent)
        run_analysis_demo()
        
        print("\n" + "=" * 60)
        print("✓ 所有功能模块演示完成！")
        print("=" * 60)
        print("\n生成的文件:")
        print("  - 配置文件: output/config_*.json")
        print("  - 静态图像: output/static/")
        print("  - 动态仿真: output/dynamic/")
        print("  - 训练模型: models/")
        print("  - 规划路径: output/planned_path.json")
        print("  - 可视化图: output/path_visualization.png")
        print("  - 多约束方案: output/multi_constraint_solutions.json")
        print("  - 分析报告: output/analysis_report.json/.txt")
        
    elif args.mode == 'config':
        run_config_demo()
    elif args.mode == 'generation':
        run_generation_demo()
    elif args.mode == 'training':
        run_training_demo()
    elif args.mode == 'planning':
        config = VolcanicAshConfig.load('output/current_config.json') if os.path.exists('output/current_config.json') else VolcanicAshConfig()
        agent = build_agent_for_config(config, 'models/final_model.pth', allow_missing_model=True)
        run_planning_demo(agent)
    elif args.mode == 'multi_constraint':
        config = VolcanicAshConfig.load('output/current_config.json') if os.path.exists('output/current_config.json') else VolcanicAshConfig()
        agent = build_agent_for_config(config, 'models/final_model.pth')
        run_multi_constraint_demo(agent)
    elif args.mode == 'analysis':
        run_analysis_demo()
    elif args.mode == 'validate_image':
        if not args.image_path:
            raise ValueError('--image-path is required for validate_image mode')
        run_image_validation_demo(args.image_path)


if __name__ == '__main__':
    main()
