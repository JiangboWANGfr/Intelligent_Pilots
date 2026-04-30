import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
from datetime import datetime

from src.path_planning.planner import PathPlanner
from src.rl_training.ddpg_agent import DDPGAgent
from src.config.volcanic_ash_config import VolcanicAshConfig


class MultiConstraintPlanner:
    def __init__(self, config: VolcanicAshConfig, agent: DDPGAgent):
        self.base_config = config
        self.base_agent = agent
        self.planner = PathPlanner(config, agent)
    
    def generate_multiple_solutions(self,
                                   start_geo: Tuple[float, float],
                                   target_geo: Tuple[float, float],
                                   risk_tolerance_levels: List[str] = ['low', 'medium', 'high'],
                                   fuel_constraints: List[float] = [50.0, 100.0, 150.0],
                                   max_steps: int = 500) -> Dict:
        
        solutions = []
        solution_id = 1
        
        for risk_level in risk_tolerance_levels:
            modified_config = self._create_risk_adjusted_config(risk_level)
            
            for fuel_limit in fuel_constraints:
                solution = self._generate_single_solution(
                    start_geo, target_geo, modified_config,
                    risk_level, fuel_limit, max_steps, solution_id
                )
                
                solutions.append(solution)
                solution_id += 1
        
        valid_solutions = [s for s in solutions if s['is_valid']]
        valid_solutions.sort(key=lambda x: x['overall_score'], reverse=True)
        
        result = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'start_position': list(start_geo),
                'target_position': list(target_geo),
                'total_generated': len(solutions),
                'valid_solutions': len(valid_solutions)
            },
            'solutions': valid_solutions,
            'comparison_metrics': self._compute_comparison_metrics(valid_solutions)
        }
        
        return result
    
    def _create_risk_adjusted_config(self, risk_level: str) -> VolcanicAshConfig:
        config = deepcopy(self.base_config)
        
        if risk_level == 'low':
            config.concentration_threshold *= 0.5
        elif risk_level == 'medium':
            pass
        elif risk_level == 'high':
            config.concentration_threshold *= 1.5
        
        return config
    
    def _generate_single_solution(self, start_geo, target_geo, config,
                                 risk_level, fuel_limit, max_steps, solution_id):
        
        temp_planner = PathPlanner(config, self.base_agent)
        
        try:
            path_result = temp_planner.plan_path_geo(start_geo, target_geo, max_steps)
            
            is_valid = True
            if path_result['total_fuel'] > fuel_limit:
                is_valid = False
            
            metrics = self._calculate_solution_metrics(
                path_result, risk_level, fuel_limit
            )
            
            solution = {
                'solution_id': solution_id,
                'risk_tolerance': risk_level,
                'fuel_limit': fuel_limit,
                'is_valid': is_valid,
                'path_data': path_result,
                'metrics': metrics,
                'overall_score': self._calculate_overall_score(metrics)
            }
            
        except Exception as e:
            solution = {
                'solution_id': solution_id,
                'risk_tolerance': risk_level,
                'fuel_limit': fuel_limit,
                'is_valid': False,
                'error': str(e),
                'metrics': {},
                'overall_score': -999
            }
        
        return solution
    
    def _calculate_solution_metrics(self, path_result, risk_level,
                                   fuel_limit) -> Dict:
        
        waypoints = path_result.get('waypoints', [])
        
        total_distance = 0
        if len(waypoints) > 1:
            for i in range(1, len(waypoints)):
                dx = waypoints[i]['pixel_x'] - waypoints[i-1]['pixel_x']
                dy = waypoints[i]['pixel_y'] - waypoints[i-1]['pixel_y']
                total_distance += np.sqrt(dx**2 + dy**2)
        
        concentrations = [w['concentration'] for w in waypoints]
        avg_concentration = np.mean(concentrations) if concentrations else 0
        max_concentration = max(concentrations) if concentrations else 0
        high_exposure_time = sum(1 for c in concentrations if c > 0.3)
        
        straight_line_dist = np.sqrt(
            (path_result['target_pixel'][0] - path_result['start_pixel'][0])**2 +
            (path_result['target_pixel'][1] - path_result['start_pixel'][1])**2
        ) if 'target_pixel' in path_result else 1
        
        deviation_ratio = total_distance / max(straight_line_dist, 1)
        
        fuel_used = path_result.get('total_fuel', 0)
        fuel_efficiency = fuel_used / max(fuel_limit, 1)
        
        success_bonus = 100 if path_result.get('success', False) else 0
        
        risk_score = self._calculate_risk_score(avg_concentration, max_concentration,
                                               high_exposure_time, len(waypoints))
        
        return {
            'total_distance': round(total_distance, 2),
            'straight_line_distance': round(straight_line_dist, 2),
            'deviation_ratio': round(deviation_ratio, 3),
            'average_concentration': round(avg_concentration, 4),
            'max_concentration': round(max_concentration, 4),
            'high_exposure_steps': high_exposure_time,
            'fuel_used': round(fuel_used, 2),
            'fuel_remaining': round(max(fuel_limit - fuel_used, 0), 2),
            'fuel_efficiency': round(fuel_efficiency, 3),
            'success': path_result.get('success', False),
            'risk_score': round(risk_score, 2),
            'success_bonus': success_bonus
        }
    
    def _calculate_risk_score(self, avg_conc, max_conc, high_exp_steps, total_steps):
        weight_avg = 0.3
        weight_max = 0.4
        weight_exp = 0.3
        
        avg_score = avg_conc * 100 * weight_avg
        max_score = min(max_conc * 150, 100) * weight_max
        
        exp_ratio = high_exp_steps / max(total_steps, 1)
        exp_score = exp_ratio * 100 * weight_exp
        
        return avg_score + max_score + exp_score
    
    def _calculate_overall_score(self, metrics) -> float:
        if not metrics or not metrics.get('success', False):
            base_score = 0
        else:
            base_score = 50
        
        distance_score = max(0, 30 - metrics.get('deviation_ratio', 1) * 10)
        fuel_score = max(0, 20 * (1 - metrics.get('fuel_efficiency', 1)))
        safety_score = max(0, 30 - metrics.get('risk_score', 50))
        success_bonus = metrics.get('success_bonus', 0)
        
        overall = base_score + distance_score + fuel_score + safety_score + success_bonus
        
        return round(overall, 2)
    
    def _compute_comparison_metrics(self, solutions: List[Dict]) -> Dict:
        if not solutions:
            return {}
        
        all_scores = [s['overall_score'] for s in solutions]
        all_fuels = [s['metrics'].get('fuel_used', 0) for s in solutions]
        all_risks = [s['metrics'].get('risk_score', 0) for s in solutions]
        all_deviations = [s['metrics'].get('deviation_ratio', 0) for s in solutions]
        
        best_by_score = max(solutions, key=lambda x: x['overall_score'])
        safest = min(solutions, key=lambda x: x['metrics'].get('risk_score', 999))
        most_efficient = min(solutions, key=lambda x: x['metrics'].get('fuel_used', 999))
        most_direct = min(solutions, key=lambda x: x['metrics'].get('deviation_ratio', 999))
        
        return {
            'statistics': {
                'avg_score': round(np.mean(all_scores), 2),
                'best_score': round(max(all_scores), 2),
                'worst_score': round(min(all_scores), 2),
                'avg_fuel': round(np.mean(all_fuels), 2),
                'avg_risk': round(np.mean(all_risks), 2),
                'avg_deviation': round(np.mean(all_deviations), 3)
            },
            'recommendations': {
                'best_overall': best_by_score['solution_id'],
                'safest_route': safest['solution_id'],
                'most_fuel_efficient': most_efficient['solution_id'],
                'most_direct_route': most_direct['solution_id']
            }
        }
    
    def export_solutions_json(self, result: Dict, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Solutions exported to: {filepath}")
    
    def generate_comparison_report(self, result: Dict) -> str:
        report_lines = [
            "=" * 80,
            "多约束路径规划方案对比报告",
            "=" * 80,
            f"\n生成时间: {result['generation_info']['timestamp']}",
            f"起点: {result['generation_info']['start_position']}",
            f"终点: {result['generation_info']['target_position']}",
            f"有效方案数: {result['generation_info']['valid_solutions']}",
            "\n" + "-" * 80,
            "各方案详细指标:",
            "-" * 80
        ]
        
        for solution in result['solutions']:
            m = solution['metrics']
            report_lines.extend([
                f"\n【方案 {solution['solution_id']}】",
                f"  风险容忍度: {solution['risk_tolerance']}",
                f"  油料限制: {solution['fuel_limit']}",
                f"  是否有效: {'是' if solution['is_valid'] else '否'}",
                f"  综合评分: {solution['overall_score']:.2f}",
                f"  总距离: {m.get('total_distance', 0):.2f} px",
                f"  航线偏离度: {m.get('deviation_ratio', 0):.3f}",
                f"  平均浓度暴露: {m.get('average_concentration', 0):.4f}",
                f"  最大浓度暴露: {m.get('max_concentration', 0):.4f}",
                f"  高风险暴露步数: {m.get('high_exposure_steps', 0)}",
                f"  油料消耗: {m.get('fuel_used', 0):.2f}",
                f"  风险评分: {m.get('risk_score', 0):.2f}",
                f"  是否成功到达: {'是' if m.get('success', False) else '否'}"
            ])
        
        if result.get('comparison_metrics'):
            comp = result['comparison_metrics']
            stats = comp.get('statistics', {})
            recs = comp.get('recommendations', {})
            
            report_lines.extend([
                "\n" + "=" * 80,
                "统计汇总:",
                "=" * 80,
                f"平均综合评分: {stats.get('avg_score', 0):.2f}",
                f"最佳评分: {stats.get('best_score', 0):.2f}",
                f"平均油料消耗: {stats.get('avg_fuel', 0):.2f}",
                f"平均风险评分: {stats.get('avg_risk', 0):.2f}",
                "\n推荐方案:",
                f"  最佳综合方案: 方案 {recs.get('best_overall', 'N/A')}",
                f"  最安全路线: 方案 {recs.get('safest_route', 'N/A')}",
                f"  最省油路线: 方案 {recs.get('most_fuel_efficient', 'N/A')}",
                f"  最直接路线: 方案 {recs.get('most_direct_route', 'N/A')}"
            ])
        
        report_lines.append("\n" + "=" * 80)
        
        return "\n".join(report_lines)
