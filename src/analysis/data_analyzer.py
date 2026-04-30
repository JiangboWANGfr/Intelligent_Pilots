import json
import os
from typing import Dict, List
from datetime import datetime
import numpy as np


class DataAnalyzer:
    def __init__(self):
        self.simulation_results = []
        self.analysis_cache = {}
    
    def add_simulation_result(self, result: Dict):
        self.simulation_results.append({
            'timestamp': datetime.now().isoformat(),
            'data': result
        })
    
    def analyze_flight_data(self, flight_data: Dict) -> Dict:
        waypoints = flight_data.get('waypoints', [])
        
        if not waypoints:
            return {'error': 'No flight data available'}
        
        positions = [(w['pixel_x'], w['pixel_y']) for w in waypoints]
        velocities = [(w['velocity_x'], w['velocity_y']) for w in waypoints]
        concentrations = [w['concentration'] for w in waypoints]
        speeds = [np.sqrt(v[0]**2 + v[1]**2) for v in velocities]
        rewards = [w['cumulative_reward'] for w in waypoints]
        fuels = [w['cumulative_fuel'] for w in waypoints]
        
        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        time_in_safe_zone = sum(1 for c in concentrations if c < 0.2)
        time_in_low_risk = sum(1 for c in concentrations if 0.2 <= c < 0.4)
        time_in_medium_risk = sum(1 for c in concentrations if 0.4 <= c < 0.6)
        time_in_high_risk = sum(1 for c in concentrations if c >= 0.6)
        total_time = len(concentrations)
        
        analysis = {
            'flight_summary': {
                'total_duration': total_time,
                'total_distance_km': round(total_distance * 0.1, 2),
                'success': flight_data.get('success', False),
                'final_reward': flight_data.get('total_reward', 0),
                'total_fuel_consumption': flight_data.get('total_fuel', 0)
            },
            'speed_analysis': {
                'avg_speed': round(np.mean(speeds), 3),
                'max_speed': round(np.max(speeds), 3),
                'min_speed': round(np.min(speeds), 3),
                'speed_std': round(np.std(speeds), 3)
            },
            'concentration_analysis': {
                'avg_concentration': round(np.mean(concentrations), 4),
                'max_concentration': round(np.max(concentrations), 4),
                'min_concentration': round(np.min(concentrations), 4),
                'time_above_threshold_03': sum(1 for c in concentrations if c > 0.3),
                'time_above_threshold_05': sum(1 for c in concentrations if c > 0.5),
                'integral_exposure': round(sum(concentrations), 2)
            },
            'risk_distribution': {
                'safe_zone_percent': round(time_in_safe_zone/total_time*100, 2) if total_time > 0 else 0,
                'low_risk_percent': round(time_in_low_risk/total_time*100, 2) if total_time > 0 else 0,
                'medium_risk_percent': round(time_in_medium_risk/total_time*100, 2) if total_time > 0 else 0,
                'high_risk_percent': round(time_in_high_risk/total_time*100, 2) if total_time > 0 else 0
            },
            'efficiency_metrics': {
                'fuel_efficiency': round(flight_data.get('total_fuel', 0) / max(total_distance, 1), 3),
                'reward_per_step': round(flight_data.get('total_reward', 0) / max(total_time, 1), 2),
                'distance_per_step': round(total_distance / max(total_time, 1), 2)
            }
        }
        
        return analysis
    
    def generate_comprehensive_report(self, simulation_results: List[Dict] = None) -> Dict:
        if simulation_results is None:
            simulation_results = self.simulation_results
        
        if not simulation_results:
            return {'error': 'No simulation data available'}
        
        individual_analyses = []
        for sim in simulation_results:
            analysis = self.analyze_flight_data(sim['data'])
            analysis['simulation_timestamp'] = sim['timestamp']
            individual_analyses.append(analysis)
        
        aggregated_stats = self._aggregate_statistics(individual_analyses)
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_simulations': len(simulation_results),
                'report_type': 'comprehensive_analysis'
            },
            'individual_analyses': individual_analyses,
            'aggregated_statistics': aggregated_stats,
            'recommendations': self._generate_recommendations(aggregated_stats)
        }
        
        return report
    
    def _aggregate_statistics(self, analyses: List[Dict]) -> Dict:
        successes = [a for a in analyses if a.get('flight_summary', {}).get('success', False)]
        
        avg_distances = [a['flight_summary']['total_distance_km'] for a in analyses]
        avg_fuels = [a['flight_summary']['total_fuel_consumption'] for a in analyses]
        avg_rewards = [a['flight_summary']['final_reward'] for a in analyses]
        avg_concentrations = [a['concentration_analysis']['avg_concentration'] for a in analyses]
        high_risk_percents = [a['risk_distribution']['high_risk_percent'] for a in analyses]
        
        return {
            'success_rate': {
                'count': len(successes),
                'percentage': round(len(successes)/len(analyses)*100, 2) if analyses else 0
            },
            'performance_averages': {
                'avg_distance': round(np.mean(avg_distances), 2),
                'std_distance': round(np.std(avg_distances), 2),
                'avg_fuel': round(np.mean(avg_fuels), 2),
                'avg_reward': round(np.mean(avg_rewards), 2),
                'avg_concentration': round(np.mean(avg_concentrations), 4)
            },
            'safety_metrics': {
                'avg_high_risk_exposure': round(np.mean(high_risk_percents), 2),
                'worst_case_high_risk': round(max(high_risk_percents), 2) if high_risk_percents else 0,
                'best_case_high_risk': round(min(high_risk_percents), 2) if high_risk_percents else 0
            },
            'extremes': {
                'best_reward': round(max(avg_rewards), 2) if avg_rewards else 0,
                'worst_reward': round(min(avg_rewards), 2) if avg_rewards else 0,
                'longest_flight': round(max(avg_distances), 2) if avg_distances else 0,
                'shortest_flight': round(min(avg_distances), 2) if avg_distances else 0
            }
        }
    
    def _generate_recommendations(self, stats: Dict) -> List[str]:
        recommendations = []
        
        success_rate = stats.get('success_rate', {}).get('percentage', 0)
        if success_rate < 70:
            recommendations.append("成功率较低，建议增加训练轮数或调整奖励函数参数")
        elif success_rate >= 90:
            recommendations.append("模型表现优秀，成功率超过90%")
        
        avg_high_risk = stats.get('safety_metrics', {}).get('avg_high_risk_exposure', 0)
        if avg_high_risk > 15:
            recommendations.append("高风险区域暴露时间较长，建议调整安全阈值或优化规避策略")
        
        avg_fuel = stats.get('performance_averages', {}).get('avg_fuel', 0)
        if avg_fuel > 100:
            recommendations.append("油料消耗偏高，可考虑优化路径平滑性以减少机动消耗")
        
        if not recommendations:
            recommendations.append("当前模型表现良好，各项指标均在合理范围内")
        
        return recommendations
    
    def export_report_json(self, report: Dict, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"分析报告已导出至: {filepath}")
    
    def format_text_report(self, report: Dict) -> str:
        lines = [
            "=" * 90,
            "火山灰云规避飞行仿真数据分析报告",
            "=" * 90,
            f"\n报告生成时间: {report['report_metadata']['generated_at']}",
            f"仿真次数总计: {report['report_metadata']['total_simulations']}"
        ]
        
        agg = report.get('aggregated_statistics', {})
        if agg:
            sr = agg.get('success_rate', {})
            lines.extend([
                "\n" + "-" * 90,
                "一、总体统计",
                "-" * 90,
                f"成功到达次数: {sr.get('count', 0)}",
                f"成功率: {sr.get('percentage', 0)}%"
            ])
            
            perf = agg.get('performance_averages', {})
            lines.extend([
                "\n" + "-" * 90,
                "二、性能指标（平均值）",
                "-" * 90,
                f"飞行距离: {perf.get('avg_distance', 0)} km (标准差: {perf.get('std_distance', 0)})",
                f"油料消耗: {perf.get('avg_fuel', 0)}",
                f"最终奖励值: {perf.get('avg_reward', 0)}",
                f"平均浓度暴露: {perf.get('avg_concentration', 0)}"
            ])
            
            safety = agg.get('safety_metrics', {})
            lines.extend([
                "\n" + "-" * 90,
                "三、安全性指标",
                "-" * 90,
                f"高风险区域平均暴露比例: {safety.get('avg_high_risk_exposure', 0)}%",
                f"最差情况高风险暴露: {safety.get('worst_case_high_risk', 0)}%",
                f"最佳情况高风险暴露: {safety.get('best_case_high_risk', 0)}%"
            ])
            
            extremes = agg.get('extremes', {})
            lines.extend([
                "\n" + "-" * 90,
                "四、极值统计",
                "-" * 90,
                f"最高奖励: {extremes.get('best_reward', 0)}",
                f"最低奖励: {extremes.get('worst_reward', 0)}",
                f"最长航程: {extremes.get('longest_flight', 0)} km",
                f"最短航程: {extremes.get('shortest_flight', 0)} km"
            ])
        
        recs = report.get('recommendations', [])
        if recs:
            lines.extend([
                "\n" + "-" * 90,
                "五、建议与改进方向",
                "-" * 90
            ])
            for i, rec in enumerate(recs, 1):
                lines.append(f"{i}. {rec}")
        
        lines.extend(["\n" + "=" * 90])
        
        return "\n".join(lines)
