# 火山灰云规避决策系统 - 使用说明

## 📋 项目概述
这是一个面向火山灰云环境的智能飞行员最优规避与脱离决策系统，包含8个核心功能模块。

## 🛠️ 技术栈
- **后端**: Python (NumPy, SciPy, Matplotlib, OpenCV, PyTorch, Gymnasium)
- **前端**: CesiumJS, HTML5, CSS3, JavaScript
- **服务端**: Flask

## 📁 项目结构
```
火山灰/
├── main.py                          # 主程序入口（命令行模式）
├── server.py                        # Web API服务器
├── requirements.txt                 # Python依赖包
├── src/                             # 源代码目录
│   ├── config/                      # 配置模块
│   │   └── volcanic_ash_config.py   # GMM模型参数配置
│   ├── model/                       # 模型模块
│   │   └── gmm_model.py            # 火山灰云GMM模型
│   ├── generation/                  # 图像生成模块
│   │   └── image_generator.py      # 静动态图像生成
│   ├── rl_env/                      # 强化学习环境
│   │   └── volcanic_ash_env.py     # Gymnasium仿真环境
│   ├── rl_training/                 # 强化学习训练
│   │   ├── ddpg_agent.py           # DDPG智能体
│   │   └── trainer.py              # 训练器
│   ├── path_planning/               # 路径规划
│   │   ├── planner.py              # 智能路径规划
│   │   └── multi_constraint.py     # 多约束方案生成
│   └── analysis/                    # 数据分析
│       └── data_analyzer.py        # 仿真数据分析
├── web/                             # 前端目录
│   └── index.html                   # CesiumJS 3D可视化页面
├── output/                          # 输出数据目录
├── models/                          # 训练模型存储
└── 项目要求文档                     # 项目需求说明
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行完整演示（命令行）
```bash
# 运行所有功能模块的完整演示
python main.py --mode full --episodes 300

# 单独运行某个功能模块
python main.py --mode config          # 参数配置演示
python main.py --mode generation      # 图像生成演示
python main.py --mode training        # 强化学习训练
python main.py --mode planning        # 路径规划演示
python main.py --mode multi_constraint # 多约束规划演示
python main.py --mode analysis        # 数据分析演示
```

### 3. 启动Web服务
```bash
python server.py
```
然后访问: http://localhost:5000

## 📊 功能模块详解

### 1️⃣ 火山灰云模型参数配置
- **文件**: `src/config/volcanic_ash_config.py`
- **功能**:
  - 支持单中心、双中心、三中心、环形4种预设模型
  - 可配置GMM参数：中心数、云团尺寸、浓度阈值、质量占比等
  - 支持参数保存与加载（JSON格式）

### 2️⃣ 火山灰云静动态图像生成
- **文件**: `src/generation/image_generator.py`, `src/model/gmm_model.py`
- **功能**:
  - 基于高斯混合模型（GMM）生成火山灰云浓度分布
  - 输出灰度图和四色危险区划图（绿-黄-橙-红）
  - 自动筛选无效图像并本地保存
  - 动态仿真：模拟云团位移、形变、浓度衰减
  - 导出地理坐标和浓度数据为JSON格式

### 3️⃣ 强化学习模型训练
- **文件**: `src/rl_env/volcanic_ash_env.py`, `src/rl_training/ddpg_agent.py`, `src/rl_training/trainer.py`
- **功能**:
  - 基于PyTorch实现DDPG算法（深度确定性策略梯度）
  - 自定义Gymnasium仿真环境，包含：
    - 飞机感知范围（9点局部浓度检测）
    - 飞行速度限制（最大15单位/步）
    - 安全阈值和危险阈值判定
  - 可配置奖励函数、学习率、训练轮数
  - 实时展示训练损失、奖励值变化曲线
  - 自动保存训练模型和训练历史

### 4️⃣ 飞机规避路径智能规划
- **文件**: `src/path_planning/planner.py`
- **功能**:
  - 输入飞机初始位置和目标航线的地理坐标
  - 调用训练好的DDPG模型进行路径规划
  - 输出最优规避路径（避开高浓度区域）
  - 导出路径坐标、航点信息为JSON格式
  - 本地可视化预览（Matplotlib）

### 5️⃣ 多约束路径方案生成
- **文件**: `src/path_planning/multi_constraint.py`
- **功能**:
  - 根据风险容忍度（低/中/高）生成差异化方案
  - 根据油料消耗限制筛选可行路径
  - 计算各方案的指标：
    - 风险等级评分
    - 油料预估消耗
    - 航线偏离度
    - 浓度暴露统计
  - 自动推荐最佳综合方案、最安全路线、最省油路线等
  - 所有方案数据导出为JSON格式

### 6️⃣ 3D地理空间可视化渲染
- **文件**: `web/index.html`
- **功能**:
  - 基于CesiumJS搭建Web端3D可视化页面
  - 加载在线地图底图（Cesium World Terrain）
  - 渲染火山灰云的静动态分布（颜色编码浓度等级）
  - 显示原始航线与规划后的规避路径
  - 支持地图缩放、平移、视角切换（3D/2D）

### 7️⃣ 飞机动态飞行仿真演示
- **文件**: `web/index.html` (JavaScript部分)
- **功能**:
  - 在3D场景中沿规划路径动态飞行
  - 同步展示实时信息：
    - 经纬度坐标
    - 飞行速度
    - 所处区域火山灰浓度
    - 飞行状态（安全/低风险/中风险/高风险）
  - 支持播放、暂停、停止控制
  - 支持0.5x-5x调速

### 8️⃣ 仿真结果数据与分析
- **文件**: `src/analysis/data_analyzer.py`
- **功能**:
  - 统计飞行数据：
    - 总航程距离、总时长
    - 平均/最大/最小飞行速度
    - 油料消耗总量
  - 分析浓度暴露情况：
    - 平均/最大浓度暴露
    - 各危险区域停留时间占比
    - 高风险暴露积分
  - 路径规划核心指标：
    - 成功率统计
    - 奖励值分布
    - 油料效率
  - 生成结构化JSON报表和文本报告

## 🔧 API接口说明

启动Web服务后，可通过以下API调用：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/presets | 获取预设配置 |
| POST | /api/generate | 生成火山灰云图像 |
| POST | /api/train | 训练强化学习模型 |
| POST | /api/plan | 规划规避路径 |
| POST | /api/multi-plan | 多约束路径规划 |
| POST | /api/analyze | 数据分析 |

## 📈 输出文件说明

运行后会生成以下输出：

```
output/
├── static/                    # 静态图像
│   ├── *_grayscale.png       # 灰度图
│   ├── *_danger.png          # 危险区划图
│   └── generation_summary.json
├── dynamic/                   # 动态仿真序列
│   ├── frame_XXX_*.png       # 各帧图像
│   └── dynamic_data.json     # 动态数据
├── planned_path.json         # 规划路径数据
├── multi_constraint_solutions.json  # 多约束方案
├── analysis_report.json      # JSON分析报告
└── analysis_report.txt       # 文本分析报告

models/
├── final_model.pth           # 最终训练模型
├── checkpoint_ep*.pth        # 中间检查点
├── training_history.json     # 训练历史
└── training_curves.png       # 训练曲线图
```

## ⚠️ 注意事项

1. **Cesium Token**: 在`web/index.html`第277行替换为您自己的Cesium Ion Access Token
2. **GPU支持**: 如有NVIDIA GPU，建议安装CUDA版本的PyTorch以加速训练
3. **内存需求**: 图像生成和强化学习训练需要较大内存（建议8GB+）
4. **浏览器兼容**: 3D可视化页面需要现代浏览器（Chrome/Firefox/Edge最新版）

## 💡 使用建议

1. **首次使用**: 先运行 `python main.py --mode full` 完成全流程演示
2. **仅查看可视化**: 启动 `python server.py` 后访问 http://localhost:5000
3. **自定义训练**: 修改 `main.py` 中的训练参数（episodes、learning_rate等）
4. **调整模型**: 通过修改预设配置或自定义GMM参数来改变火山灰云形状

## 🐛 常见问题

**Q: 训练时间过长怎么办？**
A: 减少 `--episodes` 参数（如100-200轮），或降低 `max_steps_per_episode`

**Q: Cesium地图不显示？**
A: 检查网络连接，确保能访问Cesium CDN；检查Token是否有效

**Q: 内存不足？**
A: 减少生成的图像数量（num_images）和动态帧数（dynamic_frames）

**Q: 如何提高路径规划效果？**
A: 增加训练轮数，调整奖励函数权重，优化网络结构
