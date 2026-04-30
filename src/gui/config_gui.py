import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from src.config.volcanic_ash_config import (
    VolcanicAshConfig,
    get_preset_configs,
    CloudModelType
)
from src.model.gmm_model import GMMVolcanicAshModel


class VolcanicAshConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("火山灰云模型 - 参数配置界面 (含不规则形状)")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 800)

        self.current_config = None
        self.preview_image_tk = None
        self.danger_image_tk = None
        self.preview_grayscale_array = None
        self.preview_danger_array = None
        self.current_volcano_name = ""

        style = ttk.Style()
        style.theme_use('clam')

        self._setup_ui()
        self._load_preset("单中心模型")

        self.center_editor_window = None

    def _setup_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_paned)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        main_paned.add(right_frame, weight=2)

        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)

    def _build_left_panel(self, parent):
        canvas = tk.Canvas(parent, width=480)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._build_model_type_section(scrollable_frame)
        self._build_basic_params_section(scrollable_frame)
        self._build_geo_params_section(scrollable_frame)
        self._build_irregular_params_section(scrollable_frame)  # 新增
        self._build_center_editor_section(scrollable_frame)
        self._build_action_buttons(scrollable_frame)

    def _build_model_type_section(self, parent):
        frame = ttk.LabelFrame(parent, text="模型类型（预设）", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        self.model_type_var = tk.StringVar(value="single_center")

        presets = [
            ("单中心模型", "single_center"),
            ("双中心模型", "double_center"),
            ("三中心模型", "triple_center"),
            ("环形模型", "ring")
        ]

        for i, (text, value) in enumerate(presets):
            rb = ttk.Radiobutton(
                frame, text=text, value=value,
                variable=self.model_type_var,
                command=self._on_preset_change
            )
            rb.grid(row=i // 2, column=i % 2, sticky="w", padx=5, pady=3)

    def _build_basic_params_section(self, parent):
        frame = ttk.LabelFrame(parent, text="基础参数", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        params = [
            ("高斯中心数：", "num_centers", 1, 8, 1),
            ("云团尺寸：", "cloud_size", 50, 300, 80.0),
            ("浓度阈值：", "concentration_threshold", 0.05, 1.0, 0.3),
            ("质量占比：", "mass_ratio", 0.3, 1.0, 0.8),
            ("图像宽度：", "img_width", 256, 1024, 512),
            ("图像高度：", "img_height", 256, 1024, 512),
        ]

        self.param_vars = {}
        for row, (label, key, min_val, max_val, default) in enumerate(params):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)

            var = tk.DoubleVar(value=default)
            self.param_vars[key] = var

            scale = ttk.Scale(
                frame, from_=min_val, to=max_val,
                variable=var, orient=tk.HORIZONTAL,
                command=lambda v, k=key: self._on_param_change(k)
            )
            scale.grid(row=row, column=1, sticky="ew", padx=5)

            entry = ttk.Entry(frame, textvariable=var, width=8)
            entry.grid(row=row, column=2, padx=5)
            entry.bind("<Return>", lambda e, k=key: self._on_param_change(k))

        frame.columnconfigure(1, weight=1)

    def _build_geo_params_section(self, parent):
        frame = ttk.LabelFrame(parent, text="地理坐标参数", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        select_frame = ttk.Frame(frame)
        select_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(select_frame, text="🌋 快速选地点：").pack(side=tk.LEFT)
        self.volcano_combo = ttk.Combobox(
            select_frame, values=list(self._get_volcano_presets().keys()),
            state="readonly", width=22
        )
        self.volcano_combo.pack(side=tk.LEFT, padx=5)
        self.volcano_combo.bind("<<ComboboxSelected>>", self._on_volcano_selected)

        ttk.Button(select_frame, text="应用", command=self._on_volcano_selected, width=6).pack(side=tk.LEFT)

        grid_frame = ttk.Frame(frame)
        grid_frame.pack(fill=tk.X, pady=(4, 0))

        geo_params = [
            ("中心纬度：", "geo_center_lat", -90.0, 90.0, 35.36),
            ("中心经度：", "geo_center_lon", -180.0, 180.0, 138.73),
            ("纬度跨度：", "geo_span_lat", 0.1, 20.0, 1.5),
            ("经度跨度：", "geo_span_lon", 0.1, 20.0, 1.5),
        ]

        self.geo_vars = {}
        for row, (label, key, min_val, max_val, default) in enumerate(geo_params):
            ttk.Label(grid_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)

            var = tk.DoubleVar(value=default)
            self.geo_vars[key] = var

            scale = ttk.Scale(
                grid_frame, from_=min_val, to=max_val,
                variable=var, orient=tk.HORIZONTAL,
                command=lambda v, k=key: self._on_geo_change(k)
            )
            scale.grid(row=row, column=1, sticky="ew", padx=5)

            entry = ttk.Entry(grid_frame, textvariable=var, width=10)
            entry.grid(row=row, column=2, padx=5)
            entry.bind("<Return>", lambda e, k=key: self._on_geo_change(k))

        grid_frame.columnconfigure(1, weight=1)

        coord_display = ttk.Frame(frame)
        coord_display.pack(fill=tk.X, pady=(8, 0))
        self.coord_label = ttk.Label(coord_display, text="📍 当前位置: 纬度 35.3606°N, 经度 138.7274°E",
                                     font=("Microsoft YaHei", 9, "bold"), foreground="#d35400")
        self.coord_label.pack(side=tk.LEFT)

    def _build_irregular_params_section(self, parent):
        """构建不规则形状参数配置区域 - 新增功能"""
        frame = ttk.LabelFrame(parent, text="🌪️ 不规则形状参数（新功能）", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # 启用开关
        enable_frame = ttk.Frame(frame)
        enable_frame.pack(fill=tk.X, pady=(0, 8))

        self.enable_irregular_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            enable_frame,
            text="✓ 启用不规则形状生成",
            variable=self.enable_irregular_var,
            command=self._on_irregular_enable_change
        ).pack(side=tk.LEFT)

        ttk.Label(
            enable_frame,
            text="（使火山灰云更真实）",
            font=("Microsoft YaHei", 8),
            foreground="#7f8c8d"
        ).pack(side=tk.LEFT, padx=5)

        # 参数配置区域
        self.irregular_params_frame = ttk.Frame(frame)
        self.irregular_params_frame.pack(fill=tk.X)

        # 参数列表
        irregular_params = [
            ("扰动强度：", "turbulence_scale", 0.0, 0.3, 0.15, "控制形状不规则程度"),
            ("风向角度(°)：", "wind_direction", 0.0, 360.0, 45.0, "风吹方向(0=右,90=上)"),
            ("风力强度：", "wind_strength", 0.0, 1.0, 0.3, "控制拉伸程度"),
            ("分形维度：", "fractal_dimension", 1.0, 2.0, 1.5, "控制边界粗糙度"),
            ("细丝数量：", "num_filaments", 0, 20, 5, "添加分支状细丝数量"),
        ]

        self.irregular_vars = {}
        for row, (label, key, min_val, max_val, default, tooltip) in enumerate(irregular_params):
            ttk.Label(self.irregular_params_frame, text=label).grid(
                row=row, column=0, sticky="w", pady=3
            )

            var = tk.DoubleVar(value=default)
            self.irregular_vars[key] = var

            scale = ttk.Scale(
                self.irregular_params_frame,
                from_=min_val, to=max_val,
                variable=var,
                orient=tk.HORIZONTAL,
                command=lambda v, k=key: self._on_irregular_param_change(k)
            )
            scale.grid(row=row, column=1, sticky="ew", padx=5)

            entry = ttk.Entry(self.irregular_params_frame, textvariable=var, width=8)
            entry.grid(row=row, column=2, padx=5)
            entry.bind("<Return>", lambda e, k=key: self._on_irregular_param_change(k))

            hint = ttk.Label(
                self.irregular_params_frame,
                text=tooltip,
                font=("Microsoft YaHei", 7),
                foreground="#95a5a6"
            )
            hint.grid(row=row, column=3, sticky="w", padx=5)

        self.irregular_params_frame.columnconfigure(1, weight=1)

        # 复选框
        checkbox_frame = ttk.Frame(self.irregular_params_frame)
        checkbox_frame.grid(row=len(irregular_params), column=0, columnspan=4,
                           sticky="w", pady=(8, 0))

        self.add_fractal_var = tk.BooleanVar(value=True)
        self.add_filaments_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            checkbox_frame,
            text="分形边界",
            variable=self.add_fractal_var,
            command=lambda: self._on_irregular_param_change(None)
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Checkbutton(
            checkbox_frame,
            text="细丝结构",
            variable=self.add_filaments_var,
            command=lambda: self._on_irregular_param_change(None)
        ).pack(side=tk.LEFT)

        # 随机种子
        seed_frame = ttk.Frame(self.irregular_params_frame)
        seed_frame.grid(row=len(irregular_params)+1, column=0, columnspan=4,
                       sticky="w", pady=(5, 0))

        ttk.Label(seed_frame, text="随机种子：").pack(side=tk.LEFT)
        self.random_seed_var = tk.IntVar(value=42)
        ttk.Entry(seed_frame, textvariable=self.random_seed_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(
            seed_frame,
            text="（可选，用于可重复生成）",
            font=("Microsoft YaHei", 7),
            foreground="#95a5a6"
        ).pack(side=tk.LEFT)

        # 预设快捷按钮
        preset_frame = ttk.LabelFrame(frame, text="快速预设", padding=5)
        preset_frame.pack(fill=tk.X, pady=(8, 0))

        presets = [
            ("轻度", {"turbulence_scale": 0.1, "wind_strength": 0.2, "num_filaments": 3, "fractal_dimension": 1.3}),
            ("中等", {"turbulence_scale": 0.15, "wind_strength": 0.3, "num_filaments": 5, "fractal_dimension": 1.5}),
            ("强烈", {"turbulence_scale": 0.22, "wind_strength": 0.42, "num_filaments": 12, "fractal_dimension": 1.75}),
        ]

        for i, (name, params) in enumerate(presets):
            ttk.Button(
                preset_frame,
                text=name,
                command=lambda p=params: self._apply_irregular_preset(p),
                width=8
            ).grid(row=0, column=i, padx=2)

    def _on_irregular_enable_change(self):
        """启用/禁用不规则参数时的回调"""
        enabled = self.enable_irregular_var.get()

        # 启用/禁用所有不规则参数控件
        state = "normal" if enabled else "disabled"
        for child in self.irregular_params_frame.winfo_children():
            try:
                if hasattr(child, 'configure'):
                    child.configure(state=state)
            except:
                pass
            # 递归处理子控件
            for subchild in child.winfo_children():
                try:
                    if hasattr(subchild, 'configure'):
                        subchild.configure(state=state)
                except:
                    pass

        self._sync_config_from_ui()
        self._update_info_text()

    def _on_irregular_param_change(self, key=None):
        """不规则参数改变时的回调"""
        self._sync_config_from_ui()
        self._update_info_text()

    def _apply_irregular_preset(self, params):
        """应用不规则预设参数"""
        for key, value in params.items():
            if key in self.irregular_vars:
                self.irregular_vars[key].set(value)
        self._sync_config_from_ui()
        self._update_info_text()
        messagebox.showinfo("预设应用成功", f"已应用不规则参数预设")

    def _get_volcano_presets(self):
        return {
            "-- 手动输入 --": (35.36, 138.73, 1.5, 1.5, "日本富士山"),
            "🗻 日本·富士山": (35.3606, 138.7274, 1.5, 1.5, "日本静冈县/山梨县"),
            "🏔️ 中国·长白山天池": (42.008, 128.055, 1.2, 1.2, "中国吉林/朝鲜边境"),
            "🌋 意大利·维苏威火山": (40.8219, 14.4291, 1.0, 1.0, "意大利那不勒斯"),
            "🔥 美国·圣海伦斯火山": (46.1914, -122.1956, 1.5, 1.5, "美国华盛顿州"),
            "🌊 冰岛·埃亚菲亚德拉": (63.6301, -19.6189, 2.0, 2.0, "冰岛南部"),
            "⛰️ 印尼·喀拉喀托": (-6.1025, 105.4230, 1.3, 1.3, "印尼巽他海峡"),
            "🌑 日本·樱岛火山": (31.5874, 130.6125, 1.0, 1.0, "日本鹿儿岛"),
            "🏔️ 菲律宾·皮纳图博": (15.1333, 120.3500, 1.5, 1.5, "菲律宾吕宋岛"),
            "🌋 智利·比亚里卡": (-39.4233, -71.9342, 1.5, 1.5, "智利南部"),
            "🔴 厄瓜多尔·科托帕希": (-0.6772, -78.5361, 1.2, 1.2, "厄瓜多尔安第斯山脉"),
        }

    def _on_volcano_selected(self, event=None):
        name = self.volcano_combo.get()
        presets = self._get_volcano_presets()

        if name in presets:
            lat, lon, span_lat, span_lon, desc = presets[name]
            if name == "-- 手动输入 --":
                self.current_volcano_name = ""
                return
            self.current_volcano_name = name
            self.geo_vars["geo_center_lat"].set(lat)
            self.geo_vars["geo_center_lon"].set(lon)
            self.geo_vars["geo_span_lat"].set(span_lat)
            self.geo_vars["geo_span_lon"].set(span_lon)
            self._sync_config_from_ui()
            self._update_coord_display()
            self._update_info_text()
            self._draw_map_preview()

    def _build_center_editor_section(self, parent):
        frame = ttk.LabelFrame(parent, text="GMM高斯中心编辑器", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Button(btn_frame, text="+ 添加中心", command=self._add_center).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="- 删除选中", command=self._remove_center).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="✏ 编辑选中", command=self._edit_center).pack(side=tk.LEFT, padx=2)

        columns = ("编号", "X坐标", "Y坐标", "权重", "标准差X", "标准差Y")
        self.center_tree = ttk.Treeview(frame, columns=columns, show="headings", height=6)

        for col in columns:
            self.center_tree.heading(col, text=col)
            self.center_tree.column(col, width=70, anchor="center")

        self.center_tree.pack(fill=tk.BOTH, expand=True)

    def _build_action_buttons(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=10)

        buttons = [
            ("🔍 预览效果", self._preview_config),
            ("💾 保存配置", self._save_config),
            ("📂 加载配置", self._load_config),
            ("📤 导出JSON", self._export_json),
            ("✅ 应用并关闭", self._apply_and_close),
        ]

        for text, command in buttons:
            btn = ttk.Button(frame, text=text, command=command, width=14)
            btn.pack(side=tk.LEFT, padx=3, expand=True, fill=tk.X)

    def _build_right_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        preview_tab = ttk.Frame(notebook)
        danger_tab = ttk.Frame(notebook)
        map_tab = ttk.Frame(notebook)
        info_tab = ttk.Frame(notebook)

        notebook.add(preview_tab, text="灰度图预览")
        notebook.add(danger_tab, text="危险区划图")
        notebook.add(map_tab, text="📍 地图位置")
        notebook.add(info_tab, text="配置信息")

        self._build_preview_canvas(preview_tab, "grayscale")
        self._build_preview_canvas(danger_tab, "danger")
        self._build_map_canvas(map_tab)
        self._build_info_panel(info_tab)

    def _build_preview_canvas(self, parent, name):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(frame, bg="#1a1a2e", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.bind("<Configure>", lambda e, preview_name=name: self._refresh_preview_canvas(preview_name))

        if name == "grayscale":
            self.grayscale_canvas = canvas
        else:
            self.danger_canvas = canvas

    def _refresh_preview_canvas(self, name):
        if name == "grayscale":
            if self.preview_grayscale_array is not None and hasattr(self, 'grayscale_canvas'):
                self._display_image_on_canvas(self.preview_grayscale_array, self.grayscale_canvas)
        else:
            if self.preview_danger_array is not None and hasattr(self, 'danger_canvas'):
                self._display_image_on_canvas(self.preview_danger_array, self.danger_canvas, is_rgb=True)

    def _build_info_panel(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.info_text = tk.Text(frame, wrap=tk.WORD, font=("Microsoft YaHei", 10), bg="#f5f5f5")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)

        self.info_text.pack(side="left", fill=tk.BOTH, expand=True)
        scrollbar.pack(side="right", fill=tk.Y)

    def _build_map_canvas(self, parent):
        outer_frame = ttk.Frame(parent)
        outer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        info_bar = ttk.Frame(outer_frame)
        info_bar.pack(fill=tk.X, pady=(0, 5))

        self.map_info_label = ttk.Label(info_bar, text="", font=("Microsoft YaHei", 9))
        self.map_info_label.pack(side=tk.LEFT)

        self.map_canvas = tk.Canvas(outer_frame, bg="#1a2a3a", highlightthickness=0)
        self.map_canvas.pack(fill=tk.BOTH, expand=True)
        self.map_canvas.bind("<Configure>", lambda e: self._draw_map_preview())

    def _draw_map_preview(self):
        if not hasattr(self, 'map_canvas'):
            return

        self.map_canvas.delete("all")
        canvas = self.map_canvas
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()

        if cw < 50 or ch < 50:
            return

        lat = self.geo_vars["geo_center_lat"].get()
        lon = self.geo_vars["geo_center_lon"].get()
        span_lat = max(self.geo_vars["geo_span_lat"].get(), 0.5)
        span_lon = max(self.geo_vars["geo_span_lon"].get(), 0.5)

        margin = 40
        map_w = cw - 2 * margin
        map_h = ch - 2 * margin
        cx = cw / 2
        cy = ch / 2

        canvas.create_rectangle(margin, margin, cw - margin, ch - margin,
                                outline="#3498db", width=2, fill="#0d1b2a")

        for i in range(5):
            y = margin + (map_h * (i + 1)) / 6
            canvas.create_line(margin, y, cw - margin, y, fill="#1e3a5f", dash=(2, 4))
            lat_val = lat + span_lat / 2 - span_lat * (i + 1) / 6
            canvas.create_text(margin - 5, y, text=f"{lat_val:.1f}°",
                               anchor="e", fill="#7fb3d5", font=("Consolas", 8))

        for i in range(6):
            x = margin + map_w * (i + 1) / 7
            canvas.create_line(x, margin, x, ch - margin, fill="#1e3a5f", dash=(2, 4))
            lon_val = lon - span_lon / 2 + span_lon * (i + 1) / 7
            direction = "E" if lon_val >= 0 else "W"
            canvas.create_text(x, ch - margin + 12, text=f"{abs(lon_val):.1f}°{direction}",
                               anchor="n", fill="#7fb3d5", font=("Consolas", 8))

        radius_x = min(map_w * 0.15, 60)
        radius_y = min(map_h * 0.15, 60)

        for r in range(int(radius_y), 0, -10):
            alpha = int(80 * (1 - r / radius_y))
            color = f"#{alpha:02x}2020"
            canvas.create_oval(
                cx - r * radius_x / radius_y, cy - r,
                cx + r * radius_x / radius_y, cy + r,
                fill=color, outline=""
            )

        canvas.create_oval(
            cx - 8, cy - 8, cx + 8, cy + 8,
            fill="#e74c3c", outline="white", width=2
        )
        canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="white", outline="")

        canvas.create_text(cx, cy - 18, text="🌋 火山灰云中心",
                           anchor="s", fill="#e74c3c", font=("Microsoft YaHei", 9, "bold"))

        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        pos_text = f"📍 {abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}"
        canvas.create_text(cw / 2, ch - margin + 28, text=pos_text,
                           anchor="n", fill="#f39c12", font=("Microsoft YaHei", 10, "bold"))

        volcano_name = ""
        if self.current_volcano_name and self.current_volcano_name != "-- 手动输入 --":
            volcano_name = self.current_volcano_name.split(" ", 1)[-1] if " " in self.current_volcano_name else self.current_volcano_name
        else:
            presets = self._get_volcano_presets()
            for vname, vals in presets.items():
                if vname == "-- 手动输入 --":
                    continue
                if abs(vals[0] - lat) < 0.01 and abs(vals[1] - lon) < 0.01:
                    volcano_name = vname.split(" ", 1)[-1] if " " in vname else vname
                    break

        if volcano_name:
            canvas.create_text(cw / 2, margin - 12, text=f"📍 {volcano_name}",
                               anchor="s", fill="#2ecc71", font=("Microsoft YaHei", 11, "bold"))

        if hasattr(self, 'map_info_label'):
            self.map_info_label.config(
                text=f"覆盖范围: 纬度 ±{span_lat/2:.1f}° | 经度 ±{span_lon/2:.1f}°"
            )

    def _update_coord_display(self):
        if not hasattr(self, 'coord_label'):
            return
        lat = self.geo_vars["geo_center_lat"].get()
        lon = self.geo_vars["geo_center_lon"].get()

        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"

        volcano_name = ""
        if hasattr(self, 'current_volcano_name') and self.current_volcano_name and self.current_volcano_name != "-- 手动输入 --":
            volcano_name = f" ({self.current_volcano_name})"
        else:
            presets = self._get_volcano_presets()
            for vname, vals in presets.items():
                if vname == "-- 手动输入 --":
                    continue
                if abs(vals[0] - lat) < 0.05 and abs(vals[1] - lon) < 0.05:
                    volcano_name = f" ({vname})"
                    break

        self.coord_label.config(text=f"📍 当前位置: {abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}{volcano_name}")

    def _resolve_preset_name(self, preset_name):
        legacy_preset_map = {
            "单中心模型": "单中心模型_规则",
            "双中心模型": "双中心_复杂扩散",
            "三中心模型": "三中心_多细丝",
            "环形模型": "环形_高分形"
        }
        return legacy_preset_map.get(preset_name, preset_name)

    def _load_preset(self, preset_name):
        presets = get_preset_configs()
        resolved_name = self._resolve_preset_name(preset_name)

        if resolved_name in presets:
            self.current_config = VolcanicAshConfig.from_dict(presets[resolved_name].to_dict())
            self._sync_ui_from_config()
            self._update_center_table()
            self._update_info_text()
            self._update_coord_display()
            self.root.after(200, self._draw_map_preview)

    def _on_preset_change(self):
        preset_labels = {
            "single_center": "单中心模型",
            "double_center": "双中心模型",
            "triple_center": "三中心模型",
            "ring": "环形模型"
        }
        label = preset_labels.get(self.model_type_var.get(), "单中心模型")
        self._load_preset(label)

    def _sync_ui_from_config(self):
        if not self.current_config:
            return

        cfg = self.current_config
        self.param_vars["num_centers"].set(cfg.num_centers)
        self.param_vars["cloud_size"].set(cfg.cloud_size)
        self.param_vars["concentration_threshold"].set(cfg.concentration_threshold)
        self.param_vars["mass_ratio"].set(cfg.mass_ratio)
        self.param_vars["img_width"].set(cfg.image_size[0])
        self.param_vars["img_height"].set(cfg.image_size[1])

        self.geo_vars["geo_center_lat"].set(cfg.geo_center_lat)
        self.geo_vars["geo_center_lon"].set(cfg.geo_center_lon)
        self.geo_vars["geo_span_lat"].set(cfg.geo_span_lat)
        self.geo_vars["geo_span_lon"].set(cfg.geo_span_lon)

        # 同步不规则参数
        if hasattr(self, 'enable_irregular_var'):
            self.enable_irregular_var.set(cfg.enable_irregular)
            self.irregular_vars["turbulence_scale"].set(cfg.turbulence_scale)
            self.irregular_vars["wind_direction"].set(cfg.wind_direction)
            self.irregular_vars["wind_strength"].set(cfg.wind_strength)
            self.irregular_vars["fractal_dimension"].set(cfg.fractal_dimension)
            self.irregular_vars["num_filaments"].set(cfg.num_filaments)
            self.add_fractal_var.set(cfg.add_fractal_boundary)
            self.add_filaments_var.set(cfg.add_filaments)
            if cfg.random_seed is not None:
                self.random_seed_var.set(cfg.random_seed)

        presets = self._get_volcano_presets()
        self.current_volcano_name = ""
        for vname, vals in presets.items():
            if vname == "-- 手动输入 --":
                continue
            if abs(vals[0] - cfg.geo_center_lat) < 0.01 and abs(vals[1] - cfg.geo_center_lon) < 0.01:
                self.current_volcano_name = vname
                break

        type_map = {
            "single_center": "single_center",
            "double_center": "double_center",
            "triple_center": "triple_center",
            "ring": "ring"
        }
        self.model_type_var.set(type_map.get(cfg.model_type, "single_center"))

    def _sync_config_from_ui(self):
        if not self.current_config:
            self.current_config = VolcanicAshConfig()

        cfg = self.current_config
        cfg.model_type = self.model_type_var.get()
        cfg.num_centers = int(self.param_vars["num_centers"].get())
        cfg.cloud_size = self.param_vars["cloud_size"].get()
        cfg.concentration_threshold = self.param_vars["concentration_threshold"].get()
        cfg.mass_ratio = self.param_vars["mass_ratio"].get()
        cfg.image_size = (
            int(self.param_vars["img_width"].get()),
            int(self.param_vars["img_height"].get())
        )
        cfg.geo_center_lat = self.geo_vars["geo_center_lat"].get()
        cfg.geo_center_lon = self.geo_vars["geo_center_lon"].get()
        cfg.geo_span_lat = self.geo_vars["geo_span_lat"].get()
        cfg.geo_span_lon = self.geo_vars["geo_span_lon"].get()

        # 同步不规则参数
        if hasattr(self, 'enable_irregular_var'):
            cfg.enable_irregular = self.enable_irregular_var.get()
            cfg.turbulence_scale = self.irregular_vars["turbulence_scale"].get()
            cfg.wind_direction = self.irregular_vars["wind_direction"].get()
            cfg.wind_strength = self.irregular_vars["wind_strength"].get()
            cfg.fractal_dimension = self.irregular_vars["fractal_dimension"].get()
            cfg.num_filaments = int(self.irregular_vars["num_filaments"].get())
            cfg.add_fractal_boundary = self.add_fractal_var.get()
            cfg.add_filaments = self.add_filaments_var.get()
            seed_val = self.random_seed_var.get()
            cfg.random_seed = seed_val if seed_val > 0 else None

        centers = []
        for item in self.center_tree.get_children():
            values = self.center_tree.item(item)["values"]
            centers.append({
                "x": float(values[1]),
                "y": float(values[2]),
                "weight": float(values[3]),
                "std_x": float(values[4]),
                "std_y": float(values[5])
            })
        cfg.centers = centers

    def _on_param_change(self, key):
        self._sync_config_from_ui()
        self._update_info_text()

    def _on_geo_change(self, key):
        if key in ["geo_center_lat", "geo_center_lon"]:
            lat = self.geo_vars["geo_center_lat"].get()
            lon = self.geo_vars["geo_center_lon"].get()
            presets = self._get_volcano_presets()
            matched = False
            for vname, vals in presets.items():
                if vname == "-- 手动输入 --":
                    continue
                if abs(vals[0] - lat) < 0.0001 and abs(vals[1] - lon) < 0.0001:
                    matched = True
                    break
            if not matched:
                self.current_volcano_name = ""

        self._sync_config_from_ui()
        self._update_coord_display()
        self._draw_map_preview()
        self._update_info_text()

    def _update_center_table(self):
        for item in self.center_tree.get_children():
            self.center_tree.delete(item)

        if self.current_config and self.current_config.centers:
            for i, center in enumerate(self.current_config.centers):
                self.center_tree.insert("", "end", values=(
                    i + 1,
                    round(center["x"], 1),
                    round(center["y"], 1),
                    round(center["weight"], 3),
                    round(center["std_x"], 1),
                    round(center["std_y"], 1)
                ))

    def _add_center(self):
        self._open_center_editor(None)

    def _remove_center(self):
        selected = self.center_tree.selection()
        if selected:
            self.center_tree.delete(selected[0])
            self._sync_config_from_ui()
            self._update_info_text()

    def _edit_center(self):
        selected = self.center_tree.selection()
        if selected:
            item = selected[0]
            values = self.center_tree.item(item)["values"]
            center_data = {
                "id": values[0],
                "x": values[1],
                "y": values[2],
                "weight": values[3],
                "std_x": values[4],
                "std_y": values[5]
            }
            self._open_center_editor(center_data, item)

    def _open_center_editor(self, existing_data=None, tree_item=None):
        if self.center_editor_window and tk.Toplevel.winfo_exists(self.center_editor_window):
            self.center_editor_window.lift()
            return

        win = tk.Toplevel(self.root)
        self.center_editor_window = win
        win.title("编辑高斯中心" if existing_data else "添加高斯中心")
        win.geometry("360x320")
        win.transient(self.root)
        win.grab_set()

        defaults = existing_data or {
            "id": len(self.center_tree.get_children()) + 1,
            "x": 256, "y": 256, "weight": 0.25, "std_x": 50, "std_y": 50
        }

        frame = ttk.Frame(win, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)

        entries = {}
        fields = [
            ("X 坐标：", "x", 0, 512),
            ("Y 坐标：", "y", 0, 512),
            ("权重（0~1）：", "weight", 0.01, 1.0),
            ("标准差 X：", "std_x", 1, 200),
            ("标准差 Y：", "std_y", 1, 200),
        ]

        for row, (label, key, min_v, max_v) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
            var = tk.DoubleVar(value=defaults[key])
            entries[key] = var
            ttk.Entry(frame, textvariable=var, width=12).grid(row=row, column=1, pady=5, padx=10)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=len(fields), column=0, columnspan=2, pady=15)

        def save_center():
            data = {k: var.get() for k, var in entries.items()}
            values = (
                int(defaults["id"]),
                round(data["x"], 1),
                round(data["y"], 1),
                round(data["weight"], 3),
                round(data["std_x"], 1),
                round(data["std_y"], 1)
            )

            if tree_item and self.center_tree.exists(tree_item):
                self.center_tree.item(tree_item, values=values)
            else:
                self.center_tree.insert("", "end", values=values)

            self._sync_config_from_ui()
            self._update_info_text()
            win.destroy()

        ttk.Button(btn_frame, text="确定", command=save_center, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=win.destroy, width=10).pack(side=tk.LEFT, padx=5)

        win.protocol("WM_DELETE_WINDOW", lambda: setattr(self, 'center_editor_window', None) or win.destroy())

    def _preview_config(self):
        self._sync_config_from_ui()

        if not self.current_config or not self.current_config.centers:
            messagebox.showwarning("提示", "请至少添加一个GMM高斯中心！")
            return

        try:
            model = GMMVolcanicAshModel(self.current_config)
            conc_map = model.generate_concentration_map()

            grayscale_img = model.generate_grayscale_image(conc_map)
            danger_img = model.generate_danger_zone_image(conc_map)

            self.preview_grayscale_array = grayscale_img
            self.preview_danger_array = danger_img

            self._display_image_on_canvas(grayscale_img, self.grayscale_canvas)
            self._display_image_on_canvas(danger_img, self.danger_canvas, is_rgb=True)

            irregular_status = "启用" if self.current_config.enable_irregular else "未启用"
            messagebox.showinfo("成功", f"预览生成成功！\n不规则形状：{irregular_status}")

        except Exception as e:
            messagebox.showerror("错误", f"生成预览失败：\n{str(e)}")

    def _display_image_on_canvas(self, img_array, canvas, is_rgb=False):
        canvas.delete("all")

        if not HAS_PIL:
            canvas.create_text(
                canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                text="请安装 Pillow 库：pip install Pillow",
                fill="white", font=("Microsoft YaHei", 12)
            )
            return

        try:
            if is_rgb:
                pil_img = Image.fromarray(img_array.astype(np.uint8))
            else:
                pil_img = Image.fromarray(img_array.astype(np.uint8), mode='L')

            canvas.update_idletasks()
            cw = max(canvas.winfo_width(), 200)
            ch = max(canvas.winfo_height(), 200)

            pil_img.thumbnail((cw - 20, ch - 20), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)

            x = (cw - pil_img.width) // 2
            y = (ch - pil_img.height) // 2
            canvas.create_image(x, y, anchor="nw", image=tk_img)
            canvas.image = tk_img

        except Exception as e:
            canvas.create_text(
                canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                text=f"显示错误：{str(e)}",
                fill="red", font=("Microsoft YaHei", 11)
            )

    def _save_config(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            initialdir="output",
            title="保存配置"
        )
        if filepath:
            self._sync_config_from_ui()
            self.current_config.save(filepath)
            messagebox.showinfo("成功", f"配置已保存至：\n{filepath}")

    def _load_config(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            initialdir="output",
            title="加载配置"
        )
        if filepath:
            try:
                self.current_config = VolcanicAshConfig.load(filepath)
                self._sync_ui_from_config()
                self._update_center_table()
                self._update_info_text()
                messagebox.showinfo("成功", f"配置已从以下位置加载：\n{filepath}")
            except Exception as e:
                messagebox.showerror("错误", f"加载配置失败：\n{str(e)}")

    def _export_json(self):
        self._sync_config_from_ui()
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json")],
            initialdir="output",
            title="导出配置为JSON"
        )
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.current_config.to_dict(), f, ensure_ascii=False, indent=2)
            messagebox.showinfo("成功", f"已导出至：\n{filepath}")

    def _update_info_text(self):
        self.info_text.delete(1.0, tk.END)

        if not self.current_config:
            return

        info = "=" * 45 + "\n"
        info += "           当前配置参数摘要\n"
        info += "=" * 45 + "\n\n"

        type_names = {
            "single_center": "单中心模型",
            "double_center": "双中心模型",
            "triple_center": "三中心模型",
            "ring": "环形模型"
        }

        info += "【模型类型】\n"
        info += f"  类型：{type_names.get(self.current_config.model_type, self.current_config.model_type)}\n"
        info += f"  中心数：{self.current_config.num_centers}\n\n"

        info += "【基础参数】\n"
        info += f"  云团尺寸：{self.current_config.cloud_size}\n"
        info += f"  浓度阈值：{self.current_config.concentration_threshold}\n"
        info += f"  质量占比：{self.current_config.mass_ratio}\n"
        info += f"  图像尺寸：{self.current_config.image_size[0]} × {self.current_config.image_size[1]}\n\n"

        info += "【地理坐标】\n"
        info += f"  中心位置：({self.current_config.geo_center_lat}, {self.current_config.geo_center_lon})\n"
        info += f"  跨度范围：纬度{self.current_config.geo_span_lat}，经度{self.current_config.geo_span_lon}\n\n"

        # 添加不规则参数信息
        if hasattr(self, 'enable_irregular_var'):
            info += "【不规则形状参数】\n"
            info += f"  启用状态：{'是 ✓' if self.current_config.enable_irregular else '否 ✗'}\n"
            if self.current_config.enable_irregular:
                info += f"  扰动强度：{self.current_config.turbulence_scale:.2f}\n"
                info += f"  风向角度：{self.current_config.wind_direction:.0f}°\n"
                info += f"  风力强度：{self.current_config.wind_strength:.2f}\n"
                info += f"  分形维度：{self.current_config.fractal_dimension:.2f}\n"
                info += f"  细丝数量：{self.current_config.num_filaments}\n"
                info += f"  分形边界：{'是' if self.current_config.add_fractal_boundary else '否'}\n"
                info += f"  细丝结构：{'是' if self.current_config.add_filaments else '否'}\n"
                if self.current_config.random_seed:
                    info += f"  随机种子：{self.current_config.random_seed}\n"
            info += "\n"

        info += "【GMM高斯中心详情】\n"
        total_weight = 0
        for i, c in enumerate(self.current_config.centers):
            info += f"  中心{i+1}: X={c['x']:.1f}, Y={c['y']:.1f}, "
            info += f"权重={c['weight']:.3f}, 标准差=({c['std_x']:.1f},{c['std_y']:.1f})\n"
            total_weight += c['weight']
        info += f"\n  权重总和：{total_weight:.3f}"
        if abs(total_weight - 1.0) > 0.01:
            info += "  [!] 警告：权重总和建议接近1.0"
        info += "\n"

        self.info_text.insert(tk.END, info)

    def _apply_and_close(self):
        self._sync_config_from_ui()
        output_path = os.path.join("output", "current_config.json")
        os.makedirs("output", exist_ok=True)
        self.current_config.save(output_path)

        irregular_status = "已启用" if self.current_config.enable_irregular else "未启用"
        messagebox.showinfo("完成", f"配置已保存至：\n{output_path}\n\n不规则形状：{irregular_status}")
        self.root.quit()

    def run(self):
        self.root.mainloop()
        return self.current_config


def main():
    app = VolcanicAshConfigGUI()
    config = app.run()

    if config:
        print("\n最终配置参数：")
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))

    return config


if __name__ == "__main__":
    main()
