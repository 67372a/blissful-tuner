#!/usr/bin/env python3
# gui_wan2.2_trainer.py
# - Generic defaults
# - Attention presets
# - Cache tab
# - Optimizer presets + custom
# - Custom args + env
# - Save/Checkpoint controls
# - Profiles w/ auto-save
# - Collapsible Paths & Runtime
# - Dual-phase (HIGH→LOW) with per-phase overrides (epochs, dataset TOML, extra args, suffixes)
# - Robust Stop (process group kill)
# - Datasets tab: open/edit/validate/save TOML; insert dataset templates
#
# MIT License

from __future__ import annotations
import os, shlex, json, sys, threading, subprocess, signal, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    import tomllib as _toml_r  # py3.11+
except Exception:
    try:
        import tomli as _toml_r  # fallback if installed
    except Exception:
        _toml_r = None

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QWidget, QFormLayout, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QPlainTextEdit, QFileDialog, QLabel, QHBoxLayout,
    QVBoxLayout, QGroupBox, QMessageBox, QScrollArea, QSplitter, QTabWidget,
    QToolButton, QFrame
)

# ---------- Generic Defaults ----------

DEFAULTS = {
    "paths": {
        "WAN_T2V_HIGH": "",
        "WAN_T2V_LOW":  "",
        "WAN_I2V_HIGH": "",
        "WAN_I2V_LOW":  "",
        "T5_PATH":      "",
        "VAE_PATH":     "",
        "DATASET_TOML": "",
        "LOG_DIR":      str(Path.cwd() / "wan_logs"),
        "OUT_DIR":      str(Path.cwd() / "wanOutputs"),
        "PROFILE_DIR":  str(Path.home() / "wan_profiles"),
        "PROFILE_NAME": "wan_run",
    },
    "runtime": {
        "entrypoint": "-m musubi_tuner.wan_train_network",
        "attention_backend": "SDPA",
        "seed": 420,
        "acc_cpu_threads": 1,
        "blocks_to_swap": 36,
        "lora_rank": 16,
        "lora_alpha": 16,
        "network_dropout": 0.05,
        "n_workers": 8,
        "discrete_shift": 12.0,
        "epochs": 5,
        "env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_F32_MATMUL_PRECISION": "high",
            "TORCH_FLOAT32_MATMUL_PRECISION": "high",
        },
    },
    "task": {
        "kind": "i2v-A14B",
        "noise": "high",
        "min_t": 900,
        "max_t": 1000,
    },
    "optimizer": {
        "preset": "SAEM",
        "learning_rate": 2e-4,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 28,
        "type_path": "customized_optimizers.simplifiedademamix.SimplifiedAdEMAMixExM",
        "args_kv": "alpha=1\nbeta1_warmup=280\nbetas=(0.99,0.997)\nmin_beta1=0.95\namsgrad_max_decay_rate=0.96\namsgrad_min_decay_rate=0.96\nuse_adabelief=True\ntorch_compile=False",
    },
    "cache": {
        "entry_latents": "-m musubi_tuner.wan_cache_latents",
        "entry_textenc": "-m musubi_tuner.wan_cache_text_encoder_outputs",
        "batch_size": 16,
        "mode": "t2v",
    },
    "save": {
        "save_every_n_epochs": 0,
        "save_every_n_steps": 0,
        "save_last_n_epochs": 0,
        "save_last_n_steps": 0,
        "save_last_n_epochs_state": 0,
        "save_last_n_steps_state": 0,
        "save_state": False,
        "save_state_on_train_end": False,
    },
    "dual": {
        "enabled": False,
        "high_suffix": "_HIGH",
        "low_suffix": "_LOW",
        "epochs_high": 0,  # 0 = use main epochs
        "epochs_low": 0,
        "dataset_high": "",  # empty = use main dataset TOML
        "dataset_low":  "",
        "extra_high": "",
        "extra_low":  "",
    }
}

NOISE_BANDS = {
    "t2v-A14B": {"high": (875, 1000), "low": (0, 875)},
    "i2v-A14B": {"high": (900, 1000), "low": (0, 900)},
}

OPT_PRESETS = {
    "AdamW": {
        "type_path": "torch.optim.AdamW",
        "lr": "3e-4", "sched": "cosine", "warmup": "100",
        "args": "weight_decay=0.01\neps=1e-8\nbetas=(0.9,0.999)"
    },
    "AdamW8bit": {
        "type_path": "bitsandbytes.optim.AdamW8bit",
        "lr": "3e-4", "sched": "cosine", "warmup": "100",
        "args": "weight_decay=0.01\neps=1e-8\nbetas=(0.9,0.999)"
    },
    "CAME": {
        "type_path": "customized_optimizers.came.CAME",
        "lr": "3e-5", "sched": "constant_with_warmup", "warmup": "100",
        "args": "weight_decay=0.01\neps=(1e-30,1e-16)\nbetas=(0.9,0.999,0.9999)"
    },
    "SAEM": {
        "type_path": "customized_optimizers.simplifiedademamix.SimplifiedAdEMAMixExM",
        "lr": "2e-4", "sched": "cosine", "warmup": "28",
        "args": "alpha=1\nbeta1_warmup=280\nbetas=(0.99,0.997)\nmin_beta1=0.95\namsgrad_max_decay_rate=0.96\namsgrad_min_decay_rate=0.96\nuse_adabelief=True\ntorch_compile=False"
    },
    "FFTD": {
        "type_path": "customized_optimizers.fftdescent.FFTDescent",
        "lr": "3e-4", "sched": "cosine", "warmup": "28",
        "args": "weight_decay=0.10"
    },
    "SingState": {
        "type_path": "customized_optimizers.singstate.SingState",
        "lr": "2e-4", "sched": "cosine", "warmup": "28",
        "args": "weight_decay=0.01"
    },
    "TALON": {
        "type_path": "customized_optimizers.talon.TALON",
        "lr": "2e-4", "sched": "cosine", "warmup": "28",
        "args": "weight_decay=0.01"
    },
    "SGD": {
        "type_path": "customized_optimizers.utils.SGD",
        "lr": "1e-3", "sched": "cosine", "warmup": "100",
        "args": "momentum=0.9\nweight_decay=0.0\nnesterov=True"
    },
}

# ---------- Helpers ----------

def kv_text_to_list_for(prefix: str, kv_text: str) -> List[str]:
    args: List[str] = []
    for line in kv_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"): continue
        if "=" in s:
            k, v = s.split("=", 1)
            args.extend([prefix, f"{k.strip()}={v.strip()}"])
        else:
            args.append(s)
    return args

def env_text_to_dict(text: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"): continue
        if "=" not in s: continue
        k, v = s.split("=", 1)
        env[k.strip()] = v.strip()
    return env

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def unique_path(base: Path, suffix: str = ".json") -> Path:
    candidate = base.with_suffix(suffix)
    i = 2
    while candidate.exists():
        candidate = base.with_name(f"{base.stem}_{i}").with_suffix(suffix)
        i += 1
    return candidate

def warn_missing_paths(parent: QWidget, needed: Dict[str, str]) -> bool:
    missing = [k for k, v in needed.items() if not v]
    if not missing: return True
    msg = "The following required paths are empty:\n  - " + "\n  - ".join(missing) + "\n\nContinue anyway?"
    r = QMessageBox.question(parent, "Missing paths", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    return r == QMessageBox.Yes

# ---------- Collapsible ----------

class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, collapsed: bool = True):
        super().__init__()
        self.btn = QToolButton(text=title, checkable=True, checked=not collapsed, toolButtonStyle=Qt.ToolButtonTextBesideIcon)
        self.btn.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self.btn.toggled.connect(self.on_toggled)
        self.frame = QFrame()
        lay = QVBoxLayout(self.frame); lay.setContentsMargins(12,6,0,6); lay.addWidget(content)
        root = QVBoxLayout(self); root.addWidget(self.btn); root.addWidget(self.frame)
        self.setCollapsed(collapsed)
    def on_toggled(self, checked: bool):
        self.setCollapsed(not checked)
    def setCollapsed(self, collapsed: bool):
        self.frame.setVisible(not collapsed)
        self.btn.setArrowType(Qt.RightArrow if collapsed else Qt.DownArrow)

# ---------- Proc streamer ----------

class ProcStreamer(QObject):
    line = Signal(str); done = Signal(int)
    def __init__(self): super().__init__(); self._proc=None
    def start(self, argv: List[str], env: Dict[str, str] | None = None):
        if self._proc is not None: raise RuntimeError("Process already running")
        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = os.setsid  # new process group
        elif os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        self._proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, **kwargs)
        threading.Thread(target=self._pump, daemon=True).start()
    def _pump(self):
        for line in self._proc.stdout: self.line.emit(line.rstrip("\n"))
        rc = self._proc.wait(); self.done.emit(rc); self._proc=None
    def stop(self, grace: float = 5.0):
        if self._proc is None: return
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            else:
                self._proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        except Exception:
            try: self._proc.terminate()
            except Exception: pass
        t0 = time.time()
        while self._proc and (time.time() - t0) < grace:
            rc = self._proc.poll()
            if rc is not None:
                self._proc = None; return
            time.sleep(0.1)
        if self._proc:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                else:
                    self._proc.kill()
            except Exception: pass
            self._proc = None

# ---------- Main ----------

class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minimal WAN 2.2 Trainer GUI")
        self.resize(1500, 950)

        self.streamer = ProcStreamer()
        self.streamer.line.connect(self.on_log)
        self.streamer.done.connect(self.on_done)

        # Shared path fields
        self.ed_t2v_high = self.file_row("WAN t2v HIGH", DEFAULTS["paths"]["WAN_T2V_HIGH"])
        self.ed_t2v_low  = self.file_row("WAN t2v LOW",  DEFAULTS["paths"]["WAN_T2V_LOW"])
        self.ed_i2v_high = self.file_row("WAN i2v HIGH", DEFAULTS["paths"]["WAN_I2V_HIGH"])
        self.ed_i2v_low  = self.file_row("WAN i2v LOW",  DEFAULTS["paths"]["WAN_I2V_LOW"])
        self.ed_t5       = self.file_row("T5",           DEFAULTS["paths"]["T5_PATH"])
        self.ed_vae      = self.file_row("VAE",          DEFAULTS["paths"]["VAE_PATH"])
        self.ed_toml     = self.file_row("Dataset TOML", DEFAULTS["paths"]["DATASET_TOML"])
        self.ed_logdir   = self.dir_row ("Log Dir",      DEFAULTS["paths"]["LOG_DIR"])
        self.ed_outdir   = self.dir_row ("Output Dir",   DEFAULTS["paths"]["OUT_DIR"])

        tabs = QTabWidget()
        train_tab = QWidget(); cache_tab = QWidget(); datasets_tab = QWidget()
        tabs.addTab(train_tab, "Train"); tabs.addTab(cache_tab, "Cache"); tabs.addTab(datasets_tab, "Datasets (TOML)")

        lay_train = QVBoxLayout(train_tab)
        lay_cache = QVBoxLayout(cache_tab)
        lay_ds    = QVBoxLayout(datasets_tab)

        # PATHS (collapsible, default collapsed)
        gb_paths = QGroupBox("Paths")
        lay_p = QFormLayout(gb_paths)
        for w in (self.ed_t2v_high, self.ed_t2v_low, self.ed_i2v_high, self.ed_i2v_low,
                  self.ed_t5, self.ed_vae, self.ed_toml, self.ed_logdir, self.ed_outdir):
            label, widget = w[0]; lay_p.addRow(label, widget)
        sec_paths = CollapsibleSection("Paths", gb_paths, collapsed=True)

        # TASK
        self.cb_task = QComboBox(); self.cb_task.addItems(["t2v-A14B", "i2v-A14B"])
        self.cb_noise = QComboBox(); self.cb_noise.addItems(["high", "low"])
        self.sb_min_t = QSpinBox(); self.sb_min_t.setRange(0, 10000)
        self.sb_max_t = QSpinBox(); self.sb_max_t.setRange(0, 10000)
        self.cb_task.currentTextChanged.connect(self.on_task_noise_change)
        self.cb_noise.currentTextChanged.connect(self.on_task_noise_change)
        self.cb_task.setCurrentText(DEFAULTS["task"]["kind"]); self.cb_noise.setCurrentText(DEFAULTS["task"]["noise"])
        self.on_task_noise_change()
        gb_task = QGroupBox("Task"); lay_t = QFormLayout(gb_task)
        lay_t.addRow("Task", self.cb_task); lay_t.addRow("Noise band", self.cb_noise)
        lay_t.addRow("min_timestep", self.sb_min_t); lay_t.addRow("max_timestep", self.sb_max_t)

        # RUNTIME (collapsible)
        self.ed_entry = QLineEdit(DEFAULTS["runtime"]["entrypoint"])
        self.cb_attn = QComboBox(); self.cb_attn.addItems(["SDPA", "FlashAttention", "xFormers"])
        self.cb_attn.setCurrentText(DEFAULTS["runtime"]["attention_backend"])
        self.sb_seed = QSpinBox(); self.sb_seed.setRange(0, 2**31-1); self.sb_seed.setValue(DEFAULTS["runtime"]["seed"])
        self.sb_threads = QSpinBox(); self.sb_threads.setRange(1, 64); self.sb_threads.setValue(DEFAULTS["runtime"]["acc_cpu_threads"])
        self.sb_blocks = QSpinBox(); self.sb_blocks.setRange(0, 1000); self.sb_blocks.setValue(DEFAULTS["runtime"]["blocks_to_swap"])
        self.sb_rank = QSpinBox(); self.sb_rank.setRange(1, 4096); self.sb_rank.setValue(DEFAULTS["runtime"]["lora_rank"])
        self.sb_alpha = QSpinBox(); self.sb_alpha.setRange(1, 4096); self.sb_alpha.setValue(DEFAULTS["runtime"]["lora_alpha"])
        self.dsb_dropout = QDoubleSpinBox(); self.dsb_dropout.setRange(0, 1); self.dsb_dropout.setSingleStep(0.01); self.dsb_dropout.setValue(DEFAULTS["runtime"]["network_dropout"])
        self.sb_workers = QSpinBox(); self.sb_workers.setRange(0, 128); self.sb_workers.setValue(DEFAULTS["runtime"]["n_workers"])
        self.dsb_shift = QDoubleSpinBox(); self.dsb_shift.setRange(0, 1000); self.dsb_shift.setDecimals(3); self.dsb_shift.setValue(DEFAULTS["runtime"]["discrete_shift"])
        self.sb_epochs = QSpinBox(); self.sb_epochs.setRange(1, 1000); self.sb_epochs.setValue(DEFAULTS["runtime"]["epochs"])

        self.txt_env = QPlainTextEdit("\n".join(f"{k}={v}" for k,v in DEFAULTS["runtime"]["env"].items()))
        self.txt_env.setPlaceholderText("ENV variables (key=value per line)")
        self.txt_custom_args = QPlainTextEdit(); self.txt_custom_args.setPlaceholderText("Custom extra args (one token or key=value per line).")

        gb_run = QGroupBox("Runtime"); lay_r = QFormLayout(gb_run)
        lay_r.addRow("Entrypoint", self.ed_entry)
        lay_r.addRow("Attention backend", self.cb_attn)
        lay_r.addRow("Seed", self.sb_seed)
        lay_r.addRow("CPU threads (accelerate)", self.sb_threads)
        lay_r.addRow("Blocks to swap", self.sb_blocks)
        lay_r.addRow("LoRA Rank", self.sb_rank)
        lay_r.addRow("LoRA Alpha", self.sb_alpha)
        lay_r.addRow("Network Dropout", self.dsb_dropout)
        lay_r.addRow("DataLoader workers", self.sb_workers)
        lay_r.addRow("Discrete flow shift", self.dsb_shift)
        lay_r.addRow("Max epochs", self.sb_epochs)
        lay_r.addRow(QLabel("Extra ENV (key=value per line)")); lay_r.addRow(self.txt_env)
        lay_r.addRow(QLabel("Custom extra args (appended)")); lay_r.addRow(self.txt_custom_args)
        sec_run = CollapsibleSection("Runtime", gb_run, collapsed=True)

        # OPTIMIZER
        self.cb_opt = QComboBox(); self.cb_opt.addItems(list(OPT_PRESETS.keys()) + ["Custom"])
        self.ed_opt_type = QLineEdit(DEFAULTS["optimizer"]["type_path"])
        self.ed_lr = QLineEdit(str(DEFAULTS["optimizer"]["learning_rate"]))
        self.cb_sched = QComboBox(); self.cb_sched.addItems(["cosine", "constant", "constant_with_warmup"])
        self.cb_sched.setCurrentText(DEFAULTS["optimizer"]["lr_scheduler"])
        self.sb_warmup = QSpinBox(); self.sb_warmup.setRange(0, 100000); self.sb_warmup.setValue(DEFAULTS["optimizer"]["lr_warmup_steps"])
        self.txt_opt_args = QPlainTextEdit(DEFAULTS["optimizer"]["args_kv"])
        self.cb_opt.currentTextChanged.connect(self.on_opt_preset_change); self.on_opt_preset_change(self.cb_opt.currentText())
        gb_opt = QGroupBox("Optimizer"); lay_o = QFormLayout(gb_opt)
        lay_o.addRow("Preset", self.cb_opt)
        lay_o.addRow("Optimizer type (module.Class)", self.ed_opt_type)
        lay_o.addRow("Learning rate", self.ed_lr)
        lay_o.addRow("LR scheduler", self.cb_sched)
        lay_o.addRow("Warmup steps", self.sb_warmup)
        lay_o.addRow(QLabel("Optimizer args (key=value per line; accepts raw tokens)")); lay_o.addRow(self.txt_opt_args)

        # SAVE
        self.sb_save_epochs = QSpinBox(); self.sb_save_epochs.setRange(0, 10000); self.sb_save_epochs.setValue(DEFAULTS["save"]["save_every_n_epochs"])
        self.sb_save_steps  = QSpinBox(); self.sb_save_steps.setRange(0, 1000000); self.sb_save_steps.setValue(DEFAULTS["save"]["save_every_n_steps"])
        self.sb_keep_last_epochs = QSpinBox(); self.sb_keep_last_epochs.setRange(0, 10000); self.sb_keep_last_epochs.setValue(DEFAULTS["save"]["save_last_n_epochs"])
        self.sb_keep_last_steps  = QSpinBox(); self.sb_keep_last_steps.setRange(0, 1000000); self.sb_keep_last_steps.setValue(DEFAULTS["save"]["save_last_n_steps"])
        self.sb_keep_last_epochs_state = QSpinBox(); self.sb_keep_last_epochs_state.setRange(0, 10000); self.sb_keep_last_epochs_state.setValue(DEFAULTS["save"]["save_last_n_epochs_state"])
        self.sb_keep_last_steps_state  = QSpinBox(); self.sb_keep_last_steps_state.setRange(0, 1000000); self.sb_keep_last_steps_state.setValue(DEFAULTS["save"]["save_last_n_steps_state"])
        self.cb_save_state = QCheckBox("save_state"); self.cb_save_state.setChecked(DEFAULTS["save"]["save_state"])
        self.cb_save_state_on_end = QCheckBox("save_state_on_train_end"); self.cb_save_state_on_end.setChecked(DEFAULTS["save"]["save_state_on_train_end"])
        gb_save = QGroupBox("Save / Checkpointing"); lay_s = QFormLayout(gb_save)
        lay_s.addRow("save_every_n_epochs (0=off)", self.sb_save_epochs)
        lay_s.addRow("save_every_n_steps (0=off)",  self.sb_save_steps)
        lay_s.addRow("save_last_n_epochs (0=off)", self.sb_keep_last_epochs)
        lay_s.addRow("save_last_n_steps (0=off)",  self.sb_keep_last_steps)
        lay_s.addRow("save_last_n_epochs_state (0=off)", self.sb_keep_last_epochs_state)
        lay_s.addRow("save_last_n_steps_state (0=off)",  self.sb_keep_last_steps_state)
        lay_s.addRow(self.cb_save_state)
        lay_s.addRow(self.cb_save_state_on_end)

        # DUAL-PHASE (overrides)
        self.cb_dual = QCheckBox("Dual-phase: run HIGH then LOW")
        self.cb_dual.setChecked(DEFAULTS["dual"]["enabled"])
        self.ed_dual_high_suffix = QLineEdit(DEFAULTS["dual"]["high_suffix"])
        self.ed_dual_low_suffix  = QLineEdit(DEFAULTS["dual"]["low_suffix"])
        self.sb_dual_epochs_high = QSpinBox(); self.sb_dual_epochs_high.setRange(0, 1000); self.sb_dual_epochs_high.setValue(DEFAULTS["dual"]["epochs_high"])
        self.sb_dual_epochs_low  = QSpinBox(); self.sb_dual_epochs_low.setRange(0, 1000); self.sb_dual_epochs_low.setValue(DEFAULTS["dual"]["epochs_low"])
        self.ed_dual_dataset_high = self.file_row_simple(DEFAULTS["dual"]["dataset_high"])
        self.ed_dual_dataset_low  = self.file_row_simple(DEFAULTS["dual"]["dataset_low"])
        self.txt_dual_extra_high = QPlainTextEdit(DEFAULTS["dual"]["extra_high"]); self.txt_dual_extra_high.setPlaceholderText("Extra args for HIGH (optional)")
        self.txt_dual_extra_low  = QPlainTextEdit(DEFAULTS["dual"]["extra_low"]);  self.txt_dual_extra_low.setPlaceholderText("Extra args for LOW (optional)")
        gb_dual = QGroupBox("Dual-phase (HIGH→LOW)")
        lay_d = QFormLayout(gb_dual)
        lay_d.addRow(self.cb_dual)
        lay_d.addRow("High suffix", self.ed_dual_high_suffix)
        lay_d.addRow("Low suffix",  self.ed_dual_low_suffix)
        lay_d.addRow("Epochs (HIGH, 0=use main)", self.sb_dual_epochs_high)
        lay_d.addRow("Epochs (LOW,  0=use main)", self.sb_dual_epochs_low)
        lay_d.addRow("Dataset TOML (HIGH, optional)", self.ed_dual_dataset_high)
        lay_d.addRow("Dataset TOML (LOW, optional)",  self.ed_dual_dataset_low)
        lay_d.addRow(QLabel("Extra args for HIGH")); lay_d.addRow(self.txt_dual_extra_high)
        lay_d.addRow(QLabel("Extra args for LOW"));  lay_d.addRow(self.txt_dual_extra_low)

        # META / Profiles
        self.ed_run_name = QLineEdit("Wan2.2_Run")
        self.ed_profile_dir = self.dir_row_simple(DEFAULTS["paths"]["PROFILE_DIR"])
        self.ed_profile_name = QLineEdit(DEFAULTS["paths"]["PROFILE_NAME"])
        self.cb_autosave = QCheckBox("Auto-save profile on Start"); self.cb_autosave.setChecked(True)
        gb_meta = QGroupBox("Run Meta / Profiles"); lay_m = QFormLayout(gb_meta)
        lay_m.addRow("Log prefix / Output name", self.ed_run_name)
        lay_m.addRow("Profile dir", self.ed_profile_dir)
        lay_m.addRow("Profile base name", self.ed_profile_name)
        lay_m.addRow(self.cb_autosave)

        # Buttons
        self.btn_build = QPushButton("Build Train Command"); self.btn_build.clicked.connect(self.on_build_only)
        self.btn_start = QPushButton("Start Training"); self.btn_start.clicked.connect(self.on_start)
        self.btn_stop  = QPushButton("Stop"); self.btn_stop.clicked.connect(self.on_stop)
        self.btn_save  = QPushButton("Save Profile…"); self.btn_save.clicked.connect(self.on_save)
        self.btn_load  = QPushButton("Load Profile…"); self.btn_load.clicked.connect(self.on_load)
        btn_row = QHBoxLayout()
        for b in (self.btn_build, self.btn_start, self.btn_stop, self.btn_save, self.btn_load): btn_row.addWidget(b)
        btn_row.addStretch(1)

        # Left scroll (Train)
        left_train_widget = QWidget(); left_train_layout = QVBoxLayout(left_train_widget)
        left_train_layout.addWidget(sec_paths)
        left_train_layout.addWidget(gb_task)
        left_train_layout.addWidget(sec_run)
        left_train_layout.addWidget(gb_opt)
        left_train_layout.addWidget(gb_save)
        left_train_layout.addWidget(gb_dual)
        left_train_layout.addWidget(gb_meta)
        left_train_layout.addLayout(btn_row)
        left_train_layout.addStretch(1)
        left_train_scroll = QScrollArea(); left_train_scroll.setWidgetResizable(True); left_train_scroll.setWidget(left_train_widget)

        # ===== Cache tab =====
        self.ed_cache_entry_lat = QLineEdit(DEFAULTS["cache"]["entry_latents"])
        self.ed_cache_entry_txt = QLineEdit(DEFAULTS["cache"]["entry_textenc"])
        self.sb_cache_bs = QSpinBox(); self.sb_cache_bs.setRange(1, 4096); self.sb_cache_bs.setValue(DEFAULTS["cache"]["batch_size"])
        self.cb_cache_mode = QComboBox(); self.cb_cache_mode.addItems(["t2v", "i2v"]); self.cb_cache_mode.setCurrentText(DEFAULTS["cache"]["mode"])
        self.txt_cache_custom = QPlainTextEdit(); self.txt_cache_custom.setPlaceholderText("Custom extra args for cache (applied to BOTH steps).")

        form_cache_top = QFormLayout()
        form_cache_top.addRow("Latents cache entry", self.ed_cache_entry_lat)
        form_cache_top.addRow("Text-enc cache entry", self.ed_cache_entry_txt)
        form_cache_top.addRow("Cache batch size", self.sb_cache_bs)
        form_cache_top.addRow("Mode", self.cb_cache_mode)

        latents_inner = QWidget(); lat_form = QFormLayout(latents_inner)
        lat_form.addRow("Dataset TOML", QLabel("Uses shared 'Dataset TOML' from Paths"))
        lat_form.addRow("VAE path", QLabel("Uses shared 'VAE' from Paths"))
        textenc_inner = QWidget(); txt_form = QFormLayout(textenc_inner)
        txt_form.addRow("Dataset TOML", QLabel("Uses shared 'Dataset TOML' from Paths"))
        txt_form.addRow("T5 path", QLabel("Uses shared 'T5' from Paths"))
        sec_latents = CollapsibleSection("Cache VAE Latents", latents_inner, collapsed=True)
        sec_txtenc  = CollapsibleSection("Cache Text-Encoder Outputs", textenc_inner, collapsed=True)

        self.btn_cache_build = QPushButton("Build Cache Commands"); self.btn_cache_build.clicked.connect(self.on_cache_build_only)
        self.btn_cache_run = QPushButton("Run Cache"); self.btn_cache_run.clicked.connect(self.on_cache_run)
        cache_btn_row = QHBoxLayout(); cache_btn_row.addWidget(self.btn_cache_build); cache_btn_row.addWidget(self.btn_cache_run); cache_btn_row.addStretch(1)

        cache_top = QWidget(); cache_top_lay = QVBoxLayout(cache_top)
        cache_top_lay.addLayout(form_cache_top)
        cache_top_lay.addWidget(QLabel("Custom extra args for cache (applied to BOTH latents & text-enc)"))
        cache_top_lay.addWidget(self.txt_cache_custom)
        cache_top_lay.addWidget(sec_latents); cache_top_lay.addWidget(sec_txtenc)
        cache_top_lay.addLayout(cache_btn_row); cache_top_lay.addStretch(1)

        left_cache_scroll = QScrollArea(); left_cache_scroll.setWidgetResizable(True); left_cache_scroll.setWidget(cache_top)

        # ===== Datasets (TOML) tab =====
        self.ed_ds_path = self.file_row_simple("")
        self.btn_ds_load = QPushButton("Load TOML")
        self.btn_ds_save = QPushButton("Save")
        self.btn_ds_saveas = QPushButton("Save As…")
        self.btn_ds_validate = QPushButton("Validate")
        self.btn_ds_insert_video = QPushButton("Insert VIDEO dataset template")
        self.btn_ds_insert_image = QPushButton("Insert IMAGE dataset template")
        controls = QHBoxLayout()
        controls.addWidget(QLabel("TOML path:")); controls.addWidget(self.ed_ds_path)
        controls.addWidget(self.btn_ds_load); controls.addWidget(self.btn_ds_validate)
        controls.addWidget(self.btn_ds_save); controls.addWidget(self.btn_ds_saveas)
        controls.addWidget(self.btn_ds_insert_video); controls.addWidget(self.btn_ds_insert_image)
        self.txt_ds = QPlainTextEdit(); self.txt_ds.setPlaceholderText("# Edit your TOML here. Use Validate to check syntax if tomllib/tomli is available.")

        self.btn_ds_load.clicked.connect(self.on_ds_load)
        self.btn_ds_save.clicked.connect(self.on_ds_save)
        self.btn_ds_saveas.clicked.connect(self.on_ds_save_as)
        self.btn_ds_validate.clicked.connect(self.on_ds_validate)
        self.btn_ds_insert_video.clicked.connect(self.on_ds_insert_video)
        self.btn_ds_insert_image.clicked.connect(self.on_ds_insert_image)

        ds_top = QWidget(); ds_lay = QVBoxLayout(ds_top)
        ds_lay.addLayout(controls); ds_lay.addWidget(self.txt_ds)
        ds_scroll = QScrollArea(); ds_scroll.setWidgetResizable(True); ds_scroll.setWidget(ds_top)

        # Right log
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(12000)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(tabs); splitter.addWidget(self.log)
        splitter.setStretchFactor(0, 2); splitter.setStretchFactor(1, 3)

        lay_train.addWidget(left_train_scroll)
        lay_cache.addWidget(left_cache_scroll)
        lay_ds.addWidget(ds_scroll)

        root = QVBoxLayout(self); root.addWidget(splitter)

    # ---------- UI helpers ----------
    def file_row(self, label: str, default: str):
        le = QLineEdit(default); btn = QPushButton("…")
        def browse():
            base = default or str(Path.cwd())
            path, _ = QFileDialog.getOpenFileName(self, f"Select file for {label}", base)
            if path: le.setText(path)
        btn.clicked.connect(browse)
        box = QHBoxLayout(); box.addWidget(le); box.addWidget(btn)
        w = QWidget(); w.setLayout(box)
        return (label, w), le

    def file_row_simple(self, default: str) -> QWidget:
        le = QLineEdit(default); btn = QPushButton("…")
        def browse():
            base = default or str(Path.cwd())
            path, _ = QFileDialog.getOpenFileName(self, "Select file", base)
            if path: le.setText(path)
        btn.clicked.connect(browse)
        box = QHBoxLayout(); box.addWidget(le); box.addWidget(btn)
        w = QWidget(); w.setLayout(box); w._lineedit = le
        return w

    def dir_row(self, label: str, default: str):
        le = QLineEdit(default); btn = QPushButton("…")
        def browse():
            base = default or str(Path.cwd())
            path = QFileDialog.getExistingDirectory(self, f"Select directory for {label}", base)
            if path: le.setText(path)
        btn.clicked.connect(browse)
        box = QHBoxLayout(); box.addWidget(le); box.addWidget(btn)
        w = QWidget(); w.setLayout(box)
        return (label, w), le

    def dir_row_simple(self, default: str) -> QWidget:
        le = QLineEdit(default); btn = QPushButton("…")
        def browse():
            base = default or str(Path.cwd())
            path = QFileDialog.getExistingDirectory(self, "Select directory", base)
            if path: le.setText(path)
        btn.clicked.connect(browse)
        box = QHBoxLayout(); box.addWidget(le); box.addWidget(btn)
        w = QWidget(); w.setLayout(box); w._lineedit = le
        return w

    def get_profile_dir_text(self) -> str:
        return getattr(self.ed_profile_dir, "_lineedit").text()

    # ---------- Events ----------
    def on_task_noise_change(self, *_):
        task = self.cb_task.currentText(); noise = self.cb_noise.currentText()
        lo, hi = NOISE_BANDS.get(task, {}).get(noise, (0, 1000))
        self.sb_min_t.setValue(lo); self.sb_max_t.setValue(hi)

    def on_opt_preset_change(self, name: str):
        if name in OPT_PRESETS:
            p = OPT_PRESETS[name]
            self.ed_opt_type.setText(p["type_path"])
            self.ed_lr.setText(p["lr"])
            self.cb_sched.setCurrentText(p["sched"])
            self.sb_warmup.setValue(int(float(p["warmup"])))
            self.txt_opt_args.setPlainText(p["args"])

    # ---------- Build commands ----------
    def build_train_command(self, *, noise_override: Optional[str] = None, name_suffix: str = "", dataset_override: str = "", epochs_override: int = 0, extra_override: str = "") -> Tuple[List[str], Dict[str, str]]:
        entry_parts = shlex.split(self.ed_entry.text().strip())
        task = self.cb_task.currentText()
        noise = noise_override if noise_override else self.cb_noise.currentText()
        min_t = self.sb_min_t.value(); max_t = self.sb_max_t.value()

        # Canonical band if override
        if noise_override:
            lo, hi = NOISE_BANDS.get(task, {}).get(noise_override, (min_t, max_t))
            min_t, max_t = lo, hi

        # DIT selection per task/noise
        if task.startswith("t2v"):
            dit = self.ed_t2v_high[1].text() if noise == "high" else self.ed_t2v_low[1].text()
        else:
            dit = self.ed_i2v_high[1].text() if noise == "high" else self.ed_i2v_low[1].text()

        dataset_path = dataset_override if dataset_override else self.ed_toml[1].text()
        epochs = epochs_override if epochs_override > 0 else self.sb_epochs.value()

        shared = [
            "--task", task,
            "--blocks_to_swap", str(self.sb_blocks.value()),
            "--t5", self.ed_t5[1].text(),
            "--vae", self.ed_vae[1].text(),
            "--dataset_config", dataset_path,
            "--network_module", "networks.lora_wan",
            "--network_dim", str(self.sb_rank.value()),
            "--network_alpha", str(self.sb_alpha.value()),
            "--gradient_checkpointing",
            "--max_data_loader_n_workers", str(self.sb_workers.value()),
            "--persistent_data_loader_workers",
            "--timestep_sampling", "shift",
            "--discrete_flow_shift", f"{self.dsb_shift.value()}",
            "--preserve_distribution_shape",
            "--log_with", "tensorboard",
            "--logging_dir", self.ed_logdir[1].text(),
            "--seed", str(self.sb_seed.value()),
            "--network_dropout", f"{self.dsb_dropout.value()}",
            "--mixed_precision", "fp16",
            "--fp8_base", "--fp8_scaled", "--mixed_precision_transformer",
            "--dit", dit,
            "--min_timestep", str(int(min_t)), "--max_timestep", str(int(max_t)),
            "--max_train_epochs", str(epochs),
            "--output_dir", self.ed_outdir[1].text(),
            "--output_name", self.ed_run_name.text() + name_suffix,
            "--log_prefix", self.ed_run_name.text() + name_suffix,
        ]

        # Save controls
        if self.sb_save_epochs.value() > 0: shared += ["--save_every_n_epochs", str(self.sb_save_epochs.value())]
        if self.sb_save_steps.value()  > 0: shared += ["--save_every_n_steps",  str(self.sb_save_steps.value())]
        if self.sb_keep_last_epochs.value() > 0: shared += ["--save_last_n_epochs", str(self.sb_keep_last_epochs.value())]
        if self.sb_keep_last_steps.value()  > 0: shared += ["--save_last_n_steps",  str(self.sb_keep_last_steps.value())]
        if self.sb_keep_last_epochs_state.value() > 0: shared += ["--save_last_n_epochs_state", str(self.sb_keep_last_epochs_state.value())]
        if self.sb_keep_last_steps_state.value()  > 0: shared += ["--save_last_n_steps_state",  str(self.sb_keep_last_steps_state.value())]
        if self.cb_save_state.isChecked(): shared += ["--save_state"]
        if self.cb_save_state_on_end.isChecked(): shared += ["--save_state_on_train_end"]

        # Attention backend
        attn = self.cb_attn.currentText()
        env = os.environ.copy(); env.update(env_text_to_dict(self.txt_env.toPlainText()))
        attn_flags = []
        if attn == "FlashAttention":
            attn_flags = ["--flash_attn"]; env["XFORMERS_DISABLED"] = "1"
        elif attn == "xFormers":
            attn_flags = ["--xformers"]; env.pop("XFORMERS_DISABLED", None)
        else:
            env["XFORMERS_DISABLED"] = "1"

        # Optimizer
        optimizer = [
            "--optimizer_type", self.ed_opt_type.text().strip(),
            "--lr_scheduler", self.cb_sched.currentText(),
            "--learning_rate", self.ed_lr.text(),
            "--lr_warmup_steps", str(self.sb_warmup.value()),
        ]
        optimizer += kv_text_to_list_for("--optimizer_args", self.txt_opt_args.toPlainText())

        # Custom extra args (global + override)
        extra_tokens: List[str] = []
        for line in (self.txt_custom_args.toPlainText() + "\n" + (extra_override or "")).splitlines():
            s = line.strip()
            if not s or s.startswith("#"): continue
            extra_tokens += shlex.split(s)

        # accelerate command (let it warn by default; fewer surprises)
        cmd = ["accelerate", "launch", "--num_cpu_threads_per_process", str(self.sb_threads.value())]
        cmd += entry_parts + attn_flags + shared + optimizer + extra_tokens
        return cmd, env

    def build_cache_commands(self) -> Tuple[List[str], List[str], Dict[str, str]]:
        python_bin = sys.executable or "python"
        dataset = self.ed_toml[1].text(); batch = str(self.sb_cache_bs.value())
        mode = self.cb_cache_mode.currentText(); i2v_flag = ["--i2v"] if mode == "i2v" else []
        cache_extra: List[str] = []
        for line in self.txt_cache_custom.toPlainText().splitlines():
            s = line.strip()
            if not s or s.startswith("#"): continue
            cache_extra += shlex.split(s)

        entry_lat = shlex.split(self.ed_cache_entry_lat.text().strip())
        cmd_lat = [python_bin] + entry_lat + ["--dataset_config", dataset, "--vae", self.ed_vae[1].text(), "--batch_size", batch] + i2v_flag + cache_extra

        entry_txt = shlex.split(self.ed_cache_entry_txt.text().strip())
        cmd_txt = [python_bin] + entry_txt + ["--dataset_config", dataset, "--t5", self.ed_t5[1].text(), "--batch_size", batch] + cache_extra

        env = os.environ.copy(); env.update(env_text_to_dict(self.txt_env.toPlainText()))
        return cmd_lat, cmd_txt, env

    # ---------- Handlers ----------
    def on_build_only(self):
        # Warn if obvious required paths are blank
        needed = {"T5": self.ed_t5[1].text(), "VAE": self.ed_vae[1].text(), "Dataset TOML": self.ed_toml[1].text()}
        if self.cb_task.currentText().startswith("t2v"):
            needed["DIT (t2v high/low)"] = (self.ed_t2v_high[1].text() if self.cb_noise.currentText()=="high" else self.ed_t2v_low[1].text())
        else:
            needed["DIT (i2v high/low)"] = (self.ed_i2v_high[1].text() if self.cb_noise.currentText()=="high" else self.ed_i2v_low[1].text())
        if not warn_missing_paths(self, needed): return

        if self.cb_dual.isChecked():
            cmd_h, _ = self.build_train_command(
                noise_override="high",
                name_suffix=self.ed_dual_high_suffix.text(),
                dataset_override=getattr(self.ed_dual_dataset_high, "_lineedit").text(),
                epochs_override=self.sb_dual_epochs_high.value(),
                extra_override=self.txt_dual_extra_high.toPlainText(),
            )
            cmd_l, _ = self.build_train_command(
                noise_override="low",
                name_suffix=self.ed_dual_low_suffix.text(),
                dataset_override=getattr(self.ed_dual_dataset_low, "_lineedit").text(),
                epochs_override=self.sb_dual_epochs_low.value(),
                extra_override=self.txt_dual_extra_low.toPlainText(),
            )
            self.log.appendPlainText("$ " + " ".join(shlex.quote(c) for c in cmd_h))
            self.log.appendPlainText("$ " + " ".join(shlex.quote(c) for c in cmd_l))
        else:
            cmd, env = self.build_train_command()
            self.log.appendPlainText("$ " + " ".join(shlex.quote(c) for c in cmd))
            show = " ".join(f'{k}=\"{v}\"' for k,v in env.items() if k in ("PYTORCH_CUDA_ALLOC_CONF","TORCH_F32_MATMUL_PRECISION","TORCH_FLOAT32_MATMUL_PRECISION","XFORMERS_DISABLED"))
            if show: self.log.appendPlainText(f"# ENV: {show}")

    def on_start(self):
        needed = {"T5": self.ed_t5[1].text(), "VAE": self.ed_vae[1].text(), "Dataset TOML": self.ed_toml[1].text()}
        if self.cb_task.currentText().startswith("t2v"):
            needed["DIT (t2v high/low)"] = (self.ed_t2v_high[1].text() if self.cb_noise.currentText()=="high" else self.ed_t2v_low[1].text())
        else:
            needed["DIT (i2v high/low)"] = (self.ed_i2v_high[1].text() if self.cb_noise.currentText()=="high" else self.ed_i2v_low[1].text())
        if not warn_missing_paths(self, needed): return

        if self.cb_autosave.isChecked(): self.auto_save_profile()

        try:
            if self.cb_dual.isChecked():
                cmd_h, env = self.build_train_command(
                    noise_override="high",
                    name_suffix=self.ed_dual_high_suffix.text(),
                    dataset_override=getattr(self.ed_dual_dataset_high, "_lineedit").text(),
                    epochs_override=self.sb_dual_epochs_high.value(),
                    extra_override=self.txt_dual_extra_high.toPlainText(),
                )
                cmd_l, _   = self.build_train_command(
                    noise_override="low",
                    name_suffix=self.ed_dual_low_suffix.text(),
                    dataset_override=getattr(self.ed_dual_dataset_low, "_lineedit").text(),
                    epochs_override=self.sb_dual_epochs_low.value(),
                    extra_override=self.txt_dual_extra_low.toPlainText(),
                )
                self.log.appendPlainText("$ " + " ".join(shlex.quote(c) for c in cmd_h))
                self.streamer.start(cmd_h, env=env)
                def run_low(rc):
                    self.streamer.done.disconnect(run_low)
                    if rc != 0:
                        self.log.appendPlainText("[HIGH phase failed; not running LOW phase]"); self.btn_start.setEnabled(True); return
                    self.log.appendPlainText("$ " + " ".join(shlex.quote(c) for c in cmd_l))
                    try:
                        self.streamer.start(cmd_l, env=env)
                    except Exception as e:
                        QMessageBox.critical(self, "Launch error", str(e))
                self.streamer.done.connect(run_low)
            else:
                cmd, env = self.build_train_command()
                self.log.appendPlainText("$ " + " ".join(shlex.quote(c) for c in cmd))
                self.streamer.start(cmd, env=env)
            self.btn_start.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Launch error", str(e))

    def on_stop(self):
        self.streamer.stop()
        self.btn_start.setEnabled(True)

    def on_log(self, text: str): self.log.appendPlainText(text)
    def on_done(self, rc: int): self.log.appendPlainText(f"[process exited {rc}]"); self.btn_start.setEnabled(True)

    def on_cache_build_only(self):
        cmd_lat, cmd_txt, env = self.build_cache_commands()
        self.log.appendPlainText("$ " + " ".join(shlex.quote(c) for c in cmd_lat))
        self.log.appendPlainText("$ " + " ".join(shlex.quote(c) for c in cmd_txt))

    def on_cache_run(self):
        cmd_lat, cmd_txt, env = self.build_cache_commands()
        try:
            self.log.appendPlainText("$ " + " ".join(shlex.quote(c) for c in cmd_lat))
            self.streamer.start(cmd_lat, env=env)
            def run_next(rc):
                self.streamer.done.disconnect(run_next)
                if rc != 0:
                    self.log.appendPlainText("[latents caching failed; not running text-encoder cache]"); self.btn_start.setEnabled(True); return
                self.log.appendPlainText("$ " + " ".join(shlex.quote(c) for c in cmd_txt))
                try:
                    self.streamer.start(cmd_txt, env=env)
                except Exception as e:
                    QMessageBox.critical(self, "Launch error", str(e))
            self.streamer.done.connect(run_next); self.btn_start.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Launch error", str(e))

    # ---------- Datasets (TOML) tab: actions ----------
    def _ds_path(self) -> str:
        return getattr(self.ed_ds_path, "_lineedit").text()
    def on_ds_load(self):
        p = self._ds_path()
        if not p:
            p, _ = QFileDialog.getOpenFileName(self, "Open TOML", str(Path.cwd()), "TOML (*.toml);;All (*)")
            if not p: return
            getattr(self.ed_ds_path, "_lineedit").setText(p)
        try:
            text = Path(p).read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e)); return
        self.txt_ds.setPlainText(text)
        self.log.appendPlainText(f"[loaded dataset toml] {p}")
    def on_ds_save(self):
        p = self._ds_path()
        if not p:
            QMessageBox.information(self, "Save", "Pick a TOML file path first (use the … button).")
            return
        try:
            Path(p).write_text(self.txt_ds.toPlainText(), encoding="utf-8")
            self.log.appendPlainText(f"[saved dataset toml] {p}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))
    def on_ds_save_as(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save TOML as…", str(Path.cwd()), "TOML (*.toml)")
        if not p: return
        try:
            Path(p).write_text(self.txt_ds.toPlainText(), encoding="utf-8")
            getattr(self.ed_ds_path, "_lineedit").setText(p)
            self.log.appendPlainText(f"[saved dataset toml as] {p}")
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))
    def on_ds_validate(self):
        if _toml_r is None:
            QMessageBox.information(self, "Validate", "tomllib/tomli not available in this Python; cannot parse for validation.")
            return
        try:
            # both tomllib and tomli accept str input for loads
            _toml_r.loads(self.txt_ds.toPlainText())
            self.log.appendPlainText("[TOML validation] OK")
            QMessageBox.information(self, "Validate", "TOML parsed successfully.")
        except Exception as e:
            self.log.appendPlainText(f"[TOML validation] ERROR: {e}")
            QMessageBox.critical(self, "Validate", f"Parse error:\n{e}")
    def on_ds_insert_video(self):
        tmpl = (
            '\n[[datasets]]\n'
            'video_directory = "/path/to/videos"\n'
            'cache_directory = "/path/to/cache"\n'
            'resolution = [640, 640]\n'
            'batch_size = 2\n'
            'num_repeats = 1\n'
            'frame_extraction = "uniform"\n'
            'frame_sample = 2\n'
            'target_frames = [21]\n'
            'source_fps = 16.0\n'
            'max_frames = 21\n'
        )
        self.txt_ds.appendPlainText(tmpl)
    def on_ds_insert_image(self):
        tmpl = (
            '\n[[datasets]]\n'
            'image_directory = "/path/to/images"\n'
            'cache_directory = "/path/to/cache"\n'
            'resolution = [768, 768]\n'
            'batch_size = 4\n'
            'num_repeats = 1\n'
        )
        self.txt_ds.appendPlainText(tmpl)

    # ---------- Profiles ----------
    def collect_profile(self) -> dict:
        data = {
            "paths": {
                "WAN_T2V_HIGH": self.ed_t2v_high[1].text(),
                "WAN_T2V_LOW":  self.ed_t2v_low[1].text(),
                "WAN_I2V_HIGH": self.ed_i2v_high[1].text(),
                "WAN_I2V_LOW":  self.ed_i2v_low[1].text(),
                "T5_PATH":      self.ed_t5[1].text(),
                "VAE_PATH":     self.ed_vae[1].text(),
                "DATASET_TOML": self.ed_toml[1].text(),
                "LOG_DIR":      self.ed_logdir[1].text(),
                "OUT_DIR":      self.ed_outdir[1].text(),
                "PROFILE_DIR":  self.get_profile_dir_text(),
            },
            "runtime": {
                "entrypoint": self.ed_entry.text(),
                "attention_backend": self.cb_attn.currentText(),
                "seed": self.sb_seed.value(),
                "acc_cpu_threads": self.sb_threads.value(),
                "blocks_to_swap": self.sb_blocks.value(),
                "lora_rank": self.sb_rank.value(),
                "lora_alpha": self.sb_alpha.value(),
                "network_dropout": self.dsb_dropout.value(),
                "n_workers": self.sb_workers.value(),
                "discrete_shift": self.dsb_shift.value(),
                "epochs": self.sb_epochs.value(),
                "env": env_text_to_dict(self.txt_env.toPlainText()),
            },
            "task": {
                "kind": self.cb_task.currentText(),
                "noise": self.cb_noise.currentText(),
                "min_t": self.sb_min_t.value(),
                "max_t": self.sb_max_t.value(),
            },
            "optimizer": {
                "preset": self.cb_opt.currentText(),
                "learning_rate": float(self.ed_lr.text() or 0),
                "lr_scheduler": self.cb_sched.currentText(),
                "lr_warmup_steps": self.sb_warmup.value(),
                "type_path": self.ed_opt_type.text(),
                "args_kv": self.txt_opt_args.toPlainText(),
            },
            "save": {
                "save_every_n_epochs": self.sb_save_epochs.value(),
                "save_every_n_steps": self.sb_save_steps.value(),
                "save_last_n_epochs": self.sb_keep_last_epochs.value(),
                "save_last_n_steps": self.sb_keep_last_steps.value(),
                "save_last_n_epochs_state": self.sb_keep_last_epochs_state.value(),
                "save_last_n_steps_state": self.sb_keep_last_steps_state.value(),
                "save_state": self.cb_save_state.isChecked(),
                "save_state_on_train_end": self.cb_save_state_on_end.isChecked(),
            },
            "dual": {
                "enabled": self.cb_dual.isChecked(),
                "high_suffix": self.ed_dual_high_suffix.text(),
                "low_suffix":  self.ed_dual_low_suffix.text(),
                "epochs_high": self.sb_dual_epochs_high.value(),
                "epochs_low":  self.sb_dual_epochs_low.value(),
                "dataset_high": getattr(self.ed_dual_dataset_high, "_lineedit").text(),
                "dataset_low":  getattr(self.ed_dual_dataset_low, "_lineedit").text(),
                "extra_high": self.txt_dual_extra_high.toPlainText(),
                "extra_low":  self.txt_dual_extra_low.toPlainText(),
            },
            "run_name": self.ed_run_name.text(),
            "custom_args": self.txt_custom_args.toPlainText(),
            "cache": {
                "entry_latents": self.ed_cache_entry_lat.text(),
                "entry_textenc": self.ed_cache_entry_txt.text(),
                "batch_size": self.sb_cache_bs.value(),
                "mode": self.cb_cache_mode.currentText(),
                "custom_args": self.txt_cache_custom.toPlainText(),
            },
            "profile_base_name": self.ed_profile_name.text(),
        }
        return data

    def apply_profile(self, data: dict):
        self.ed_t2v_high[1].setText(data["paths"]["WAN_T2V_HIGH"])
        self.ed_t2v_low [1].setText(data["paths"]["WAN_T2V_LOW"])
        self.ed_i2v_high[1].setText(data["paths"]["WAN_I2V_HIGH"])
        self.ed_i2v_low [1].setText(data["paths"]["WAN_I2V_LOW"])
        self.ed_t5      [1].setText(data["paths"]["T5_PATH"])
        self.ed_vae     [1].setText(data["paths"]["VAE_PATH"])
        self.ed_toml    [1].setText(data["paths"]["DATASET_TOML"])
        self.ed_logdir  [1].setText(data["paths"]["LOG_DIR"])
        self.ed_outdir  [1].setText(data["paths"]["OUT_DIR"])
        if "PROFILE_DIR" in data["paths"] and hasattr(self.ed_profile_dir, "_lineedit"):
            self.ed_profile_dir._lineedit.setText(data["paths"]["PROFILE_DIR"])

        self.ed_entry.setText(data["runtime"]["entrypoint"])
        self.cb_attn.setCurrentText(data["runtime"].get("attention_backend", "SDPA"))
        self.sb_seed.setValue(int(data["runtime"]["seed"]))
        self.sb_threads.setValue(int(data["runtime"]["acc_cpu_threads"]))
        self.sb_blocks.setValue(int(data["runtime"]["blocks_to_swap"]))
        self.sb_rank.setValue(int(data["runtime"]["lora_rank"]))
        self.sb_alpha.setValue(int(data["runtime"]["lora_alpha"]))
        self.dsb_dropout.setValue(float(data["runtime"]["network_dropout"]))
        self.sb_workers.setValue(int(data["runtime"]["n_workers"]))
        self.dsb_shift.setValue(float(data["runtime"]["discrete_shift"]))
        self.sb_epochs.setValue(int(data["runtime"]["epochs"]))
        self.txt_env.setPlainText("\n".join(f"{k}={v}" for k,v in data["runtime"].get("env", {}).items()))

        self.cb_task.setCurrentText(data["task"]["kind"])
        self.cb_noise.setCurrentText(data["task"]["noise"])
        self.sb_min_t.setValue(int(data["task"]["min_t"]))
        self.sb_max_t.setValue(int(data["task"]["max_t"]))

        self.cb_opt.setCurrentText(data["optimizer"]["preset"] if data["optimizer"]["preset"] in list(OPT_PRESETS.keys()) + ["Custom"] else "Custom")
        self.ed_opt_type.setText(data["optimizer"]["type_path"])
        self.ed_lr.setText(str(data["optimizer"]["learning_rate"]))
        self.cb_sched.setCurrentText(data["optimizer"]["lr_scheduler"])
        self.sb_warmup.setValue(int(data["optimizer"]["lr_warmup_steps"]))
        self.txt_opt_args.setPlainText(data["optimizer"]["args_kv"])

        save = data.get("save", {})
        self.sb_save_epochs.setValue(int(save.get("save_every_n_epochs", 0)))
        self.sb_save_steps.setValue(int(save.get("save_every_n_steps", 0)))
        self.sb_keep_last_epochs.setValue(int(save.get("save_last_n_epochs", 0)))
        self.sb_keep_last_steps.setValue(int(save.get("save_last_n_steps", 0)))
        self.sb_keep_last_epochs_state.setValue(int(save.get("save_last_n_epochs_state", 0)))
        self.sb_keep_last_steps_state.setValue(int(save.get("save_last_n_steps_state", 0)))
        self.cb_save_state.setChecked(bool(save.get("save_state", False)))
        self.cb_save_state_on_end.setChecked(bool(save.get("save_state_on_train_end", False)))

        dual = data.get("dual", {})
        self.cb_dual.setChecked(bool(dual.get("enabled", False)))
        self.ed_dual_high_suffix.setText(dual.get("high_suffix", "_HIGH"))
        self.ed_dual_low_suffix.setText(dual.get("low_suffix", "_LOW"))
        self.sb_dual_epochs_high.setValue(int(dual.get("epochs_high", 0)))
        self.sb_dual_epochs_low.setValue(int(dual.get("epochs_low", 0)))
        getattr(self.ed_dual_dataset_high, "_lineedit").setText(dual.get("dataset_high", ""))
        getattr(self.ed_dual_dataset_low, "_lineedit").setText(dual.get("dataset_low", ""))
        self.txt_dual_extra_high.setPlainText(dual.get("extra_high", ""))
        self.txt_dual_extra_low.setPlainText(dual.get("extra_low", ""))

        self.ed_run_name.setText(data.get("run_name", "Wan2.2_Run"))
        self.txt_custom_args.setPlainText(data.get("custom_args", ""))

        cache = data.get("cache", {})
        self.ed_cache_entry_lat.setText(cache.get("entry_latents", DEFAULTS["cache"]["entry_latents"]))
        self.ed_cache_entry_txt.setText(cache.get("entry_textenc", DEFAULTS["cache"]["entry_textenc"]))
        self.sb_cache_bs.setValue(int(cache.get("batch_size", DEFAULTS["cache"]["batch_size"])))
        self.cb_cache_mode.setCurrentText(cache.get("mode", DEFAULTS["cache"]["mode"]))
        self.txt_cache_custom.setPlainText(cache.get("custom_args", ""))

        if "profile_base_name" in data: self.ed_profile_name.setText(data["profile_base_name"])

    def on_save(self):
        data = self.collect_profile()
        fn, _ = QFileDialog.getSaveFileName(self, "Save profile", str(Path(self.get_profile_dir_text()) / (self.ed_profile_name.text() + ".json")), "JSON (*.json)")
        if fn:
            ensure_dir(Path(fn).parent)
            with open(fn, "w") as f: json.dump(data, f, indent=2)
            self.log.appendPlainText(f"[saved profile] {fn}")

    def on_load(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load profile", str(Path(self.get_profile_dir_text() or str(Path.home()))), "JSON (*.json)")
        if not fn: return
        try:
            with open(fn, "r") as f: data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e)); return
        self.apply_profile(data); self.log.appendPlainText(f"[loaded profile] {fn}")

    def auto_save_profile(self):
        data = self.collect_profile()
        base_dir = Path(self.get_profile_dir_text() or (Path.home() / "wan_profiles"))
        ensure_dir(base_dir); base = base_dir / self.ed_profile_name.text()
        path = unique_path(base, ".json")
        try:
            with open(path, "w") as f: json.dump(data, f, indent=2)
            self.log.appendPlainText(f"[auto-saved profile] {path}")
        except Exception as e:
            self.log.appendPlainText(f"[auto-save failed] {e}")

def main():
    app = QApplication(sys.argv); w = Main(); w.show(); sys.exit(app.exec())

if __name__ == "__main__":
    main()
