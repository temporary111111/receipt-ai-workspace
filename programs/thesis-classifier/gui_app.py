#!/usr/bin/env python
"""
GUI Application for AI vs Real Receipt Classifier
Simple tkinter desktop app - select image, get prediction with explanations
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import subprocess
import sys
from pathlib import Path
import os

# Paths
PREDICT_SCRIPT = Path(__file__).parent / "predict_receipt.py"
MODEL_PATH = Path(__file__).parent / "models" / "best_classifier.pkl"
PYTHON_EXE = r"C:\Users\dev\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
PYTHONPATH = r"C:\Users\dev\Workspace\school\elective\programs\receipt-ai-text-cropper-paddleocr\python_deps"


class ReceiptClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI vs Real Receipt Classifier")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        self.current_image_path = None
        self.result_data = None

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="AI vs Real Receipt Classifier",
                                font=("Segoe UI", 16, "bold"))
        title_label.pack(pady=(0, 10))

        # Top section: Image selection and preview
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # Left: Image preview
        preview_frame = ttk.LabelFrame(top_frame, text="Selected Image", padding="10")
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.image_label = ttk.Label(preview_frame, text="No image selected\nClick 'Browse' to select a receipt image",
                                     anchor=tk.CENTER, font=("Segoe UI", 10))
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Right: Controls
        control_frame = ttk.LabelFrame(top_frame, text="Controls", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0))

        self.browse_btn = ttk.Button(control_frame, text="Browse Image",
                                     command=self.browse_image, width=20)
        self.browse_btn.pack(pady=(0, 10), fill=tk.X)

        self.predict_btn = ttk.Button(control_frame, text="Analyze Receipt",
                                      command=self.start_prediction, width=20, state=tk.DISABLED)
        self.predict_btn.pack(pady=(0, 10), fill=tk.X)

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 10))

        self.status_label = ttk.Label(control_frame, text="Ready", font=("Segoe UI", 9))
        self.status_label.pack(pady=(0, 10))

        # Device selection
        ttk.Label(control_frame, text="Device:").pack(anchor=tk.W)
        self.device_var = tk.StringVar(value="cpu")
        device_combo = ttk.Combobox(control_frame, textvariable=self.device_var,
                                    values=["cpu", "gpu:0"], state="readonly", width=15)
        device_combo.pack(fill=tk.X, pady=(0, 10))

        # Keep temp files checkbox
        self.keep_temps_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Keep temp crop files",
                        variable=self.keep_temps_var).pack(anchor=tk.W)

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Prediction Summary
        self.summary_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.summary_frame, text="Summary")

        self.setup_summary_tab()

        # Tab 2: Detailed Explanation
        self.explanation_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.explanation_frame, text="Why?")

        self.setup_explanation_tab()

        # Tab 3: Raw Features
        self.features_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.features_frame, text="Features")

        self.setup_features_tab()

        # Tab 4: JSON Output
        self.json_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.json_frame, text="Raw JSON")

        self.setup_json_tab()

    def setup_summary_tab(self):
        # Prediction result
        self.pred_label = ttk.Label(self.summary_frame, text="",
                                    font=("Segoe UI", 24, "bold"), foreground="#2E86AB")
        self.pred_label.pack(pady=20)

        self.conf_label = ttk.Label(self.summary_frame, text="",
                                    font=("Segoe UI", 12))
        self.conf_label.pack(pady=5)

        # Separator
        ttk.Separator(self.summary_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)

        # Key insights
        ttk.Label(self.summary_frame, text="Key Insights", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        self.insights_text = tk.Text(self.summary_frame, height=12, wrap=tk.WORD,
                                     font=("Consolas", 10), state=tk.DISABLED)
        self.insights_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Scrollbar for insights
        insights_scroll = ttk.Scrollbar(self.insights_text, command=self.insights_text.yview)
        insights_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.insights_text.config(yscrollcommand=insights_scroll.set)

    def setup_explanation_tab(self):
        ttk.Label(self.explanation_frame, text="Feature Comparison to Training Data (900 AI + 900 Real)",
                  font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Treeview for explanation
        columns = ("Feature", "Value", "Real Mean", "AI Mean", "Closer To", "Signal")
        self.expl_tree = ttk.Treeview(self.explanation_frame, columns=columns, show="headings", height=16)

        for col in columns:
            self.expl_tree.heading(col, text=col)
            if col == "Feature":
                self.expl_tree.column(col, width=220, anchor=tk.W)
            elif col in ("Closer To", "Signal"):
                self.expl_tree.column(col, width=100, anchor=tk.CENTER)
            else:
                self.expl_tree.column(col, width=120, anchor=tk.CENTER)

        self.expl_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Scrollbar
        expl_scroll = ttk.Scrollbar(self.explanation_frame, orient=tk.VERTICAL, command=self.expl_tree.yview)
        expl_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.expl_tree.config(yscrollcommand=expl_scroll.set)

        # Legend
        legend_frame = ttk.Frame(self.explanation_frame)
        legend_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(legend_frame, text="Legend: [AI] = closer to AI-generated mean  |  [RL] = closer to Real mean  |  Signal = how discriminative",
                  font=("Segoe UI", 9), foreground="gray").pack(anchor=tk.W)

    def setup_features_tab(self):
        columns = ("Feature", "Value")
        self.feat_tree = ttk.Treeview(self.features_frame, columns=columns, show="headings", height=16)

        self.feat_tree.heading("Feature", text="Feature")
        self.feat_tree.heading("Value", text="Extracted Value")
        self.feat_tree.column("Feature", width=300, anchor=tk.W)
        self.feat_tree.column("Value", width=200, anchor=tk.CENTER)

        self.feat_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        feat_scroll = ttk.Scrollbar(self.features_frame, orient=tk.VERTICAL, command=self.feat_tree.yview)
        feat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.feat_tree.config(yscrollcommand=feat_scroll.set)

    def setup_json_tab(self):
        self.json_text = tk.Text(self.json_frame, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.json_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        json_scroll = ttk.Scrollbar(self.json_frame, orient=tk.VERTICAL, command=self.json_text.yview)
        json_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.json_text.config(yscrollcommand=json_scroll.set)

    def browse_image(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select Receipt Image", filetypes=filetypes)
        if path:
            self.current_image_path = Path(path)
            self.display_image_preview(path)
            self.predict_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Selected: {Path(path).name}")
            self.clear_results()

    def display_image_preview(self, path):
        try:
            from PIL import Image, ImageTk
            img = Image.open(path)
            # Resize to fit preview area (max 400x300)
            img.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
        except Exception as e:
            self.image_label.config(image="", text=f"Preview error: {e}")

    def clear_results(self):
        self.pred_label.config(text="")
        self.conf_label.config(text="")
        self.insights_text.config(state=tk.NORMAL)
        self.insights_text.delete(1.0, tk.END)
        self.insights_text.config(state=tk.DISABLED)
        for item in self.expl_tree.get_children():
            self.expl_tree.delete(item)
        for item in self.feat_tree.get_children():
            self.feat_tree.delete(item)
        self.json_text.config(state=tk.NORMAL)
        self.json_text.delete(1.0, tk.END)
        self.json_text.config(state=tk.DISABLED)

    def start_prediction(self):
        if not self.current_image_path:
            return

        self.predict_btn.config(state=tk.DISABLED)
        self.browse_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Running text detection...")
        self.notebook.select(0)  # Switch to summary tab

        # Run in background thread
        thread = threading.Thread(target=self.run_prediction, daemon=True)
        thread.start()

    def run_prediction(self):
        try:
            cmd = [
                PYTHON_EXE,
                str(PREDICT_SCRIPT),
                str(self.current_image_path),
                "--model", str(MODEL_PATH),
                "--device", self.device_var.get(),
            ]
            if self.keep_temps_var.get():
                cmd.append("--keep-temps")

            env = os.environ.copy()
            env["PYTHONPATH"] = PYTHONPATH

            # Run subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)

            # Find JSON output
            json_path = self.current_image_path.with_suffix('.prediction.json')
            if json_path.exists():
                with open(json_path) as f:
                    self.result_data = json.load(f)
                self.root.after(0, self.on_prediction_complete)
            else:
                error_msg = result.stderr if result.stderr else "No output file generated"
                self.root.after(0, lambda: self.on_prediction_error(error_msg))

        except subprocess.TimeoutExpired:
            self.root.after(0, lambda: self.on_prediction_error("Timeout (5 min)"))
        except Exception as e:
            self.root.after(0, lambda: self.on_prediction_error(str(e)))

    def on_prediction_complete(self):
        self.progress.stop()
        self.predict_btn.config(state=tk.NORMAL)
        self.browse_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Done!")

        if self.result_data:
            self.display_results()

    def on_prediction_error(self, error):
        self.progress.stop()
        self.predict_btn.config(state=tk.NORMAL)
        self.browse_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Error")
        messagebox.showerror("Prediction Error", f"Failed to analyze image:\n{error}")

    def display_results(self):
        r = self.result_data

        # Summary tab
        pred = r.get('prediction', -1)
        prob_ai = r.get('probability_ai', 0)
        prob_real = r.get('probability_real', 0)

        if pred == 1:
            self.pred_label.config(text="AI-GENERATED", foreground="#E74C3C")
            self.conf_label.config(text=f"Confidence: {prob_ai:.1%} AI")
        else:
            self.pred_label.config(text="REAL RECEIPT", foreground="#27AE60")
            self.conf_label.config(text=f"Confidence: {prob_real:.1%} Real")

        # Insights
        self.insights_text.config(state=tk.NORMAL)
        self.insights_text.delete(1.0, tk.END)

        insights = []
        if 'explanation' in r and r['explanation']:
            insights.append("TOP DISCRIMINATIVE FEATURES:")
            insights.append("-" * 50)
            for i, (feat, info) in enumerate(list(r['explanation'].items())[:6], 1):
                val = info['value']
                closer = info['closer_to']
                strength = info['signal_strength']
                real_m = info['real_mean']
                ai_m = info['ai_mean']
                arrow = "AI" if closer == "AI" else "Real"
                insights.append(f"{i}. {feat}")
                insights.append(f"   Value: {val:.4f}  |  Real mean: {real_m:.4f}  |  AI mean: {ai_m:.4f}")
                insights.append(f"   → Closer to: {arrow}  (signal: {strength:.0%})")
                insights.append("")

        self.insights_text.insert(1.0, "\n".join(insights))
        self.insights_text.config(state=tk.DISABLED)

        # Explanation tab
        for item in self.expl_tree.get_children():
            self.expl_tree.delete(item)

        if 'explanation' in r:
            for feat, info in list(r['explanation'].items())[:14]:
                val = info['value']
                closer = info['closer_to']
                strength = info['signal_strength']
                real_m = info['real_mean']
                ai_m = info['ai_mean']

                # Visual signal bar
                bar_len = int(strength * 15)
                bar = "█" * bar_len + "░" * (15 - bar_len)

                tag = "ai" if closer == "AI" else "real"
                self.expl_tree.insert("", tk.END, values=(
                    feat, f"{val:.4f}", f"{real_m:.4f}", f"{ai_m:.4f}",
                    f"[{closer}]", f"{bar} {strength:.0%}"
                ), tags=(tag,))

        self.expl_tree.tag_configure("ai", foreground="#E74C3C")
        self.expl_tree.tag_configure("real", foreground="#27AE60")

        # Features tab
        for item in self.feat_tree.get_children():
            self.feat_tree.delete(item)

        if 'features' in r:
            for feat, val in r['features'].items():
                self.feat_tree.insert("", tk.END, values=(feat, f"{val:.4f}" if val is not None else "N/A"))

        # JSON tab
        self.json_text.config(state=tk.NORMAL)
        self.json_text.delete(1.0, tk.END)
        self.json_text.insert(1.0, json.dumps(r, indent=2, default=str))
        self.json_text.config(state=tk.DISABLED)


def main():
    # Check if model exists
    if not MODEL_PATH.exists():
        messagebox.showerror("Model Not Found",
                             f"Trained model not found at:\n{MODEL_PATH}\n\n"
                             "Please run train_classifier.py first.")
        return

    # Check if predict script exists
    if not PREDICT_SCRIPT.exists():
        messagebox.showerror("Script Not Found",
                             f"predict_receipt.py not found at:\n{PREDICT_SCRIPT}")
        return

    root = tk.Tk()
    app = ReceiptClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()