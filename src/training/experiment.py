import csv
import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(
        self,
        results_dir: str = "results",
        run_name: str | None = None,
        config: dict[str, Any] | Any | None = None,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"run_{timestamp}"
        self.run_dir = Path(results_dir) / run_name
        self.samples_dir = self.run_dir / "samples"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        self.summary_path = self.run_dir / "summary.json"
        self.curve_path = self.run_dir / "loss_curve.svg"
        self.experiment_log_path = self.run_dir / "experiment_log.md"

        self.metrics: list[dict[str, Any]] = []
        self.start_time = time.time()

        if config is not None:
            self.save_config(config)

    def save_config(self, config: dict[str, Any] | Any) -> None:
        if is_dataclass(config):
            config = asdict(config)
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        record = dict(metrics)
        record.setdefault("wallclock_time", time.time() - self.start_time)
        self.metrics.append(record)

        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._write_metrics_csv()
        self._write_loss_curve()
        self._write_summary()

    def save_text_sample(
        self,
        name: str,
        prompt: str,
        generated_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "prompt": prompt,
            "generated_text": generated_text,
            "metadata": metadata or {},
        }
        with open(self.samples_dir / f"{name}.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        with open(self.samples_dir / f"{name}.txt", "w", encoding="utf-8") as handle:
            handle.write(generated_text)

    def append_experiment_note(self, title: str, body: str) -> None:
        with open(self.experiment_log_path, "a", encoding="utf-8") as handle:
            handle.write(f"## {title}\n\n{body}\n\n")

    def _write_metrics_csv(self) -> None:
        if not self.metrics:
            return

        fieldnames: list[str] = []
        for record in self.metrics:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.metrics:
                writer.writerow(record)

    def _write_summary(self) -> None:
        if not self.metrics:
            return

        latest = self.metrics[-1]
        summary = {
            "num_records": len(self.metrics),
            "latest": latest,
            "best_train_loss": min(
                record["train_loss"]
                for record in self.metrics
                if "train_loss" in record
            ),
        }
        val_losses = [record["val_loss"] for record in self.metrics if "val_loss" in record]
        if val_losses:
            summary["best_val_loss"] = min(val_losses)

        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    def _write_loss_curve(self) -> None:
        train_points = [
            (float(record["iteration"]), float(record["train_loss"]))
            for record in self.metrics
            if "iteration" in record and "train_loss" in record
        ]
        val_points = [
            (float(record["iteration"]), float(record["val_loss"]))
            for record in self.metrics
            if "iteration" in record and "val_loss" in record
        ]

        if not train_points and not val_points:
            return

        all_points = train_points + val_points
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)

        width = 720
        height = 360
        pad = 40

        def project(points: list[tuple[float, float]]) -> str:
            if not points:
                return ""
            if max_x == min_x:
                x_scale = 1.0
            else:
                x_scale = (width - 2 * pad) / (max_x - min_x)
            if max_y == min_y:
                y_scale = 1.0
            else:
                y_scale = (height - 2 * pad) / (max_y - min_y)

            projected = []
            for x, y in points:
                px = pad + (x - min_x) * x_scale
                py = height - pad - (y - min_y) * y_scale
                projected.append(f"{px:.2f},{py:.2f}")
            return " ".join(projected)

        train_polyline = project(train_points)
        val_polyline = project(val_points)

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#222" stroke-width="1"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#222" stroke-width="1"/>
<text x="{pad}" y="24" font-size="16" font-family="monospace">Loss Curve</text>
<text x="{pad}" y="{height - 10}" font-size="12" font-family="monospace">step</text>
<text x="8" y="{pad}" font-size="12" font-family="monospace">loss</text>
{f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{escape(train_polyline)}"/>' if train_polyline else ''}
{f'<polyline fill="none" stroke="#d62728" stroke-width="2" points="{escape(val_polyline)}"/>' if val_polyline else ''}
<text x="{width - 180}" y="24" font-size="12" font-family="monospace" fill="#1f77b4">train_loss</text>
<text x="{width - 90}" y="24" font-size="12" font-family="monospace" fill="#d62728">val_loss</text>
</svg>
"""
        with open(self.curve_path, "w", encoding="utf-8") as handle:
            handle.write(svg)
