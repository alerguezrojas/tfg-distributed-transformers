"""ReportFormatter — prints the human-readable report and writes the CSV."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from src.benchmark.value_objects import (
    ModelInfo, CPUInfo, DiskInfo, DatasetProfile, PerformancePrediction, BenchmarkReport,
)
from src.benchmark.time_estimator import TimeEstimator
from src.benchmark.benchmarker import Benchmarker


class ReportFormatter:
    W = 72

    def __init__(self, output_path: Path | None = None):
        self._output_path = output_path
        self._lines: list[str] = []

    def _emit(self, line: str = ""):
        print(line)
        self._lines.append(line)

    def flush(self):
        if self._output_path:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_path.write_text("\n".join(self._lines) + "\n")
            print(f"\n  Informe guardado en: {self._output_path}")

    def print(self, report: BenchmarkReport):
        self._header(report)
        self._model_section(report.model_info)
        self._hardware_section(report)
        self._cpu_section(report.cpu_info)
        self._disk_section(report.disk_info, report.dataset_profile)
        self._memory_section(report)
        self._benchmark_section(report)
        self._estimates_section(report)
        self._ddp_section(report)
        self._prediction_section(report.performance_prediction)
        self._study_section(report)
        self._recommendations_section(report)
        self.flush()

    def _header(self, report: BenchmarkReport):
        self._emit("═" * self.W)
        self._emit("  ANÁLISIS DE VIABILIDAD — BigEarthNet ViT  (v3)")
        self._emit(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        if report.nfs_factor != 1.0:
            self._emit(f"  Factor NFS aplicado: ×{report.nfs_factor:.2f}")
        self._emit("═" * self.W)

    def _model_section(self, m: ModelInfo):
        self._emit(f"\n{'─'*self.W}  MODELO")
        self._emit(f"  Nombre:               {m.name}")
        self._emit(f"  Parámetros totales:   {m.total_params:,} ({m.total_params/1e6:.1f}M)")
        if m.flops_per_image_mflops:
            self._emit(f"  FLOPs/imagen:         {m.flops_per_image_mflops:.1f} MFLOPs")
        self._emit(f"  Memoria estática:     {m.total_static_mb/1024:.2f} GB")

    def _hardware_section(self, report: BenchmarkReport):
        h = report.hardware_info
        self._emit(f"\n{'─'*self.W}  GPU")
        if h.is_cuda:
            self._emit(f"  {h.device_name}  |  VRAM: {h.total_vram_gb:.1f} GB total / {h.free_vram_gb:.1f} GB libre")
            if h.compute_capability:
                self._emit(f"  Compute Capability: {h.compute_capability}  ({h.architecture})")
            if h.cuda_cores:
                self._emit(f"  {h.sm_count} SMs  |  {h.cuda_cores:,} CUDA cores  |  "
                           f"{h.tensor_cores:,} Tensor cores")
            self._emit(f"  Precisión del benchmark: {report.precision} "
                       f"({'Tensor cores' if report.precision != 'fp32' else 'CUDA cores'})")
            pc = report.precision_comparison
            if pc:
                self._emit(f"  FP32 {pc['fp32_imgs_s']:.0f} img/s  vs  "
                           f"{pc['tc_precision'].upper()} {pc['tc_imgs_s']:.0f} img/s  "
                           f"→ {pc['speedup']}× con Tensor cores "
                           f"(VRAM {pc['fp32_vram_gb']} → {pc['tc_vram_gb']} GB)")
        else:
            self._emit("  GPU: no disponible (CUDA no activo)")

    def _cpu_section(self, cpu: CPUInfo | None):
        if cpu is None:
            return
        self._emit(f"\n{'─'*self.W}  CPU / SISTEMA")
        self._emit(f"  Núcleos: {cpu.logical_cores} lógicos / {cpu.physical_cores} físicos")
        if cpu.freq_mhz:
            self._emit(f"  Frecuencia: {cpu.freq_mhz:.0f} MHz")
        self._emit(f"  RAM: {cpu.ram_total_gb:.1f} GB total / {cpu.ram_free_gb:.1f} GB libre")

    def _disk_section(self, disk: DiskInfo | None, profile: DatasetProfile | None):
        if disk is None and profile is None:
            return
        self._emit(f"\n{'─'*self.W}  DISCO / DATASET I/O")
        if disk:
            self._emit(f"  Tipo: {disk.disk_type}  |  NFS: {'sí' if disk.is_nfs else 'no'}")
            if disk.read_mb_per_s > 0:
                self._emit(f"  Velocidad lectura: {disk.read_mb_per_s:.0f} MB/s  |  {disk.files_per_second:.0f} patches/s")
        if profile:
            self._emit(f"  Patches encontrados: ~{profile.n_files_total_est:,}")
            if profile.io_bottleneck_ratio > 0:
                if profile.io_bottleneck_ratio > 1.2:
                    self._emit(f"  Aviso: I/O-BOUND (ratio={profile.io_bottleneck_ratio:.2f}) — data loading más lento que cómputo")
                else:
                    self._emit(f"  Compute-bound (ratio={profile.io_bottleneck_ratio:.2f}) — GPU es el cuello de botella")

    def _memory_section(self, report: BenchmarkReport):
        m, h = report.model_info, report.hardware_info
        if not m.activation_mb_per_image:
            return
        self._emit(f"\n{'─'*self.W}  MEMORIA POR BATCH SIZE")
        self._emit(f"  {'Batch':>5}  {'Total est.':>11}  {'Estado':>10}")
        for bs in report.batch_sizes:
            total_gb = m.total_mb(bs) / 1024
            if h.is_cuda and total_gb > h.total_vram_gb:
                estado = "OOM"
            elif h.is_cuda and total_gb > h.total_vram_gb * 0.85:
                estado = "Límite"
            else:
                estado = "OK"
            self._emit(f"  {bs:>5}  {total_gb:>9.2f} GB  {estado:>10}")

    def _benchmark_section(self, report: BenchmarkReport):
        self._emit(f"\n{'─'*self.W}  BENCHMARK  ({Benchmarker.N_MEASURE} batches sintéticos)")
        self._emit(f"  {'Batch':>5}  {'Modo':<8}  {'imgs/s(train)':>13}  {'imgs/s(eval)':>12}  {'VRAM':>7}")
        for r in report.results:
            if r.oom:
                self._emit(f"  {r.batch_size:>5}  {r.trace_mode:<8}  {'OOM':>13}")
            else:
                self._emit(
                    f"  {r.batch_size:>5}  {r.trace_mode:<8}  "
                    f"{r.images_per_second_train:>11.1f}  "
                    f"{r.images_per_second_eval:>10.1f}  "
                    f"{r.peak_vram_gb:>5.2f} GB"
                )

    def _estimates_section(self, report: BenchmarkReport):
        estimator = TimeEstimator()
        nfs, mi = report.nfs_factor, report.model_info
        self._emit(f"\n{'─'*self.W}  ESTIMACIONES DE TIEMPO")
        for epochs in report.epochs_list:
            self._emit(f"\n  {epochs} epochs:")
            self._emit(f"  {'Batch':>5}  {'Modo':<8}  {'Train/ep':>8}  {'Eval/ep':>7}  {'Total/ep':>8}  {'TOTAL':>8}")
            for r in report.results:
                est = estimator.estimate(r, report.dataset_train, report.dataset_val, epochs, nfs, mi)
                if est is None:
                    self._emit(f"  {r.batch_size:>5}  {r.trace_mode:<8}  OOM")
                else:
                    self._emit(
                        f"  {r.batch_size:>5}  {r.trace_mode:<8}  "
                        f"{TimeEstimator.format_time(est['train_per_epoch']):>8}  "
                        f"{TimeEstimator.format_time(est['eval_per_epoch']):>7}  "
                        f"{TimeEstimator.format_time(est['total_per_epoch']):>8}  "
                        f"{TimeEstimator.format_time(est['total']):>8}"
                    )

    def _ddp_section(self, report: BenchmarkReport):
        if not report.ddp_scenarios:
            return
        self._emit(f"\n{'─'*self.W}  ANÁLISIS DDP — DISTRIBUCIÓN DE RECURSOS")
        self._emit(f"  {'GPUs':>4}  {'Batch/GPU':>9}  {'Global batch':>12}  {'Workers':>7}  {'Speedup':>7}  {'Efic.':>6}  {'Cuello':>8}")
        for s in report.ddp_scenarios:
            self._emit(
                f"  {s.n_gpus:>4}  {s.batch_per_gpu:>9}  {s.global_batch:>12}  "
                f"{s.num_workers_per_gpu:>7}  {s.estimated_speedup:>6.2f}×  "
                f"{s.scaling_efficiency:>5.1f}%  {s.bottleneck:>8}"
            )
        best = max(report.ddp_scenarios, key=lambda s: s.estimated_speedup / max(s.n_gpus, 1))
        self._emit(f"\n  Configuración recomendada: {best.n_gpus} GPU(s) con batch={best.batch_per_gpu}/GPU")
        if best.bottleneck == "io":
            self._emit("  Aviso: I/O es el cuello de botella — más GPUs no ayudarán sin disco más rápido")
        elif best.bottleneck == "sync":
            self._emit("  Aviso: Sincronización de gradientes es el cuello de botella — red lenta")

    def _prediction_section(self, pred: PerformancePrediction | None):
        if pred is None:
            return
        self._emit(f"\n{'─'*self.W}  PREDICCIÓN DE RENDIMIENTO (empírica)")
        self._emit(f"  Modelo: {pred.model_name}")
        self._emit(f"  Val F1 esperado:    ~{pred.predicted_best_f1:.3f}  (epoch ≈ {pred.predicted_best_epoch})")
        self._emit(f"  Early stop aprox.:  epoch ≈ {pred.predicted_early_stop_epoch}  (patience=10)")
        self._emit(f"  Confianza:          {pred.confidence}")
        self._emit(f"  Nota: {pred.notes[:100]}")

    def _study_section(self, report: BenchmarkReport):
        study = report.study_report
        if study is None:
            return
        self._emit(f"\n{'─'*self.W}  ESTUDIO EMPÍRICO DE CONVERGENCIA (medido)")

        if getattr(study, "lr_range", None):
            lr = study.lr_range
            self._emit("  LR range test:")
            self._emit(f"    LR sugerido (mayor descenso): {lr.suggested_lr:.2e}")
            self._emit(f"    LR del mínimo de loss:        {lr.min_loss_lr:.2e}")
            if lr.diverged_lr:
                self._emit(f"    LR de divergencia:            {lr.diverged_lr:.2e}")

        if getattr(study, "convergence", None):
            cv = study.convergence
            self._emit("  Convergencia (mini-training real):")
            self._emit(f"    Steps medidos:        {len(cv.steps)}  |  throughput {cv.measured_imgs_per_s:.1f} imgs/s")
            self._emit(f"    Ajuste loss=a·t^-b+c: a={cv.fit_a:.3f}, b={cv.fit_b:.3f}, c={cv.fit_c:.3f}  (R²={cv.r_squared:.3f})")
            self._emit(f"    Loss extrapolada 1 epoch:  {cv.extrapolated_loss_1ep:.4f}")
            self._emit(f"    Loss extrapolada final:    {cv.extrapolated_loss_final:.4f}")
            self._emit(f"    Val F1 estimado (medido):  ~{cv.extrapolated_best_f1:.3f}")
            self._emit(f"    Plateau estimado:          epoch ≈ {cv.epochs_to_plateau}")

        if getattr(study, "gradient_noise", None):
            gn = study.gradient_noise
            self._emit("  Gradient noise scale:")
            self._emit(f"    Norma gradiente: {gn.grad_norm_mean:.3f} ± {gn.grad_norm_std:.3f}  (CV={gn.cv:.3f})")
            self._emit(f"    Batch size sugerido: {gn.suggested_batch_size}  (noise scale ≈ {gn.noise_scale:.1f})")

    def _recommendations_section(self, report: BenchmarkReport):
        self._emit(f"\n{'─'*self.W}  RECOMENDACIONES")
        estimator = TimeEstimator()
        nfs = report.nfs_factor
        target_epochs = max(report.epochs_list)
        viable = [r for r in report.results if not r.oom and r.trace_mode == "off"]
        if not viable:
            self._emit("  Ningún batch size viable")
            return
        best = max(viable, key=lambda r: r.images_per_second_train)
        self._emit(f"  Batch size óptimo: {best.batch_size} ({best.images_per_second_train:.1f} imgs/s)")
        for mode in report.trace_modes:
            r = next((x for x in report.results if x.batch_size == best.batch_size and x.trace_mode == mode), None)
            if r and not r.oom:
                est = estimator.estimate(r, report.dataset_train, report.dataset_val, target_epochs, nfs, report.model_info)
                if est:
                    # Single-GPU estimate only here; the accurate distributed scaling
                    # (compute/IO/sync-aware) is the DDP-scenarios section below, not a
                    # flat-efficiency guess.
                    self._emit(
                        f"  → --trace {mode:<8} {target_epochs} epochs: "
                        f"~{TimeEstimator.format_time(est['total'])}"
                    )
        if nfs == 1.0:
            self._emit("\n  Nota: En Verode (NFS), usa --nfs-factor 1.3 para estimaciones más precisas")
        self._emit()

    def write_csv(self, report: BenchmarkReport, env: str = "local"):
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        out_dir = Path(f"logs/{env}/benchmark")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"benchmark_{timestamp}.csv"

        estimator = TimeEstimator()
        target_epochs = max(report.epochs_list) if report.epochs_list else 0
        mi, hi = report.model_info, report.hardware_info

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Metadata del modelo
            writer.writerow(["#meta", "model_name", "total_params_M", "flops_mflops",
                              "hardware_name", "total_vram_gb", "free_vram_gb"])
            writer.writerow(["#meta", mi.name, round(mi.total_params/1e6, 2),
                              round(mi.flops_per_image_mflops, 1), hi.device_name,
                              round(hi.total_vram_gb, 2), round(hi.free_vram_gb, 2)])

            # GPU hardware specs (compute capability × SM count)
            writer.writerow(["#gpu", "compute_capability", "architecture",
                              "sm_count", "cuda_cores", "tensor_cores"])
            writer.writerow(["#gpu", hi.compute_capability, hi.architecture,
                              hi.sm_count, hi.cuda_cores, hi.tensor_cores])

            # Precision used + optional FP32-vs-Tensor-core comparison
            writer.writerow(["#precision", "mode"])
            writer.writerow(["#precision", report.precision])
            pc = report.precision_comparison
            if pc:
                writer.writerow(["#precision_cmp", "batch_size", "tc_precision",
                                  "fp32_imgs_s", "tc_imgs_s", "speedup",
                                  "fp32_vram_gb", "tc_vram_gb"])
                writer.writerow(["#precision_cmp", pc.get("batch_size"), pc.get("tc_precision"),
                                  pc.get("fp32_imgs_s"), pc.get("tc_imgs_s"), pc.get("speedup"),
                                  pc.get("fp32_vram_gb"), pc.get("tc_vram_gb")])

            # Tamaño REAL del dataset usado (n imágenes por split) — clave para
            # que la comparación estimación-vs-real no asuma el full set.
            writer.writerow(["#sizes", "n_train", "n_val", "nfs_factor"])
            writer.writerow(["#sizes", report.dataset_train, report.dataset_val,
                              report.nfs_factor])

            # Memoria del modelo
            writer.writerow(["#model_mem", "weight_mb", "gradient_mb", "optimizer_mb",
                              "activation_mb_per_image", "total_static_mb"])
            writer.writerow(["#model_mem", round(mi.weight_mb, 1), round(mi.gradient_mb, 1),
                              round(mi.optimizer_mb, 1), round(mi.activation_mb_per_image, 1),
                              round(mi.total_static_mb, 1)])

            # CPU info
            if report.cpu_info:
                cpu = report.cpu_info
                writer.writerow(["#cpu", "logical_cores", "physical_cores", "freq_mhz",
                                  "ram_total_gb", "ram_free_gb"])
                writer.writerow(["#cpu", cpu.logical_cores, cpu.physical_cores,
                                  round(cpu.freq_mhz, 0), round(cpu.ram_total_gb, 1),
                                  round(cpu.ram_free_gb, 1)])

            # Disk info
            if report.disk_info:
                disk = report.disk_info
                writer.writerow(["#disk", "type", "is_nfs", "read_mb_per_s", "files_per_second"])
                writer.writerow(["#disk", disk.disk_type, "yes" if disk.is_nfs else "no",
                                  round(disk.read_mb_per_s, 1), round(disk.files_per_second, 1)])

            # Dataset profile
            if report.dataset_profile:
                dp = report.dataset_profile
                writer.writerow(["#dataset", "n_files_est", "read_mb_per_s",
                                  "files_per_second", "io_bottleneck_ratio"])
                writer.writerow(["#dataset", dp.n_files_total_est,
                                  round(dp.sample_read_mb_per_s, 1),
                                  round(dp.files_per_second, 1),
                                  round(dp.io_bottleneck_ratio, 3)])

            # Predicción de rendimiento
            if report.performance_prediction:
                pred = report.performance_prediction
                writer.writerow(["#prediction", "predicted_best_f1", "predicted_best_epoch",
                                  "predicted_early_stop_epoch", "confidence"])
                writer.writerow(["#prediction", pred.predicted_best_f1,
                                  pred.predicted_best_epoch, pred.predicted_early_stop_epoch,
                                  pred.confidence])
                # Curva F1 (una fila por epoch)
                if pred.curve_epochs:
                    writer.writerow(["#curve_val_f1"] + pred.curve_f1_val)
                    writer.writerow(["#curve_train_f1"] + pred.curve_f1_train)
                    writer.writerow(["#curve_epochs"] + pred.curve_epochs)

            # Escenarios DDP
            if report.ddp_scenarios:
                writer.writerow(["#ddp", "n_gpus", "batch_per_gpu", "global_batch",
                                  "workers_per_gpu", "speedup", "efficiency_pct",
                                  "sync_overhead_pct", "bottleneck",
                                  "time_train_epoch_min", "time_total_h"])
                for s in report.ddp_scenarios:
                    writer.writerow([
                        "#ddp", s.n_gpus, s.batch_per_gpu, s.global_batch,
                        s.num_workers_per_gpu, round(s.estimated_speedup, 2),
                        round(s.scaling_efficiency, 1), round(s.sync_overhead_pct, 1),
                        s.bottleneck,
                        round(s.time_train_per_epoch_s / 60, 1),
                        round(s.time_total_s / 3600, 2),
                    ])

            # Estudio empírico de convergencia (v4)
            study = report.study_report
            if study is not None:
                lr = getattr(study, "lr_range", None)
                cv = getattr(study, "convergence", None)
                gn = getattr(study, "gradient_noise", None)
                if lr is not None:
                    writer.writerow(["#study_lr", "suggested_lr", "min_loss_lr", "diverged_lr"])
                    writer.writerow(["#study_lr", f"{lr.suggested_lr:.3e}",
                                      f"{lr.min_loss_lr:.3e}",
                                      f"{lr.diverged_lr:.3e}" if lr.diverged_lr else ""])
                    writer.writerow(["#study_lr_curve_lrs"] + [f"{x:.3e}" for x in lr.lrs])
                    writer.writerow(["#study_lr_curve_losses"] + [round(x, 5) for x in lr.losses])
                if cv is not None:
                    writer.writerow(["#study_conv", "fit_a", "fit_b", "fit_c", "r_squared",
                                      "loss_1ep", "loss_final", "best_f1", "epochs_to_plateau",
                                      "measured_imgs_per_s"])
                    writer.writerow(["#study_conv", round(cv.fit_a, 5), round(cv.fit_b, 5),
                                      round(cv.fit_c, 5), round(cv.r_squared, 4),
                                      round(cv.extrapolated_loss_1ep, 5),
                                      round(cv.extrapolated_loss_final, 5),
                                      round(cv.extrapolated_best_f1, 4),
                                      cv.epochs_to_plateau, round(cv.measured_imgs_per_s, 1)])
                    writer.writerow(["#study_conv_steps"] + cv.steps)
                    writer.writerow(["#study_conv_losses"] + [round(x, 5) for x in cv.losses])
                    writer.writerow(["#study_conv_f1s"] + [round(x, 5) for x in cv.f1s])
                if gn is not None:
                    writer.writerow(["#study_grad", "grad_norm_mean", "grad_norm_std",
                                      "noise_scale", "suggested_batch_size", "cv"])
                    writer.writerow(["#study_grad", round(gn.grad_norm_mean, 5),
                                      round(gn.grad_norm_std, 5), round(gn.noise_scale, 2),
                                      gn.suggested_batch_size, round(gn.cv, 4)])

            # Benchmarks (filas principales)
            writer.writerow([
                "batch_size", "trace_mode",
                "s_per_batch_train", "imgs_per_s_train",
                "s_per_batch_eval", "imgs_per_s_eval",
                "peak_vram_gb", "avg_power_w", "oom",
                "est_train_min_per_epoch", "est_eval_min_per_epoch",
                "est_total_min_per_epoch",
                f"est_total_h_{target_epochs}ep",
                "est_energy_train_wh_per_epoch", "est_energy_eval_wh_per_epoch",
                "est_energy_total_wh",
                "flops_train_gflops_per_epoch", "flops_eval_gflops_per_epoch",
                "optimizer_steps_per_epoch",
                f"est_ddp_2gpu_h_{target_epochs}ep",
                f"est_ddp_4gpu_h_{target_epochs}ep",
            ])
            for r in report.results:
                est = (estimator.estimate(r, report.dataset_train, report.dataset_val,
                                          target_epochs, report.nfs_factor, model_info=mi)
                       if not r.oom else None)
                writer.writerow([
                    r.batch_size, r.trace_mode,
                    round(r.seconds_per_batch_train, 4) if not r.oom else "",
                    round(r.images_per_second_train, 1) if not r.oom else "",
                    round(r.seconds_per_batch_eval, 4) if not r.oom else "",
                    round(r.images_per_second_eval, 1) if not r.oom else "",
                    round(r.peak_vram_gb, 2) if not r.oom else "",
                    round(r.avg_power_w, 1) if not r.oom and r.avg_power_w > 0 else "",
                    "yes" if r.oom else "no",
                    round(est["train_per_epoch"] / 60, 1) if est else "",
                    round(est["eval_per_epoch"] / 60, 1) if est else "",
                    round(est["total_per_epoch"] / 60, 1) if est else "",
                    round(est["total"] / 3600, 2) if est else "",
                    round(est["energy_train_wh_per_epoch"], 2) if est and est["energy_train_wh_per_epoch"] else "",
                    round(est["energy_eval_wh_per_epoch"], 2) if est and est["energy_eval_wh_per_epoch"] else "",
                    round(est["energy_total_wh"], 1) if est and est["energy_total_wh"] else "",
                    round(est["flops_train_gflops_per_epoch"], 1) if est and est["flops_train_gflops_per_epoch"] else "",
                    round(est["flops_eval_gflops_per_epoch"], 1) if est and est["flops_eval_gflops_per_epoch"] else "",
                    est["optimizer_steps_per_epoch"] if est else "",
                    round(est["ddp_total_2gpu_h"], 2) if est else "",
                    round(est["ddp_total_4gpu_h"], 2) if est else "",
                ])

        print(f"  → CSV guardado: {csv_path}")
        return csv_path
