"""Resource-bounded isolated execution of generated CadQuery programs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .ast_graph import build_ast_graph, validate_cadquery_code
from .dag_astar.metrics import DEFAULT_PROTOCOL, GeometryMetricProtocol, compare_stl


RUNNER = r'''
import builtins as _builtins
import json as _json
import functools as _functools
import math as _math
import os as _os
import socket as _socket
import sys as _sys
import traceback as _traceback
import cadquery as _cq
import numpy as _np

class _ModuleProxy:
    __slots__ = ("_module", "_allowed")
    def __init__(self, module, allowed):
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_allowed", frozenset(allowed))
    def __getattr__(self, name):
        if name not in object.__getattribute__(self, "_allowed"):
            raise AttributeError("module_attribute_not_allowed:" + str(name))
        return getattr(object.__getattribute__(self, "_module"), name)

_safe_cq = _ModuleProxy(_cq, [name for name in dir(_cq) if not name.startswith("_") and name not in {
    "exporters", "importers", "plugins",
}])
_safe_linalg = _ModuleProxy(_np.linalg, {"norm"})
_safe_numpy = _ModuleProxy(_np, {
    "abs", "all", "any", "arange", "arctan2", "array", "asarray", "clip",
    "concatenate", "cos", "cross", "deg2rad", "degrees", "dot", "e", "empty",
    "exp", "float32", "float64", "full", "hstack", "int32", "int64", "isfinite",
    "linspace", "log", "maximum", "mean", "meshgrid", "minimum", "ndarray", "ones",
    "pi", "radians", "rad2deg", "reshape", "sin", "sqrt", "stack", "sum", "tan",
    "transpose", "vstack", "where", "zeros",
})
object.__setattr__(_safe_numpy, "_allowed", frozenset({
    *object.__getattribute__(_safe_numpy, "_allowed"), "linalg",
}))
class _NumpyProxy:
    def __getattr__(self, name):
        if name == "linalg":
            return _safe_linalg
        return getattr(_safe_numpy, name)

_safe_functools = _ModuleProxy(_functools, {"reduce"})
_ALLOWED_MODULES = {
    "cadquery": _safe_cq, "functools": _safe_functools,
    "math": _math, "numpy": _NumpyProxy(),
}
def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".")[0]
    if level != 0 or root not in _ALLOWED_MODULES:
        raise ImportError("import_not_allowed:" + str(name))
    if root == "cadquery" and any(item in {"exporters", "importers"} for item in (fromlist or ())):
        raise ImportError("cadquery_io_import_not_allowed")
    return _ALLOWED_MODULES[root]

_printed = 0
def _limited_print(*values, sep=" ", end="\n", file=None, flush=False):
    global _printed
    text = sep.join(str(value) for value in values) + end
    remaining = max(0, 65536 - _printed)
    if remaining:
        _sys.stdout.write(text[:remaining])
        _printed += min(len(text), remaining)
    if flush:
        _sys.stdout.flush()

def _deny_network(*args, **kwargs):
    raise PermissionError("network_disabled")
_socket.socket = _deny_network
_socket.create_connection = _deny_network

_source = _builtins.open(__USER_CODE_PATH__, "r", encoding="utf-8").read()
_phase = "user"
_allowed_export_paths = {_os.path.abspath(__STEP_PATH__), _os.path.abspath(__STL_PATH__)}
def _audit(event, args):
    if event == "open":
        path = args[0] if args else ""
        if isinstance(path, (str, bytes, _os.PathLike)):
            resolved = _os.path.abspath(_os.fspath(path))
            if _phase == "user" or resolved not in _allowed_export_paths:
                raise PermissionError("filesystem_access_disabled:" + str(path))
    if event.startswith("socket.") or event in {
        "subprocess.Popen", "os.system", "os.exec", "os.spawn", "ctypes.dlopen",
    }:
        raise PermissionError("process_or_network_access_disabled:" + event)
_sys.addaudithook(_audit)

_safe_builtins = {
    "__import__": _restricted_import, "abs": abs, "all": all, "any": any,
    "bool": bool, "dict": dict, "enumerate": enumerate, "Exception": Exception,
    "float": float, "int": int, "isinstance": isinstance, "len": len,
    "list": list, "map": map, "max": max, "min": min, "object": object,
    "pow": pow, "print": _limited_print, "range": range, "reversed": reversed,
    "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "ValueError": ValueError, "RuntimeError": RuntimeError, "zip": zip,
}

def _collect(value):
    if isinstance(value, _cq.Workplane):
        return [item for item in value.vals() if isinstance(item, _cq.Shape)]
    if isinstance(value, _cq.Assembly):
        return [value.toCompound()]
    if isinstance(value, _cq.Shape):
        return [value]
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(_collect(item))
        return output
    return []

try:
    _namespace = {"__builtins__": _safe_builtins, "__name__": "__cadquery_user__"}
    exec(compile(_source, __USER_CODE_PATH__, "exec"), _namespace, _namespace)
    if "result" not in _namespace:
        raise ValueError("missing_top_level_result")
    _shapes = _collect(_namespace["result"])
    if not _shapes:
        raise ValueError("top_level_result_contains_no_shape")
    _compound = _shapes[0] if len(_shapes) == 1 else _cq.Compound.makeCompound(_shapes)
    _solids = list(_compound.Solids())
    if not _solids:
        raise ValueError("top_level_result_contains_no_solid")
    _volumes = [float(solid.Volume()) for solid in _solids]
    if any(volume <= __MIN_VOLUME__ for volume in _volumes):
        raise ValueError("zero_or_negative_solid_volume")
    if not _compound.isValid() or not all(solid.isValid() for solid in _solids):
        raise ValueError("invalid_topology")
    _bbox = _compound.BoundingBox()
    _faces = list(_compound.Faces())
    _cylinders = []
    for _face in _faces:
        try:
            if _face.geomType() != "CYLINDER":
                continue
            _cylinder = _face._geomAdaptor().Cylinder()
            _center = _face.Center()
            _cylinders.append({
                "radius": float(_cylinder.Radius()),
                "center": [float(_center.x), float(_center.y), float(_center.z)],
                "area": float(_face.Area()),
            })
        except Exception:
            continue
    _phase = "export"
    _cq.exporters.export(_compound, __STEP_PATH__)
    _cq.exporters.export(_compound, __STL_PATH__, tolerance=0.001, angularTolerance=0.1)
    if _os.path.getsize(__STEP_PATH__) <= 0 or _os.path.getsize(__STL_PATH__) <= 0:
        raise ValueError("empty_export")
    _roundtrip = _cq.importers.importStep(__STEP_PATH__)
    _roundtrip_solids = list(_roundtrip.solids().vals())
    if not _roundtrip_solids or not all(solid.isValid() for solid in _roundtrip_solids):
        raise ValueError("step_roundtrip_invalid")
    _metrics = {
        "solid_count": len(_solids), "shape_count": len(_shapes),
        "volume": float(sum(_volumes)), "solid_volumes": _volumes,
        "is_valid": True,
        "bbox": {
            "min": [float(_bbox.xmin), float(_bbox.ymin), float(_bbox.zmin)],
            "max": [float(_bbox.xmax), float(_bbox.ymax), float(_bbox.zmax)],
            "size": [float(_bbox.xlen), float(_bbox.ylen), float(_bbox.zlen)],
        },
        "topology": {
            "face_count": len(_faces), "edge_count": len(_compound.Edges()),
            "vertex_count": len(_compound.Vertices()), "shell_count": len(_compound.Shells()),
            "cylindrical_faces": _cylinders,
        },
        "step": __STEP_PATH__, "stl": __STL_PATH__,
        "step_bytes": _os.path.getsize(__STEP_PATH__), "stl_bytes": _os.path.getsize(__STL_PATH__),
    }
    _limited_print("CADQUERY_METRICS=" + _json.dumps(_metrics, ensure_ascii=True, sort_keys=True), flush=True)
except BaseException:
    _traceback.print_exc()
    _sys.stderr.flush()
    _sys.stdout.flush()
    _os._exit(1)
_sys.stdout.flush()
_os._exit(0)
'''


def _safe_task_id(value: str) -> str:
    raw = str(value)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not safe:
        raise ValueError("task_id_has_no_safe_characters")
    # Windows commonly keeps the legacy 260-character path limit.  Experiment
    # roots are already long, so a readable short prefix plus a stable digest
    # is safer than truncating at 120 characters.  The digest also prevents
    # collisions between conditions that share the same prefix.
    digest = __import__("hashlib").sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:20]}_{digest}"


def _classify_failure(stderr: str, stdout: str, resource_failure: str | None) -> str:
    if resource_failure:
        return resource_failure
    text = (stderr + "\n" + stdout).lower()
    if any(marker in text for marker in (
        "cannot allocate memory for thread-local data",
        "cannot allocate memory",
        "std::bad_alloc",
        "memoryerror",
    )):
        return "MemoryLimitExceeded"
    if "syntaxerror" in text:
        return "SyntaxError"
    if ("import_not_allowed" in text or "network_disabled" in text or "permissionerror" in text
            or "filesystem_access_disabled" in text or "process_or_network_access_disabled" in text):
        return "SandboxViolation"
    if "no_solid" in text or "no_shape" in text or "missing_top_level_result" in text:
        return "EmptyGeometry"
    if "zero_or_negative" in text:
        return "ZeroVolumeGeometry"
    if "invalid_topology" in text or "roundtrip_invalid" in text:
        return "InvalidTopology"
    if "empty_export" in text or "export" in text:
        return "ExportFailure"
    return "CadQueryExecutionError"


class CadQueryRuntime:
    def __init__(
        self, output_dir: str | Path, timeout_seconds: float = 120.0, *,
        max_memory_mb: int = 8192, max_cpu_seconds: float | None = None,
        minimum_volume: float = 1e-9,
        metric_protocol: GeometryMetricProtocol = DEFAULT_PROTOCOL,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        configured_python = os.getenv("CADQUERY_RUNTIME_PYTHON", "").strip()
        # Do not call Path.resolve() here.  Virtual-environment launchers are
        # commonly symbolic links on POSIX; dereferencing one can replace the
        # venv entry point with its base interpreter and consequently hide the
        # venv's site-packages (including CadQuery) from the isolated process.
        self.python_executable = (
            os.path.abspath(os.path.expanduser(configured_python))
            if configured_python else sys.executable
        )
        if not Path(self.python_executable).is_file():
            raise FileNotFoundError(
                f"cadquery_runtime_python_missing:{self.python_executable}"
            )
        self.timeout_seconds = float(timeout_seconds)
        self.max_memory_mb = int(max_memory_mb)
        self.max_cpu_seconds = float(max_cpu_seconds or timeout_seconds)
        self.minimum_volume = float(minimum_volume)
        self.metric_protocol = metric_protocol

    def _new_run_dir(self, task_id: str, attempt: int, code_hash: str) -> Path:
        task_root = (self.output_dir / _safe_task_id(task_id)).resolve()
        if self.output_dir not in task_root.parents:
            raise ValueError("task_path_escapes_output_dir")
        task_root.mkdir(parents=True, exist_ok=True)
        stem = f"a{attempt:03d}_{code_hash[:8]}"
        for suffix in range(1000):
            candidate = task_root / (stem if suffix == 0 else f"{stem}_r{suffix:03d}")
            try:
                candidate.mkdir(exist_ok=False)
                return candidate
            except FileExistsError:
                continue
        raise RuntimeError("unable_to_allocate_unique_run_directory")

    @staticmethod
    def _sanitised_environment() -> dict[str, str]:
        allowed = (
            "SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP",
            "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
        )
        environment = {key: os.environ[key] for key in allowed if key in os.environ}
        environment.update({
            "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
            # BLAS libraries size thread pools from the host CPU count.  On
            # large servers that can reserve several GiB of virtual address
            # space per isolated runner and conflict with RLIMIT_AS despite a
            # tiny RSS.  Fixed single-thread execution is deterministic and
            # preserves the real memory/CPU/timeout limits below.
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        })
        return environment

    @staticmethod
    def _terminate_tree(process: subprocess.Popen) -> None:
        try:
            import psutil
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except Exception:
            process.kill()

    def execute_code(
        self, code: str, *, task_id: str, attempt: int = 0,
        reference_stl: str | Path | None = None, timeout_seconds: float | None = None,
    ) -> dict:
        issues = validate_cadquery_code(code)
        if issues:
            return {
                "success": False, "stage": "validate", "error_type": "CadQueryCodeValidationError",
                "error": "|".join(issues), "validation_issues": issues,
            }
        code_hash = __import__("hashlib").sha256(code.encode("utf-8")).hexdigest()
        run_dir = self._new_run_dir(task_id, attempt, code_hash)
        user_path = (run_dir / "user_program.py").resolve()
        runner_path = (run_dir / "runner.py").resolve()
        stdout_path, stderr_path = run_dir / "stdout.log", run_dir / "stderr.log"
        stl_path, step_path = (run_dir / "model.stl").resolve(), (run_dir / "model.step").resolve()
        user_path.write_text(code, encoding="utf-8")
        runner = (
            RUNNER.replace("__USER_CODE_PATH__", repr(str(user_path)))
            .replace("__STEP_PATH__", repr(str(step_path)))
            .replace("__STL_PATH__", repr(str(stl_path)))
            .replace("__MIN_VOLUME__", repr(self.minimum_volume))
        )
        runner_path.write_text(runner, encoding="utf-8")
        try:
            ast_graph = build_ast_graph(code)
        except Exception as error:
            return {
                "success": False, "stage": "ast", "error_type": type(error).__name__,
                "error": str(error), "run_dir": str(run_dir),
            }
        effective_timeout = min(self.timeout_seconds, float(timeout_seconds)) if timeout_seconds is not None else self.timeout_seconds
        if effective_timeout <= 0:
            return {
                "success": False, "stage": "execute", "error_type": "TimeoutExpired",
                "error": "no_runtime_budget_remaining", "run_dir": str(run_dir),
            }
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        preexec_fn = None
        if os.name != "nt":
            def preexec_fn():
                import resource
                memory = self.max_memory_mb * 1024 * 1024
                cpu = max(1, int(self.max_cpu_seconds) + 1)
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024, 256 * 1024 * 1024))
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        started = time.perf_counter()
        resource_failure: str | None = None
        resource_limits_available = False
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                [self.python_executable, str(runner_path)], cwd=str(run_dir), stdout=stdout_file,
                stderr=stderr_file, text=True, env=self._sanitised_environment(),
                creationflags=creationflags, preexec_fn=preexec_fn,
                start_new_session=(os.name != "nt"),
            )
            try:
                import psutil
                monitored = psutil.Process(process.pid)
                resource_limits_available = True
            except Exception:
                monitored = None
            while process.poll() is None:
                elapsed = time.perf_counter() - started
                if elapsed > effective_timeout:
                    resource_failure = "TimeoutExpired"
                    self._terminate_tree(process)
                    break
                if monitored is not None:
                    try:
                        family = [monitored, *monitored.children(recursive=True)]
                        memory = sum(item.memory_info().rss for item in family) / (1024 * 1024)
                        cpu = sum(sum(item.cpu_times()[:2]) for item in family)
                        if memory > self.max_memory_mb:
                            resource_failure = "MemoryLimitExceeded"
                            self._terminate_tree(process)
                            break
                        if cpu > self.max_cpu_seconds:
                            resource_failure = "CpuLimitExceeded"
                            self._terminate_tree(process)
                            break
                    except Exception:
                        monitored = None
                time.sleep(0.05)
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._terminate_tree(process)
                returncode = -1
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")[-65536:]
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")[-65536:]
        metrics: dict = {}
        for line in stdout.splitlines():
            if line.startswith("CADQUERY_METRICS="):
                try:
                    metrics = json.loads(line.split("=", 1)[1])
                except json.JSONDecodeError:
                    metrics = {}
        success = bool(
            resource_failure is None and returncode == 0
            and metrics.get("is_valid") is True
            and int(metrics.get("solid_count", 0)) >= 1
            and float(metrics.get("volume", 0.0)) > self.minimum_volume
            and stl_path.is_file() and stl_path.stat().st_size > 0
            and step_path.is_file() and step_path.stat().st_size > 0
        )
        geometry_metrics = None
        if success and reference_stl:
            try:
                geometry_metrics = compare_stl(
                    str(stl_path), str(reference_stl), protocol=self.metric_protocol,
                ).to_dict()
            except Exception as error:
                geometry_metrics = {"error": f"{type(error).__name__}:{error}"}
        error_type = None if success else _classify_failure(stderr, stdout, resource_failure)
        return {
            "success": success, "stage": "execute", "returncode": returncode,
            "stdout": stdout, "stderr": stderr, "error_type": error_type,
            "error": "" if success else (resource_failure or stderr[-4000:] or stdout[-2000:]),
            "runtime_seconds": time.perf_counter() - started,
            "metrics": metrics, "geometry_metrics": geometry_metrics,
            "run_dir": str(run_dir), "script": str(user_path), "ast_graph": ast_graph,
            "resource_limits_available": resource_limits_available,
            "isolation": {
                "python_audit_hook": True, "network_disabled": True,
                "filesystem_user_access": "deny", "sanitized_environment": True,
                "process_tree_termination": True,
                "posix_rlimits": os.name != "nt",
                "windows_resource_monitor": os.name == "nt" and resource_limits_available,
            },
            "limits": {
                "timeout_seconds": effective_timeout, "max_memory_mb": self.max_memory_mb,
                "max_cpu_seconds": self.max_cpu_seconds,
            },
        }
