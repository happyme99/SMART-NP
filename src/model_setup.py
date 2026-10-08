"""Download a GGUF model and build llama.cpp in this repository."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    filename: str
    revision: str = "main"
    sha256: str = ""


def registry() -> dict[str, ModelSpec]:
    raw = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))
    return {name: ModelSpec(name=name, **item) for name, item in raw.items()}


def resolve_model(name: str, repo_id: str | None = None, filename: str | None = None, revision: str = "main") -> ModelSpec:
    if repo_id or filename:
        if not repo_id or not filename:
            raise ValueError("--model-repo and --model-file must be used together")
        return ModelSpec(name=name, repo_id=repo_id, filename=filename, revision=revision)
    try:
        return registry()[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; add it to configs/models.json or pass --model-repo and --model-file") from exc


def model_path(spec: ModelSpec) -> Path:
    return ROOT / "models" / spec.name / spec.filename


def installed_model_path(name: str) -> Path:
    metadata = ROOT / "models" / name / "model.json"
    if metadata.is_file():
        return model_path(ModelSpec(**json.loads(metadata.read_text(encoding="utf-8"))))
    return model_path(resolve_model(name))


def command(args: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def need(executable: str) -> None:
    if shutil.which(executable) is None:
        raise RuntimeError(f"Missing required executable: {executable}")


def find_cuda_compiler() -> Path:
    paths = [Path(path) for path in [shutil.which("nvcc")] if path]
    paths.extend(Path("/usr/local").glob("cuda*/bin/nvcc"))
    candidates = {path.resolve() for path in paths if path.is_file()}
    if not candidates:
        raise RuntimeError("Missing required executable: nvcc")

    def version(path: Path) -> tuple[int, int]:
        output = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=False).stdout
        match = re.search(r"release (\d+)\.(\d+)", output)
        return (int(match.group(1)), int(match.group(2))) if match else (0, 0)

    return max(candidates, key=lambda path: (version(path), str(path)))


def build_llama(ref: str, cuda: bool) -> Path:
    for executable in ("git", "cmake"):
        need(executable)
    source = ROOT / ".tools/llama.cpp"
    build = source / "build"
    binary = source / "build/bin/llama-server"
    if not source.is_dir():
        command(["git", "clone", "--depth", "1", "--branch", ref, LLAMA_REPO, str(source)])
    cmake = ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
    if cuda:
        cuda_compiler = find_cuda_compiler()
        cuda_root = cuda_compiler.parent.parent
        cache = build / "CMakeCache.txt"
        if cache.is_file():
            cached = cache.read_text(errors="ignore")
            if f"CMAKE_CUDA_COMPILER:STRING={cuda_compiler}" not in cached:
                shutil.rmtree(build)
        cmake.extend(
            [
                "-DGGML_CUDA=ON",
                f"-DCMAKE_CUDA_COMPILER={cuda_compiler}",
                f"-DCUDAToolkit_ROOT={cuda_root}",
                "-DLLAMA_USE_PREBUILT_UI=OFF",
            ]
        )
    command(cmake)
    command(["cmake", "--build", str(build), "--config", "Release", "--target", "llama-server", "-j"])
    if not binary.is_file():
        raise RuntimeError(f"llama.cpp build did not create {binary}")
    return binary


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(spec: ModelSpec) -> Path:
    destination = model_path(spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=destination.parent,
        )
    )
    if spec.sha256 and sha256(path) != spec.sha256:
        raise RuntimeError(f"Downloaded GGUF checksum differs for {path}; remove it and retry")
    (path.parent / "model.json").write_text(json.dumps(asdict(spec), indent=2) + "\n", encoding="utf-8")
    return path


def setup_model(spec: ModelSpec, llama_ref: str = "master", cuda: bool = True) -> tuple[Path, Path]:
    binary = ROOT / ".tools/llama.cpp/build/bin/llama-server"
    gguf = model_path(spec)
    cuda_library = binary.parent / "libggml-cuda.so.0"
    ready = binary.is_file() and gguf.is_file() and (not cuda or cuda_library.is_file())
    if ready:
        print(f"{'[Model setup already done]':<41}  {'':<12}  llama_server={binary}\n{'':<41}  {'':<12}  model={gguf}", flush=True)
        return binary, gguf
    binary = binary if binary.is_file() and (not cuda or cuda_library.is_file()) else build_llama(llama_ref, cuda=cuda)
    gguf = gguf if gguf.is_file() else download_model(spec)
    print(json.dumps({"status": "ready", "llama_server": str(binary), "model": str(gguf), "model_sha256": sha256(gguf)}, indent=2))
    return binary, gguf
