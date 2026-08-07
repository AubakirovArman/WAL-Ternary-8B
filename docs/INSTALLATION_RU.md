# Полная инструкция по установке и запуску

## Требования

- Linux x86-64, macOS arm64 или Windows через WSL2;
- Python 3.10–3.12 и PyTorch 2.2+;
- около 3 ГБ для модели и 2.3 ГБ для compute cache;
- для NVIDIA: совместимый driver, CUDA Toolkit/`nvcc`, Ninja, C++ compiler;
- для CPU: C++17 compiler и Ninja; AVX-512 желателен;
- для Apple: arm64 Python/PyTorch и Xcode Command Line Tools.

Проверенная CUDA-конфигурация — H200/SM90. Другие GPU могут работать, но пока
не имеют официального correctness/performance подтверждения.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install '.[runtime]'
wal-runtime doctor
```

Без clone:

```bash
python -m pip install 'git+https://github.com/AubakirovArman/WAL-Ternary-8B.git#egg=wal-tat[runtime]'
```

## Проверка модели и подготовка кэша

```bash
wal-runtime inspect armanibadboy/WAL-Ternary-8B
wal-runtime prepare armanibadboy/WAL-Ternary-8B \
  --output ~/.cache/wal/WAL-Ternary-8B-v77
```

Папка output должна быть новой. Не отделяйте `manifest.json` и
`attestation.json` от файлов `.walhw`. Для offline-запуска перенесите полный
HF snapshot и всю папку кэша, затем передайте локальный путь к snapshot.

## NVIDIA

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --device cuda:0 --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --prompt 'Кратко объясни фотосинтез.' --max-new-tokens 256
```

Первый запуск компилирует CUDA extension и создаёт CUDA Graph. Это cold start;
следующие запуски используют кэш компиляции.

## CPU и Apple Silicon

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --device cpu --backend cpu-v4 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --threads 16 --prompt 'Кратко объясни фотосинтез.'
```

На Apple установите `xcode-select --install`, используйте native arm64 Python
и PyTorch. Начните с автоматического числа потоков, затем сравните 4, 8 и число
performance cores. ARM64/NEON путь экспериментальный до физического теста.

Portable fallback:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --device mps --backend portable --prompt 'Привет' --max-new-tokens 16
```

Он корректностный и медленный; это не оптимизированное Metal-ядро.

## Память и KV cache

`--max-cache-len 0` выбирает минимальную степень двойки, достаточную для
prompt и ответа, но не меньше 256. Большой KV cache увеличивает память и может
снизить скорость single-stream decode.

## Benchmark

```bash
wal-runtime benchmark armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --max-cache-len 2048 --tokens 128
```

Всегда записывайте GPU/CPU, версии PyTorch/CUDA, длину prompt, KV cache и
warm/cold статус. Иначе сравнение скоростей некорректно.

## Проверки разработчика

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_runtime_cli.py -q
python -m build --wheel
```

Не используйте `--skip-hashes` в обычном запуске. Если кэш отвергнут,
пересоберите конкретную папку из нужной ревизии модели, а не отключайте
проверку.

## Частые проблемы

- `nvcc not found`: установите CUDA Toolkit и проверьте `nvcc --version`.
- `ninja not found`: `python -m pip install ninja`.
- CUDA OOM: оставьте auto KV, сократите контекст и проверьте чужие процессы.
- Медленный первый запуск: ожидаемая JIT-компиляция.
- Медленный CPU prompt: prefill пока слабее decode и будет улучшаться отдельно.
- Ошибка Apple build: Python, PyTorch и terminal должны быть arm64;
  `xcode-select -p` должен работать.

Модель загружается с `trust_remote_code=False`; checkpoint рассматривается
как данные, а его манифесты и hashes проверяются fail-closed.
