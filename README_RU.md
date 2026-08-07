# WAL-Ternary-8B

[English](README.md) · [Русский](README_RU.md) · [Қазақша](README_KK.md)

WAL-Ternary-8B V77 — самостоятельно построенная low-bit версия
`Qwen/Qwen3-8B`. Полное дерево checkpoint занимает **2.898417 бита на
исходный параметр (BPW)**. Все 252 линейные матрицы Transformer body
исполняются прямо из packed-представления; плотные веса Qwen или Neutrino при
запуске не нужны.

Модель и копия runtime находятся в одном месте:
[`armanibadboy/WAL-Ternary-8B`](https://huggingface.co/armanibadboy/WAL-Ternary-8B).

> Это исследовательский релиз. CUDA на H200 и x86-64 CPU проверены.
> ARM64/NEON для Apple реализован, но пока не проверен на физическом Mac.

## Что хранится в модели

```text
W(x) = (alpha · T + beta · R)x + Σ Uᵣ(Vᵣᵀx)
T ∈ {-1, 0, +1}
R = ровно 8 знаковых residual-позиций на группу из 128 весов
```

- 252/252 матрицы Transformer body: T3 + sparse-k8.
- 213 матриц: дополнительная бинарная low-rank коррекция WALB2.
- 39 матриц: только базовый слой.
- Embedding: INT3/g128.
- LM head: INT4/g128.
- Norm и малые тензоры: BF16.
- Полезная нагрузка модели: 2.882034 BPW.
- Полное дерево релиза: 2.898417 BPW.

Это не чистая 1.58-битная BitNet. Это тернарно-производная гибридная система
2.898 BPW с sparse и binary correction lanes.

## Проверенные результаты

| Исходный замороженный gate | Порог | V77 |
|---|---:|---:|
| Полный checkpoint | ≤2.952031 BPW | **2.898417 BPW** |
| C4 perplexity | ≤22.5637 | **20.1743** |
| MMLU-Redux, 5-shot | ≥57.28% | **59.29%** |
| GSM8K strict | ≥58.66% | **76.57%** |

Дополнительно: GSM8K flexible **77.10%**, IFEval strict **51.57%**, IFEval
loose **54.90%**.

Паритет прямого packed runtime:

| Метрика | Материализованный reference | Direct packed |
|---|---:|---:|
| C4 PPL | 20.1743 | **20.2156** |
| MMLU-Redux | 59.29% | **59.26%** |
| GSM8K strict | 76.57% | **76.19%** |
| GSM8K flexible | 77.10% | **76.35%** |

## Скорость и память

Runtime не держит постоянную плотную INT8/BF16-копию Transformer body.

| Backend | Результат | Память | Статус |
|---|---:|---:|---|
| H200, CUDA V18 Graph, KV 2048 | 72.41 ток/с | 3.93 ГБ | проверено |
| H200, auto KV 256 | 99.36 ток/с | 3.66 ГБ | short-context тест |
| Xeon CPU V4, 64 потока | 5.93 ток/с | 4.12 ГБ RSS | проверено |
| Apple ARM64/NEON | — | — | реализовано, нужен Mac-тест |

Числа относятся к конкретному железу и протоколу и не являются гарантией для
любого компьютера.

## Быстрый запуск

Установка из GitHub:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'git+https://github.com/AubakirovArman/WAL-Ternary-8B.git#egg=wal-tat[runtime]'
wal-runtime doctor
```

Установка из копии кода внутри Hugging Face:

```bash
git clone https://huggingface.co/armanibadboy/WAL-Ternary-8B
cd WAL-Ternary-8B/runtime
python -m pip install '.[runtime]'
wal-runtime doctor
```

Подготовка исполняемого кэша примерно на 2.3 ГБ:

```bash
wal-runtime prepare armanibadboy/WAL-Ternary-8B \
  --output ~/.cache/wal/WAL-Ternary-8B-v77
```

Это не распакованные dense-веса: кэш хранит packed ternary symbols, позиции и
знаки sparse-k8, а также FP16 scales. Проверяются parent manifest, ABI,
converter, происхождение каждой матрицы и SHA-256.

Запуск на NVIDIA:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --prompt 'Объясни простыми словами, почему небо голубое.' \
  --max-new-tokens 256
```

Запуск на CPU:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --backend cpu-v4 --device cpu \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --threads 16 --prompt 'Кратко объясни фотосинтез.'
```

Benchmark:

```bash
wal-runtime benchmark armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 --tokens 128
```

Первый CUDA-запуск JIT-компилирует ядра: нужны CUDA Toolkit/NVCC и Ninja. Для
CPU нужен C++17 compiler. На Apple нужен arm64 Python/PyTorch и Xcode Command
Line Tools; используйте `cpu-v4`. Portable MPS fallback доступен как
`--backend portable --device mps`, но это ещё не fused Metal runtime.

Полная инструкция: [docs/INSTALLATION_RU.md](docs/INSTALLATION_RU.md).

## Команды CLI

```text
wal-runtime doctor
wal-runtime inspect MODEL
wal-runtime prepare MODEL --output CACHE
wal-runtime generate MODEL --hardware-cache CACHE --prompt TEXT
wal-runtime benchmark MODEL --hardware-cache CACHE --tokens 128
```

## Ограничения

- Runtime исследовательский, а не production serving engine.
- Готовых precompiled wheels пока нет.
- CPU prefill требует отдельного batched packed-GEMM.
- Apple Silicon и CUDA кроме SM90 ещё не подтверждены.
- IFEval чувствителен к численной арифметике; детали в
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
- Проверки coding, tool use, safety, multilingual и long context неполны.

Полный путь исследования описан в
[статье](docs/RESEARCH_ARTICLE_RU.md).

## Лицензия

Apache-2.0 в соответствии с лицензией
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B). Тензоры Neutrino не
используются. Проект независим от Alibaba/Qwen и Fermion Research.

Автор: Арман Аубакиров; экспериментальный процесс выполнялся с AI-помощью.
