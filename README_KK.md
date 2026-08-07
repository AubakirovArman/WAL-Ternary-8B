# WAL-Ternary-8B

[English](README.md) · [Русский](README_RU.md) · [Қазақша](README_KK.md)

WAL-Ternary-8B V77 — `Qwen/Qwen3-8B` моделінен дербес құрастырылған low-bit
нұсқа. Толық checkpoint бастапқы бір параметрге **2.898417 бит (BPW)** алады.
Transformer body ішіндегі 252 сызықтық матрицаның барлығы packed-пішімнен
тікелей орындалады. Іске қосу үшін dense Qwen немесе Neutrino салмақтары қажет
емес.

Модель мен runtime коды бір Hugging Face репозиторийінде:
[`armanibadboy/WAL-Ternary-8B`](https://huggingface.co/armanibadboy/WAL-Ternary-8B).

> Бұл — зерттеу релизі. H200 CUDA және x86-64 CPU жолдары тексерілді.
> Apple ARM64/NEON коды бар, бірақ нақты Mac құрылғысында әлі расталмаған.

## Модель пішімі

```text
W(x) = (alpha · T + beta · R)x + Σ Uᵣ(Vᵣᵀx)
T ∈ {-1, 0, +1}
R = әр 128 салмақ тобына дәл 8 таңбалы residual позиция
```

- Transformer body: 252/252 T3 + sparse-k8 матрицасы.
- 213 матрицада binary low-rank WALB2 түзетуі бар.
- 39 матрица тек base жолын пайдаланады.
- Embedding: INT3/g128.
- LM head: INT4/g128.
- Norm және шағын тензорлар: BF16.
- Модель payload: 2.882034 BPW; толық релиз: 2.898417 BPW.

Бұл таза 1.58-bit BitNet емес. Бұл sparse және binary түзету жолдары бар
2.898-BPW ternary-derived гибрид жүйе.

## Расталған нәтижелер

| Бастапқы frozen gate | Шек | V77 |
|---|---:|---:|
| Толық checkpoint | ≤2.952031 BPW | **2.898417 BPW** |
| C4 perplexity | ≤22.5637 | **20.1743** |
| MMLU-Redux, 5-shot | ≥57.28% | **59.29%** |
| GSM8K strict | ≥58.66% | **76.57%** |

Қосымша: GSM8K flexible **77.10%**, IFEval strict **51.57%**, IFEval loose
**54.90%**.

Direct-packed сәйкестігі:

| Метрика | Materialized reference | Direct packed |
|---|---:|---:|
| C4 PPL | 20.1743 | **20.2156** |
| MMLU-Redux | 59.29% | **59.26%** |
| GSM8K strict | 76.57% | **76.19%** |
| GSM8K flexible | 77.10% | **76.35%** |

## Жылдамдық пен жад

Runtime Transformer body-дің тұрақты dense INT8/BF16 көшірмесін сақтамайды.

| Backend | Нәтиже | Жад | Күйі |
|---|---:|---:|---|
| H200, CUDA V18 Graph, KV 2048 | 72.41 ток/с | 3.93 ГБ | расталды |
| H200, auto KV 256 | 99.36 ток/с | 3.66 ГБ | қысқа context тесті |
| Xeon CPU V4, 64 thread | 5.93 ток/с | 4.12 ГБ RSS | расталды |
| Apple ARM64/NEON | — | — | іске асқан, Mac тесті қажет |

Бұл сандар нақты hardware және хаттамаға тәуелді.

## Орнату

GitHub арқылы:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'git+https://github.com/AubakirovArman/WAL-Ternary-8B.git#egg=wal-tat[runtime]'
wal-runtime doctor
```

Hugging Face ішіндегі код көшірмесі арқылы:

```bash
git clone https://huggingface.co/armanibadboy/WAL-Ternary-8B
cd WAL-Ternary-8B/runtime
python -m pip install '.[runtime]'
wal-runtime doctor
```

Шамамен 2.3 ГБ executable cache дайындау:

```bash
wal-runtime prepare armanibadboy/WAL-Ternary-8B \
  --output ~/.cache/wal/WAL-Ternary-8B-v77
```

Бұл dense салмақтар емес: cache ішінде packed ternary symbols, sparse-k8
позициялары/таңбалары және FP16 scales сақталады. Parent manifest, ABI,
converter, matrix lineage және SHA-256 fail-closed түрде тексеріледі.

NVIDIA-да іске қосу:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --prompt 'Аспан неге көк екенін қарапайым тілмен түсіндір.' \
  --max-new-tokens 256
```

CPU-да іске қосу:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --backend cpu-v4 --device cpu \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --threads 16 --prompt 'Фотосинтезді қысқаша түсіндір.'
```

Benchmark:

```bash
wal-runtime benchmark armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 --tokens 128
```

Бірінші CUDA іске қосу included kernels-ті JIT-компиляциялайды; CUDA
Toolkit/NVCC және Ninja керек. CPU үшін C++17 compiler қажет. Apple Silicon
үшін native arm64 Python/PyTorch және Xcode Command Line Tools орнатыңыз да,
`cpu-v4` қолданыңыз. `--backend portable --device mps` fallback бар, бірақ
ол fused Metal runtime емес.

Толық нұсқаулық: [docs/INSTALLATION_KK.md](docs/INSTALLATION_KK.md).

## CLI командалары

```text
wal-runtime doctor
wal-runtime inspect MODEL
wal-runtime prepare MODEL --output CACHE
wal-runtime generate MODEL --hardware-cache CACHE --prompt TEXT
wal-runtime benchmark MODEL --hardware-cache CACHE --tokens 128
```

## Шектеулер

- Бұл research runtime; production serving engine емес.
- Precompiled wheels әлі жарияланбаған.
- CPU prefill үшін batched packed-GEMM қажет.
- Apple Silicon және SM90-нан басқа CUDA архитектуралары тексерілмеген.
- IFEval numerical arithmetic-ке сезімтал; [benchmark есебін](docs/BENCHMARKS.md)
  қараңыз.
- Coding, tool use, safety, multilingual және long-context тесттері толық емес.

Зерттеудің толық тарихы
[қазақ тіліндегі мақалада](docs/RESEARCH_ARTICLE_KK.md) берілген. Сонымен бірге
[орысша](docs/RESEARCH_ARTICLE_RU.md) және
[ағылшынша](docs/RESEARCH_ARTICLE_EN.md) нұсқалар бар.

## Лицензия

Apache-2.0, upstream
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) лицензиясына сәйкес.
Neutrino тензорлары пайдаланылмаған. Жоба Alibaba/Qwen және Fermion
Research ұйымдарынан тәуелсіз.

Автор: Арман Аубакиров; эксперименттер AI көмегімен жүргізілді.
