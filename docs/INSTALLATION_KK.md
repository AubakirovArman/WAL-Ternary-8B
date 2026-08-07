# Толық орнату және іске қосу нұсқаулығы

## Талаптар

- Linux x86-64, macOS arm64 немесе WSL2;
- Python 3.10–3.12 және PyTorch 2.2+;
- модельге шамамен 3 ГБ, compute cache үшін 2.3 ГБ;
- NVIDIA үшін driver, CUDA Toolkit/`nvcc`, Ninja және C++ compiler;
- CPU үшін C++17 compiler; AVX-512 ұсынылады;
- Apple үшін arm64 Python/PyTorch және Xcode Command Line Tools.

Расталған CUDA жүйесі — H200/SM90. Басқа GPU-ларға бөлек тексеру қажет.

## Орнату

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install '.[runtime]'
wal-runtime doctor
```

Clone жасамай:

```bash
python -m pip install 'git+https://github.com/AubakirovArman/WAL-Ternary-8B.git#egg=wal-tat[runtime]'
```

## Модельді тексеру және cache дайындау

```bash
wal-runtime inspect armanibadboy/WAL-Ternary-8B
wal-runtime prepare armanibadboy/WAL-Ternary-8B \
  --output ~/.cache/wal/WAL-Ternary-8B-v77
```

Output бумасы алдын ала болмауы керек. `manifest.json`,
`attestation.json` және `.walhw` файлдарын бірге сақтаңыз. Offline режимі
үшін толық HF snapshot пен cache бумасын көшіріп, model id орнына local path
беріңіз.

## NVIDIA

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --device cuda:0 --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --prompt 'Фотосинтезді қысқаша түсіндір.' --max-new-tokens 256
```

Бірінші іске қосу CUDA extension-ды JIT-компиляциялап, CUDA Graph жасайды.
Келесі warm run компиляция кэшін пайдаланады.

## CPU және Apple Silicon

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --device cpu --backend cpu-v4 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --threads 16 --prompt 'Фотосинтезді қысқаша түсіндір.'
```

Apple-де `xcode-select --install` орындаңыз және native arm64 Python/PyTorch
пайдаланыңыз. Auto threads-тен бастаңыз, кейін 4, 8 және performance core саны
бойынша салыстырыңыз. ARM64/NEON жолы нақты Mac тестіне дейін experimental.

Portable MPS fallback:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --device mps --backend portable --prompt 'Сәлем' --max-new-tokens 16
```

Бұл correctness fallback; fused Metal kernel емес.

## KV cache және жад

`--max-cache-len 0` prompt пен output-қа жеткілікті ең кіші екілік дәрежені
таңдайды, минимум 256. Үлкен KV cache жадты көбейтіп, single-stream speed-ті
азайтуы мүмкін.

## Benchmark

```bash
wal-runtime benchmark armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --max-cache-len 2048 --tokens 128
```

Hardware, PyTorch/CUDA нұсқасы, prompt length, KV cache және warm/cold күйін
міндетті түрде жазыңыз.

## Тесттер

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_runtime_cli.py -q
python -m build --wheel
```

Кәдімгі inference кезінде `--skip-hashes` пайдаланбаңыз. Cache rejected
болса, verification-ды өшірмей, керекті model revision-нан cache-ті қайта
жасаңыз.

## Жиі кездесетін мәселелер

- `nvcc not found`: CUDA Toolkit орнатып, `nvcc --version` тексеріңіз.
- `ninja not found`: `python -m pip install ninja`.
- CUDA OOM: auto KV қолданыңыз, context-ті азайтыңыз, басқа GPU process-терді
  тексеріңіз.
- Бірінші run баяу: бұл JIT compilation.
- CPU prompt баяу: prefill decode-тан аз оңтайландырылған.
- Apple build қатесі: Python, PyTorch және shell arm64 болуы керек;
  `xcode-select -p` жұмыс істеуі тиіс.

Model `trust_remote_code=False` арқылы жүктеледі; manifest пен hash
fail-closed түрде тексеріледі.
