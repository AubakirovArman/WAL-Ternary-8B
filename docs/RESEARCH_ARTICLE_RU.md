# WAL-Ternary-8B: самостоятельное преобразование Qwen3-8B в исполняемую low-bit систему 2.898 BPW

## Аннотация

В этой работе описан полный путь построения WAL-Ternary-8B V77 —
самодостаточной low-bit версии Qwen3-8B, в которой все 252 линейные матрицы
Transformer body хранятся и исполняются в собственной packed-системе. Полное
дерево checkpoint занимает 2.898417 бита на исходный параметр. Базовый оператор
сочетает тернарные значения с восемью sparse residual-позициями на группу из
128 весов; 213 матриц дополнительно используют бинарные low-rank коррекции
WALB2. Embedding хранится в INT3, LM head — в INT4.

Работа началась с попытки максимально сохранить знания готовой Qwen3-8B при
переходе к тернарному представлению. Чистая post-training тернаризация
оказалась разрушительной: C4 perplexity достигла примерно 7952.54. Добавление
sparse-k8 восстановило модель до порядка 41 PPL, а co-adapted WALB2 и
последовательные структурные эксперименты дали V73 с C4 28.58. Серия причинных
опытов показала важный отрицательный результат: локальное уменьшение ошибки
оператора не гарантирует улучшение полной autoregressive trajectory. Прорыв
произошёл в R12, где 465 уже существующих амплитуд были оптимизированы
совместно без добавления новых байтов. Финальная V77 достигла C4 PPL 20.1743,
MMLU-Redux 59.29% и GSM8K strict 76.57%, пройдя четыре исходных gate проекта.

Отдельная runtime-ветка превратила компактный файл в реальную исполняемую
low-bit модель. Direct-packed runtime не создаёт постоянную dense INT8/BF16
копию Transformer body. После оптимизации скорость выросла с 4.62 до
72.41 токена/с на H200 при 3.93 ГБ peak VRAM; short-context измерение с
автоматическим KV cache достигло 99.36 токена/с при 3.66 ГБ. Native CPU V4
получил 5.93 токена/с на Xeon при 4.12 ГБ RSS.

## 1. Мотивация

Современные 8B-модели достаточно умны для локального применения, но BF16-веса
требуют около 16 ГБ только на параметры. Обычная 4-битная квантизация уменьшает
это примерно в четыре раза, однако цель WAL была жёстче: исследовать, можно ли
перевести уже обученную Qwen3-8B в тернарно-производную систему примерно
3 BPW, не уничтожив знания и reasoning.

Задача принципиально отличается от обучения BitNet b1.58 с нуля. В
[BitNet b1.58](https://arxiv.org/abs/2402.17764) тернарные веса {-1,0,+1}
встраиваются в процесс обучения. Здесь исходная модель уже обучена, и её
необходимо преобразовать post-training или очень малым constrained recovery,
сохранив распределение функций. В качестве практического ориентира изучался
Neutrino-8B, но ни одного его тензора WAL не использует. Базовая архитектура —
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), описанная в
[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).

Исходная цель была зафиксирована четырьмя абсолютными воротами:

| Gate | Требование |
|---|---:|
| Полное дерево checkpoint | ≤2.952031 BPW |
| C4 perplexity | ≤22.5637 |
| MMLU-Redux | ≥57.28% |
| GSM8K strict | ≥58.66% |

Эти пороги отвечали на главный вопрос: удалось ли вообще получить
жизнеспособную автономную low-bit Qwen3-8B. Более поздние сравнительные
guardrails относительно V73 использовались как строгая диагностика, но не
переписывали исходную цель задним числом.

## 2. Что именно было преобразовано

Qwen3-8B — dense Transformer с 36 блоками. В каждом блоке есть четыре attention
проекции Q/K/V/O и три MLP-проекции gate/up/down: всего 36×7 = 252 большие
линейные матрицы. В WAL все 252 матрицы выведены из BF16-представления.

Для группы из 128 исходных весов базовый оператор имеет вид:

```text
W_base = alpha * T + beta * R,
T[i] ∈ {-1, 0, +1},
R содержит ровно 8 ненулевых знаковых позиций.
```

Тернарный код даёт грубую глобальную форму, а sparse-k8 возвращает несколько
наиболее важных отклонений внутри каждой группы. Для 213 матриц добавлена
коррекция:

```text
DeltaW(x) = Σ diag(s_row) U_sign diag(s_latent)
              V_sign diag(s_column) x.
```

Матрицы `U_sign` и `V_sign` состоят из {-1,+1} и хранятся bit-packed.
Непрерывная информация остаётся в небольших FP16 scales. Этот формат получил
название WALB2. Оставшиеся 39 матриц работают только на T3+sparse-k8 базе.

Endpoint-слои также не остались dense: embedding — INT3/g128, LM head —
INT4/g128. Малые norms и служебные тензоры сохранены в BF16, потому что их вклад
в размер невелик, а функциональная чувствительность высока.

## 3. Учёт битности

У модели 8,190,735,360 уникальных исходных параметров. Полезная нагрузка
serialized model занимает 2,950,747,732 байта:

```text
8 * 2,950,747,732 / 8,190,735,360 = 2.882034496 BPW.
```

Полное дерево релиза с обязательными метаданными и runtime-файлами занимает
2,967,521,309 байт, или 2.898417472 BPW.

Это важно для терминологии. WAL нельзя называть «чистой 1.58-битной моделью»:
тернарный алфавит сам по себе не определяет физическую стоимость всей системы.
Sparse positions, signs, scales, WALB2, endpoints и metadata входят в реальный
размер.

## 4. Экспериментальная методология

Работа велась как последовательность design cards. До открытия confirmation
набора фиксировались:

- гипотеза и причинный механизм;
- разрешённые параметры;
- train/selection/confirmation firewall;
- метрики и gates;
- условия остановки;
- неизменность checkpoint bytes, если опыт byte-neutral.

Наборы, использованные для fitting или model selection, не переиспользовались
как независимое подтверждение. Для generation-метрик сохранялись парные
переходы `wrong→correct` и `correct→wrong`, а не только среднее accuracy.
Token-level confidence intervals считались по примерам, поскольку токены одного
ответа статистически зависимы.

Важным методологическим правилом было разделение уровней:

```text
меньше weight MSE
≠ лучше operator output
≠ меньше teacher-forced KL
≠ лучше полный autoregressive ответ.
```

Именно это правило спасло проект от публикации нескольких локально красивых,
но функционально нейтральных или вредных кандидатов.

## 5. Этап I: почему чистая тернаризация не сработала

Первоначально предполагалось, что хорошо подобранные scales позволят перенести
знания напрямую в {-1,0,+1}. Полная абляция R12-0 позже дала точный ответ:

| Представление | Mean NLL | PPL |
|---|---:|---:|
| Pure T3 | 8.981246 | **7952.54** |
| T3 + sparse-k8 | около 3.71 | около **41** |
| T3 + sparse-k8 + WALB2, V73 | около 3.35 | **28.58** |

Чистая тернарная база потеряла практически всё. Sparse-k8 дала крупнейшее
первое восстановление. WALB2 сама по себе не являлась универсальным «патчем»:
её эффект зависел от уже присутствующей sparse-базы. R12-0 показал сильную
неаддитивность: k8 улучшал pure T3 примерно на 5.2588 nat/token, тогда как
WALB2 без правильной базы могла даже ухудшать NLL. Значит, итоговая модель —
co-adapted residual system, а не сумма независимых квантователей.

## 6. Этап II: построение V73

Следующие циклы распределяли sparse capacity, подбирали группы и роли,
добавляли бинарные low-rank corrections и проверяли каждое изменение на
реальном NLL. Ключевым практическим результатом стала V73:

- 252/252 low-bit body matrices;
- 213 WALB2 corrections;
- complete tree 2.898417 BPW;
- C4 PPL 28.5811;
- MMLU-Redux 60.36%;
- GSM8K strict 75.28%;
- GSM8K flexible 76.57%.

V73 не прошла только исходный C4 gate. При этом knowledge и mathematical
reasoning сохранились намного лучше, чем ожидалось по perplexity. Это показало,
что одно число PPL нельзя трактовать как полное описание полезности модели.

## 7. Протокол Qwen3 и ложный GSM8K-провал

Ранний bare-prompt GSM8K дал 0% strict, что выглядело как уничтожение
математических способностей. После применения правильного Qwen3 chat template
с `enable_thinking=False` та же V73 получила около 73% strict на
диагностическом срезе. BF16 на том же режиме давала около 94%.

Следовательно, первая «катастрофа» была в большой степени ошибкой протокола.
Но оставшийся разрыв был реален. Token-level forensics показал:

- top-1 agreement с BF16 около 85.9%;
- KL около 0.42–0.50 nat/token;
- немного более низкую entropy V73;
- раннее расхождение semantic trajectory;
- избыточную уверенность уже после выбора другой ветви;
- отдельные проблемы EOS/repetition в thinking mode.

Вывод: знания и capacity сохранились, но геометрия logits стала менее точной.

## 8. Этап III: причинная локализация R5–R7

Head/body swaps и activation patching локализовали заметную часть drift не в
LM head как единственном источнике, а внутри Transformer body. Особенно
чувствительными оказались L4 `gate_proj` и позже L10 attention V/O.

В R7d существующие row scales L4 gate были перекалиброваны на реальных
V73-states. Это был первый настоящий локальный byte-neutral PASS:

- failure KL уменьшился на 4.98–5.81% на трёх независимых наборах;
- gate pre-activation error снизилась примерно на 35%;
- восстановлено около 53–55% эффекта BF16 gate oracle;
- controls не пострадали;
- размер остался 2.898417 BPW.

Однако полная генерация дала:

```text
wrong → correct: 14
correct → wrong: 18
unchanged:       568
```

Из 600 ответов изменились только 32. Локальный оператор стал устойчиво ближе к
BF16, но sequence accuracy слегка ухудшилась. Это стало первым сильным
доказательством различия между operator recovery и trajectory recovery.

## 9. Этап IV: почему простое масштабирование R8 не помогло

R8a проверил всю линию
`theta(lambda)=theta_V73+lambda(theta_R7d-theta_V73)`. На selection Set D
кандидат выглядел полезно: 24 исправления против 15 поломок. Независимый Set E
дал 15 исправлений против 29 поломок и -2.33 percentage points с confidence
interval, не включавшим ноль. Это был классический winner's curse:
оптимальный локальный шаг не переносился на новые autoregressive boundaries.

R8b-0 вычислил gradients sequence margin по 36,864 row-scale параметрам. В
среднем repair и preserve directions были согласованы
(`cos≈0.729`), но 42.54% отдельных пар имели отрицательный cosine. Rank-64 QP
улучшал все 256 repair trajectories и сохранял 41.94% unconstrained signal, но
нарушал preserve constraint для 52 из 256 примеров при лимите 51. Формально —
FAIL, научно — near-feasible repair cone.

Последняя strict-preserve проверка R8b-1 закрыла amplitude-only ветку. Для 15
защищаемых trajectories норма gradient была меньше требуемого safety margin
при зафиксированном trust radius. По неравенству Коши—Буняковского constraints
были невыполнимы до запуска solver. Это не доказывает невозможность любого
нелинейного scale update, но в сочетании с R7d/R8a/R8b-0 достаточно, чтобы не
продолжать sweep learning rate, ridge или rank.

## 10. Этап V: directional oracles R9–R10

R9 сменил вопрос с «как масштабировать старую correction» на «можно ли
перестроить её направление при том же rank и byte budget». Continuous
same-rank oracle показал, что локальная directional capacity существует, но
full generation снова не перенесла улучшение. Sequence-aware R9a2 с explicit
repair/preserve objective также не дал устойчивого gain на свежем Set L.

R10 добавил второй причинный узел — L10 attention V/O — и совместный 64-мерный
oracle L4+L10. Fit был технически успешен, но mechanism не перенёсся на
fresh holdout. К этому моменту было достаточно доказательств:

> Статические локальные correction directions способны исправлять средний
> оператор, но не обладают достаточной context-specific selectivity для
> безопасного переключения autoregressive trajectories.

Эти отрицательные результаты не были потерей времени. Они сузили пространство
решений и запретили бесконечный локальный hyperparameter search.

## 11. Этап VI: R12 и глобальная коадаптация

После локальных веток проект вернулся к единственному непройденному top-level
gate: C4. R12-0 показал, что representation co-adapted глобально. Поэтому R12-A
не добавлял новые codes, sparse positions, ranks или factors. Он совместно
настраивал 465 уже сохранённых амплитуд:

- 252 коэффициента для T3+sparse-k8 base;
- 213 коэффициентов для WALB2 corrections.

Коэффициенты оптимизировались на train-only C4 windows, а release C4,
MMLU/GSM8K/IFEval test и будущие confirmation splits были закрыты. После
fitting значения складывались в существующие FP16 scales, поэтому byte count
не менялся.

Результат V76 R12-A резко снизил C4. Однако generation audit обнаружил
state-dependent termination regression. Вместо открытия всей модели был
проведён один заранее разрешённый R12-B1-Lite: те же 465 foldable amplitudes
получили termination-safe constrained calibration с отдельными stop/continue
traces и preservation anchors. Финальный folded checkpoint получил имя V77.

Главная находка R12:

> В V73 уже была достаточная дискретная information capacity. Большая часть
> оставшегося ущерба происходила не из отсутствия codes, а из неверной
> глобальной совместной калибровки residual lanes.

## 12. Финальные результаты V77

| Метрика | V73 | V77 | Изменение |
|---|---:|---:|---:|
| Complete tree BPW | 2.898417 | **2.898417** | 0 bytes |
| C4 PPL | 28.5811 | **20.1743** | -29.4% |
| MMLU-Redux | **60.36%** | 59.29% | -1.07 pp |
| GSM8K strict | 75.28% | **76.57%** | +1.29 pp |
| GSM8K flexible | 76.57% | **77.10%** | +0.53 pp |
| IFEval strict, cap 256 | 48.24% | **51.57%** | +3.33 pp |
| IFEval loose | 53.97% | **54.90%** | +0.93 pp |
| IFEval repeated tail | 35.86% | **28.65%** | -7.21 pp |

V77 прошла все четыре исходных gate. Поздний V73-relative promotion protocol
формально дал FAIL из-за MMLU -1.07 pp и IFEval max-length +2.40 pp. Эти факты
сохранены. Итоговый статус поэтому двухуровневый:

```text
WAL research goal: ACHIEVED
V77 original gates: 4/4 PASS
strict dominance over every V73 diagnostic: NOT ACHIEVED
```

Такой отчёт честнее как объявления «полного превосходства», так и сведения
всего результата к одному слову FAIL.

## 13. Runtime: от компактного архива к исполняемой модели

Первые WAL checkpoints действительно были компактны на диске, но reference
loader материализовал BF16 body. Это позволяло измерять качество, но не давало
реальной экономии inference memory. Direct-packed runtime должен был вычислять:

```text
y = (T3+sparse-k8)x + Σ U_r(V_r^T x)
```

не создавая постоянную dense матрицу.

Portable reference первым доказал корректность формата, но один forward
занимал минуты. Затем был создан hardware cache `.walhw`: вычислительная
раскладка 2-bit ternary symbols, sparse positions/signs и FP16 scales. Это не
dense checkpoint. Cache детерминированно связан с canonical model manifest,
converter SHA, ABI и hashes всех 252 матриц.

GPU-путь развивался по шагам:

| Версия | Основное изменение | Decode |
|---|---|---:|
| первый direct-packed | correctness kernels | 4.62 tok/s |
| первый WALB2 fusion | меньше launches и traffic | 5.13 tok/s |
| V17/V18 | paired activations, fused paths | около 32 tok/s non-graph |
| V18 + CUDA Graph | steady-state replay | **72.41 tok/s** |
| V18 + auto KV 256 | short-context CLI | **99.36 tok/s** |

Nsight Systems опроверг раннюю гипотезу, что весь runtime launch-bound:
основное время находилось внутри packed T3/WALB2 kernels. Сначала были улучшены
чтение, unpack и fusion; после этого host overhead стал заметен, и CUDA Graph
дал большой дополнительный gain.

Память снизилась с 12.60 ГБ старого materializing/cache пути до 3.75–3.93 ГБ
direct-packed, а cold start — примерно с 355 до 10.1 секунд после подготовки
кэша. На Xeon native CPU V4 объединил пять проходов corrected matrix в два и
достиг 5.93 токена/с на 64 threads при 4.12 ГБ RSS.

## 14. Проверка runtime parity

Прямое packed-исполнение слегка меняет reduction order и rounding. Поэтому
требовать bit-exact logits между BF16 materialized, CUDA, TF32 и CPU
неправильно. Были измерены конечные задачи:

| Метрика | Materialized V77 | Direct packed |
|---|---:|---:|
| C4 PPL | 20.1743 | 20.2156 |
| MMLU-Redux | 59.29% | 59.26% |
| GSM8K strict | 76.57% | 76.19% |
| GSM8K flexible | 77.10% | 76.35% |

C4, MMLU и GSM8K дают сильный numerical parity. IFEval оказался более
чувствительным: исходный direct-packed cap-256 получил 49.35% strict против
51.57% reference. После замороженного discovery на первых 100 prompts был
выбран deployment profile: cap 512 плюс live repeated-tail guard. На нетронутых
441 prompts он дал 56.69% strict и 60.54% loose против frozen materialized
cap-256 52.38/55.56. При этом materialized cap-512 всё ещё выше —
58.96/63.72. Следовательно, runtime практически полезен, но численный IFEval
gap полностью не закрыт.

## 15. Сравнение с Neutrino

Neutrino-8B был важным внешним ориентиром: он показал, что sub-2-bit
Transformer linears и специализированный runtime могут быть практичными.
Текущая карточка
[Neutrino-8B](https://huggingface.co/FermionResearch/Neutrino-8B) описывает
собственный five-valued TRTC/FV5 формат и нативные runtimes.

Однако прямое числовое сравнение карточек некорректно без одного harness:
различаются prompt templates, subsets, parsers, output caps и runtime stack.
Поэтому корректные утверждения WAL ограничены:

- обе модели основаны на Qwen3-8B topology;
- WAL построена независимо из Qwen3-8B;
- Neutrino tensors в WAL отсутствуют;
- WAL имеет собственную систему T3+sparse-k8+WALB2;
- полный same-harness functional/runtime parity не заявляется.

## 16. Главные научные уроки

### 16.1. Pure PTQ ternary для готовой 8B-модели недостаточно

PPL около 7952 показывает, что сохранение алфавита {-1,0,+1} не означает
сохранение функции. Нужна дополнительная residual capacity или обучение,
которое заранее формирует ternary-friendly geometry.

### 16.2. Representation должна быть co-adapted

Sparse-k8 и WALB2 не являются независимыми улучшателями. Их эффект зависит от
совместных scales и состояния всей сети.

### 16.3. Локальная fidelity не равна trajectory quality

R7d устойчиво улучшил оператор на трёх наборах, но не улучшил ответы. R8–R10
повторили этот урок в более сильных пространствах. Autoregressive argmax
дискретен: один ранний flip полностью меняет следующий контекст.

### 16.4. Уже правильные trajectories — защищаемый ресурс

Teacher imitation может ухудшить правильный альтернативный reasoning path.
Repair objective должен отдельно учитывать baseline-wrong и baseline-correct
примеры.

### 16.5. Глобальная калибровка может быть сильнее локального rebuild

R12 добился крупнейшего gain, изменив только 465 foldable amplitudes. Это
подчёркивает скрытую ёмкость уже сохранённых low-bit codes.

### 16.6. Физический размер и runtime — разные задачи

2.898 BPW на диске не даёт 4 ГБ inference автоматически. Экономия появляется
только тогда, когда ядра умеют считать непосредственно по packed bytes.

## 17. Ограничения

- Apple Silicon backend реализован на ARM64/NEON, но не измерен на реальном Mac.
- Precompiled binaries пока не опубликованы; первый запуск JIT-компилирует.
- CPU prefill значительно слабее CPU decode.
- CUDA performance подтверждён прежде всего на H200/SM90.
- IFEval direct-packed gap не закрыт полностью.
- Не проведён полный same-harness Neutrino audit.
- Coding, tool use/BFCL, safety, multilingual и long-context batteries неполны.
- Работа исследует преобразование одной 8B architecture; перенос на 27B/70B не
  доказан.

## 18. Воспроизводимость и открытая публикация

Публичный комплект включает:

- immutable model checkpoint и tokenizer на Hugging Face;
- форматные manifests и hashes;
- CLI `wal-runtime`;
- portable, native CPU V4 и CUDA V18 backends;
- CUDA C++ sources;
- deterministic hardware-cache converter;
- focused tests и CI;
- benchmark protocols и этот отчёт на трёх языках.

Быстрый путь:

```bash
pip install 'git+https://github.com/AubakirovArman/WAL-Ternary-8B.git#egg=wal-tat[runtime]'
wal-runtime prepare armanibadboy/WAL-Ternary-8B \
  --output ~/.cache/wal/WAL-Ternary-8B-v77
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --prompt 'Explain ternary quantization simply.'
```

## 19. Заключение

WAL-Ternary-8B не стала чистой 1.58-битной моделью и не доказала абсолютный
паритет с каждым внешним продуктом. Результат другой и, с инженерной точки
зрения, более конкретный:

> Готовая Qwen3-8B была независимо преобразована в автономную систему
> 2.898417 BPW со 100% low-bit покрытием Transformer body, прошла четыре
> исходных quality/rate gate и получила прямой CPU/CUDA packed runtime без
> постоянной dense INT8/BF16 копии.

Путь к этому результату показал, почему большая часть простых идей не работает:
тернарный алфавит не сохраняет функцию сам по себе, локальная реконструкция не
равна autoregressive quality, а компактный checkpoint не равен компактному
inference. В то же время R12 и runtime V18 показали, что co-adapted low-bit
capacity и специализированные kernels способны вернуть как качество, так и
практическую эффективность без увеличения размера модели.

## Ссылки

1. Qwen Team. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), 2025.
2. Qwen. [Qwen3-8B model card and Apache-2.0 weights](https://huggingface.co/Qwen/Qwen3-8B).
3. Ma et al. [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764), 2024.
4. Jin et al. [PARQ: Piecewise-Affine Regularized Quantization](https://arxiv.org/abs/2503.15748), 2025.
5. Fermion Research. [Neutrino-8B model card](https://huggingface.co/FermionResearch/Neutrino-8B), accessed 2026-08-07.
