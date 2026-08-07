# WAL-Ternary-8B: Qwen3-8B моделін 2.898 BPW тікелей орындалатын low-bit жүйеге дербес түрлендіру

## Аннотация

Бұл мақалада WAL-Ternary-8B V77 моделін нөлден бастап құрудың толық жолы
сипатталады. WAL — `Qwen/Qwen3-8B` архитектурасынан дербес жасалған,
өзін-өзі қамтамасыз ететін low-bit жүйе. Transformer body ішіндегі 252 үлкен
сызықтық матрицаның барлығы dense BF16 пішімінен шығарылып, жеке packed
representation ішінде сақталады және тікелей сол пішімнен орындалады.

Толық релиз ағашының физикалық құны — бастапқы бір параметрге
**2.898417 бит (BPW)**. Негізгі оператор әр 128 салмақ тобына арналған
тернарлық `{-1, 0, +1}` кодты, сегіз sparse residual позицияны және 213
матрицада қолданылатын binary low-rank WALB2 түзетулерін біріктіреді.
Embedding `INT3/g128`, ал LM head `INT4/g128` түрінде сақталады.

Зерттеу дайын Qwen3-8B моделінің білімін тернарлық representation-ға ауысқанда
мүмкіндігінше сақтап қалу әрекетінен басталды. Қарапайым post-training
тернарлау модельді іс жүзінде бұзды: C4 perplexity шамамен **7952.54** болды.
Sparse-k8 модельді шамамен **41 PPL** деңгейіне қайтарды. Co-adapted WALB2 және
кейінгі құрылымдық тәжірибелер V73 нұсқасын **28.58 C4 PPL** деңгейіне жеткізді.

R7–R10 сериясы өте маңызды теріс нәтиже берді: жеке оператордың activation
error немесе teacher-forced KL көрсеткішін жақсарту толық autoregressive
жауапты автоматты түрде жақсартпайды. Негізгі серпіліс R12 кезеңінде болды:
жаңа кодтар мен байттар қоспай, модельде бұрыннан бар 465 амплитуда бірлесіп
қайта калибрленді. Финалдық V77:

- C4 PPL — **20.1743**;
- MMLU-Redux — **59.29%**;
- GSM8K strict — **76.57%**;
- толық көлем — **2.898417 BPW**.

Осы нәтижемен жоба бастапқы төрт gate-тің барлығын орындады.

Бөлек runtime бағыты компакт checkpoint-ті нақты орындалатын low-bit модельге
айналдырды. Direct-packed runtime Transformer body үшін тұрақты dense
INT8/BF16 көшірме жасамайды. GPU decode жылдамдығы алғашқы **4.62 токен/с**
деңгейінен **72.41 токен/с** деңгейіне дейін өсті; short-context режимінде
**99.36 токен/с** өлшенді. Peak GPU memory **3.66–3.93 ГБ**, ал native CPU V4
64 thread режимінде **5.93 токен/с** және **4.12 ГБ RSS** көрсетті.

Бұл жұмыс «таза 1.58-bit модель» туралы мәлімдеме емес. Нақты нәтиже —
Қазақстанда жасалған, ашық жарияланған, тернарлық негізі бар **2.898-BPW
гибридті low-bit Qwen3-8B**, оның model format, checkpoint, CPU/CUDA runtime,
сынақ хаттамалары, сәтсіз тәжірибелері мен шектеулері бірге жарияланған.

## 1. Идея қайдан пайда болды

8B класындағы қазіргі модельдер локалды қолдану үшін жеткілікті қабілетті,
бірақ BF16 салмақтарының өзі шамамен 16 ГБ орын алады. Қалыпты 4-bit
quantization бұл көлемді едәуір азайтады, алайда WAL жобасының сұрағы бұдан
қатаң болды:

> Дайын оқытылған Qwen3-8B моделін шамамен 3 BPW тернарлық жүйеге ауыстырып,
> оның білімін, reasoning қабілетін және практикалық орындалуын сақтауға бола ма?

Жобаның сыртқы шабыт көзі — Fermion Research жасаған Neutrino-8B. Neutrino
sub-2-bit Transformer linears және арнайы runtime практикалық болуы мүмкін
екенін көрсетті. Бірақ WAL Neutrino-ның көшірмесі емес және оның бір де бір
тензорын, scale мәнін, packed кодын, контейнерін немесе runtime компонентін
қолданбайды.

Екі модельдің ортақ бастапқы архитектурасы —
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B). Осы нүктеден кейін
олардың representation, training/recovery және runtime жолдары бөлек.

WAL-дың мақсаты Neutrino нәтижелерін жүктеп алу немесе қайта орау емес еді.
Мақсат — сыртқы benchmark-ті бағдар ретінде қабылдап, Qwen3-8B-ден бастап
толық тәуелсіз жолды өзіміз өту:

```text
Qwen3-8B
→ low-bit representation іздеу
→ quality recovery
→ автономды checkpoint
→ packed CPU/GPU kernels
→ соңғы benchmark және parity тексеруі
```

## 2. Low-bit LLM өрісіндегі орны

WAL жеке вакуумда пайда болған жоқ. Ultra-low-bit және ternary LLM бағытына
бірнеше маңызды жоба кіреді:

- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — тернарлық салмақтарды
  оқыту процесінің өзіне енгізетін native low-bit architecture;
- [bitnet.cpp](https://github.com/microsoft/BitNet) — BitNet модельдеріне
  арналған ресми CPU/GPU inference framework;
- [Spectra](https://arxiv.org/abs/2407.12327) — ternary language models-ті
  pretraining кезінде зерттейтін академиялық жұмыс;
- [TernaryLLM](https://arxiv.org/abs/2406.07177) — learnable ternarization және
  feature distillation арқылы low-bit quality recovery зерттеуі;
- [PrismML Ternary Bonsai](https://huggingface.co/collections/prism-ml/ternary-bonsai)
  — binary/ternary модельдер отбасы;
- [Neutrino-8B](https://huggingface.co/FermionResearch/Neutrino-8B) — жеке
  five-valued TRTC/FV5 форматы және арнайы runtime бар Qwen3-class жүйе.

Бұл жобаларды бір кестеге қойып, тек жарияланған сандар бойынша «кім жеңді» деп
айту дұрыс емес. Олардың:

- бастапқы модельдері;
- pretraining немесе post-training әдістері;
- physical packing есебі;
- prompt template-тері;
- benchmark subset-тері;
- parser-лері;
- output cap-тері;
- runtime hardware және execution stack-тері

әртүрлі.

WAL осы кең өрісте ерекше бір нүктені зерттейді: **дайын Qwen3-8B моделін full
retraining жасамай, өзіміздің T3+sparse-k8+WALB2 жүйесіне ауыстыру және сол
representation-ды тікелей орындайтын ашық runtime құру**.

Сондықтан WAL-ды жай «3-bit quant» деп те, «таза 1.58-bit BitNet» деп те атау
дұрыс емес. Ол — тернарлық негіз, sparse residual және binary low-rank
correction жолдары бар нақты **2.898-BPW** гибридті low-bit жүйе.

## 3. Бастапқы мақсат және алдын ала бекітілген gate-тер

Жобаның табысы эксперименттер аяқталғаннан кейін таңдалған ыңғайлы метрикамен
емес, алдын ала бекітілген төрт абсолютті gate арқылы бағаланды:

| Gate | Талап |
|---|---:|
| Толық checkpoint ағашы | ≤2.952031 BPW |
| C4 perplexity | ≤22.5637 |
| MMLU-Redux | ≥57.28% |
| GSM8K strict | ≥58.66% |

Бұл gate-тер негізгі сұраққа жауап берді:

> Практикалық көлемде өміршең, автономды low-bit Qwen3-8B жасалды ма?

Кейінірек V73-пен салыстыратын қатаң non-regression guardrail-дар қосылды.
Олар пайдалы диагностикалық құрал болды, бірақ бастапқы зерттеу мақсатын
нәтижеден кейін қайта жазған жоқ.

## 4. Qwen3-8B моделінің қай бөліктері түрлендірілді

Qwen3-8B — 36 Transformer блогынан тұратын dense модель. Әр блокта:

- attention үшін Q, K, V және O проекциялары;
- MLP үшін gate, up және down проекциялары бар.

Сондықтан үлкен сызықтық матрицалар саны:

```text
36 × 7 = 252
```

WAL ішінде осы 252 матрицаның барлығы BF16 representation-нан шығарылды.

Әр 128 бастапқы салмақ тобы үшін base operator:

```text
W_base = alpha · T + beta · R

T[i] ∈ {-1, 0, +1}
R = дәл 8 нөлден өзге таңбалы sparse позиция
```

Мұндағы:

- `T` — жалпы weight geometry-дің өте дөрекі тернарлық қаңқасы;
- `R` — әр топтағы ең маңызды сегіз ауытқуды қайтарады;
- `alpha` және `beta` — осы екі lane амплитудасын анықтайды.

213 матрица үшін қосымша WALB2 correction қолданылады:

```text
DeltaW(x) = Σ diag(s_row) U_sign diag(s_latent)
              V_sign diag(s_column) x
```

`U_sign` және `V_sign` тек `{-1, +1}` мәндерінен тұрады және bit-packed түрде
сақталады. Үздіксіз ақпарат шағын FP16 scale мәндерінде қалады. Қалған 39
матрица T3+sparse-k8 base жолымен ғана жұмыс істейді.

Endpoint layers те dense емес:

- embedding — `INT3/g128`;
- LM head — `INT4/g128`;
- norm және шағын қызметтік тензорлар — BF16.

Шағын тензорларды BF16 күйінде қалдырудың себебі — олардың жалпы файлға қосатын
көлемі аз, ал сапаға сезімталдығы жоғары.

## 5. Нақты биттік бюджет

Модельде **8,190,735,360** бірегей бастапқы параметр бар. Serialized model
payload көлемі — **2,950,747,732 байт**:

```text
8 × 2,950,747,732 / 8,190,735,360 = 2.882034496 BPW
```

Metadata, manifests және релизге міндетті runtime файлдары кіретін толық ағаш:

```text
2,967,521,309 байт = 2.898417472 BPW
```

Бұл есеп терминология үшін өте маңызды. `{-1, 0, +1}` алфавитінің теориялық
энтропиясы шамамен 1.585 бит болғанымен, толық модельге мыналар да кіреді:

- sparse позициялар;
- residual signs;
- group scales;
- WALB2 binary factors;
- embedding және LM head;
- metadata және integrity manifests.

Сол себепті WAL өзін «1.58-bit модель» деп атамайды. Есеп physical artifact-тің
толық құны бойынша жүргізіледі.

## 6. Эксперименттік әдістеме

Жұмыс design card тізбегі ретінде жүргізілді. Әр claim-bearing тәжірибеге дейін
мыналар бекітілді:

- гипотеза және себептік механизм;
- өзгертуге рұқсат етілген параметрлер;
- fitting, selection және confirmation деректерінің бөлінуі;
- негізгі және control метрикалар;
- PASS/FAIL gate;
- тоқтау шарты;
- byte-neutral тәжірибелер үшін checkpoint өлшемінің өзгермеуі.

Fitting немесе candidate selection үшін қолданылған split кейін тәуелсіз
confirmation ретінде қолданылмады. Generation метрикаларында тек орташа
accuracy емес, жұптық ауысулар сақталды:

```text
wrong → correct
correct → wrong
unchanged
```

Token-level confidence interval жеке токендер бойынша емес, жауаптар бойынша
bootstrap арқылы есептелді, себебі бір жауаптың токендері статистикалық
тәуелсіз емес.

Жобаны бірнеше қате promotion-нан сақтаған негізгі ереже:

```text
аз weight MSE
≠ жақсы operator output
≠ аз teacher-forced KL
≠ жақсы толық autoregressive жауап
```

## 7. I кезең: pure T3 неге модельді бұзды

Бастапқы гипотеза қарапайым болды: жақсы group scale тернарлық `{-1,0,+1}`
торына дайын модельдің білімін жеткілікті деңгейде тасымалдайды.

R12-0 ablation кейін бұл сұраққа нақты жауап берді:

| Representation | Mean NLL | PPL |
|---|---:|---:|
| Pure T3 | 8.981246 | **7952.54** |
| T3 + sparse-k8 | шамамен 3.71 | шамамен **41** |
| T3 + sparse-k8 + WALB2, V73 | шамамен 3.35 | **28.58** |

Pure T3 модель функциясының басым бөлігін жоғалтты. Формалды түрде weights
тернарлық болды, бірақ language model іс жүзінде бұзылды.

Sparse-k8 ең үлкен алғашқы recovery берді. Ол pure T3-ті шамамен
**5.2588 nat/token** жақсартты. Бірақ WALB2 жеке әмбебап patch болып шықпады:
дұрыс sparse base болмаса, binary correction кей жағдайда NLL-ді нашарлата
алды.

Осыдан маңызды қорытынды шықты:

> Финалдық WAL representation — тәуелсіз quantizer-лер қосындысы емес,
> бір-біріне бейімделген co-adapted residual system.

## 8. II кезең: V73 моделін құру

Келесі циклдерде:

- sparse capacity layer және role бойынша бөлінді;
- group geometry зерттелді;
- binary low-rank correction қосылды;
- әр өзгеріс нақты NLL және generation арқылы тексерілді;
- checkpoint byte budget тұрақты бақыланды.

Нәтижесінде V73 алынды:

- 252/252 low-bit Transformer body матрицасы;
- 213 WALB2 correction;
- толық көлем — 2.898417 BPW;
- C4 PPL — 28.5811;
- MMLU-Redux — 60.36%;
- GSM8K strict — 75.28%;
- GSM8K flexible — 76.57%.

V73 бастапқы gate-тердің ішінен тек C4 шегінен өтпеді. Соған қарамастан,
knowledge және mathematical reasoning perplexity көрсеткішінен күткеннен әлдеқайда
жақсы сақталды.

Бұл бір маңызды практикалық сабақ берді:

> Бір ғана PPL саны модельдің барлық пайдалы қабілетін толық сипаттамайды.

## 9. Qwen3 protocol және жалған GSM8K апаты

Алғашқы bare-prompt GSM8K тексеруі V73 үшін 0% strict көрсетті. Бұл модельдің
математикалық қабілеті толық жойылғандай көрінді.

Бірақ Qwen3-тің дұрыс chat template-і және `enable_thinking=False` режимі
қолданылғанда, сол V73 диагностикалық slice-та шамамен **73% strict** алды.
BF16 teacher сол режимде шамамен **94%** көрсетті.

Демек, алғашқы 0% нәтижесінің үлкен бөлігі model failure емес, evaluation
protocol қатесі болды. Бірақ BF16-мен қалған айырмашылық нақты еді.

Token-level forensics мынаны көрсетті:

- BF16-пен top-1 agreement шамамен 85.9%;
- KL шамамен 0.42–0.50 nat/token;
- V73 entropy сәл төмен;
- semantic trajectory ерте ажырайды;
- қате branch таңдалған соң модель оған шамадан тыс сенімді болады;
- thinking mode ішінде EOS және repetition мәселелері бар.

Яғни knowledge capacity сақталды, бірақ logits geometry дәлдігі төмендеді.

## 10. III кезең: R5–R7 себептік локализация

Head/body swap және activation patching drift-тің бәрі LM head-та емес екенін
көрсетті. Негізгі сезімтал аймақтардың бірі L4 `gate_proj`, кейін L10 attention
V/O болды.

R7d кезінде L4 gate-тің бұрыннан бар row scales мәндері нақты V73 states арқылы
қайта калибрленді. Бұл алғашқы жергілікті byte-neutral mechanism PASS болды:

- failure KL үш тәуелсіз set-те 4.98–5.81% төмендеді;
- gate pre-activation error шамамен 35% азайды;
- BF16 gate oracle әсерінің 53–55%-ы қалпына келді;
- control states бұзылмады;
- model size 2.898417 BPW болып қалды.

Бірақ 600 толық жауап нәтижесі:

```text
wrong → correct: 14
correct → wrong: 18
unchanged:       568
```

Локалды оператор BF16-ге сенімді түрде жақындады, бірақ sequence accuracy
жақсармады. Бұл operator recovery мен trajectory recovery екі бөлек міндет
екенінің алғашқы күшті дәлелі болды.

## 11. IV кезең: R8 және amplitude-only тармағының жабылуы

R8a бүкіл shrink жолын тексерді:

```text
theta(lambda) = theta_V73 + lambda(theta_R7d - theta_V73)
```

Selection Set D ішінде нәтиже пайдалы көрінді:

```text
24 wrong→correct
15 correct→wrong
```

Бірақ тәуелсіз Set E:

```text
15 wrong→correct
29 correct→wrong
-2.33 percentage points
```

берді. Confidence interval нөлді қамтымады. Бұл selection winner's curse-тің
нақты мысалы болды: жергілікті ең жақсы `lambda` жаңа autoregressive boundary
үшін тасымалданбады.

R8b-0 36,864 row-scale параметріне sequence-margin gradient есептеді.
Орта есеппен repair және preserve direction келісімді болды (`cos≈0.729`),
бірақ жеке жұптардың **42.54%**-ында cosine теріс болды.

Rank-64 QP:

- 256 repair trajectory-дің барлығына оң predicted gain берді;
- unconstrained repair signal-дың 41.94%-ын сақтады;
- бірақ 256 preserve мысалдың 52-сін бұзды, ал алдын ала бекітілген лимит 51 еді.

Формалды verdict — FAIL; ғылыми мағынасы — near-feasible repair cone.

R8b-1 соңғы strict-preserve тексеруі amplitude-only тармағын жапты. 15
қорғалатын trajectory үшін gradient norm бекітілген trust radius ішінде талап
етілген safety margin-нан кіші болды. Коши–Буняковский теңсіздігі бойынша бұл
constraints solver іске қосылмай тұрып-ақ орындалмайтыны дәлелденді.

Бұл кез келген nonlinear scale update мүмкін емес дегенді білдірмейді. Бірақ
R7d, R8a және R8b-0 нәтижелерімен бірге ескі row-scale direction үшін тағы
learning-rate, ridge немесе rank sweep жасауға ғылыми негіз қалдырмады.

## 12. V кезең: R9–R10 directional oracle-дары

R9 сұрақты өзгертті:

```text
Ескі correction-ды қанша масштабтау керек?
```

дегеннің орнына:

```text
Сол rank және byte budget ішінде correction бағытын қайта құруға бола ма?
```

Continuous same-rank oracle жергілікті directional capacity бар екенін
көрсетті, бірақ full generation improvement жаңа set-ке тасымалданбады.
Repair/preserve objective бар R9a2 де fresh holdout ішінде тұрақты gain бермеді.

R10 екінші causal node — L10 attention V/O — және L4+L10 joint 64-dimensional
oracle қосты. Fit техникалық тұрғыда сәтті өтті, бірақ mechanism fresh holdout
үшін қайтадан әлсіз болды.

Осы жерде жеткілікті дәлел жиналды:

> Статикалық жергілікті correction direction орташа операторды жақсарта алады,
> бірақ әр контекст үшін дұрыс autoregressive trajectory-ді қауіпсіз таңдауға
> қажетті context-specific selectivity бермейді.

Бұл сәтсіз тәжірибелер бос шығын болған жоқ. Олар solution space-ті тарылтып,
жобаны шексіз local hyperparameter search-тен тоқтатты.

## 13. VI кезең: R12 жаһандық ко-бейімдеу

Жергілікті тармақтар жабылғаннан кейін жоба бастапқы өтпеген негізгі gate — C4
көрсеткішіне қайта оралды.

R12-0 бүкіл representation жаһандық co-adapted екенін көрсетті. Сондықтан R12-A:

- жаңа ternary codes қоспады;
- жаңа sparse positions қоспады;
- WALB2 rank үлкейтпеді;
- жаңа binary factors қоспады;
- metadata өсірмеді.

Оның орнына бұрыннан сақталған 465 амплитуда бірге оңтайландырылды:

- T3+sparse-k8 base үшін 252 коэффициент;
- WALB2 corrections үшін 213 коэффициент.

Fit тек train-only C4 windows ішінде жүргізілді. Release C4, MMLU, GSM8K,
IFEval test және confirmation split жабық қалды. Соңғы коэффициенттер бұрыннан
бар FP16 scale мәндеріне folded болды, сондықтан checkpoint көлемі өзгермеді.

R12-A жасаған V76 C4 көрсеткішін қатты жақсартты, бірақ generation audit
state-dependent termination regression тапты. Осыдан кейін тек бір алдын ала
рұқсат етілген R12-B1-Lite орындалды: сол 465 foldable amplitude stop/continue
trace және preservation anchor арқылы termination-safe calibration алды.

Финалдық folded checkpoint — V77.

R12-нің негізгі ғылыми жаңалығы:

> V73 ішінде жеткілікті discrete information capacity бар еді. Қалған
> quality loss-тың үлкен бөлігі жаңа кодтардың жетіспеуінен емес, residual
> lane-дардың жаһандық калибрациясының қате болуынан шықты.

## 14. V77 финалдық нәтижелері

| Метрика | V73 | V77 | Өзгеріс |
|---|---:|---:|---:|
| Complete tree BPW | 2.898417 | **2.898417** | 0 байт |
| C4 PPL | 28.5811 | **20.1743** | -29.4% |
| MMLU-Redux | **60.36%** | 59.29% | -1.07 pp |
| GSM8K strict | 75.28% | **76.57%** | +1.29 pp |
| GSM8K flexible | 76.57% | **77.10%** | +0.53 pp |
| IFEval strict, cap 256 | 48.24% | **51.57%** | +3.33 pp |
| IFEval loose | 53.97% | **54.90%** | +0.93 pp |
| IFEval repeated tail | 35.86% | **28.65%** | -7.21 pp |

V77 бастапқы төрт gate-тің бәрінен өтті.

Кейін енгізілген V73-relative promotion protocol екі guardrail бойынша FAIL
берді:

- MMLU: V73-тен -1.07 pp;
- IFEval max-length: +2.40 pp.

Бұл нәтижелер жасырылған жоқ. Сондықтан статус екі деңгейлі:

```text
WAL research goal: ACHIEVED
V77 original gates: 4/4 PASS
strict dominance over every V73 diagnostic: NOT ACHIEVED
```

Бұл «бәрінен толық оздық» деген асыра мәлімдемеден де, бүкіл жұмысты бір FAIL
сөзіне қысқартудан да әділ.

## 15. Compact checkpoint-тен нақты runtime-ға дейін

Алғашқы WAL checkpoint-тері дискіде шынымен компакт болды, бірақ reference
loader Transformer body-ді BF16 representation-ға материализациялады. Бұл
quality evaluation үшін жеткілікті еді, алайда inference memory үнемін
бермеді.

Осы айырмашылық өте маңызды:

```text
3 ГБ файл
≠
4 ГБ inference
```

Егер runtime compact bytes-ті толық precision көшірмеге ашса, physical storage
артықшылығы нақты орындалуға айналмайды.

Direct-packed runtime мына операторды:

```text
y = (T3+sparse-k8)x + Σ U_r(V_r^T x)
```

тұрақты dense матрица құрмай орындауы тиіс болды.

Portable reference алдымен формат дұрыстығын дәлелдеді, бірақ бір forward
минуттар алды. Кейін `.walhw` hardware cache жасалды. Ол:

- 2-bit packed ternary symbols;
- sparse positions және signs;
- binary WALB2 factors;
- FP16 scales;
- ABI/layout metadata

сақтайды. Бұл dense checkpoint емес.

Cache canonical checkpoint manifest, converter SHA, ABI version және 252
матрицаның hash мәндерімен fail-closed түрде байланысады. Parent checkpoint
сәйкес болмаса, runtime stale cache-ті үнсіз пайдаланбайды.

## 16. GPU runtime эволюциясы

| Нұсқа | Негізгі өзгеріс | Decode |
|---|---|---:|
| Алғашқы direct-packed | correctness kernels | 4.62 ток/с |
| Бірінші WALB2 fusion | launches және traffic азайды | 5.13 ток/с |
| V17/V18 | paired activations, fused paths | шамамен 32 ток/с non-graph |
| V18 + CUDA Graph | steady-state replay | **72.41 ток/с** |
| V18 + auto KV 256 | short-context CLI | **99.36 ток/с** |

Nsight Systems ерте айтылған «runtime түгел launch-bound» гипотезасын
теріске шығарды. Уақыттың көбі packed T3 және WALB2 kernels ішінде өтетіні
анықталды.

Сондықтан оңтайландыру реті:

1. data layout және vectorized read;
2. unpack тиімділігі;
3. WALB2 fusion;
4. shared-input projection reuse;
5. содан кейін CUDA Graph болды.

Kernel уақыты азайған соң ғана host overhead елеулі үлеске айналып, CUDA Graph
үлкен қосымша gain берді.

Нәтижелер:

- GPU decode: 4.62 → 72.41 ток/с, шамамен 15.7×;
- short-context: 99.36 ток/с;
- old materializing/cache memory: 12.60 ГБ;
- direct-packed peak: 3.75–3.93 ГБ;
- prepared cache қолданылған cold start: шамамен 355 → 10.1 секунд.

72.41 және 99.36 ток/с сандары нақты H200/SM90 configuration, prompt, KV және
benchmark protocol-ға байланысты. Оларды кез келген GPU үшін кепілдік ретінде
қарауға болмайды.

## 17. CPU runtime

Native CPU V4 corrected matrix үшін бес бөлек өтуді екі fused pass-қа дейін
қысқартты. Xeon платформасында 64 thread режимінде:

```text
decode: 5.93 ток/с
RSS:    4.12 ГБ
```

Бір same-machine 16-thread салыстыруда WAL 3.26 ток/с, ал Neutrino 2.16 ток/с
көрсетті. Бірақ бұл тек сол бір hardware/protocol өлшемі; оны барлық CPU үшін
жалпы артықшылық деп жариялауға болмайды.

CPU decode практикалық деңгейге жеткенімен, CPU prefill әлі әлсіз. Қазіргі
decode-style path prompt token-дерін жеткілікті batch reuse-пен өңдемейді.
Келесі инженерлік міндеттер:

- batched packed GEMM;
- weight tile reuse;
- NUMA-aware pinning;
- физикалық core/SMT sweep;
- Apple Silicon-дағы нақты NEON тесті;
- кейін арнайы Metal backend.

## 18. Direct-packed quality parity

Packed arithmetic reduction order мен rounding-ті сәл өзгертеді. Сондықтан
BF16 materialized, CUDA, TF32 және CPU арасында bit-exact logits талап ету
әдістемелік тұрғыдан дұрыс емес. Маңыздысы — task-level quality және bounded
numerical drift.

| Метрика | Materialized V77 | Direct packed |
|---|---:|---:|
| C4 PPL | 20.1743 | **20.2156** |
| MMLU-Redux | 59.29% | **59.26%** |
| GSM8K strict | 76.57% | **76.19%** |
| GSM8K flexible | 77.10% | **76.35%** |

C4, MMLU және GSM8K direct-packed runtime-ның күшті numerical parity
көрсетті.

IFEval сезімтал болып шықты:

- materialized cap-256 strict — 51.57%;
- алғашқы direct-packed cap-256 strict — 49.35%.

Алғашқы 100 prompt тек discovery үшін қолданылып, deployment profile алдын ала
бекітілді: cap 512 және live repeated-tail guard. Қалған untouched 441 prompt:

- direct-packed strict — 56.69%;
- direct-packed loose — 60.54%;
- frozen materialized cap-256 strict/loose — 52.38/55.56%;
- materialized cap-512 strict/loose — 58.96/63.72%.

Практикалық profile пайдалы, бірақ IFEval numerical gap толық жабылды деп айтуға
болмайды.

## 19. Neutrino-мен адал салыстыру

Neutrino-8B WAL үшін маңызды сыртқы бағдар болды. Оның ресми model card-ы
five-valued TRTC/FV5 форматты, native runtime және жеке benchmark battery-ді
сипаттайды.

2026-08-07 күніндегі жарияланған Neutrino көрсеткіштері арасында:

- executable container — шамамен 3.88 ГБ;
- lossless transport artifact — шамамен 2.56 ГБ;
- C4 PPL — 21.48;
- MMLU-Redux — 67.84%;
- IFEval prompt-strict — 73.17%;
- BFCL v3 — 65.31%;
- GSM8K flexible — 51.00%;
- GSM8K stated/strict format — 49.33%.

WAL-дың өз frozen protocol нәтижелері:

- complete release tree — 2.967 ГБ / 2.898417 BPW;
- C4 PPL — 20.1743;
- MMLU-Redux — 59.29%;
- IFEval strict — 51.57%;
- GSM8K strict — 76.57%.

Бұл сандарды тікелей ranking ретінде қолдануға болмайды. Same-harness audit әлі
толық орындалмаған. Prompt template, parser, output cap және subset өзгеше.

Сондықтан дұрыс қорытынды:

- Neutrino жарияланған MMLU, IFEval және BFCL бойынша күшті әрі runtime жағынан
  анағұрлым mature;
- WAL өз GSM8K protocol-ында жоғары mathematical result көрсетті;
- екі жүйе C4 бойынша бір сандық диапазонда;
- WAL Neutrino tensors қолданбай, Qwen3-8B-ден тәуелсіз жасалды;
- WAL-дың жеке T3+sparse-k8+WALB2 форматы бар;
- толық functional/runtime parity тек ортақ harness-тен кейін ғана айтылуы тиіс.

## 20. WAL-дың күшті жақтары

### 20.1. Толық low-bit Transformer coverage

252 үлкен body matrix-тің барлығы low-bit representation ішінде. Runtime dense
Qwen3 немесе Neutrino body checkpoint-ін қажет етпейді.

### 20.2. Физикалық көлем адал есептелген

Тек `log2(3)` немесе selected layer bits емес, payload, endpoints, scales және
релиз metadata-сы бірге есептеледі.

### 20.3. Direct-packed execution дәлелденген

Checkpoint жай архив емес. CPU/CUDA kernels packed representation-ды тұрақты
dense INT8/BF16 body көшірмесінсіз орындайды.

### 20.4. Model-level quality gate-тер орындалған

V77 бастапқы C4, MMLU, GSM8K және size gate-тердің 4/4-інен өтті.

### 20.5. Теріс нәтижелер де жарияланған

R7–R10 локалды fidelity мен sequence quality арасындағы қайшылықты жасырып
қалмады. Бұл басқа зерттеушілердің сол тұйық тармақтарды қайталау ықтималдығын
азайтады.

### 20.6. Ашық reproduction жолы бар

Model, runtime source, CLI, manifests, hashes, installation guides және research
article бір public release ішінде берілген.

### 20.7. Қазақстанда жасалған дербес зерттеу

Бұл ресми ұлттық foundation model немесе мемлекеттік жоба емес. Ол — Арман
Аубакиров бастаған, AI-assisted эксперименттік процесс арқылы Қазақстанда
жасалған ашық зерттеу жұмысы. Осы нақты тұжырым асыра айтпай-ақ жобаның
шығу тегін көрсетеді.

## 21. Әлсіз жақтары мен шектеулері

WAL нәтижесін дұрыс түсіну үшін мына шектеулерді бірге көрсету қажет:

- бұл таза 1.58-bit BitNet емес;
- full retraining немесе native ternary pretraining орындалмады;
- MMLU V73-пен салыстырғанда 1.07 pp төмендеді;
- IFEval direct-packed numerical gap толық жабылған жоқ;
- precompiled wheels/binaries әлі жоқ, алғашқы іске қосу JIT compile жасайды;
- CPU prefill decode-ке қарағанда әлсіз;
- ARM64/NEON коды жазылған, бірақ физикалық Mac-та benchmark жасалмаған;
- арнайы Metal kernel жоқ;
- CUDA performance негізінен H200/SM90 үшін дәлелденді;
- басқа NVIDIA, AMD және consumer GPU батареясы толық емес;
- coding, BFCL/tool use, safety және multilingual бағалау жеткіліксіз;
- long-context architecture мүмкіндігі бар, бірақ custom runtime 32K/131K
  battery-ден өтпеген;
- Neutrino/Bonsai/BitNet-пен толық same-harness comparison жасалмаған;
- әдіс тек Qwen3-8B үшін дәлелденді, 27B/70B-ге автоматты тасымалдануы белгісіз;
- runtime production serving engine емес: batching, concurrency, monitoring және
  fault isolation әлі жетілдірілуі керек.

Бұл тармақтар нәтижені жоққа шығармайды. Олар claim шекарасын анықтайды.

## 22. Қай бөлігін басқа зерттеуші қайталай алады

Жарияланған материалдар мына толық циклді зерттеуге мүмкіндік береді:

1. Canonical checkpoint пен tokenizer-ді жүктеу.
2. Manifest және SHA арқылы artifact тұтастығын тексеру.
3. `.walhw` hardware cache-ті deterministic converter арқылы дайындау.
4. Portable, CPU немесе CUDA backend таңдау.
5. Packed inference іске қосу.
6. C4/MMLU/GSM8K/IFEval хаттамаларын қайта орындау.
7. Runtime parity-ді materialized reference-пен салыстыру.
8. Format пен kernel source-ты өзгертіп, жаңа тәжірибелер құру.

Бірақ «мақаланы оқып, бір командамен кез келген 8B модельді 2.898 BPW сапамен
түрлендіру» әзірге мүмкін емес. V77 жасау жолында architecture-specific
selection, көп сатылы recovery және ұзақ experimental campaign болды.

Яғни ашық жарияланған жол:

- проблема decomposition-ын;
- representation формуласын;
- evaluation firewall-ын;
- сәтсіз механизмдерді;
- runtime ABI мен source-ты;
- финалдық artifact-ті

қайталай алады. Бірақ басқа base model үшін quality recovery қайта дәлелденуі
керек.

## 23. Жылдам орнату және іске қосу

GitHub арқылы:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  'git+https://github.com/AubakirovArman/WAL-Ternary-8B.git#egg=wal-tat[runtime]'
wal-runtime doctor
```

Hardware cache дайындау:

```bash
wal-runtime prepare armanibadboy/WAL-Ternary-8B \
  --output ~/.cache/wal/WAL-Ternary-8B-v77
```

GPU іске қосу:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --backend cuda-v18 \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --prompt 'Тернарлық модель деген не екенін қарапайым тілмен түсіндір.' \
  --max-new-tokens 256
```

CPU іске қосу:

```bash
wal-runtime generate armanibadboy/WAL-Ternary-8B \
  --backend cpu-v4 --device cpu \
  --hardware-cache ~/.cache/wal/WAL-Ternary-8B-v77 \
  --threads 16 \
  --prompt 'Фотосинтезді қысқаша түсіндір.'
```

Толық нұсқаулық:
[INSTALLATION_KK.md](INSTALLATION_KK.md).

## 24. Жарияланған материалдар

Public release құрамына:

- Hugging Face ішіндегі immutable model checkpoint және tokenizer;
- GitHub ішіндегі runtime source;
- CUDA C++ және native CPU kernels;
- `wal-runtime` CLI;
- portable reference backend;
- deterministic hardware-cache converter;
- parent/model/cache manifests және SHA-256;
- benchmark evidence;
- security және integrity ережелері;
- қазақ, орыс және ағылшын тіліндегі installation guide;
- қазақ, орыс және ағылшын тіліндегі research article

кіреді.

Негізгі сілтемелер:

- Модель: <https://huggingface.co/armanibadboy/WAL-Ternary-8B>
- Код және runtime: <https://github.com/AubakirovArman/WAL-Ternary-8B>
- Қазақша мақала: <https://github.com/AubakirovArman/WAL-Ternary-8B/blob/main/docs/RESEARCH_ARTICLE_KK.md>
- Орысша мақала: <https://github.com/AubakirovArman/WAL-Ternary-8B/blob/main/docs/RESEARCH_ARTICLE_RU.md>
- English paper: <https://github.com/AubakirovArman/WAL-Ternary-8B/blob/main/docs/RESEARCH_ARTICLE_EN.md>

## 25. Негізгі ғылыми сабақтар

### 25.1. Тернарлық алфавит функцияны өзі сақтамайды

Pure T3 үшін PPL ≈7952 болуы `{-1,0,+1}` кодқа көшу автоматты түрде knowledge
сақтайды деген ойды теріске шығарады.

### 25.2. Residual capacity бірге бейімделуі тиіс

Sparse-k8 және WALB2 әсерлері additive емес. Олардың scales және бағыттары
бүкіл модель күйімен бірге калибрленуі керек.

### 25.3. Operator fidelity sequence quality-ге тең емес

Локалды KL немесе activation error төмендегенімен, ерте бір argmax flip барлық
келесі context-ті өзгерте алады.

### 25.4. Дұрыс бастапқы trajectory — қорғалатын ресурс

Teacher imitation дұрыс, бірақ teacher-ден өзгеше reasoning path-ты бұзуы
мүмкін. Repair objective baseline-wrong және baseline-correct мысалдарды бөлек
қарауы керек.

### 25.5. Глобалдық calibration жаңа кодтан күшті болуы мүмкін

R12 жаңа байт қоспай, 465 foldable amplitude арқылы C4-ті 29.4% жақсартты.
Бұл existing low-bit codes ішінде жасырын capacity бар екенін көрсетті.

### 25.6. Compact storage және compact inference екі бөлек міндет

2.898 BPW файл тек packed kernels бар кезде ғана 4 ГБ шамасындағы inference
memory артықшылығына айналады.

### 25.7. Дұрыс protocol — модель сапасының бір бөлігі

Bare-prompt GSM8K 0% нәтижесі дұрыс Qwen3 chat protocol қолданылғанда шамамен
73%-ға өзгерді. Evaluation stack модельдің шынайы қабілетін бүркемелеуі мүмкін.

## 26. Қорытынды

WAL-Ternary-8B таза 1.58-bit модель емес және барлық сыртқы low-bit жобамен
толық паритет дәлелдеген жоқ. Оның нақты, тексерілген нәтижесі мынау:

> Дайын Qwen3-8B моделі тәуелсіз түрде 2.898417-BPW автономды жүйеге
> түрлендірілді; Transformer body-дің 252/252 матрицасы low-bit coverage алды;
> модель бастапқы төрт size/quality gate-тен өтті және тұрақты dense INT8/BF16
> body көшірмесінсіз тікелей packed CPU/CUDA runtime алды.

Жолдың ең құнды бөлігі тек финалдық checkpoint емес. Зерттеу:

- неге pure post-training ternary сәтсіз болатынын;
- sparse және low-rank residual не үшін керегін;
- local reconstruction неге full generation-ды кепілдемейтінін;
- global co-adaptation қалай byte-neutral үлкен gain бере алатынын;
- compact file-ды нақты compact inference-ке айналдыру үшін жаңа kernels неге
  қажет екенін

бір эксперименттік тізбек ішінде көрсетті.

Нәтиже мінсіз емес. MMLU trade-off, IFEval numerical sensitivity, CPU prefill,
Apple Silicon, long context, tool use және production packaging әлі ашық.

Бірақ негізгі жол салынды және ашық жарияланды:

```text
дайын Qwen3-8B
→ дербес low-bit representation
→ quality recovery
→ 2.898-BPW checkpoint
→ direct-packed runtime
→ task-level verification
```

Бұл басқа зерттеушілерге барлық қадамды сынға алуға, қайталауға, жақсартуға
және жаңа low-bit модельдерге бейімдеуге мүмкіндік береді.

## Әдебиет және дереккөздер

1. Qwen Team. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), 2025.
2. Qwen. [Qwen3-8B model card және Apache-2.0 weights](https://huggingface.co/Qwen/Qwen3-8B).
3. Ma et al. [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764), 2024.
4. Microsoft. [bitnet.cpp official inference framework](https://github.com/microsoft/BitNet).
5. Jin et al. [PARQ: Piecewise-Affine Regularized Quantization](https://arxiv.org/abs/2503.15748), 2025.
6. TriLM Team. [Spectra: Surprising Effectiveness of Pretraining Ternary Language Models at Scale](https://arxiv.org/abs/2407.12327), 2024.
7. Chen et al. [TernaryLLM: Ternarized Large Language Model](https://arxiv.org/abs/2406.07177), 2024.
8. PrismML. [Ternary Bonsai model collection](https://huggingface.co/collections/prism-ml/ternary-bonsai).
9. Fermion Research. [Neutrino-8B model card](https://huggingface.co/FermionResearch/Neutrino-8B), қаралған күні: 2026-08-07.

