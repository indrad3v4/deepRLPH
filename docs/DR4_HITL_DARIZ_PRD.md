# Stage 4 – HITL DARIZ Engine PRD

You’re already very close to a real HITL/DARIZ loop; what’s missing is mostly wiring, centralization of the DARIZ logic, and a few concrete routing/robustness fixes in UI + orchestrator + engine.

---

## 1) Clarifying questions

- After a crash in the Terminal tab, do we want a fully automatic mode (immediately DARIZ → Model Council → run agents), or always a small confirmation dialog?
- What is the maximum number of auto‑DARIZ fix attempts before marking a case as "manual only" (e.g., 2–3 cycles)?

(If unanswered, default: auto‑mode with 2 cycles and soft‑confirm via existing messagebox dialogs.)

---

## 2) PRD Backlog (Stage 4: HITL DARIZ Engine)

### DR4-01 – Smart Crash Detector & Auto-Routing in UI

**Why / value**  
Автоматически распознавать краши/OOM в 💻 Terminal & Feedback и переводить проект в режим "DARIZ → Model Council" без ручных кликов, убирая трение между падением кода и стартом фикса.

**Acceptance criteria**

- `_run_in_terminal` помечает в состоянии UI факт сбоя (exit_code != 0, "Killed: 9", traceback в stdout/stderr).
- При первом краше появляется диалог: "Запустить DARIZ Auto‑Fix в режиме Model Council?" с опцией авто‑запомнить выбор.
- При согласии автоматически вызывается `_trigger_dariz_fix(auto_execute=True)` с текущими логами.
- После успешного DARIZ‑цикла UI переключает Execution tab, ставит `mode='council'`, `agents=4` и запускает `_execute_agents()` на обновлённом PRD.

**Verification**

Ручной сценарий:

1. Создать простой проект, подложить скрипт, который падает (например, `raise RuntimeError`).
2. Запустить его через "Run Target Script", получить crash.
3. Подтвердить авто‑DARIZ: PRD перезаписан, Execution tab в "council", прогон агентов стартует без доп. кликов.

**Files/modules likely touched**

- `src/ui/setup_window.py` (`_run_in_terminal`, `_trigger_dariz_fix`, `_create_execution_tab`, состояние текущего краша).
- Вспомогательный небольшой модуль/класс для хранения "crash snapshot" в UI (можно прямо в `RalphUI`).

---

### DR4-02 – Централизованный DARIZ Engine в Orchestrator (не в UI)

**Why / value**  
Сейчас DARIZ‑логика и 8‑шаговый анализ живут напрямую в UI: `_trigger_dariz_fix` сам строит промпт и парсит JSON‑PRD. Лучше иметь единый DARIZ‑движок в orchestrator, чтобы UI был тонким, а тот же механизм можно было вызывать и из бэкенда/CLI.

**Acceptance criteria**

- Новый класс/функция `DarizEngine` или метод `RalphOrchestrator.run_dariz_fix(crash_snapshot, strategy_text, auto_execute: bool)` возвращает нормализованный PRD и, опционально, указания по режиму (`parallel`/`council`, `num_agents`).
- Вся heavy‑логика: system prompt 8‑шагового Kukalev DARIZ, извлечение JSON из ответа, нормализация PRD → перенесена из `_trigger_dariz_fix` в orchestrator/отдельный модуль.
- `_trigger_dariz_fix` в UI только собирает `terminal_dump + exec_logs + user_strategy` и вызывает orchestrator‑метод; UI больше не парсит JSON руками.
- Orchestrator после успешного DARIZ автозаписывает новый `prd.json` и возвращает путь/структуру для UI.

**Verification**

- Юнит‑тест на уровне orchestrator: подать фиктивный `crash_snapshot` + синтетический LLM‑ответ с ```json { ... } ```, убедиться, что PRD нормализован и записан в `prd.json`.
- Ручной прогон через UI: нажать "Auto‑Fix & Run Agents" и убедиться, что поведение осталось прежним, но все LLM‑обращения идут через orchestrator.

**Files/modules likely touched**

- `src/orchestrator.py` (новый DARIZ‑метод, использование `_normalize_prd_for_engine`, запись `prd.json`).
- Новый модуль, например `src/dariz_engine.py`.
- Упрощённый `_trigger_dariz_fix` в `src/ui/setup_window.py`.
- Возможно, лёгкая интеграция с `src/prd_generator.py` / `src/prompt_generator.py` для повторного использования шаблонов.

---

### DR4-03 – Crash-Aware Context Ingestor (Grounded Auto-Fix)

**Why / value**  
DARIZ сейчас смотрит только на traceback и свободный текст, но уже есть `context_ingestor` и инспекторы датасетов/ONNX. При краше логично автоматически подтягивать мини‑саммари схем, датасетов и моделей, чтобы стратегия фикса опиралась на реальные структуры, а не только на текст.

**Acceptance criteria**

- Новый тип "crash snapshot context": `ContextIngestor` умеет принимать `crash_snapshot` (traceback + путь к проекту) и возвращать доп. поля (например, задействованные таблицы/фичи, размер данных, схема ONNX).
- Orchestrator при вызове DARIZ автоматически вызывает `ingest_project_context` и добавляет краткий текстовый summary (1–2 абзаца) в prompt, который получает DeepSeek.
- Если в проекте есть Parquet/CSV или ONNX, в DARIZ‑промпт попадает хотя бы: количество строк/колонок, тип задачи (regression/classification), основные слои модели.

**Verification**

- Юнит‑/интеграционный тест: синтетический проект с dummy Parquet/ONNX, фиктивный crash → вызов DARIZ → в сформированном `system_prompt/user_message` присутствуют артефакты из `dataset_inspector`/`onnx_inspector`.
- Ручной прогон: запустить Terminal‑краш в таком проекте и посмотреть в логах orchestrator, что контекст реально добавлен.

**Files/modules likely touched**

- `src/context_ingestor.py`, `src/dataset_inspector.py`, `src/onnx_inspector.py`.
- `src/orchestrator.py` (DARIZ‑метод добавляет `ctx.to_text_summary()` к промпту).

---

### DR4-04 – Escalation & Retry Policy in ExecutionEngine + Coordinator

**Why / value**  
Сейчас council‑логика умеет последовательно пробегать по ролям (Architect/Reviewer/Refiner/Verifier), фиксируя успех или провал, но нет явной политики эскалации после нескольких фейлов подряд на одном story и связи с DARIZ/Terminal‑loop.

**Acceptance criteria**

- В `_handle_story_failure` добавлен лимит попыток (`max_attempts`, например 3); при превышении статус story становится `blocked` и помечается флагом, что требуется DARIZ/manual.
- `ExecutionEngine` возвращает в ответе список `failed` с полем `requires_dariz=True` для таких историй.
- UI Execution tab, получив такой результат, отображает баннер "Некоторые stories требуют DARIZ‑цикла" и предлагает либо открыть Terminal tab с автоподстановкой последних ошибок, либо напрямую вызвать DARIZ‑движок без отдельного запуска скрипта.

**Verification**

- Интеграционный тест (engine + mock coordinator): имитировать повторные фейлы (возвращать `status="failed"`), убедиться, что после N попыток история помечена `blocked` и попадает в `failed` с флагом.
- Ручной прогон: одна из PRD‑историй намеренно невыполнима → после council‑прогона UI показывает, что она требует DARIZ‑анализа.

**Files/modules likely touched**

- `src/execution_engine.py` (`_handle_story_failure`, контракт ответа `execute`).
- `src/agent_coordinator_extensions.py` (опционально – более аккуратное описание попыток/эскалации, обогащённый результат).
- `src/ui/setup_window.py` (обработка статуса `partial`/`failed` в `_execute_agents`).

---

### DR4-05 – HITL UX Fixes & Bugfix в Execution UI

**Why / value**  
UI уже богатый, но есть несостыки: `_update_agents_hint` жёстко считает `mode='council'`, игнорируя реальное значение combobox, и Validation tab пока заглушка. Это мешает понять, как именно пройдёт запуск и где пользователь может вмешаться.

**Acceptance criteria**

- `_update_agents_hint` реально читает текущий режим через `self.mode_cb.get()` и корректно отображает `[Parallel Swarm]` vs `[Model Council]`, без хардкода `'council'`.
- В Execution tab текст статуса и hint всегда синхронен с тем, что реально передаётся в `execute_prd_loop(execution_mode=mode)`.
- В Validation tab хотя бы базовый wired‑up путь: кнопка "Run Validation" запускает оркестраторский/координаторский скрипт (например, `make validate` или `pytest`), а результат пишется в `validation_text`.
- В HITL Terminal tab явно показано, когда последний запуск был "Auto‑Fix (DARIZ + Council)" vs ручной.

**Verification**

- Ручной: переключать режимы в комбобоксе, изменять количество агентов, смотреть, что hint и `agents_status_label` отражают правильные значения, а в логах orchestrator видно совпадающий `execution_mode`.
- Ручной: нажать "Run Validation" → отрабатывает команда, вывод появляется в Validation tab.

**Files/modules likely touched**

- `src/ui/setup_window.py` (`_update_agents_hint`, `_execute_agents`, `_run_validation`).
- Опционально, небольшой helper в orchestrator/координаторе для единой команды валидации.

---

## 3) Loop execution plan

Для каждого DR4‑item:

- **Implement**  
  Выделяешь ветку (`feature/dr4-0X-...`), делаешь минимальные изменения только в указанных файлах.

- **Run checks**  
  Минимум: `python main.py` → ручной сценарий через UI для этого item.  
  По мере готовности: `pytest` для DARIZ/orchestrator/engine, плюс целевой скрипт в Terminal tab.

- **Fix**  
  Чинишь баги, приводишь интерфейсы в согласованное состояние (DARIZ‑движок возвращает одну и ту же форму PRD, Execution UI/engine её одинаково читают).

- **Commit**  
  Коммит вида `DR4-0X: short summary`, опционально – маленький PR к самому себе.

- **Mark pass**  
  Отмечаешь item как выполненный в своём PRD, при необходимости добавляешь пару строк в `docs/` (например, "Как работает Auto-Fix DARIZ").

---

## 4) Definition of Done (Stage 4)

- Любой краш в Terminal tab может быть переведён в DARIZ‑цикл одним кликом или автоматически, без ручного копирования логов.
- DARIZ‑логика живёт в orchestrator (или отдельном модуле), возвращает нормализованный PRD и не зависит от UI.
- ExecutionEngine и UI корректно различают `parallel` vs `council`, подсказки/логи совпадают с фактическим режимом.
- После нескольких неудачных попыток по одной истории ExecutionEngine явно помечает её как требующую DARIZ/HITL, а UI даёт понятный путь продолжения.
- Базовый контекст проекта (ingested docs/datasets/ONNX) автоматически приходит в DARIZ‑промпты, делая авто‑фиксы осмысленными по отношению к реальным данным.
