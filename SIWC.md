# Sign in with ChatGPT → public Responses API

Дополнительный **Python-профиль `siwc`** использует официальный OAuth для open-source
клиентов и публичный Responses API с ChatGPT-планом. Сам `codex-as-api` остаётся
сторонним совместимым шлюзом, а не продуктом или одобренным клиентом OpenAI.

Статус ветки: **preview**. Доступность зависит от аккаунта, разрешённых scopes,
политик OpenAI и реально выданного каталога. Публичная API-карточка движка не
доказывает такое же окно или набор инструментов на подписочном маршруте.

## Два явно разделённых профиля

| Профиль | Авторизация | Upstream |
|---|---|---|
| `codex` — по умолчанию | существующий standalone Codex OAuth upstream | ChatGPT Codex backend |
| `siwc` — явно | собственная регистрация и direct ChatGPT-plan grant | `https://api.openai.com/v1` |

Профиль `siwc` **не использует API key или API-кредиты** и не переходит на них
при ошибке. Credentials другого OAuth-клиента не импортируются. TypeScript и Rust
в этой ветке не получают SIWC-поддержку автоматически: адаптация относится к Python.

## Отдельная установка и вход

Для разработки допускается установка ветки; для воспроизводимого запуска
в ссылке зависимости закрепляется проверенный полный commit SHA.

```sh
uv venv .venv
uv pip install --python .venv/bin/python \
  'git+https://github.com/olegopro/codex-as-api.git@siwc-oauth'
```

Профиль и его auth-файл выбираются **до входа**, отдельно от старой установки:

```sh
export CODEX_AS_API_BACKEND=siwc
export CODEX_AS_API_SIWC_AUTH_PATH="$HOME/.codex-as-api/siwc/auth.json"
.venv/bin/codex-as-api login
.venv/bin/codex-as-api status
.venv/bin/codex-as-api preflight
```

Вход требует действия владельца в браузере. Используются PKCE S256, state и nonce;
callback слушает только `127.0.0.1`. Выданный client ID сохраняется вместе с его
регистрацией. Идентификатор хоста создаётся как UUID URN (`urn:uuid:…`), как в
официальном примере; незавершённые регистрации старого формата исправляются при
следующем входе, без изменения уже выданных client ID и токенов.
Не следует копировать client ID из другого приложения или переносить
токены из `~/.codex/auth.json`.

После входа сервер можно проверить на отдельном loopback-порту:

```sh
CODEX_AS_API_HOST=127.0.0.1 CODEX_AS_API_PORT=18081 \
  .venv/bin/codex-as-api
```

`status` — локальное состояние credentials, не проверка отзыва токена и не баланс
подписки. `preflight` проверяет авторизацию и доступность настроенного движка через
GET каталога; при необходимости обновляет grant, но не выполняет генерацию.
Файлы авторизации не должны попадать в git, логи, отчёты или аргументы процесса;
каталог защищается режимом `0700`, файлы — `0600`.

## Протокол

- Каталог аккаунта: Bearer GET `https://api.openai.com/v1/models`.
- Генерация по умолчанию: `wss://api.openai.com/v1/responses`, событие
  `response.create`, `store:false`, полный `input` на каждом ходу. Сокет сохраняется
  отдельно для аккаунта, сессии (`prompt_cache_key`) и модели. Канал другой модели
  не вытесняет основной. История по `previous_response_id` в шлюзе не хранится.
- Прямой контроль 2026-10-31: повтор на том же WebSocket прочитал 1792 из 1926
  токенов; переподключение дало 0 из 1929. HTTP-контроль дал 0. Эти результаты
  относятся к текущему grant и Sol, а не обещают одинаковую экономию каждому плану.
- До 16 каналов; неактивные вытесняются при следующем обращении по LRU/30 минутам
  простоя. Канал обновляется при смене токена и перед часовым пределом соединения.
  Отдельного уборщика или дополнительного сервиса нет.
- `CODEX_AS_API_SIWC_TRANSPORT=http` — явный диагностический HTTP/SSE; автоматического
  отката на него, legacy или API key нет. Каталог и OAuth остаются HTTP-запросами.
- Явный HTTPS proxy учитывается каталогом и WebSocket. Соединение не перенаправляется
  на ChatGPT `backend-api`; credentials по-прежнему только из SIWC-профиля.
- Инструкции передаются как `instructions`/developer, не system-role input items.
- Function/custom tools приводятся к разрешённому preview формату; namespace содержит
  обязательный непустой `description`. Вызовы и результаты сохраняют имена и call IDs
  на стороне Anthropic-клиента.
- Успех требует `response.completed`: EOF, failed/incomplete и потоковая ошибка не
  означают успешного завершения и не завершают недописанный tool-call.
- Anthropic SSE отправляет `ping` каждые 10 секунд тишины. Отмена незавершённого
  запроса закрывает его канал, не затрагивая соседние. Таймаут SIWC upstream — 600 с.
- `stop_sequences` Anthropic-фасада обрабатываются локально, включая маркер
  `</block>` внутренних вызовов Claude Code. Клиент получает обрезанный текст,
  `stop_reason: stop_sequence` и сам маркер. Upstream дочитывается до завершения:
  usage учитывает весь фактически созданный вывод, ошибки после маркера не скрываются.
- Запрещённые preview-параметры не отправляются, включая `max_output_tokens`,
  `metadata`, `temperature` и HTTP `previous_response_id`.
- Потолок Anthropic `max_tokens` нельзя объявлять гарантированным upstream cap:
  подписочный preview запрещает `max_output_tokens`.
- Hosted MCP, Responses `tool_search` и другие неподдерживаемые hosted tools не
  появляются от включения этого профиля. Инструменты локального клиента — другой слой.

Набор движков и метаданные окна запрашиваются для текущего аккаунта. Несуществующий
slug не должен молча подменяться другой моделью. Значения default context, maximum
configurable context и effective input budget не взаимозаменяемы.

## Безопасная миграция

Старую установку и её credentials следует сохранить. Новый профиль проверяется
на отдельном порту; действующий сервер не заменяется просто ради установки пакета.
Перед продолжением старого чата отдельно проверяется перенос encrypted reasoning:
совпадение имени движка само по себе не доказывает переносимость opaque items.

Минимальная проверка — короткий синтетический tool-roundtrip и продолжение в обе
стороны. Для определения окна не требуются промпты на сотни тысяч токенов. Если
каталог не сообщает достоверный предел, большой лимит не следует выдумывать.

## Первичные источники

- [Официальный sign-in contract](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [Каталог и inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
- [Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
- [Responses WebSocket](https://developers.openai.com/api/docs/guides/websocket-mode)
- [Аккаунты, refresh и хранение](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions)
- [OIDC discovery](https://auth.openai.com/.well-known/openid-configuration)
