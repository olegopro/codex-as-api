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
регистрацией. Не следует копировать client ID из другого приложения или переносить
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
- Генерация: POST `/v1/responses`, `store:false`, `stream:true`, полный `input`.
- Инструкции передаются как `instructions`/developer, не system-role input items.
- Function/custom tools приводятся к разрешённому preview формату; вызовы и результаты
  сохраняют имена и call IDs на стороне Anthropic-клиента.
- Успех требует `response.completed`: EOF, failed/incomplete и потоковая ошибка не
  означают успешного завершения и не завершают недописанный tool-call.
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
- [Аккаунты, refresh и хранение](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions)
- [OIDC discovery](https://auth.openai.com/.well-known/openid-configuration)
