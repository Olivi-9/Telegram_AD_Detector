# Telegram AD Detector Bot

基于 Pyrogram + 规则引擎 + 机器学习 的 Telegram 群广告检测与自动处置机器人。

项目目标：在尽量降低误杀的前提下，对新成员高风险行为（广告文案、引流简介、可疑内容类型）进行自动检测、分级告警和处置。

广告识别能力不仅限于 Telegram, 可以拓展使用于其他社交平台的广告检测场景（如评论区 弹幕等），核心逻辑和模型可复用，适配层负责不同平台的消息解析与动作执行。

本项目仅作为了解机器学习并作出实战的落地应用，实际使用请修改代码以适配自己的需求，并做好充分测试。

## 1. 核心能力

- 多源检测：规则检测 + ML 文本分类联合判定。
- 风险分级：按风险分数映射 LOW / MEDIUM / HIGH。
- 自动处置：支持 NO_ACTION / MONITOR / WARN / KICK。
- 新成员保护策略：重点检测新成员，降低对老成员干扰。
- 用户白名单：支持管理员命令动态添加/移除白名单用户。
- 持久化状态：记录用户消息计数、违规计数、白名单状态。

## 2. 技术架构

### 2.1 模块划分

- `ad_bot.py`：Bot 主入口，负责事件监听、消息处理、执行处置动作、管理员命令。
- `detection_engine.py`：检测核心，负责消息解析、规则引擎、ML 调用、风险评分与动作建议。
- `user_state.py`：用户状态管理与持久化（`bot_user_states.json`）。
- `bot_config.py`：配置中心，支持 `.env` 与环境变量读取、配置校验。
- `main.py`：训练脚本，输出可部署模型工件。
- `predict.py`：推理脚本（单条/批量/交互模式）。

### 2.2 处理流程（线上）

1. `ad_bot.py` 接收群消息（新消息/编辑消息/新成员加入事件）。
2. 过滤无需处理的消息：bot 消息、匿名管理员消息、命令消息、重复消息。
3. 加载用户状态并更新消息计数；必要时校准真实入群时间。
4. 调用 `DetectionEngine.analyze(...)` 完成检测决策。
5. 根据建议动作执行删除、警告、踢人等操作。
6. 状态持久化并输出日志；可选调试报告回帖。

## 3. 检测引擎实现细节

### 3.1 MessageParser（消息解析）

`detection_engine.py` 中 `MessageParser.parse()` 会统一解析以下信息：

- 消息类型：`TEXT / PHOTO / VIDEO / DOCUMENT / CONTACT / STICKER / VOICE / OTHER`
- 文本字段：`text` 或 `caption`
- 结构特征：是否包含链接、是否包含 `@`、是否为回复、是否疑似外部回复

### 3.2 RuleEngine（规则检测）

规则命中项（默认只对新成员更严格）：

- 外部引用消息
- 联系人卡片消息
- 新成员图片消息
- 新成员贴纸消息
- 新成员简介包含 `t.me`

规则结果聚合为 `RuleResult`，并提供 `any_hit()` 与 `hit_count()` 供后续决策使用。

### 3.3 MLDetector（模型检测）

仅对文本消息调用 ML：

- 先做文本清洗（NFKC 归一化、控制字符/代理项过滤）
- 若出现编码异常，触发更激进清洗并重试
- 推理入口：`predict.py` 的 `predict_text()`

模型工件默认读取：

- `ad_model.pkl`
- `tfidf_vectorizer.pkl`
- `embedding_model.pkl`
- `emb_weight.pkl`

### 3.4 风险评分与分级

系统会将规则信号与 ML 信号叠加为风险分数，最终截断到 `[0, 1]`。

主要加分逻辑（当前实现）：

- `external_reply_hit`: `+0.85`
- `contact_message_hit`: `+0.85`
- `image_message_hit`: `+0.25`
- `bio_link_hit`: `+0.25`
- ML 判定广告：`+min(confidence, 1.0)`（若无置信度则 `+0.5`）

风险等级阈值（`bot_config.py`）：

- `HIGH`：`risk_score >= 0.7`
- `MEDIUM`：`risk_score >= 0.5`
- `LOW`：其余

### 3.5 动作策略

- `LOW` 且无违规：`NO_ACTION`
- `LOW` 且有违规：`MONITOR`（可删消息，累计到阈值踢人）
- `MEDIUM`：`WARN`（可删消息，累计到阈值踢人）
- `HIGH`：`KICK`（可立即踢人）

管理员/群主默认受保护，不执行踢人动作。

## 4. 新成员判定逻辑

定义于 `user_state.py` 的 `UserState.is_new_member`：

- 若已获得可信入群时间（`join_time_verified=True`），按入群时长判定。
- 否则满足任一条件即判定为新成员：
        - 消息数 `<= NEW_MEMBER_MESSAGE_THRESHOLD`
        - 入群时间 `< NEW_MEMBER_TIME_THRESHOLD`

默认阈值：

- 消息数阈值：5
- 入群时长阈值：5 天

## 5. 模型训练实现（`main.py`）

训练流程：

1. 读取 `train.csv`（列：`label,text`）。
2. 构建双分支 TF-IDF：
         - Word TF-IDF（`max_features=6000`）
         - Char n-gram TF-IDF（`ngram_range=(2,8)`，`max_features=6000`）
3. 句向量编码：`SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")`
4. 在训练集内部搜索最佳 `EMB_WEIGHT`。
5. 特征融合：`X_final = hstack([X_tfidf, X_emb * EMB_WEIGHT])`
6. 训练分类器：`LinearSVC(class_weight="balanced")` + `GridSearchCV`
7. 导出模型工件到项目根目录

## 6. 快速开始（本地运行）

### 6.1 环境准备

建议 Python 3.10+，推荐使用 uv：

```bash
uv sync
```

### 6.2 配置环境变量

在项目根目录创建 `.env`：

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_API_ID=123456
TELEGRAM_API_HASH=your_api_hash
```

`bot_config.py` 会自动读取 `.env`（不会覆盖已存在的系统环境变量）。

### 6.3 训练模型（首次或重训时）

```bash
uv run telegram-ad-train
```

训练完成后应生成：

- `ad_model.pkl`
- `tfidf_vectorizer.pkl`
- `embedding_model.pkl`
- `emb_weight.pkl`

### 6.4 启动 Bot

```bash
uv run telegram-ad-bot
```

### 6.5 单机推理测试

```bash
uv run telegram-ad-predict
```

### 6.6 测试 Pytorch 是否可用

```bash
uv run python test_pytorch.py
```

### 6.7 代码测试

```bash
uv run python -m unittest discover -s tests -v
```

## 7. 推荐目录结构

- `telegram_ad_detector/`：`uv` 启动入口与后续可扩展的包代码
- `tests/`：基础烟雾测试
- `artifacts/`：后续建议放置训练产物、导出模型和临时结果
- 根目录脚本：保留现有实现，兼容旧的直接运行方式

## 8. 管理命令

常用群内命令：

- `/start`：查看运行模式摘要
- `/stats`：查看用户与违规统计
- `/config`：查看关键配置
- `/save`：手动保存状态
- `/get_group_id`：获取当前群 ID
- `/get_user_id`：获取自己或回复目标用户 ID
- `/ad_whitelist add <user_id>`：添加白名单用户（管理员）
- `/ad_whitelist remove <user_id>`：移除白名单用户（管理员）
- `/ad_whitelist list`：查看白名单（管理员）

## 9. 关键配置项（`bot_config.py`）

- 运行模式
        - `DRY_RUN`：只分析不执行动作
        - `DEBUG_REPLY`：回帖输出调试报告
        - `ONLY_NEW_MEMBER`：仅检测新成员
- 检测开关
        - `USE_ML_MODEL`
        - `CHECK_EXTERNAL_REPLY`
        - `CHECK_CONTACT_MESSAGE`
        - `CHECK_IMAGE_FROM_NEW_MEMBER`
        - `CHECK_STICKER_FROM_NEW_MEMBER`
- 风险阈值
        - `RISK_SCORE_HIGH`
        - `RISK_SCORE_MEDIUM`
- 处置阈值
        - `MONITOR_KICK_THRESHOLD`
        - `WARN_KICK_THRESHOLD`

## 10. 持久化与产物说明

- `bot_user_states.json`：用户状态（自动生成）
- `group_whitelist.json`：群白名单（自动生成，可手改）
- `bot_debug.log`：运行日志
- `*.pkl`：模型与特征工件


## 11. 常见问题

1. 启动报错缺少 API 配置

- 确认已设置 `TELEGRAM_API_ID` 与 `TELEGRAM_API_HASH`。

2. 启动报错缺少模型文件

- 先执行 `python main.py` 生成模型工件，或关闭 `USE_ML_MODEL`。

3. 误判较多

- 优先调整 `RISK_SCORE_*` 与 `*_KICK_THRESHOLD`。
- 扩充训练数据并重训模型。

## 12. 许可证

本项目使用仓库中 `LICENSE` 指定的许可证。