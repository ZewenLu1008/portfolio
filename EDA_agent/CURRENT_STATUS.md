# 项目进度状态 - Current Status

**最后更新**: 2026-07-29  
**项目名称**: Adaptive Data Cleaning & QA Agent with EDA  
**当前阶段**: Stage 2 已完成 ✅

---

## 一、项目概览

### 核心技术栈
- **LangGraph**: Multi-Agent 工作流编排
- **LangChain**: LLM 接口与 Prompt 管理
- **Pandas**: 数据清洗与处理
- **Matplotlib + Seaborn**: 数据可视化
- **DeepSeek API**: LLM 模型（代替 OpenAI）

### 工作流架构

```
START → Profiler → Coder → Executor → [Self-Correction Loop]
                              ↑            ↓
                              └────────────┘ (retry < 3)
                                           ↓
                                          QA → [QA Passed?]
                                           ↓         ↓
                                          EDA       END
                                           ↓
                                          END
```

---

## 二、Stage 1：数据清洗与 Self-Correction ✅

### 已完成功能

#### 1. **Profiler Node（数据质量诊断）**
- **文件**: `src/nodes/profiler.py`
- **职责**: 扮演"资深数据分析师"，分析数据质量问题并制定清洗策略
- **输入**: `original_df_info`（原始数据元信息）
- **输出**: `cleaning_plan`（Markdown 格式的清洗策略）
- **模型**: DeepSeek-chat, temperature=0.1

#### 2. **Coder Node（代码生成）**
- **文件**: `src/nodes/coder.py`
- **职责**: 扮演"Python 数据工程师"，将清洗策略翻译为可执行代码
- **输入**: `cleaning_plan`, `original_df_info`, `execution_error`（重试场景）
- **输出**: `generated_code`（包含 `clean_data(df)` 函数）
- **模型**: DeepSeek-coder, temperature=0.0
- **核心机制**: Self-Correction（基于 `execution_error` 重新生成代码）

#### 3. **Executor Node（代码执行）**
- **文件**: `src/nodes/executor.py`
- **职责**: 在沙箱环境中执行清洗代码，捕获错误信息
- **输入**: `generated_code`, 原始数据文件
- **输出**: `cleaned_df_info`, `execution_success`, `execution_error`
- **安全措施**: 独立命名空间、白名单机制、禁止文件 I/O

#### 4. **QA Node（质量检验）**
- **文件**: `src/nodes/qa.py`
- **职责**: 扮演"数据质量工程师"，验证清洗后数据质量
- **输入**: `original_df_info`, `cleaned_df_info`
- **输出**: `qa_result`（包含 passed, score, reason, issues, suggestions）
- **模型**: DeepSeek-chat, temperature=0.0

#### 5. **Self-Correction 机制**
- **文件**: `src/core/graph.py`（路由函数 `route_after_executor`）
- **逻辑**:
  - 执行失败 + retry_count < 3 → 回到 Coder Node
  - 执行失败 + retry_count >= 3 → 强制结束
  - 执行成功 → 进入 QA Node

### 核心设计亮点

1. **职责分离**: Profiler（诊断）+ Coder（编码）
2. **闭环纠错**: 通过 `execution_error` 形成 Self-Correction Loop
3. **沙箱执行**: 强制 `clean_data(df)` 函数签名，独立命名空间
4. **成本优化**: 分层模型选择（Profiler 用小模型，Coder 用专用模型）

---

## 三、Stage 2：EDA 自动化 ✅

### 已完成功能

#### 1. **EDA Node（探索性数据分析）**
- **文件**: `src/nodes/eda.py`
- **职责**: 扮演"数据洞察专家"，自动生成可视化代码和业务洞察
- **输入**: `cleaned_df_info`, `qa_result`
- **输出**: `eda_plan`（英文洞察）, `eda_code`（画图代码）, `eda_error`
- **模型**: DeepSeek-chat, temperature=0.2

#### 2. **生成的 3 张核心图表**
- **数值分布图**: 观察数值列的分布形态、异常值
- **分类柱状图**: 观察类别变量的分布、Top N 类别
- **相关性热力图**: 观察数值列之间的相关性

#### 3. **中文乱码问题的解决方案**
- **策略**: 在 Prompt 中强制要求 LLM 生成的代码在绘图前进行"数据英文化"
- **实现**: 使用 `.rename(columns={...})` 翻译列名，使用 `.map()` 翻译类别值
- **优势**: 无需安装中文字体，适用于任何环境，图表可国际化复用

#### 4. **路由扩展**
- **文件**: `src/core/graph.py`（新增 `route_after_qa` 函数）
- **逻辑**:
  - QA 通过 → 流向 EDA Node
  - QA 未通过 → 直接 END（数据质量不达标，不进行 EDA）

#### 5. **报告持久化存储**
- **文件**: `scripts/run_agent.py`（修改 `print_final_report` 函数）
- **功能**: 将完整执行报告保存到 `outputs/final_report.md`（UTF-8 编码）
- **报告内容**: 执行状态、数据清洗对比、QA 结果、完整的英文 Insights

### 核心设计亮点

1. **优雅的中文字体问题解决**: 数据英文化而非字体配置
2. **结构化输出解析**: 使用 `<CODE>` 和 `<INSIGHTS>` 标记分隔
3. **100% 纯英文输出**: 代码和洞察均使用英文，避免乱码
4. **报告精简**: 只保留业务价值内容（不包含代码和文件路径）

---

## 四、项目文件结构

### 核心模块

```
src/
├── core/
│   ├── state.py           # 全局状态定义（TypedDict）
│   └── graph.py           # LangGraph 工作流构建与路由逻辑
├── nodes/
│   ├── profiler.py        # 数据质量诊断节点
│   ├── coder.py           # 代码生成节点
│   ├── executor.py        # 代码执行节点
│   ├── qa.py              # 质量检验节点
│   └── eda.py             # EDA 分析节点
└── utils/
    └── data_loader.py     # 数据加载与元信息提取工具
```

### 脚本与配置

```
scripts/
├── run_agent.py           # 主运行脚本（含 Final Report 生成）
└── generate_dirty_data.py # 测试数据生成脚本
```

### 输出目录

```
outputs/
├── cleaned_data.csv       # 清洗后的数据
├── final_report.md        # 完整执行报告（精简版）
└── plots/                 # EDA 生成的 3 张图表（PNG 格式）
    ├── plot_1_*.png
    ├── plot_2_*.png
    └── plot_3_*.png
```

### 文档

```
docs/
├── interview_prep_stage1.md  # Stage 1 面试准备文档
└── interview_prep_stage2.md  # Stage 2 面试准备文档
```

### 配置文件

```
.env.example               # 环境变量示例（API Key 配置）
pyproject.toml             # 项目依赖管理（uv）
README.md                  # 项目说明文档
CURRENT_STATUS.md          # 当前进度状态（本文件）
```

---

## 五、状态定义（State Schema）

### DataCleaningState 字段列表

| 字段 | 类型 | 说明 | 生产者 | 消费者 |
|-----|------|------|--------|--------|
| `original_df_info` | Dict | 原始数据元信息 | 外部输入 | Profiler, Coder |
| `cleaning_plan` | str | 清洗策略 | Profiler | Coder |
| `generated_code` | str | 清洗代码 | Coder | Executor |
| `execution_error` | str | 执行错误信息 | Executor | Coder (Self-Correction) |
| `execution_success` | bool | 执行成功标志 | Executor | 路由函数 |
| `retry_count` | int | 重试次数 | Executor | 路由函数, Coder |
| `cleaned_df_info` | Dict | 清洗后数据元信息 | Executor | QA, EDA |
| `qa_result` | Dict | 质检结果 | QA | 路由函数 |
| `eda_plan` | str | EDA 洞察（英文） | EDA | Final Report |
| `eda_code` | str | EDA 画图代码 | EDA | - |
| `eda_error` | str | EDA 执行错误 | EDA | Final Report |
| `input_file_path` | str | 输入文件路径 | 外部输入 | Executor |
| `output_file_path` | str | 输出文件路径 | 外部输入 | Executor, EDA |
| `execution_history` | List | 执行历史记录 | Executor | 调试用 |

---

## 六、运行方式

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API Key（创建 .env 文件）
DEEPSEEK_API_KEY=sk-your-key-here
```

### 2. 生成测试数据（可选）

```bash
python scripts/generate_dirty_data.py
```

### 3. 运行 Agent

```bash
python scripts/run_agent.py
```

### 4. 查看结果

- **清洗后的数据**: `outputs/cleaned_data.csv`
- **执行报告**: `outputs/final_report.md`
- **可视化图表**: `outputs/plots/*.png`

---

## 七、当前限制与未来改进

### 当前限制

1. **EDA 无 Self-Correction**: 如果 EDA 代码执行失败，不会重试（基于成本考虑）
2. **图表数量固定**: 只生成 3 张预设的图表（分布、分类、相关性）
3. **Profiler 策略不可重试**: 如果 Profiler 的策略本身有问题，Coder 重试也无法解决

### 未来改进方向

1. **EDA Self-Correction**: 参考 Coder Node 的设计，为 EDA 添加重试机制
2. **动态图表生成**: 根据数据特征自动选择合适的图表类型
3. **Profiler + Coder 联合重试**: QA 未通过时，让 Profiler 重新诊断
4. **交互式可视化**: 引入 Plotly、Bokeh 等交互式图表库
5. **特征工程建议**: 基于 EDA 结果，自动生成特征工程代码
6. **代码安全加固**: 使用 Docker 容器或 E2B 沙箱服务执行代码

---

## 八、关键技术决策记录

### 1. 为什么使用 DeepSeek 而不是 OpenAI？
- **成本**: DeepSeek 的价格约为 GPT-4 的 1/10
- **性能**: DeepSeek-coder 在代码生成任务上表现优秀
- **兼容性**: 兼容 OpenAI API 格式，可无缝切换

### 2. 为什么强制 `clean_data(df)` 函数签名？
- **沙箱执行安全**: 独立命名空间，避免污染全局环境
- **输入输出标准化**: 便于后续节点统一处理
- **避免文件 I/O**: 纯函数式转换，降低风险

### 3. 为什么 EDA 使用"数据英文化"而非"中文字体配置"？
- **环境无关性**: 不同操作系统的字体路径不同
- **国际化友好**: 英文图表可直接用于国际化报告
- **代码简洁性**: Pandas 的 `.rename()` 和 `.map()` 是通用操作

### 4. 为什么报告不包含生成的代码？
- **聚焦业务价值**: 报告的目标读者是业务方，而非技术团队
- **减少噪音**: 代码细节会分散对洞察的注意力
- **可追溯性**: 代码仍保存在 State 中，需要时可查看

---

## 九、快速命令参考

### 查看工作流图

```bash
# 运行 Agent 时自动生成
# 输出路径: docs/graph.mmd
# 可在 https://mermaid.live/ 中查看
```

### 调试单个节点

```python
from src.core.state import StateFactory
from src.nodes.profiler import profiler_node

state = StateFactory.create_initial_state("data/dirty_data.csv")
state["original_df_info"] = {...}  # 手动填充数据
result = profiler_node(state)
print(result["cleaning_plan"])
```

### 清理输出文件

```bash
rm -rf outputs/*
```

---

## 十、联系方式与资源

- **项目文档**: `docs/interview_prep_stage1.md`, `docs/interview_prep_stage2.md`
- **LangGraph 文档**: https://langchain-ai.github.io/langgraph/
- **DeepSeek API**: https://platform.deepseek.com/

---

**状态**: ✅ Stage 1 和 Stage 2 均已完成  
**下一步**: 可考虑添加 Stage 3（高级功能扩展）或进行性能优化  
**更新时间**: 2026-07-29
