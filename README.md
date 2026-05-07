# Sepsis-Agent 代码提交说明

本目录为论文《基于智能体的脓毒症诊断与治疗建议研究》的核心代码提交包，包含智能体主流程、诊断与治疗评测、自动打分和消融实验代码。提交包不包含真实 API key、原始数据集、模型权重、运行缓存和画图脚本。 
值得注意的是，imaging.py模块因为效果不好，所被我关闭，改用了附带的影像学检查报告作为输入。
## 目录结构

```text
submit/
├── Agent/                                  # 智能体核心模块
│   ├── agent.py                            # 主流程：诊断、治疗 ReAct、治疗定稿、反思修订
│   ├── tools.py                            # Sepsis-3 规则引擎、SOFA 工具、代码工具
│   ├── utils.py                            # 病例清洗、字段屏蔽、上下文构建
│   ├── summary.py                          # 病例/工具/RAG/影像证据摘要
│   ├── rag.py                              # 本地 PDF/DOCX 知识库检索
│   ├── imaging.py                          # 影像文本和 DICOM 证据处理（这个模块因为效果不好没有被激活）
│   ├── modeling.py                         # 模型后端封装
│   ├── config.py                           # 运行配置
│   ├── llm_presets.py                      # DeepSeek、Qwen、Baichuan、AntAngel 预设
│   ├── cli.py                              # 单病例命令行入口
│   └── ablation.py                         # 消融配置
├── test/
│   ├── evaluate_agent_on_testset.py         # 智能体诊断评测
│   ├── evaluate_direct_llm_baseline.py      # 直接 LLM 诊断基线
│   ├── evaluate_direct_treatment_baseline.py# 直接治疗建议基线
│   ├── run_diagnosis_ablation_matrix.py     # 诊断消融矩阵实验
│   ├── run_curetest_treatment_matrix.py     # 治疗建议矩阵实验
│   ├── run_minitest_llm_matrix.py           # 多模型小规模诊断矩阵实验
│   └── score_curetest_treatment_with_qwen.py# 治疗建议自动评分
├── miniTest/curetest_treatment_matrix/
│   ├── run_deepseek_tool_ablation_20260430.py
│   └── score_deepseek_tool_ablation_20260430.py
├── evaluate_sofa_baseline.py                # SOFA/传统规则基线
└── requirements.txt                         # Python 依赖
```

## 环境准备

建议使用 Python 3.10 及以上版本。

```powershell
cd your_dir
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如使用在线模型，需要在运行前设置环境变量。请不要把真实密钥写入代码文件。

```powershell
$env:DASHSCOPE_API_KEY = "<your-dashscope-api-key>"
$env:DEEPSEEK_API_KEY = "<your-deepseek-api-key>"
$env:BAICHUAN_API_KEY = "<your-baichuan-api-key>"
$env:ANTANGEL_API_KEY = "<your-antangel-api-key>"

# 治疗建议自动评分器使用 OpenAI-compatible 接口时：
$env:TREATMENT_JUDGE_API_KEY = "<your-judge-api-key>"
$env:TREATMENT_JUDGE_BASE_URL = "<your-openai-compatible-base-url>"
```

也可以使用 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 作为治疗评分器的后备环境变量。

## 单病例运行

从 `submit` 目录运行：

```powershell
python -m Agent.cli `
  --case-json "<case.json>" `
  --task-mode diagnosis `
  --base-text-preset deepseek `
  --output-json ".runtime\diagnosis_result.json"
```

治疗建议任务：

```powershell
python -m Agent.cli `
  --case-json "<case.json>" `
  --task-mode treatment `
  --base-text-preset deepseek `
  --enable-treatment-round-reminders `
  --output-json ".runtime\treatment_result.json"
```

同时输出诊断与治疗建议：

```powershell
python -m Agent.cli `
  --case-json "<case.json>" `
  --task-mode diagnosis_and_treatment `
  --base-text-preset deepseek `
  --output-json ".runtime\full_result.json"
```

可选模型预设包括 `deepseek`、`qwen_plus`、`qwen_max`；治疗定稿医学模型可使用 `baichuan` 或 `antangel`。

## 诊断实验入口

智能体诊断评测：

```powershell
python test\evaluate_agent_on_testset.py --help
```

直接 LLM 诊断基线：

```powershell
python test\evaluate_direct_llm_baseline.py --help
```

SOFA/传统规则基线：

```powershell
python evaluate_sofa_baseline.py --help
```

诊断消融矩阵实验：

```powershell
python test\run_diagnosis_ablation_matrix.py --help
```

## 治疗实验入口

治疗建议智能体与基线矩阵实验：

```powershell
python test\run_curetest_treatment_matrix.py --help
```

直接治疗建议基线：

```powershell
python test\evaluate_direct_treatment_baseline.py --help
```

治疗工具消融：

```powershell
python miniTest\curetest_treatment_matrix\run_deepseek_tool_ablation_20260430.py
```

## 治疗建议自动评分

治疗建议评分脚本按照五个维度打分：临床合理性、安全性、证据依据性、完整性、可执行性。

```powershell
python test\score_curetest_treatment_with_qwen.py --help
```

评分脚本只从环境变量读取密钥，不会从示例 Python 文件中读取或保存 API key。

<!-- ## 数据与知识库说明

本提交包只包含代码。复现实验时需要另外提供：

- 病例 CSV 或 JSON 数据。
- 如需启用本地 RAG，请将 PDF/DOCX 指南或文献放入 `Agent/rag/`。
- 如需使用本地 MedGemma 或 LoRA 模型，请按 `Agent/config.py` 中的路径配置模型权重。
- API key 通过环境变量提供。

如果没有本地 RAG 文档，可以在运行时加 `--disable-rag`；如果没有影像数据或影像模型，可以加 `--disable-imaging`。 -->

## 备注

- 本目录保留的是论文实验复现所需的核心代码和实验脚本。
- 画图脚本未包含。
- 由于涉及患者隐私，实验数据未包含。

