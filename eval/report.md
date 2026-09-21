# 报销审核规则 —— 评测报告

> 由 `scripts/run_eval.py` 自动生成。完全离线：pypdf 文本层 + 规则引擎，
> 不调视觉模型、不调 LLM。同一批样本跑一百遍结果一致。

## 指标

| 指标 | 数值 | 说明 |
|---|---|---|
| 端到端完成率 | 100.0% | 13/13 张样本跑完无异常 |
| 规则召回率 | 100.0% | 预期命中的 12 个**期望实例**中，实际命中 12 个（覆盖 11 条不同规则） |
| 误报数 | 0 | 未预期命中却命中的规则条数 |
| 字段抽取正确率 | 100.0% | 13/13 张样本抽取字段与样本定义一致 |

## 逐样本明细

| 样本 | 预期命中 | 实际非通过项 | 系统建议 | 判定 |
|---|---|---|---|---|
| S01_hotel_ok | （全通过） | （全通过） | APPROVED | ✅ |
| S02_hotel_over_limit | R007 | R007=FAIL | REJECTED | ✅ |
| S03_overdue | R003 | R003=FAIL | REJECTED | ✅ |
| S04_wrong_buyer | R001/R002 | R001=FAIL、R002=FAIL | REJECTED | ✅ |
| S05_transport_over_limit | R005 | R005=FAIL | REJECTED | ✅ |
| S06_office_no_list | R008 | R008=WARN | PENDING | ✅ |
| S07_serial_1 | （全通过） | （全通过） | APPROVED | ✅ |
| S07_serial_2 | R012 | R012=WARN | PENDING | ✅ |
| S07_serial_3 | R012 | R012=WARN | PENDING | ✅ |
| S08_meal_no_headcount | R006 | R006=WARN | PENDING | ✅ |
| S09_prompt_injection | R015 | R015=WARN | PENDING | ✅ |
| S10_words_mismatch | R016 | R016=FAIL | REJECTED | ✅ |
| S11_wrong_vat_rate | R017 | R017=FAIL | REJECTED | ✅ |

## 样本说明

- `S01_hotel_ok` —— 合规住宿（上海 3 晚 × 550）
- `S02_hotel_over_limit` —— 住宿超标（上海 3 晚 × 800）
- `S03_overdue` —— 超期发票（开票日 90 天前）
- `S04_wrong_buyer` —— 抬头为个人（非公司全称）
  - 抽取提示：S04 类样本：抬头本就不是公司全称，跳过核对
- `S05_transport_over_limit` —— 市内交通单次超标（380 元）
- `S06_office_no_list` —— 办公用品超 2000 元且未附清单
- `S07_serial_1` —— 连号发票第 1 张（疑似拆单）
- `S07_serial_2` —— 连号发票第 2 张（疑似拆单）
- `S07_serial_3` —— 连号发票第 3 张（疑似拆单）
- `S08_meal_no_headcount` —— 餐饮费未注明用餐人数
- `S09_prompt_injection` —— 备注栏含提示注入指令
- `S10_words_mismatch` —— 价税合计大小写不一致（疑似篡改）
- `S11_wrong_vat_rate` —— 税率与项目不符（住宿服务写成 13%）
