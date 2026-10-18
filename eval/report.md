# 报销审核规则 —— 评测报告

> 由 `scripts/run_eval.py` 自动生成。完全离线：pypdf 文本层 + 规则引擎，
> 不调视觉模型、不调 LLM。同一批样本跑一百遍结果一致。

## 指标

| 指标 | 数值 | 说明 |
|---|---|---|
| 端到端完成率 | 100.0% | 27/27 张样本跑完无异常 |
| 规则召回率 | 100.0% | 预期命中的 25 个**期望实例**中，实际命中 25 个（覆盖 17 条不同规则） |
| 误报数 | 0 | 未预期命中却命中的规则条数（**计入失败**） |
| 严重度一致 | 0 | 命中但严重度与预期不符的条数（FAIL 与 WARN 走向完全不同的结论） |
| 字段抽取正确率 | 100.0% | 27/27 张跑完的样本抽取字段与样本清单一致 |

## 逐样本明细

| 样本 | 预期命中 | 实际非通过项 | 系统建议 | 判定 |
|---|---|---|---|---|
| S01_hotel_ok | （全通过） | （全通过） | APPROVED | ✅ |
| S02_hotel_over_limit | R007=FAIL | R007=FAIL | REJECTED | ✅ |
| S03_overdue | R003=FAIL | R003=FAIL | REJECTED | ✅ |
| S04_wrong_buyer | R001=FAIL、R002=FAIL | R001=FAIL、R002=FAIL | REJECTED | ✅ |
| S05_transport_over_limit | R005=FAIL | R005=FAIL | REJECTED | ✅ |
| S06_office_no_list | R008=WARN | R008=WARN | PENDING | ✅ |
| S07_serial_1 | （全通过） | （全通过） | APPROVED | ✅ |
| S07_serial_2 | R012=WARN | R012=WARN | PENDING | ✅ |
| S07_serial_3 | R012=WARN | R012=WARN | PENDING | ✅ |
| S08_meal_no_headcount | R006=WARN | R006=WARN | PENDING | ✅ |
| S09_prompt_injection | R015=WARN | R015=WARN | PENDING | ✅ |
| S10_words_mismatch | R016=FAIL | R016=FAIL | REJECTED | ✅ |
| S11_wrong_vat_rate | R017=FAIL | R017=FAIL | REJECTED | ✅ |
| S15_hotel_limit_edge | （全通过） | （全通过） | APPROVED | ✅ |
| S16_hotel_limit_plus_one | R007=FAIL | R007=FAIL | REJECTED | ✅ |
| S17_overdue_and_over_limit | R003=FAIL、R007=FAIL | R003=FAIL、R007=FAIL | REJECTED | ✅ |
| S18_injection_english | R015=WARN | R015=WARN | PENDING | ✅ |
| S19_buyer_fullwidth_space | （全通过） | （全通过） | APPROVED | ✅ |
| S20_meal_per_head_over | R006=FAIL | R006=FAIL | REJECTED | ✅ |
| S21_role_marker_injection | R015=WARN | R015=WARN | PENDING | ✅ |
| S22_amount_mismatch | R010=FAIL | R010=FAIL | REJECTED | ✅ |
| S23_account_unmapped | R011=WARN、R017=WARN | R011=WARN、R017=WARN | PENDING | ✅ |
| S24_voucher_imbalance | R013=FAIL | R013=FAIL | REJECTED | ✅ |
| S25_dept_not_in_budget | R014=WARN | R014=WARN | PENDING | ✅ |
| S12_train_ticket | （全通过） | （全通过） | APPROVED | ✅ |
| S13_foreign_invoice | R019=WARN | R019=WARN | PENDING | ✅ |
| S14_fixed_voucher | R009=FAIL | R009=FAIL | REJECTED | ✅ |

## 样本说明

- `S01_hotel_ok` —— 合规住宿（上海 3 晚 × 550）
- `S02_hotel_over_limit` —— 住宿超标（上海 3 晚 × 800）
- `S03_overdue` —— 超期发票（开票日 90 天前）
- `S04_wrong_buyer` —— 抬头为个人（非公司全称）
- `S05_transport_over_limit` —— 市内交通单次超标（380 元）
- `S06_office_no_list` —— 办公用品超 2000 元且未附清单
- `S07_serial_1` —— 连号发票第 1 张（疑似拆单）
- `S07_serial_2` —— 连号发票第 2 张（疑似拆单）
- `S07_serial_3` —— 连号发票第 3 张（疑似拆单）
- `S08_meal_no_headcount` —— 餐饮费未注明用餐人数
- `S09_prompt_injection` —— 备注栏含提示注入指令
- `S10_words_mismatch` —— 价税合计大小写不一致（疑似篡改）
- `S11_wrong_vat_rate` —— 税率与项目不符（住宿服务写成 13%）
- `S15_hotel_limit_edge` —— 住宿正好压线（上海 600/晚）
- `S16_hotel_limit_plus_one` —— 住宿超限一分钱（上海 601/晚）
- `S17_overdue_and_over_limit` —— 超期且住宿超标（两条 FAIL 叠加）
- `S18_injection_english` —— 备注栏英文提示注入
- `S19_buyer_fullwidth_space` —— 抬头带全角空格（应归一化后通过）
- `S20_meal_per_head_over` —— 餐饮人均超标（700 元 / 4 人 = 175）
- `S21_role_marker_injection` —— 备注栏伪造对话角色标记
- `S22_amount_mismatch` —— 申请金额与价税合计不一致（申请 500 / 票面 550）
- `S23_account_unmapped` —— 费用类型与票面项目都无法归类（团建费 + 娱乐服务）
- `S24_voucher_imbalance` —— 票面三要素不勾稽（500 + 30 ≠ 550）
- `S25_dept_not_in_budget` —— 部门不在预算表内（无法校验预算）
- `S12_train_ticket` —— 铁路电子客票（交通票据，票种路由不误杀）
- `S13_foreign_invoice` —— 境外 INVOICE（汇率人工核验）
- `S14_fixed_voucher` —— 定额发票（制度 3.4 明文不接受）
