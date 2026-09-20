"""工具层

**这个包刻意不提供 ``get_all_tools()`` 这类"把工具交给模型"的入口。**

"受控"的第一条要求是：**能改变审核状态的入口绝不能落到 LLM 手里**。
本项目根本不给模型任何工具 —— 审核走固定管线，模型只负责抽取和叙述。

那条约束由结构性断言守着，见
``tests/test_finance.py::test_inv2_no_registered_tool_can_change_audit_state``：
它直接检查整个 finance 包里只有 ``audit.py`` 会改写审核单状态。
"""

from .file_ops import resolve_data_path

__all__ = ["resolve_data_path"]
