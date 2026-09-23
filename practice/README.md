# 自定义实践区

初始状态下，`memory.jsonl` 有 3 条良性记录，`fusion_instruction.txt` 提供了一条按时间顺序合并这些记录的良性示例。可以先直接运行，再把它替换为自己的融合指令。

## 操作步骤

1. 按 JSONL 格式向 `memory.jsonl` 追加自己的合成测试碎片。

   ```json
   {"kind":"memory","ts":4,"prompt":"<你的测试片段>","response":"Stored memory record.","meta":{"id":"record-0004"}}
   ```

2. 在 `fusion_instruction.txt` 中填写一条融合指令。

3. 在 `DEMO` 目录运行传统 Baseline：

   ```cmd
   python run_demo.py --practice --baseline llm_ac --mode composition
   ```

4. 再运行 V2：

   ```cmd
   python run_demo.py --practice --baseline v2 --mode composition
   ```

也可以替换为 `agrail`、`guardagent` 或 `shieldagent`。

