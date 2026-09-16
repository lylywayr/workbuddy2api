package upstream

import (
	"encoding/json"
	"fmt"
	"strings"
)

// continuationInstruction 是追加轮次的控制消息。
// 它不改变用户原始消息，只告诉模型从已发出的 assistant 文本之后继续，
// 防止模型把整段答案重新生成一遍。
const continuationInstruction = "请紧接上一段回答继续，不要重复已经输出的内容。"

// BuildContinuationBody 根据上一轮已输出的纯文本 assistant 内容构造追加请求体。
// 续接请求沿用原请求的模型、参数、会话元数据；调用方再经 ChatStreamContext
// 出站，因此 stream、usage、缓存键和能力归一化保持正常链路。
func BuildContinuationBody(src []byte, turn StreamOutcome) ([]byte, error) {
	if !turn.canContinue() {
		return nil, fmt.Errorf("continuation is not safe: %s", turn.refuseReason())
	}
	content := turn.Content
	if strings.TrimSpace(content) == "" {
		return nil, fmt.Errorf("continuation requires non-empty assistant content")
	}
	var obj map[string]any
	if err := json.Unmarshal(src, &obj); err != nil {
		return nil, fmt.Errorf("decode continuation request: %w", err)
	}
	msgs, ok := obj["messages"].([]any)
	if !ok {
		return nil, fmt.Errorf("continuation request has no messages array")
	}
	assistant := map[string]any{"role": "assistant", "content": content}
	if strings.TrimSpace(turn.Reasoning) != "" {
		assistant["reasoning_content"] = turn.Reasoning
	}
	msgs = append(msgs, assistant, map[string]any{
		"role":    "user",
		"content": continuationInstruction,
	})
	obj["messages"] = msgs
	obj["stream"] = true
	out, err := json.Marshal(obj)
	if err != nil {
		return nil, fmt.Errorf("encode continuation request: %w", err)
	}
	return out, nil
}
