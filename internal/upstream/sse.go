// sse.go 处理上游 SSE 流：聚合成单个 OpenAI 响应，或透传给客户端。
package upstream

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// Aggregate 读取完整 SSE 流，聚合 delta.content 为单个 OpenAI chat.completion 响应。
// 分片/半行由 bufio.Reader.ReadString 处理；遇到 "data: [DONE]" 结束。
// tool_calls 以流式 delta 到达（按 index 合并：首片带 id/type/name，后续只带 arguments 片段）。
func Aggregate(r io.Reader) (map[string]any, error) {
	br := bufio.NewReaderSize(r, 64*1024)
	var (
		id, model     string
		created       float64
		content       strings.Builder
		reasoning     strings.Builder
		role          = "assistant"
		finishReason  = "stop"
		usage         map[string]any
		gotAnyContent bool
		validEvents   int
		toolCalls     = map[int]map[string]any{}
		toolOrder     []int
	)
	for {
		line, err := br.ReadString('\n')
		if err != nil && err != io.EOF {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if strings.HasPrefix(line, "data: ") {
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" {
				// 上游显式结束：停止读取，DONE 之后的任何数据一律忽略。
				break
			} else {
				var chunk map[string]any
				if json.Unmarshal([]byte(payload), &chunk) == nil {
					// 有效事件计数：仅 JSON 解析成功的数据帧计入（解析失败沿用静默 continue）。
					validEvents++
					if v, ok := chunk["id"].(string); ok && id == "" {
						id = v
					}
					if v, ok := chunk["model"].(string); ok && model == "" {
						model = v
					}
					if v, ok := chunk["created"].(float64); ok && created == 0 {
						created = v
					}
					if u, ok := chunk["usage"].(map[string]any); ok {
						usage = u
					}
					if ch, ok := chunk["choices"].([]any); ok {
						for _, ci := range ch {
							c, _ := ci.(map[string]any)
							if c == nil {
								continue
							}
							if fr, ok := c["finish_reason"].(string); ok && fr != "" {
								finishReason = fr
							}
							if delta, ok := c["delta"].(map[string]any); ok {
								if r2, ok := delta["role"].(string); ok && r2 != "" {
									role = r2
								}
								if txt, ok := delta["content"].(string); ok {
									content.WriteString(txt)
									gotAnyContent = true
								}
								if rc, ok := delta["reasoning_content"].(string); ok {
									reasoning.WriteString(rc)
								}
								if tcs, ok := delta["tool_calls"].([]any); ok {
									for _, tc := range tcs {
										call, ok := tc.(map[string]any)
										if !ok {
											continue
										}
										idx := 0
										if v, ok := call["index"].(float64); ok {
											idx = int(v)
										}
										merged, seen := toolCalls[idx]
										if !seen {
											merged = map[string]any{"index": idx}
											toolCalls[idx] = merged
											toolOrder = append(toolOrder, idx)
										}
										mergeToolCallDelta(merged, call)
									}
								}
							}
							// 有的上游把完整消息放在 message 里（非 delta）
							if msg, ok := c["message"].(map[string]any); ok && !gotAnyContent {
								if txt, ok := msg["content"].(string); ok {
									content.WriteString(txt)
								}
							}
						}
					}
				}
			}
		}
		if err == io.EOF {
			break
		}
	}
	if validEvents == 0 {
		// 上游返回 200 但没有任何有效数据事件（空流/只有 [DONE]/只有注释行）：
		// 不再合成空 content 的假成功响应，直接报错，由 handler 映射为 502 upstream_parse。
		return nil, fmt.Errorf("upstream stream contained no valid data events")
	}
	if id == "" {
		id = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	}
	if created == 0 {
		created = float64(time.Now().Unix())
	}
	message := map[string]any{
		"role":    role,
		"content": content.String(),
	}
	if reasoning.Len() > 0 {
		message["reasoning_content"] = reasoning.String()
	}
	if len(toolOrder) > 0 {
		sort.Ints(toolOrder)
		calls := make([]map[string]any, 0, len(toolOrder))
		for _, idx := range toolOrder {
			calls = append(calls, toolCalls[idx])
		}
		// P1b：finish_reason==length 且 tool_call 的 arguments 是残缺 JSON（解析失败）
		// 时不把脏参数交给客户端——残留分片会被客户端解析成非法 JSON 卡死会话。
		// 完整参数原样保留（正例零改动）；空参数（无参工具）不是截断，同样保留。
		if finishReason == "length" {
			calls = dropTruncatedToolCalls(calls)
		}
		if len(calls) > 0 {
			message["tool_calls"] = calls
		}
	}
	resp := map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": int64(created),
		"model":   model,
		"choices": []any{
			map[string]any{
				"index":         0,
				"message":       message,
				"finish_reason": finishReason,
			},
		},
	}
	if usage != nil {
		resp["usage"] = usage
	}
	return resp, nil
}

// mergeToolCallDelta 把流式 tool_call 片段合并到累计对象：
// id/type/function.name 直覆盖（后续分片通常缺省），function.arguments 拼接。
func mergeToolCallDelta(merged, delta map[string]any) {
	if v, ok := delta["id"].(string); ok && v != "" {
		merged["id"] = v
	}
	if v, ok := delta["type"].(string); ok && v != "" {
		merged["type"] = v
	}
	df, _ := delta["function"].(map[string]any)
	if df == nil {
		return
	}
	mf, _ := merged["function"].(map[string]any)
	if mf == nil {
		mf = map[string]any{}
		merged["function"] = mf
	}
	if v, ok := df["name"].(string); ok && v != "" {
		mf["name"] = v
	}
	if v, ok := df["arguments"].(string); ok && v != "" {
		if prev, _ := mf["arguments"].(string); prev != "" {
			mf["arguments"] = prev + v
		} else {
			mf["arguments"] = v
		}
	}
}

// stripToolCallNames 收敛流式 tool_calls 的 name 语义为「每个 index 只出现一次」：
// 首片保留 function.name，同一 index 后续分片里的 name 键一律删除（无论上游是
// 空串还是重复非空串）。这是 OpenAI 官方流的真实形态——首帧带 name，后续帧只带
// arguments 片段、不再出现 name 键——因此是累加型与覆盖型客户端的共同祖先行为。
//
// 两类消费模型在该形态下同时正确：
//   - 累加型（官方 WorkBuddy/CodeBuddy `name += tc_function?.name || ""`）：
//     后续分片 name 键缺失 → 追加空串，累积 name 保持唯一，不再拼成 Bash×帧数（issue #82）。
//   - 覆盖型（hawklithm#2 / Grok Build `name ?? state.name` 或 `if (name) state.name = name`）：
//     后续分片 name 键缺失 → 保留已建好的首帧 name，不被空串意外清空。
//     键缺失是比空串更安全的形态：`??` 与 truthy 守卫对缺失键必然保留旧值，
//     而对空串，`??` 会误判为重设并清空工具名。
//
// seen 记录每个 index 是否已发过首片（与 name 是否非空无关）；删除是幂等的。
// 只动 function.name 键，id/type/arguments 原样透传。
func stripToolCallNames(obj map[string]any, seen map[int]bool) {
	choices, _ := obj["choices"].([]any)
	for _, ci := range choices {
		c, _ := ci.(map[string]any)
		if c == nil {
			continue
		}
		delta, _ := c["delta"].(map[string]any)
		if delta == nil {
			continue
		}
		tcs, _ := delta["tool_calls"].([]any)
		for _, tci := range tcs {
			tc, _ := tci.(map[string]any)
			if tc == nil {
				continue
			}
			idx := 0
			if v, ok := tc["index"].(float64); ok {
				idx = int(v)
			}
			if seen[idx] {
				// 已发过首片：删除本分片的 name 键（存在即删，幂等）。
				if fn, _ := tc["function"].(map[string]any); fn != nil {
					delete(fn, "name")
				}
				continue
			}
			// 首现：保留 name 键原样（上游首片通常带非空 name；空 name 也照发，
			// 与 OpenAI 对「首帧无 name」的容忍一致），随后分片统一删除。
			seen[idx] = true
		}
	}
}

// normalizeFrame 以 OpenAI 流式规范白名单重建帧：仅保留标准字段，
// 剔除上游噪声（finish_reason:"" → null、空 content/refusal、空 tool_calls 列表、
// 空占位 function_call、顶层未知字段），空 delta 键一律省略，
// usage 缺失 → null，保证任意标准客户端按规范解析。
func normalizeFrame(obj map[string]any) map[string]any {
	out := map[string]any{}
	for _, k := range []string{"id", "object", "created", "model", "system_fingerprint", "service_tier"} {
		if v, ok := obj[k]; ok && v != nil {
			out[k] = v
		}
	}
	if _, ok := out["object"]; !ok {
		out["object"] = "chat.completion.chunk"
	}
	if _, ok := out["id"]; !ok {
		out["id"] = "chatcmpl-wb2api"
	}
	if chs, ok := obj["choices"].([]any); ok {
		nchs := make([]any, 0, len(chs))
		for _, ci := range chs {
			c, ok := ci.(map[string]any)
			if !ok {
				continue
			}
			nc := map[string]any{}
			if idx, ok := c["index"]; ok {
				nc["index"] = idx
			}
			delta := map[string]any{}
			if d, ok := c["delta"].(map[string]any); ok {
				if v, ok := d["role"].(string); ok && v != "" {
					delta["role"] = v
				}
				if v, ok := d["content"].(string); ok && v != "" {
					delta["content"] = v
				}
				if v, ok := d["reasoning_content"].(string); ok && v != "" {
					delta["reasoning_content"] = v
				}
				if v, ok := d["refusal"].(string); ok && v != "" {
					delta["refusal"] = v
				}
				if tcs, ok := d["tool_calls"].([]any); ok && len(tcs) > 0 {
					delta["tool_calls"] = tcs
				}
				if fc, ok := d["function_call"]; ok && fc != nil {
					// 空占位 function_call（name/arguments 全空）视为噪声剔除
					keep := false
					if fcm, ok2 := fc.(map[string]any); ok2 {
						n, _ := fcm["name"].(string)
						a, _ := fcm["arguments"].(string)
						keep = n != "" || a != ""
					} else {
						keep = true
					}
					if keep {
						delta["function_call"] = fc
					}
				}
			}
			nc["delta"] = delta
			if fr, ok := c["finish_reason"].(string); ok && fr != "" {
				nc["finish_reason"] = fr
			} else {
				nc["finish_reason"] = nil
			}
			nchs = append(nchs, nc)
		}
		out["choices"] = nchs
	}
	if u, ok := obj["usage"]; ok {
		out["usage"] = u
	} else {
		out["usage"] = nil
	}
	return out
}

// sseWriter 单条 SSE 响应的帧写出器：持有「一套 headers + 恰好一个 [DONE]」的
// 生命周期不变式。Stream / StreamAutoContinue 共用，保证多轮续接复用同一响应时
// 不会重复写 headers、也不会向客户端暴露中间轮次的 [DONE]。
type sseWriter struct {
	w  http.ResponseWriter
	fl http.Flusher

	headersSent bool

	// toolCallSeen 跨帧记录 delta.tool_calls 里已发过首片的 index，
	// 供逐 chunk 透传时收敛 name 为「每 index 一次」（对齐 OpenAI 官方流）。
	toolCallSeen map[int]bool

	// firstID 透传流的消息级 id 基准：缓存首个非空上游 id，后续帧缺失/空串时复用
	// （issue #35：同一条 SSE 消息所有帧共用一个真实 id，后台按 id 归并；此前中间帧
	// 一律补 chatcmpl-wb2api 哨兵，造成同流 id 分裂）。全流无真实 id → 才出现哨兵。
	firstID string

	// forceFirstID 用于自动续接：把后续上游轮次可能生成的新 id 统一收敛到首轮 id，
	// 保证客户端把多轮内容视为同一条 assistant 流。普通 Stream 保持既有语义，
	// 允许不同上游帧保留其自带的非空 id。
	forceFirstID bool

	validFrames int
	doneWritten bool
}

func newSSEWriter(w http.ResponseWriter) *sseWriter {
	fl, _ := w.(http.Flusher)
	return &sseWriter{w: w, fl: fl, toolCallSeen: map[int]bool{}}
}

// writeHeaders 首次写出前设置 SSE headers（幂等）。
func (s *sseWriter) writeHeaders() {
	if s.headersSent {
		return
	}
	s.headersSent = true
	h := s.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
}

// writeRaw 原样写出一帧（绕过 normalizeFrame）并 flush。上游 error 帧（error-passthrough）
// 与空流错误帧需保留 error 字段，不能被白名单剥掉，故经此写出。
func (s *sseWriter) writeRaw(payload string) error {
	s.writeHeaders()
	if _, err := io.WriteString(s.w, "data: "+payload+"\n\n"); err != nil {
		return err
	}
	if s.fl != nil {
		s.fl.Flush()
	}
	return nil
}

// writeFrame 把 payload 按规范白名单重建后以 data: 帧写出并 flush。
// 仅 JSON 解析成功时计数记为一次有效转发（JSON 解析失败照常降级原样写出，但不计数）。
func (s *sseWriter) writeFrame(payload string) (int, error) {
	var obj map[string]any
	valid := 0
	if json.Unmarshal([]byte(payload), &obj) == nil {
		// 上游错误帧透传（error-passthrough）：带 error 键的帧**原样写出**，不走
		// normalizeFrame 白名单——白名单会剥掉 error 字段，客户端就看不到上游
		// code/msg/requestId。error.message 即上游原文（如 6004 限流、审核拦截），
		// 计入有效帧（避免误判空流补写 "empty upstream stream"）。
		if _, hasErr := obj["error"]; hasErr {
			if err := s.writeRaw(payload); err != nil {
				return 0, err
			}
			return 1, nil
		}
		// 先按 index 收敛 tool_calls name（每 index 仅首片保留，后续分片删 name 键），再规范化透传。
		stripToolCallNames(obj, s.toolCallSeen)
		// id 续传：首帧非空真实 id 缓存；后续帧缺 id / 空 id 一律用缓存值，
		// 有自己 id 的帧保持原样（不同流分裂的帧允许各自 id）。
		if s.firstID == "" {
			if v, ok := obj["id"].(string); ok && v != "" {
				s.firstID = v
			} else if s.forceFirstID {
				// 自动续接必须给无 id 的首轮也建立稳定消息 ID，避免第二轮
				// 的真实 id 破坏同一条客户端消息的归并。
				s.firstID = "chatcmpl-wb2api"
			}
		} else {
			if v, ok := obj["id"].(string); !ok || v == "" || s.forceFirstID {
				obj["id"] = s.firstID
			}
		}
		if raw, err := json.Marshal(normalizeFrame(obj)); err == nil {
			payload = string(raw)
		}
		valid = 1
	}
	s.writeHeaders()
	if _, err := io.WriteString(s.w, "data: "+payload+"\n\n"); err != nil {
		return 0, err
	}
	if s.fl != nil {
		s.fl.Flush()
	}
	return valid, nil
}

// writeErrorFrame 以 error-passthrough 形态写出一帧错误（绕过 normalizeFrame
// 白名单，保证 error 字段原文可见）。
func (s *sseWriter) writeErrorFrame(message, typ string) {
	raw, err := json.Marshal(map[string]any{"error": map[string]any{"message": message, "type": typ}})
	if err != nil {
		return
	}
	_ = s.writeRaw(string(raw))
}

// writeDone 写出唯一的 [DONE] 并 flush。重复调用是空操作（幂等）。
func (s *sseWriter) writeDone() error {
	if s.doneWritten {
		return nil
	}
	s.doneWritten = true
	s.writeHeaders()
	if _, err := io.WriteString(s.w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	if s.fl != nil {
		s.fl.Flush()
	}
	return nil
}

// finishReasonLength 上游因输出上限截断的 finish_reason。
const finishReasonLength = "length"

// StreamOutcome 是一轮上游 SSE 流的读出结果，供自动续接回调使用。
// Content/Reasoning 是本轮 delta 增量的拼接结果。
type StreamOutcome struct {
	// FinishReason 本轮最后一个非空 finish_reason（"length" / "stop" / "tool_calls" …）。
	FinishReason string
	// Content 本轮增量文本（delta.content 拼接）。
	Content string
	// Reasoning 本轮增量思维链（delta.reasoning_content 拼接；上游不给则为空）。
	Reasoning string
	// HasToolCalls 本轮出现过 tool_calls / function_call（含空参数占位）。
	HasToolCalls bool
	// HasRefusal 本轮出现过 refusal（内容拒绝）。
	HasRefusal bool
	// ParseErr 本轮出现过无法解析的 data JSON；不允许据此续接。
	ParseErr bool
	// HasValidFrames 本轮至少出现一个可解析的 JSON data 帧（error 帧也算有效帧）。
	HasValidFrames bool
	// HadErrorFrame 本轮出现上游 error 帧；错误流绝不触发追加请求。
	HadErrorFrame bool
	// HasNonAssistantRole 本轮显式出现非 assistant 的消息角色；不视为纯文本回答。
	HasNonAssistantRole bool
	// HasUsage 本轮出现 usage 对象；各 token 字段按最后一个 usage 采信。
	HasUsage         bool
	PromptTokens     int
	CompletionTokens int
	// HasCredit 区分 usage.credit 缺失与显式 0。
	HasCredit bool
	Credit    float64
	// pending 暂存本轮最后一个带 finish_reason 的帧；外层决定是否续接后再写，
	// 否则达到续接上限时会错误地把最终 length 改成 null。
	pending *pendingSSEFrame
	// usageFrames 暂存 include_usage 产生的独立 usage-only 帧；续接中间轮丢弃，
	// 最终轮才发给客户端，避免中间 usage 破坏“单一最终结果”的语义。
	usageFrames []*pendingSSEFrame
	// ReadErr 读上游时的非 EOF 错误（截断/重置）。非 nil 时不得续接。
	ReadErr error
}

// outputLimitFinishReason 只接受上游明确表示「达到生成上限」的终态。
// length 是 OpenAI 标准值；其余是部分兼容上游使用的等价明确值。
// 不接受 incomplete/unknown 等模糊状态，避免网络截断被误续接。
func outputLimitFinishReason(reason string) bool {
	switch strings.ToLower(strings.TrimSpace(reason)) {
	case finishReasonLength, "max_tokens", "max_output_tokens", "token_limit", "output_limit":
		return true
	default:
		return false
	}
}

// refuseReason 不续接的原因（空串 = 满足续接条件；仅用于日志，不影响协议行为）。
func (o StreamOutcome) refuseReason() string {
	switch {
	case o.ReadErr != nil:
		return "read_error"
	case o.ParseErr:
		return "parse_error"
	case o.HadErrorFrame:
		return "error_frame"
	case o.HasNonAssistantRole:
		return "non_assistant_role"
	case o.HasToolCalls:
		return "tool_calls"
	case o.HasRefusal:
		return "refusal"
	case strings.TrimSpace(o.Content) == "":
		return "empty_content"
	case o.FinishReason == "":
		return "no_finish_reason"
	case !outputLimitFinishReason(o.FinishReason):
		return "finish_reason_" + o.FinishReason
	}
	return ""
}

// canContinue 本轮是否满足自动续接条件：明确 length 截断 + 纯文本 assistant
// （至少非空 content；无 tool_calls/function_call/refusal）+ 无读/解析错误。
func (o StreamOutcome) canContinue() bool { return o.refuseReason() == "" }

type pendingSSEFrame struct {
	payload string
	obj     map[string]any
}

// contentDelta 一帧里可累计的增量文本，以及是否出现 tool_calls/refusal
// （标准 delta 形态；部分上游把完整消息放在 message 里，一并兼容）。
func contentDelta(obj map[string]any) (content, reasoning string, toolCalls, refusal bool) {
	choices, ok := obj["choices"].([]any)
	if !ok {
		return "", "", false, false
	}
	var contentSB, reasonSB strings.Builder
	for _, ci := range choices {
		c, _ := ci.(map[string]any)
		if c == nil {
			continue
		}
		for _, holder := range []map[string]any{
			asMap(c["delta"]),
			asMap(c["message"]),
		} {
			if holder == nil {
				continue
			}
			contentSB.WriteString(stringField(holder, "content"))
			reasonSB.WriteString(stringField(holder, "reasoning_content"))
			if tcs, ok := holder["tool_calls"].([]any); ok && len(tcs) > 0 {
				toolCalls = true
			}
			if _, ok := holder["function_call"]; ok {
				toolCalls = true
			}
			if stringField(holder, "refusal") != "" {
				refusal = true
			}
		}
	}
	return contentSB.String(), reasonSB.String(), toolCalls, refusal
}

// asMap 把 any 断言为 map[string]any；非映射返回 nil。
func asMap(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

// stringField 取 obj[key] 的字符串值（非字符串/缺失 → 空串）。
func stringField(obj map[string]any, key string) string {
	v, _ := obj[key].(string)
	return v
}

// finishReasonOf 取一帧第一个非空 finish_reason（"" 表示本帧未结束）。
func finishReasonOf(obj map[string]any) string {
	choices, _ := obj["choices"].([]any)
	for _, ci := range choices {
		c, _ := ci.(map[string]any)
		if c == nil {
			continue
		}
		if v, ok := c["finish_reason"].(string); ok && v != "" {
			return v
		}
	}
	return ""
}

// trimFinishReasonInFrames 把帧内非空 finish_reason 改为 null：
// 中间轮（其后还有续接）的 length 终态对客户端不可见——客户端应看到「一轮连续
// 生成」，由最终轮给出真实 finish_reason。
func trimFinishReasonInFrames(obj map[string]any) {
	choices, _ := obj["choices"].([]any)
	for _, ci := range choices {
		c, _ := ci.(map[string]any)
		if c == nil {
			continue
		}
		if v, ok := c["finish_reason"].(string); ok && v != "" {
			c["finish_reason"] = nil
		}
	}
}

// streamUntilReader 用 Stream 的既有语义转发「一轮」上游流，直到 [DONE] 或 EOF，
// 并把累计文本/finish_reason 写入 out。
// holdFinish=true 时暂存每轮最后一个带 finish_reason 的帧，交给调用方决定是
// 原样收尾，还是把明确的输出上限终态改成 null 后继续下一轮。
func streamUntilReader(s *sseWriter, r io.Reader, out *StreamOutcome, holdFinish bool) error {
	if out == nil {
		out = &StreamOutcome{}
	}
	var contentSB, reasonSB strings.Builder
	br := bufio.NewReaderSize(r, 64*1024)
	var pending *pendingSSEFrame
	finish := func() {
		out.Content = contentSB.String()
		out.Reasoning = reasonSB.String()
		out.pending = pending
	}
readLoop:
	for {
		line, readErr := br.ReadString('\n')
		trimmed := strings.TrimRight(line, "\r\n")
		switch {
		case strings.HasPrefix(trimmed, "data: [DONE]"):
			// 上游显式结束：停止读取，DONE 由调用方统一写出，保证恰好一个。
			break readLoop
		case strings.HasPrefix(trimmed, "data: "):
			payload := strings.TrimPrefix(trimmed, "data: ")
			var obj map[string]any
			validJSON := json.Unmarshal([]byte(payload), &obj) == nil
			hasFinish := false
			if validJSON {
				out.HasValidFrames = true
				if _, hasErr := obj["error"]; hasErr {
					out.HadErrorFrame = true
				}
				if content, reasoning, toolCalls, refusal := contentDelta(obj); true {
					contentSB.WriteString(content)
					reasonSB.WriteString(reasoning)
					if toolCalls {
						out.HasToolCalls = true
					}
					if refusal {
						out.HasRefusal = true
					}
				}
				if choices, ok := obj["choices"].([]any); ok {
					for _, rawChoice := range choices {
						choice, _ := rawChoice.(map[string]any)
						if choice == nil {
							continue
						}
						for _, holderKey := range []string{"delta", "message"} {
							holder, _ := choice[holderKey].(map[string]any)
							if role, _ := holder["role"].(string); role != "" && role != "assistant" {
								out.HasNonAssistantRole = true
							}
						}
					}
				}
				if u, ok := obj["usage"].(map[string]any); ok && u != nil {
					out.HasUsage = true
					if v, ok := intField(u, "prompt_tokens"); ok {
						out.PromptTokens = v
					}
					if v, ok := intField(u, "completion_tokens"); ok {
						out.CompletionTokens = v
					}
					if v, ok := floatField(u, "credit"); ok {
						out.HasCredit = true
						out.Credit = v
					}
				}
				if fr := finishReasonOf(obj); fr != "" {
					out.FinishReason = fr
					hasFinish = true
				}
			} else {
				out.ParseErr = true
			}

			if holdFinish && validJSON && hasFinish {
				// 正常情况下只有最后一帧带 finish_reason；如果上游重复发送终态，
				// 先冲刷前一个，避免丢失可见内容。
				if pending != nil {
					if err := writePendingFrame(s, pending, false); err != nil {
						finish()
						return err
					}
				}
				pending = &pendingSSEFrame{payload: payload, obj: obj}
				out.pending = pending
			} else {
				if holdFinish && validJSON && isUsageOnlyFrame(obj) {
					out.usageFrames = append(out.usageFrames, &pendingSSEFrame{payload: payload, obj: obj})
				} else {
					n, writeErr := s.writeFrame(payload)
					s.validFrames += n
					if writeErr != nil {
						finish()
						return writeErr
					}
				}
			}
		case trimmed != "":
			// 注释/其他行原样透传；空行由 sseWriter 统一产生。
			s.writeHeaders()
			if _, writeErr := io.WriteString(s.w, line); writeErr != nil {
				finish()
				return writeErr
			}
			if s.fl != nil {
				s.fl.Flush()
			}
		}
		if readErr != nil {
			finish()
			if readErr == io.EOF {
				break
			}
			out.ReadErr = readErr
			return readErr
		}
	}
	finish()
	return nil
}

// writePendingFrame 写出暂存的终态帧。continuing=true 只用于确认已经成功打开
// 下一轮时：去掉当前轮 usage，且把明确输出上限终态改成 null，避免客户端把中间
// 轮误认为整条响应已经结束。
func writePendingFrame(s *sseWriter, p *pendingSSEFrame, continuing bool) error {
	if p == nil {
		return nil
	}
	payload := p.payload
	if continuing && p.obj != nil {
		trimFinishReasonInFrames(p.obj)
		p.obj["usage"] = nil
		if raw, err := json.Marshal(p.obj); err == nil {
			payload = string(raw)
		}
	}
	n, err := s.writeFrame(payload)
	s.validFrames += n
	return err
}

// writeFinalTurn 输出当前轮末尾暂存的 finish 帧与 usage-only 帧，保持上游的
// finish -> usage 顺序。中间续接轮次不会调用它，因此 usage 不会泄漏给客户端。
func writeFinalTurn(s *sseWriter, turn StreamOutcome) error {
	if turn.pending != nil {
		if err := writePendingFrame(s, turn.pending, false); err != nil {
			return err
		}
	}
	for _, usageFrame := range turn.usageFrames {
		if err := writePendingFrame(s, usageFrame, false); err != nil {
			return err
		}
	}
	return nil
}

// Stream 透传上游 SSE 到 w（逐帧规范化后 flush），保证至少写一个 [DONE]。
// 调用方必须先设置过 status 200；本函数自设 SSE headers。
func Stream(w http.ResponseWriter, r io.Reader) error {
	s := newSSEWriter(w)
	streamErr := streamUntilReader(s, r, nil, false)
	if s.validFrames == 0 {
		s.writeErrorFrame("empty upstream stream", "upstream_error")
	}
	if err := s.writeDone(); err != nil {
		return err
	}
	if streamErr != nil {
		return streamErr
	}
	if s.validFrames == 0 {
		return fmt.Errorf("upstream stream contained no valid data events")
	}
	return nil
}

// ContinueOpener 打开下一轮上游流（由 internal/server 提供：复用同账号、ctx、
// clientIP、ChatMeta）。返回 nil reader 表示不再续接；error 表示续接请求失败。
type ContinueOpener func(turn StreamOutcome) (io.ReadCloser, error)

// AutoContinueOptions 自动续接参数。
type AutoContinueOptions struct {
	// Max 最多续接轮数（<=0 = 关闭，行为与 Stream 完全一致）。内部硬上限为 3，
	// 防止配置错误把一次客户端请求放大成无界上游费用。
	Max int
	// Next 打开下一轮上游流；最多被调用 Max 次（仅在本轮满足续接条件时）。
	Next ContinueOpener
	// Context 客户端请求上下文；取消后不再发起下一轮。nil 等同 Background。
	Context context.Context
}

// AutoContinueResult 是自动续接期间各轮 usage 的累计结果。
type AutoContinueResult struct {
	Turns            int
	Continuations    int
	HasUsage         bool
	PromptTokens     int
	CompletionTokens int
	HasCredit        bool
	Credit           float64
	creditComplete   bool
}

func (r *AutoContinueResult) add(turn StreamOutcome) {
	r.Turns++
	if turn.HasUsage {
		r.HasUsage = true
		r.PromptTokens += turn.PromptTokens
		r.CompletionTokens += turn.CompletionTokens
		r.Credit += turn.Credit
	} else {
		r.creditComplete = false
	}
	if !turn.HasCredit {
		r.creditComplete = false
	}
	r.HasCredit = r.HasUsage && r.creditComplete
}

// StreamAutoContinue 保持简洁兼容入口；需要累计 usage/成本时使用
// StreamAutoContinueWithResult。
func StreamAutoContinue(w http.ResponseWriter, r io.Reader, opts AutoContinueOptions) error {
	_, err := StreamAutoContinueWithResult(w, r, opts)
	return err
}

// StreamAutoContinueWithResult 在同一 HTTP 响应内安全地续接明确的输出上限截断。
// 只有 finish_reason 为已知 output-limit 值、已有非空文本、无工具调用/拒绝/解析错、
// 且本轮读取正常时才会调用 Next。中间轮不发送 [DONE]，最终只发送一个 [DONE]。
func StreamAutoContinueWithResult(w http.ResponseWriter, r io.Reader, opts AutoContinueOptions) (AutoContinueResult, error) {
	s := newSSEWriter(w)
	s.forceFirstID = opts.Max > 0 && opts.Next != nil
	max := opts.Max
	if max < 0 {
		max = 0
	}
	if max > 3 {
		max = 3
	}
	holdFinish := max > 0 && opts.Next != nil
	result := AutoContinueResult{creditComplete: true}
	reader := r
	var lastErr error
	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}
	for {
		turn := StreamOutcome{}
		streamErr := streamUntilReader(s, reader, &turn, holdFinish)
		result.add(turn)
		lastErr = streamErr

		// 当前 reader 的数据已读完；关闭它后再打开下一轮，及时释放连接和取消句柄。
		if reader != r {
			if c, ok := reader.(io.Closer); ok {
				_ = c.Close()
			}
		}

		if streamErr != nil {
			if turn.pending != nil {
				_ = writePendingFrame(s, turn.pending, false)
			}
			break
		}
		if turn.pending == nil || !turn.canContinue() || result.Continuations >= max || opts.Next == nil {
			lastErr = writeFinalTurn(s, turn)
			break
		}
		select {
		case <-ctx.Done():
			lastErr = ctx.Err()
			_ = writeFinalTurn(s, turn)
			goto finish
		default:
		}

		nextReader, err := opts.Next(turn)
		if err != nil {
			_ = writePendingFrame(s, turn.pending, false)
			s.writeErrorFrame(err.Error(), "upstream_error")
			lastErr = err
			break
		}
		if nextReader == nil {
			lastErr = writePendingFrame(s, turn.pending, false)
			break
		}
		if err := writePendingFrame(s, turn.pending, true); err != nil {
			_ = nextReader.Close()
			lastErr = err
			break
		}
		// 中间轮 usage-only 帧不发送；下一轮的 usage 会继续累计。
		result.Continuations++
		reader = nextReader
	}

finish:
	if s.validFrames == 0 {
		s.writeErrorFrame("empty upstream stream", "upstream_error")
	}
	if err := s.writeDone(); err != nil {
		return result, err
	}
	if lastErr != nil {
		return result, lastErr
	}
	if s.validFrames == 0 {
		return result, fmt.Errorf("upstream stream contained no valid data events")
	}
	return result, nil
}

func intField(m map[string]any, key string) (int, bool) {
	v, ok := m[key]
	if !ok || v == nil {
		return 0, false
	}
	switch n := v.(type) {
	case float64:
		return int(n), true
	case float32:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	case json.Number:
		i, err := n.Int64()
		return int(i), err == nil
	default:
		return 0, false
	}
}

func floatField(m map[string]any, key string) (float64, bool) {
	v, ok := m[key]
	if !ok || v == nil {
		return 0, false
	}
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

func isUsageOnlyFrame(obj map[string]any) bool {
	if _, ok := obj["usage"].(map[string]any); !ok {
		return false
	}
	choices, exists := obj["choices"]
	if !exists || choices == nil {
		return true
	}
	list, ok := choices.([]any)
	return ok && len(list) == 0
}
