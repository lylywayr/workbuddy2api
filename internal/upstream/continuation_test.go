package upstream

import (
	"encoding/json"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestBuildContinuationBodyAppendsAssistantAndInstruction(t *testing.T) {
	turn := StreamOutcome{FinishReason: "length", Content: "前半段", Reasoning: "思考"}
	got, err := BuildContinuationBody([]byte(`{"model":"m","messages":[{"role":"user","content":"问题"}],"stream":false}`), turn)
	if err != nil {
		t.Fatal(err)
	}
	var obj map[string]any
	if err := json.Unmarshal(got, &obj); err != nil {
		t.Fatal(err)
	}
	msgs := obj["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages=%d want 3", len(msgs))
	}
	assistant := msgs[1].(map[string]any)
	if assistant["role"] != "assistant" || assistant["content"] != "前半段" || assistant["reasoning_content"] != "思考" {
		t.Errorf("assistant=%#v", assistant)
	}
	instruction := msgs[2].(map[string]any)
	if instruction["role"] != "user" || !strings.Contains(instruction["content"].(string), "继续") {
		t.Errorf("instruction=%#v", instruction)
	}
	if obj["stream"] != true {
		t.Errorf("stream=%v want true", obj["stream"])
	}
}

func TestStreamAutoContinueLengthOnlyOnce(t *testing.T) {
	first := `data: {"id":"a","choices":[{"index":0,"delta":{"content":"一"}}]}` + "\n\n" +
		`data: {"id":"a","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"credit":0.5}}` + "\n\n" +
		"data: [DONE]\n\n"
	second := `data: {"id":"b","choices":[{"index":0,"delta":{"content":"二"}}]}` + "\n\n" +
		`data: {"id":"b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":4,"credit":0.6}}` + "\n\n" +
		"data: [DONE]\n\n"
	calls := 0
	rec := httptest.NewRecorder()
	result, err := StreamAutoContinueWithResult(rec, strings.NewReader(first), AutoContinueOptions{
		Max: 1,
		Next: func(turn StreamOutcome) (io.ReadCloser, error) {
			calls++
			if turn.FinishReason != "length" || turn.Content != "一" {
				t.Fatalf("turn=%+v", turn)
			}
			return io.NopCloser(strings.NewReader(second)), nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || result.Continuations != 1 || result.CompletionTokens != 7 || result.Credit != 1.1 {
		t.Fatalf("calls=%d result=%+v", calls, result)
	}
	body := rec.Body.String()
	if strings.Count(body, "data: [DONE]") != 1 || !strings.Contains(body, `"content":"一"`) || !strings.Contains(body, `"content":"二"`) {
		t.Fatalf("body=%q", body)
	}
	if strings.Contains(body, `"finish_reason":"length"`) {
		t.Fatalf("intermediate length leaked: %q", body)
	}
	if strings.Count(body, `"finish_reason":"stop"`) != 1 {
		t.Fatalf("final stop missing: %q", body)
	}
}

func TestStreamAutoContinueDoesNotContinueOnReadError(t *testing.T) {
	rec := httptest.NewRecorder()
	calls := 0
	_, err := StreamAutoContinueWithResult(rec, &readErrorReader{
		data: []byte(`data: {"id":"a","choices":[{"index":0,"delta":{"content":"x"}}]}`),
	}, AutoContinueOptions{
		Max:  2,
		Next: func(StreamOutcome) (io.ReadCloser, error) { calls++; return nil, nil },
	})
	if err == nil || calls != 0 || strings.Count(rec.Body.String(), "data: [DONE]") != 1 {
		t.Fatalf("err=%v calls=%d body=%q", err, calls, rec.Body.String())
	}
}

type readErrorReader struct {
	data []byte
	done bool
}

func (r *readErrorReader) Read(p []byte) (int, error) {
	if !r.done {
		r.done = true
		return copy(p, r.data), nil
	}
	return 0, io.ErrUnexpectedEOF
}

func TestStreamAutoContinueStopsAtMaxAndKeepsLength(t *testing.T) {
	raw := `data: {"id":"a","choices":[{"index":0,"delta":{"content":"x"}}]}` + "\n\n" +
		`data: {"id":"a","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}` + "\n\n" + "data: [DONE]\n\n"
	calls := 0
	rec := httptest.NewRecorder()
	_, err := StreamAutoContinueWithResult(rec, strings.NewReader(raw), AutoContinueOptions{
		Max:  0,
		Next: func(StreamOutcome) (io.ReadCloser, error) { calls++; return nil, nil },
	})
	if err != nil || calls != 0 || !strings.Contains(rec.Body.String(), `"finish_reason":"length"`) {
		t.Fatalf("err=%v calls=%d body=%q", err, calls, rec.Body.String())
	}
}

func TestStreamAutoContinueDoesNotContinueToolCallsOrStop(t *testing.T) {
	for _, reason := range []string{"stop", "tool_calls"} {
		raw := `data: {"id":"a","choices":[{"index":0,"delta":{"content":"x","tool_calls":[{"index":0,"function":{"name":"f"}}]}}]}` + "\n\n" +
			`data: {"id":"a","choices":[{"index":0,"delta":{},"finish_reason":"` + reason + `"}]}` + "\n\n" + "data: [DONE]\n\n"
		calls := 0
		rec := httptest.NewRecorder()
		_, err := StreamAutoContinueWithResult(rec, strings.NewReader(raw), AutoContinueOptions{Max: 2, Next: func(StreamOutcome) (io.ReadCloser, error) { calls++; return nil, nil }})
		if err != nil || calls != 0 {
			t.Fatalf("reason=%s err=%v calls=%d body=%q", reason, err, calls, rec.Body.String())
		}
	}
}
