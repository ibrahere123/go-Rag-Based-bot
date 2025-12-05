package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

// Using Ollama local embeddings
const ollamaEmbeddingURL = "http://localhost:11434/api/embeddings"

// Embedder generates embeddings using Ollama locally.
type Embedder struct {
	httpClient *http.Client
	model      string
}

// OllamaRequest is the request format for Ollama embeddings.
type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// OllamaResponse is the response format from Ollama embeddings.
type OllamaResponse struct {
	Embedding []float64 `json:"embedding"`
}

// NewEmbedder creates a new embedder using Ollama.
func NewEmbedder(_ string) *Embedder {
	return &Embedder{
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		model: "nomic-embed-text:latest",
	}
}

// Embed generates embeddings for the given texts.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))

	for i, text := range texts {
		emb, err := e.embedSingle(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("embed text %d: %w", i, err)
		}
		embeddings[i] = emb

		if (i+1)%10 == 0 {
			log.Printf("Embedded %d/%d texts", i+1, len(texts))
		}
	}

	return embeddings, nil
}

func (e *Embedder) embedSingle(ctx context.Context, text string) ([]float32, error) {
	// Truncate if too long
	if len(text) > 8000 {
		text = text[:8000]
	}

	reqBody := OllamaRequest{
		Model:  e.model,
		Prompt: text,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ollamaEmbeddingURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama error: status %d, body: %s", resp.StatusCode, string(respBody))
	}

	var ollamaResp OllamaResponse
	if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	if len(ollamaResp.Embedding) == 0 {
		return nil, fmt.Errorf("empty embedding returned")
	}

	return float64ToFloat32(ollamaResp.Embedding), nil
}

// EmbedSingle generates an embedding for a single text.
func (e *Embedder) EmbedSingle(ctx context.Context, text string) ([]float32, error) {
	return e.embedSingle(ctx, text)
}

func float64ToFloat32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}
