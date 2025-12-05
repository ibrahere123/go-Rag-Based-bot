package vector

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"log"
	"net/http"
	"time"
)

// Client wraps the Qdrant HTTP REST client.
type Client struct {
	baseURL        string
	httpClient     *http.Client
	collectionName string
	vectorSize     int
}

// Point represents a vector point to upsert.
type Point struct {
	ID      string
	Vector  []float32
	Payload map[string]interface{}
}

// SearchResult represents a search result.
type SearchResult struct {
	ID      string
	Score   float32
	Payload map[string]interface{}
}

// NewClient creates a new Qdrant HTTP client.
func NewClient(host string, port int, collectionName string, vectorSize int) (*Client, error) {
	// Use HTTP port (6333) instead of gRPC (6334)
	baseURL := fmt.Sprintf("http://%s:%d", host, port-1) // 6334 -> 6333

	log.Printf("Connecting to Qdrant at %s", baseURL)

	return &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
		collectionName: collectionName,
		vectorSize:     vectorSize,
	}, nil
}

// EnsureCollection creates the collection if it doesn't exist.
func (c *Client) EnsureCollection(ctx context.Context) error {
	// Check if collection exists by getting its info
	resp, err := c.httpClient.Get(fmt.Sprintf("%s/collections/%s", c.baseURL, c.collectionName))
	if err != nil {
		return fmt.Errorf("check collection: %w", err)
	}
	defer resp.Body.Close()

	// If collection exists (200 OK), we're done
	if resp.StatusCode == http.StatusOK {
		log.Printf("Collection %s already exists", c.collectionName)
		return nil
	}

	// If 404, collection doesn't exist - create it
	if resp.StatusCode == http.StatusNotFound {
		return c.createCollection(ctx)
	}

	// For other status codes, log and try to create
	respBody, _ := io.ReadAll(resp.Body)
	log.Printf("Collection check returned status %d: %s, attempting to create", resp.StatusCode, string(respBody))
	return c.createCollection(ctx)
}

func (c *Client) createCollection(ctx context.Context) error {
	createReq := map[string]interface{}{
		"vectors": map[string]interface{}{
			"size":     c.vectorSize,
			"distance": "Cosine",
		},
	}

	body, _ := json.Marshal(createReq)
	req, err := http.NewRequestWithContext(ctx, http.MethodPut,
		fmt.Sprintf("%s/collections/%s", c.baseURL, c.collectionName),
		bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("create collection: %w", err)
	}
	defer resp.Body.Close()

	// 200 OK or 409 Conflict (already exists) are both acceptable
	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusConflict {
		log.Printf("Collection %s ready", c.collectionName)
		return nil
	}

	respBody, _ := io.ReadAll(resp.Body)
	return fmt.Errorf("create collection failed (status %d): %s", resp.StatusCode, string(respBody))
}

// stringToNumericID converts a string ID to a numeric ID using FNV hash.
func stringToNumericID(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// UpsertPoints inserts or updates points in the collection.
func (c *Client) UpsertPoints(ctx context.Context, points []Point) error {
	qdrantPoints := make([]map[string]interface{}, len(points))

	for i, p := range points {
		qdrantPoints[i] = map[string]interface{}{
			"id":      stringToNumericID(p.ID),
			"vector":  p.Vector,
			"payload": p.Payload,
		}
	}

	upsertReq := map[string]interface{}{
		"points": qdrantPoints,
	}

	body, _ := json.Marshal(upsertReq)
	req, err := http.NewRequestWithContext(ctx, http.MethodPut,
		fmt.Sprintf("%s/collections/%s/points?wait=true", c.baseURL, c.collectionName),
		bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("upsert points: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("upsert failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	log.Printf("Upserted %d points", len(points))
	return nil
}

// Search performs a vector similarity search.
func (c *Client) Search(ctx context.Context, vector []float32, topK int) ([]SearchResult, error) {
	searchReq := map[string]interface{}{
		"vector":       vector,
		"limit":        topK,
		"with_payload": true,
	}

	body, _ := json.Marshal(searchReq)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		fmt.Sprintf("%s/collections/%s/points/search", c.baseURL, c.collectionName),
		bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("search failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	var searchResp struct {
		Result []struct {
			ID      interface{}            `json:"id"`
			Score   float32                `json:"score"`
			Payload map[string]interface{} `json:"payload"`
		} `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	results := make([]SearchResult, len(searchResp.Result))
	for i, r := range searchResp.Result {
		id := ""
		if idVal, ok := r.Payload["id"].(string); ok {
			id = idVal
		} else {
			id = fmt.Sprintf("%v", r.ID)
		}

		results[i] = SearchResult{
			ID:      id,
			Score:   r.Score,
			Payload: r.Payload,
		}
	}

	return results, nil
}

// Close closes the client (no-op for HTTP client).
func (c *Client) Close() error {
	return nil
}
