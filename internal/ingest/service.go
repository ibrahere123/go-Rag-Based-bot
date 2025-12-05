package ingest

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"go-bot/internal/llm"
	"go-bot/internal/vector"
)

// KnowledgeEntry represents a single entry from Knowledgebase.json.
type KnowledgeEntry struct {
	ID              string   `json:"id"`
	Module          string   `json:"module"`
	Topic           string   `json:"topic"`
	Roles           []string `json:"roles"`
	QueryVariations []string `json:"query_variations"`
	Answer          string   `json:"answer"`
}

// Service handles document ingestion.
type Service struct {
	embedder     *llm.Embedder
	vectorClient *vector.Client
}

// NewService creates a new ingestion service.
func NewService(embedder *llm.Embedder, vectorClient *vector.Client) *Service {
	return &Service{
		embedder:     embedder,
		vectorClient: vectorClient,
	}
}

// IngestJSONFile parses and ingests a knowledge base JSON file.
func (s *Service) IngestJSONFile(ctx context.Context, filePath string) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	var entries []KnowledgeEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return fmt.Errorf("unmarshal json: %w", err)
	}

	log.Printf("Loaded %d entries from %s", len(entries), filePath)

	// Process in batches
	batchSize := 10
	for i := 0; i < len(entries); i += batchSize {
		end := i + batchSize
		if end > len(entries) {
			end = len(entries)
		}

		batch := entries[i:end]
		if err := s.processBatch(ctx, batch); err != nil {
			return fmt.Errorf("process batch %d: %w", i/batchSize, err)
		}

		log.Printf("Processed batch %d/%d", (i/batchSize)+1, (len(entries)+batchSize-1)/batchSize)
	}

	return nil
}

func (s *Service) processBatch(ctx context.Context, entries []KnowledgeEntry) error {
	// Generate text for embedding
	texts := make([]string, len(entries))
	for i, entry := range entries {
		texts[i] = s.entryToText(entry)
	}

	// Get embeddings
	embeddings, err := s.embedder.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed texts: %w", err)
	}

	// Create points
	points := make([]vector.Point, len(entries))
	for i, entry := range entries {
		points[i] = vector.Point{
			ID:     entry.ID,
			Vector: embeddings[i],
			Payload: map[string]interface{}{
				"id":               entry.ID,
				"module":           entry.Module,
				"topic":            entry.Topic,
				"roles":            entry.Roles,
				"query_variations": entry.QueryVariations,
				"answer":           entry.Answer,
				"text":             texts[i],
			},
		}
	}

	// Upsert to Qdrant
	if err := s.vectorClient.UpsertPoints(ctx, points); err != nil {
		return fmt.Errorf("upsert points: %w", err)
	}

	return nil
}

func (s *Service) entryToText(entry KnowledgeEntry) string {
	var sb strings.Builder
	sb.WriteString("Module: ")
	sb.WriteString(entry.Module)
	sb.WriteString("\nTopic: ")
	sb.WriteString(entry.Topic)
	sb.WriteString("\nQuestions: ")
	sb.WriteString(strings.Join(entry.QueryVariations, "; "))
	sb.WriteString("\nAnswer: ")
	sb.WriteString(entry.Answer)
	return sb.String()
}
