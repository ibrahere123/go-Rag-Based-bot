package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"go-bot/config"
	"go-bot/internal/ingest"
	"go-bot/internal/llm"
	"go-bot/internal/vector"
)

func main() {
	// Parse flags
	filePath := flag.String("file", "Knowledgebase.json", "Path to the knowledge base JSON file")
	flag.Parse()

	// Load config
	cfg := config.Load()

	if cfg.GroqAPIKey == "" {
		log.Fatal("GROQ_API_KEY is required")
	}

	// Setup context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle shutdown signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		log.Println("Shutting down...")
		cancel()
	}()

	// Initialize clients
	log.Println("Connecting to Qdrant...")
	vectorClient, err := vector.NewClient(cfg.QdrantHost, cfg.QdrantPort, cfg.CollectionName, cfg.EmbeddingDim)
	if err != nil {
		log.Fatalf("Failed to create vector client: %v", err)
	}
	defer vectorClient.Close()

	// Ensure collection exists
	if err := vectorClient.EnsureCollection(ctx); err != nil {
		log.Fatalf("Failed to ensure collection: %v", err)
	}

	// Initialize embedder
	embedder := llm.NewEmbedder(cfg.GroqAPIKey)

	// Initialize ingestion service
	ingestService := ingest.NewService(embedder, vectorClient)

	// Run ingestion
	log.Printf("Starting ingestion from %s...", *filePath)
	if err := ingestService.IngestJSONFile(ctx, *filePath); err != nil {
		log.Fatalf("Ingestion failed: %v", err)
	}

	log.Println("Ingestion completed successfully!")
}
