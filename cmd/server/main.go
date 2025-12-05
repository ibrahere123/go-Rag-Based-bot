package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go-bot/config"
	"go-bot/internal/llm"
	"go-bot/internal/rag"
	"go-bot/internal/vector"
)

// ChatRequest represents an incoming chat request.
type ChatRequest struct {
	Query  string `json:"query"`
	Stream bool   `json:"stream"`
}

// ChatResponse represents the response.
type ChatResponse struct {
	Answer  string   `json:"answer"`
	Sources []Source `json:"sources,omitempty"`
}

// Source is a simplified source reference.
type Source struct {
	ID     string  `json:"id"`
	Module string  `json:"module"`
	Topic  string  `json:"topic"`
	Score  float32 `json:"score"`
}

func main() {
	// Load config
	cfg := config.Load()

	if cfg.GroqAPIKey == "" {
		log.Fatal("GROQ_API_KEY is required")
	}

	// Setup context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initialize clients
	log.Println("Connecting to Qdrant...")
	vectorClient, err := vector.NewClient(cfg.QdrantHost, cfg.QdrantPort, cfg.CollectionName, cfg.EmbeddingDim)
	if err != nil {
		log.Fatalf("Failed to create vector client: %v", err)
	}
	defer vectorClient.Close()

	// Initialize LLM and embedder
	llmClient := llm.NewClient(cfg.GroqAPIKey)
	embedder := llm.NewEmbedder(cfg.GroqAPIKey)

	// Initialize RAG service
	ragService := rag.NewService(llmClient, embedder, vectorClient)

	// Setup HTTP server
	mux := http.NewServeMux()

	// Health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	})

	// Chat endpoint
	mux.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		if req.Query == "" {
			http.Error(w, "Query is required", http.StatusBadRequest)
			return
		}

		if req.Stream {
			// Streaming response
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")

			flusher, ok := w.(http.Flusher)
			if !ok {
				http.Error(w, "Streaming not supported", http.StatusInternalServerError)
				return
			}

			// Create a writer that flushes after each write
			streamWriter := &flushWriter{w: w, f: flusher}

			if err := ragService.StreamQuery(r.Context(), req.Query, streamWriter); err != nil {
				log.Printf("Stream error: %v", err)
			}
		} else {
			// Non-streaming response
			result, err := ragService.Query(r.Context(), req.Query)
			if err != nil {
				log.Printf("Query error: %v", err)
				http.Error(w, "Internal server error", http.StatusInternalServerError)
				return
			}

			sources := make([]Source, len(result.Sources))
			for i, s := range result.Sources {
				sources[i] = Source{
					ID:     s.ID,
					Module: s.Module,
					Topic:  s.Topic,
					Score:  s.Score,
				}
			}

			resp := ChatResponse{
				Answer:  result.Answer,
				Sources: sources,
			}

			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}
	})

	// Create server
	server := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      corsMiddleware(loggingMiddleware(mux)),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in goroutine
	go func() {
		log.Printf("Server starting on port %s", cfg.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down server...")
	shutdownCtx, shutdownCancel := context.WithTimeout(ctx, 10*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("Shutdown error: %v", err)
	}

	log.Println("Server stopped")
}

// flushWriter wraps a ResponseWriter and Flusher for streaming.
type flushWriter struct {
	w http.ResponseWriter
	f http.Flusher
}

func (fw *flushWriter) Write(p []byte) (int, error) {
	n, err := fw.w.Write(p)
	fw.f.Flush()
	return n, err
}

// loggingMiddleware logs incoming requests.
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %v", r.Method, r.URL.Path, time.Since(start))
	})
}

// corsMiddleware adds CORS headers.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}
