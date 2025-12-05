package config

import (
	"log"
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

// Config holds all configuration for the application.
type Config struct {
	GroqAPIKey     string
	QdrantHost     string
	QdrantPort     int
	Port           string
	CollectionName string
	EmbeddingDim   int
}

// Load reads configuration from environment variables.
func Load() *Config {
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, reading from environment")
	}

	qdrantPort, _ := strconv.Atoi(getEnv("QDRANT_PORT", "6334"))
	embeddingDim, _ := strconv.Atoi(getEnv("EMBEDDING_DIM", "384"))

	return &Config{
		GroqAPIKey:     getEnv("GROQ_API_KEY", ""),
		QdrantHost:     getEnv("QDRANT_HOST", "localhost"),
		QdrantPort:     qdrantPort,
		Port:           getEnv("PORT", "8080"),
		CollectionName: getEnv("COLLECTION_NAME", "knowledge_base"),
		EmbeddingDim:   embeddingDim,
	}
}

func getEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}
