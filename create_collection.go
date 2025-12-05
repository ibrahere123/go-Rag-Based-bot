package main

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
)

func main() {
	// Create collection with 768 dimensions for nomic-embed-text
	body := []byte(`{"vectors": {"size": 768, "distance": "Cosine"}}`)
	
	req, _ := http.NewRequest("PUT", "http://localhost:6333/collections/knowledge_base", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer resp.Body.Close()
	
	respBody, _ := io.ReadAll(resp.Body)
	fmt.Printf("Status: %d\nResponse: %s\n", resp.StatusCode, string(respBody))
}
