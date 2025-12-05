package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

type ChatRequest struct {
	Query string `json:"query"`
}

func main() {
	// Flags
	url := flag.String("url", "http://localhost:8080/chat", "API endpoint")
	concurrent := flag.Int("c", 10, "Number of concurrent users")
	requests := flag.Int("n", 100, "Total number of requests")
	flag.Parse()

	queries := []string{
		"How do I sign in?",
		"What is SyntraFlow?",
		"How do I request leave?",
		"What is the dashboard?",
		"How do I reset my password?",
	}

	var (
		successCount int64
		failCount    int64
		totalLatency int64
		minLatency   int64 = 999999
		maxLatency   int64
		mu           sync.Mutex
	)

	fmt.Printf("🚀 Load Test Starting...\n")
	fmt.Printf("   URL: %s\n", *url)
	fmt.Printf("   Concurrent users: %d\n", *concurrent)
	fmt.Printf("   Total requests: %d\n\n", *requests)

	startTime := time.Now()

	// Create a semaphore for concurrency control
	sem := make(chan struct{}, *concurrent)
	var wg sync.WaitGroup

	client := &http.Client{
		Timeout: 60 * time.Second,
	}

	for i := 0; i < *requests; i++ {
		wg.Add(1)
		sem <- struct{}{} // Acquire

		go func(reqNum int) {
			defer wg.Done()
			defer func() { <-sem }() // Release

			query := queries[reqNum%len(queries)]
			reqBody := ChatRequest{Query: query}
			body, _ := json.Marshal(reqBody)

			reqStart := time.Now()
			resp, err := client.Post(*url, "application/json", bytes.NewReader(body))
			latency := time.Since(reqStart).Milliseconds()

			if err != nil {
				atomic.AddInt64(&failCount, 1)
				fmt.Printf("❌ Request %d failed: %v\n", reqNum+1, err)
				return
			}
			defer resp.Body.Close()

			if resp.StatusCode == 200 {
				atomic.AddInt64(&successCount, 1)
			} else {
				atomic.AddInt64(&failCount, 1)
				fmt.Printf("❌ Request %d: status %d\n", reqNum+1, resp.StatusCode)
			}

			atomic.AddInt64(&totalLatency, latency)

			mu.Lock()
			if latency < minLatency {
				minLatency = latency
			}
			if latency > maxLatency {
				maxLatency = latency
			}
			mu.Unlock()

			if (reqNum+1)%10 == 0 {
				fmt.Printf("✓ Completed %d/%d requests\n", reqNum+1, *requests)
			}
		}(i)
	}

	wg.Wait()
	totalTime := time.Since(startTime)

	// Results
	total := successCount + failCount
	avgLatency := float64(totalLatency) / float64(total)
	rps := float64(total) / totalTime.Seconds()

	fmt.Println("\n" + "══════════════════════════════════════════════════")
	fmt.Println("📊 LOAD TEST RESULTS")
	fmt.Println("══════════════════════════════════════════════════")
	fmt.Printf("Total Requests:     %d\n", total)
	fmt.Printf("Successful:         %d (%.1f%%)\n", successCount, float64(successCount)/float64(total)*100)
	fmt.Printf("Failed:             %d (%.1f%%)\n", failCount, float64(failCount)/float64(total)*100)
	fmt.Printf("Total Time:         %.2fs\n", totalTime.Seconds())
	fmt.Printf("Requests/sec:       %.2f\n", rps)
	fmt.Println("──────────────────────────────────────────────────")
	fmt.Printf("Avg Latency:        %.0fms\n", avgLatency)
	fmt.Printf("Min Latency:        %dms\n", minLatency)
	fmt.Printf("Max Latency:        %dms\n", maxLatency)
	fmt.Println("══════════════════════════════════════════════════")
}
