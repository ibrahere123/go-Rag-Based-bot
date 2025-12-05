package rag

import (
	"context"
	"fmt"
	"io"
	"strings"

	"go-bot/internal/llm"
	"go-bot/internal/vector"
)

// Service handles RAG queries.
type Service struct {
	llmClient    *llm.Client
	embedder     *llm.Embedder
	vectorClient *vector.Client
	topK         int
}

// NewService creates a new RAG service.
func NewService(llmClient *llm.Client, embedder *llm.Embedder, vectorClient *vector.Client) *Service {
	return &Service{
		llmClient:    llmClient,
		embedder:     embedder,
		vectorClient: vectorClient,
		topK:         5,
	}
}

// QueryResult represents the result of a RAG query.
type QueryResult struct {
	Answer   string
	Sources  []Source
}

// Source represents a retrieved document source.
type Source struct {
	ID     string
	Module string
	Topic  string
	Score  float32
}

// Query performs a RAG query and returns the answer.
func (s *Service) Query(ctx context.Context, userQuery string) (*QueryResult, error) {
	// 1. Embed the query
	queryEmbedding, err := s.embedder.EmbedSingle(ctx, userQuery)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	// 2. Search for relevant documents
	results, err := s.vectorClient.Search(ctx, queryEmbedding, s.topK)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}

	// 3. Build context from results
	context_text := s.buildContext(results)

	// 4. Build messages
	messages := []llm.Message{
		{
			Role: "system",
			Content: `You are the official Support Assistant for SyntraFlow - a comprehensive employee management system.

## About SyntraFlow:
SyntraFlow is an all-in-one Employee Management System (EMS) designed to streamline HR operations for organizations of all sizes. Key features include:
- **Authentication & Access Control**: Secure sign-in, sign-up, password management, and role-based permissions
- **Employee Management**: Complete employee lifecycle management including onboarding, profiles, and document handling
- **Attendance & Rota Management**: Shift scheduling, clock in/out tracking, terminals, and live attendance monitoring
- **Leave Management**: Leave requests, approvals, balances, WFH requests, and policy configuration
- **Payroll & Salary**: Salary elements, payroll processing, and payslip generation
- **Dashboard**: Real-time performance metrics, attendance insights, meetings, and company events
- **Calendar**: Meeting scheduling, time insights, and team availability
- **Policy Manager**: Configure leave policies, shift policies, WFH rules, and compensation structures
- **Reports**: Time & attendance reports, lateness tracking, and live tracking

## Your Role:
- You are the primary support resource for SyntraFlow users
- Help employees and administrators navigate the platform
- Provide clear, step-by-step guidance for all features

## Guidelines:
1. For questions about what SyntraFlow is, use the About SyntraFlow section above
2. For specific feature questions, use the provided context from the knowledge base
3. Be concise but thorough - include all necessary steps
4. Use numbered lists for step-by-step instructions
5. If the context doesn't have specific details, say so politely and offer to help with something else
6. Never make up features or steps
7. Be professional, friendly, and helpful

## Response Format:
- Start with a direct answer
- Follow with step-by-step instructions if applicable
- End with a helpful tip if relevant`,
		},
		{
			Role:    "user",
			Content: fmt.Sprintf("Context from SyntraFlow Knowledge Base:\n%s\n\nUser Question: %s", context_text, userQuery),
		},
	}

	// 5. Get LLM response
	resp, err := s.llmClient.CreateChatCompletion(ctx, messages, 1024)
	if err != nil {
		return nil, fmt.Errorf("llm completion: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no response from LLM")
	}

	// 6. Build result
	sources := make([]Source, len(results))
	for i, r := range results {
		module, _ := r.Payload["module"].(string)
		topic, _ := r.Payload["topic"].(string)
		id, _ := r.Payload["id"].(string)
		sources[i] = Source{
			ID:     id,
			Module: module,
			Topic:  topic,
			Score:  r.Score,
		}
	}

	return &QueryResult{
		Answer:  resp.Choices[0].Message.Content,
		Sources: sources,
	}, nil
}

// StreamQuery performs a RAG query with streaming response.
func (s *Service) StreamQuery(ctx context.Context, userQuery string, writer io.Writer) error {
	// 1. Embed the query
	queryEmbedding, err := s.embedder.EmbedSingle(ctx, userQuery)
	if err != nil {
		return fmt.Errorf("embed query: %w", err)
	}

	// 2. Search for relevant documents
	results, err := s.vectorClient.Search(ctx, queryEmbedding, s.topK)
	if err != nil {
		return fmt.Errorf("search: %w", err)
	}

	// 3. Build context from results
	context_text := s.buildContext(results)

	// 4. Build messages
	messages := []llm.Message{
		{
			Role: "system",
			Content: `You are the official Support Assistant for SyntraFlow - a comprehensive employee management system.

## About SyntraFlow:
SyntraFlow is an all-in-one Employee Management System (EMS) designed to streamline HR operations for organizations of all sizes. Key features include:
- **Authentication & Access Control**: Secure sign-in, sign-up, password management, and role-based permissions
- **Employee Management**: Complete employee lifecycle management including onboarding, profiles, and document handling
- **Attendance & Rota Management**: Shift scheduling, clock in/out tracking, terminals, and live attendance monitoring
- **Leave Management**: Leave requests, approvals, balances, WFH requests, and policy configuration
- **Payroll & Salary**: Salary elements, payroll processing, and payslip generation
- **Dashboard**: Real-time performance metrics, attendance insights, meetings, and company events
- **Calendar**: Meeting scheduling, time insights, and team availability
- **Policy Manager**: Configure leave policies, shift policies, WFH rules, and compensation structures
- **Reports**: Time & attendance reports, lateness tracking, and live tracking

## Your Role:
- You are the primary support resource for SyntraFlow users
- Help employees and administrators navigate the platform
- Provide clear, step-by-step guidance for all features

## Guidelines:
1. For questions about what SyntraFlow is, use the About SyntraFlow section above
2. For specific feature questions, use the provided context from the knowledge base
3. Be concise but thorough - include all necessary steps
4. Use numbered lists for step-by-step instructions
5. If the context doesn't have specific details, say so politely and offer to help with something else
6. Never make up features or steps
7. Be professional, friendly, and helpful

## Response Format:
- Start with a direct answer
- Follow with step-by-step instructions if applicable
- End with a helpful tip if relevant`,
		},
		{
			Role:    "user",
			Content: fmt.Sprintf("Context from SyntraFlow Knowledge Base:\n%s\n\nUser Question: %s", context_text, userQuery),
		},
	}

	// 5. Stream LLM response
	return s.llmClient.StreamChatCompletion(ctx, messages, 1024, writer)
}

func (s *Service) buildContext(results []vector.SearchResult) string {
	var sb strings.Builder
	for i, r := range results {
		text, ok := r.Payload["text"].(string)
		if !ok {
			continue
		}
		sb.WriteString(fmt.Sprintf("--- Document %d (score: %.2f) ---\n", i+1, r.Score))
		sb.WriteString(text)
		sb.WriteString("\n\n")
	}
	return sb.String()
}
