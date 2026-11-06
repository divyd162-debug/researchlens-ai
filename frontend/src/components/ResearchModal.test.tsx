import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ResearchModal } from './ResearchModal'
import { api, ApiError } from '../services/api'
import { RAGResponse } from '../types'

describe('ResearchModal Component — Scope Handling & Grounded RAG Synthesis', () => {
  const mockRagResponse: RAGResponse = {
    query: 'How does self-attention scale with sequence length?',
    answer: 'Self-attention scales quadratically with sequence length O(n^2), as computed across dot-product matrix products.',
    citations: [
      {
        citation_frequency: 1,
        document_id: 'doc-uuid-1234',
        document_title: 'Attention Is All You Need',
        page_number: 4,
        chunk_id: 101,
        quoted_text: 'The computational complexity per layer is O(n^2 * d) for self-attention layers.',
        similarity_score: 0.942,
      },
    ],
    unknown_chunk_ids: [],
    malformed_markers: [],
    unsupported_numbers: [],
    status: 'ANSWERED',
    model_name: 'test-llm',
    provider_name: 'Gemini RAG Engine',
    retrieved_chunk_count: 5,
  }

  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(api.documents, 'list').mockResolvedValue({
      total: 1,
      limit: 50,
      offset: 0,
      items: [
        {
          id: 'doc-uuid-1234',
          filename: 'vaswani2017.pdf',
          title: 'Attention Is All You Need',
          page_count: 15,
          file_size_bytes: 204800,
          processing_status: 'INDEXED',
          created_at: '2026-01-01T00:00:00Z',
          updated_at: '2026-01-01T00:00:00Z',
          error_message: null,
        },
      ],
    })
    vi.spyOn(api.collections, 'list').mockResolvedValue({
      total: 1,
      limit: 50,
      offset: 0,
      items: [
        {
          id: 'col-uuid-5678',
          name: 'Foundation Models',
          description: 'Key papers in NLP and CV',
          created_at: '2026-01-01T00:00:00Z',
          updated_at: '2026-01-01T00:00:00Z',
          document_count: 3,
        },
      ],
    })
  })

  it('executes GLOBAL query without scope_id and renders grounded synthesis with citations', async () => {
    const ragSpy = vi.spyOn(api.query, 'rag').mockResolvedValue(mockRagResponse)
    const onClose = vi.fn()

    render(
      <ResearchModal
        isOpen={true}
        onClose={onClose}
        initialData={{
          prompt: 'How does self-attention scale with sequence length?',
          scope: 'GLOBAL',
        }}
      />
    )

    // Execute synthesis
    const executeBtn = screen.getByRole('button', { name: /Execute with ResearchLens/i })
    fireEvent.click(executeBtn)

    await waitFor(() => {
      expect(ragSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          query: 'How does self-attention scale with sequence length?',
          scope_type: 'GLOBAL',
          scope_id: undefined,
        })
      )
    })

    // Grounded synthesis answer rendered
    expect(await screen.findByText(/Self-attention scales quadratically with sequence length O\(n\^2\)/i)).toBeDefined()

    // Verified citation evidence rendered with page number and similarity score
    expect(screen.getByText(/Attention Is All You Need/i)).toBeDefined()
    expect(screen.getByText(/Page 4/i)).toBeDefined()
    expect(screen.getByText(/94.2% match/i)).toBeDefined()
    expect(screen.getByText(/The computational complexity per layer is O\(n\^2 \* d\)/i)).toBeDefined()
  })

  it('displays demonstration notice when inspecting sample papers without indexed ID', async () => {
    render(
      <ResearchModal
        isOpen={true}
        onClose={vi.fn()}
        initialData={{
          prompt: 'Inspect Transformer architecture',
          scope: 'GLOBAL',
          isDemonstration: true,
          demonstrationNotice: 'Reference paper — Vaswani et al. (2017). Queries search across all indexed library documents.',
        }}
      />
    )

    expect(screen.getByText(/Demonstration Reference Paper/i)).toBeDefined()
    expect(screen.getByText(/Queries search across all indexed library documents/i)).toBeDefined()
  })

  it('switches to DOCUMENT scope, loads user documents, and sends selected document UUID', async () => {
    const ragSpy = vi.spyOn(api.query, 'rag').mockResolvedValue(mockRagResponse)

    render(
      <ResearchModal
        isOpen={true}
        onClose={vi.fn()}
        initialData={{
          prompt: 'What is scaled dot-product attention?',
          scope: 'GLOBAL',
        }}
      />
    )

    // Click Single Paper button
    const singlePaperBtn = screen.getByRole('button', { name: 'Single Paper' })
    fireEvent.click(singlePaperBtn)

    // Document selector appears with loaded doc
    await waitFor(() => {
      expect(screen.getByLabelText(/Select Target Document/i)).toBeDefined()
    })

    const select = screen.getByLabelText(/Select Target Document/i) as HTMLSelectElement
    expect(select.value).toBe('doc-uuid-1234')

    // Click execute
    const executeBtn = screen.getByRole('button', { name: /Execute with ResearchLens/i })
    fireEvent.click(executeBtn)

    await waitFor(() => {
      expect(ragSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          query: 'What is scaled dot-product attention?',
          scope_type: 'DOCUMENT',
          scope_id: 'doc-uuid-1234',
        })
      )
    })
  })

  it('switches to COLLECTION scope, loads user collections, and sends selected collection UUID', async () => {
    const ragSpy = vi.spyOn(api.query, 'rag').mockResolvedValue(mockRagResponse)

    render(
      <ResearchModal
        isOpen={true}
        onClose={vi.fn()}
        initialData={{
          prompt: 'Compare self-attention and RNNs across literature',
          scope: 'GLOBAL',
        }}
      />
    )

    // Click Collection button
    const collectionBtn = screen.getByRole('button', { name: 'Collection' })
    fireEvent.click(collectionBtn)

    await waitFor(() => {
      expect(screen.getByLabelText(/Select Target Collection/i)).toBeDefined()
    })

    const select = screen.getByLabelText(/Select Target Collection/i) as HTMLSelectElement
    expect(select.value).toBe('col-uuid-5678')

    // Click execute
    const executeBtn = screen.getByRole('button', { name: /Execute with ResearchLens/i })
    fireEvent.click(executeBtn)

    await waitFor(() => {
      expect(ragSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          query: 'Compare self-attention and RNNs across literature',
          scope_type: 'COLLECTION',
          scope_id: 'col-uuid-5678',
        })
      )
    })
  })

  it('prevents scoped execution when no documents exist and shows fallback notice', async () => {
    vi.spyOn(api.documents, 'list').mockResolvedValue({ total: 0, limit: 50, offset: 0, items: [] })

    render(
      <ResearchModal
        isOpen={true}
        onClose={vi.fn()}
        initialData={{
          prompt: 'Test query',
          scope: 'GLOBAL',
        }}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Single Paper' }))

    await waitFor(() => {
      expect(screen.getByText(/No indexed documents found in your library/i)).toBeDefined()
    })

    const executeBtn = screen.getByRole('button', { name: /Execute with ResearchLens/i }) as HTMLButtonElement
    expect(executeBtn.disabled).toBe(true)
  })

  it('displays descriptive error when backend returns 401 Unauthorized', async () => {
    vi.spyOn(api.query, 'rag').mockRejectedValue(new ApiError(401, 'Unauthorized access'))

    render(
      <ResearchModal
        isOpen={true}
        onClose={vi.fn()}
        initialData={{
          prompt: 'Private inquiry',
          scope: 'GLOBAL',
        }}
      />
    )

    const executeBtn = screen.getByRole('button', { name: /Execute with ResearchLens/i })
    fireEvent.click(executeBtn)

    expect(
      await screen.findByText(/Authentication required: Sign in via the navigation header/i)
    ).toBeDefined()
  })
})
