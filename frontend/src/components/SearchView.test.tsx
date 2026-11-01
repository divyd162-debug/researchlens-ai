import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { SearchView } from './SearchView'
import { api, ApiError } from '../services/api'
import { DocumentResponse, CollectionResponse, SearchResponse } from '../types'

const mockDocs: DocumentResponse[] = [
  {
    id: 'doc-uuid-1',
    filename: 'attention_is_all_you_need.pdf',
    title: 'Attention Is All You Need',
    file_size_bytes: 1048576,
    page_count: 8,
    processing_status: 'READY',
    created_at: '2026-10-18T10:00:00Z',
    updated_at: '2026-10-18T10:00:00Z',
  },
  {
    id: 'doc-uuid-2',
    filename: 'bert_paper.pdf',
    title: null,
    file_size_bytes: 2097152,
    page_count: 12,
    processing_status: 'READY',
    created_at: '2026-10-18T11:00:00Z',
    updated_at: '2026-10-18T11:00:00Z',
  },
]

const mockCollections: CollectionResponse[] = [
  {
    id: 'col-uuid-1',
    name: 'Transformer Architectures',
    description: 'Core transformer papers',
    document_count: 2,
    created_at: '2026-10-18T09:00:00Z',
    updated_at: '2026-10-18T09:00:00Z',
  },
]

const mockSearchSuccessResponse: SearchResponse = {
  query: 'attention mechanisms in deep learning',
  scope_type: 'GLOBAL',
  scope_id: null,
  total_results: 2,
  result_count: 2,
  results: [
    {
      chunk_id: 101,
      document_id: 'doc-uuid-1',
      document_title: 'Attention Is All You Need',
      page_number: 2,
      content: 'Self-attention connects different positions of a single sequence to compute a representation.',
      similarity_score: 0.8845,
      char_start: 120,
      char_end: 213,
    },
    {
      chunk_id: 102,
      document_id: 'doc-uuid-2',
      document_title: 'bert_paper.pdf',
      page_number: 5,
      content: 'BERT uses bidirectional self-attention to pre-train deep representations.',
      similarity_score: 0.7231,
      char_start: 350,
      char_end: 425,
    },
  ],
}

describe('SearchView Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(api.documents, 'list').mockResolvedValue({
      items: mockDocs,
      total: mockDocs.length,
      limit: 100,
      offset: 0,
    })
    vi.spyOn(api.collections, 'list').mockResolvedValue({
      items: mockCollections,
      total: mockCollections.length,
      limit: 100,
      offset: 0,
    })
  })

  it('renders search interface with input, scope radio options, and initial guidance state', async () => {
    render(<SearchView />)

    expect(screen.getByRole('heading', { name: /Semantic Passage Search/i })).toBeDefined()
    expect(screen.getByPlaceholderText(/Ask a question or enter concepts/i)).toBeDefined()
    expect(screen.getByRole('button', { name: /Search Passages/i })).toBeDefined()

    // Scope selection controls
    expect(screen.getByRole('radiogroup', { name: /Search Scope/i })).toBeDefined()
    expect(screen.getByLabelText(/All Documents \(Global\)/i)).toBeDefined()
    expect(screen.getByLabelText(/Document Scoped/i)).toBeDefined()
    expect(screen.getByLabelText(/Collection Scoped/i)).toBeDefined()

    // Guidance placeholder
    expect(screen.getByText(/Grounded Semantic Search/i)).toBeDefined()

    // Await async options loading to cleanly wrap state updates in act
    await waitFor(() => {
      expect(api.documents.list).toHaveBeenCalled()
      expect(api.collections.list).toHaveBeenCalled()
    })
  })

  it('validates empty or whitespace queries client-side without calling API', async () => {
    const searchSpy = vi.spyOn(api.search, 'query')
    render(<SearchView />)

    const submitBtn = screen.getByRole('button', { name: /Search Passages/i })
    fireEvent.click(submitBtn)

    expect(await screen.findByText(/Please enter a search query/i)).toBeDefined()
    expect(searchSpy).not.toHaveBeenCalled()

    // Whitespace only
    const input = screen.getByPlaceholderText(/Ask a question or enter concepts/i)
    fireEvent.change(input, { target: { value: '    \t   ' } })
    fireEvent.click(submitBtn)

    expect(await screen.findByText(/Please enter a search query/i)).toBeDefined()
    expect(searchSpy).not.toHaveBeenCalled()
  })

  it('executes successful GLOBAL search and renders ranked passages with metadata', async () => {
    const searchSpy = vi.spyOn(api.search, 'query').mockResolvedValueOnce(mockSearchSuccessResponse)
    render(<SearchView />)

    const input = screen.getByPlaceholderText(/Ask a question or enter concepts/i)
    fireEvent.change(input, { target: { value: '  attention mechanisms in deep learning  ' } })

    const submitBtn = screen.getByRole('button', { name: /Search Passages/i })
    fireEvent.click(submitBtn)

    expect(searchSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        query: 'attention mechanisms in deep learning',
        scope_type: 'GLOBAL',
        top_k: 5,
        similarity_threshold: 0,
      }),
      expect.any(AbortSignal)
    )

    // Results header
    expect(await screen.findByText(/Attention Is All You Need/i)).toBeDefined()
    expect(screen.getByTestId('results-count').textContent).toContain('Found 2 ranked passages')

    // Passage 1
    expect(screen.getByText(/#1/i)).toBeDefined()
    expect(screen.getByText(/Attention Is All You Need/i)).toBeDefined()
    expect(screen.getByText(/Page 2/i)).toBeDefined()
    expect(screen.getByText(/Similarity: 88%/i)).toBeDefined()
    expect(screen.getByText(/Offsets: \[120–213\]/i)).toBeDefined()
    expect(screen.getByText(/Self-attention connects different positions/i)).toBeDefined()

    // Passage 2
    expect(screen.getByText(/#2/i)).toBeDefined()
    expect(screen.getByText(/bert_paper\.pdf/i)).toBeDefined()
    expect(screen.getByText(/Page 5/i)).toBeDefined()
    expect(screen.getByText(/Similarity: 72%/i)).toBeDefined()
    expect(screen.getByText(/Offsets: \[350–425\]/i)).toBeDefined()
  })

  it('switches to DOCUMENT scoped search, selects document and sends scope_id', async () => {
    const docResponse: SearchResponse = {
      query: 'recurrent connections',
      scope_type: 'DOCUMENT',
      scope_id: 'doc-uuid-1',
      total_results: 1,
      result_count: 1,
      results: [
        {
          chunk_id: 201,
          document_id: 'doc-uuid-1',
          document_title: 'Attention Is All You Need',
          page_number: 1,
          content: 'Recurrent models typically factor computation along the symbol positions.',
          similarity_score: 0.812,
          char_start: 50,
          char_end: 125,
        },
      ],
    }
    const searchSpy = vi.spyOn(api.search, 'query').mockResolvedValueOnce(docResponse)

    render(<SearchView />)

    // Wait for documents to load
    await waitFor(() => expect(screen.getByLabelText(/Document Scoped/i)).toBeDefined())

    // Switch to Document scope
    fireEvent.click(screen.getByLabelText(/Document Scoped/i))

    // Document select dropdown appears
    const docSelect = screen.getByLabelText(/Select target document/i)
    expect(docSelect).toBeDefined()
    fireEvent.change(docSelect, { target: { value: 'doc-uuid-1' } })

    const input = screen.getByPlaceholderText(/Ask a question or enter concepts/i)
    fireEvent.change(input, { target: { value: 'recurrent connections' } })
    fireEvent.click(screen.getByRole('button', { name: /Search Passages/i }))

    expect(searchSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        query: 'recurrent connections',
        scope_type: 'DOCUMENT',
        scope_id: 'doc-uuid-1',
      }),
      expect.any(AbortSignal)
    )

    expect(await screen.findByText(/Recurrent models typically factor computation/i)).toBeDefined()
    expect(screen.getByText(/Scope: Attention Is All You Need/i)).toBeDefined()
  })

  it('switches to COLLECTION scoped search and sends scope_id', async () => {
    const colResponse: SearchResponse = {
      query: 'attention layer',
      scope_type: 'COLLECTION',
      scope_id: 'col-uuid-1',
      total_results: 1,
      result_count: 1,
      results: [
        {
          chunk_id: 301,
          document_id: 'doc-uuid-1',
          document_title: 'Attention Is All You Need',
          page_number: 3,
          content: 'An attention function can be described as mapping a query and a set of key-value pairs.',
          similarity_score: 0.79,
          char_start: 0,
          char_end: 90,
        },
      ],
    }
    const searchSpy = vi.spyOn(api.search, 'query').mockResolvedValueOnce(colResponse)

    render(<SearchView />)

    fireEvent.click(screen.getByLabelText(/Collection Scoped/i))
    const colSelect = await screen.findByLabelText(/Select target collection/i)
    await screen.findByRole('option', { name: /Transformer Architectures/i })
    fireEvent.change(colSelect, { target: { value: 'col-uuid-1' } })

    const input = screen.getByPlaceholderText(/Ask a question or enter concepts/i)
    fireEvent.change(input, { target: { value: 'attention layer' } })
    fireEvent.click(screen.getByRole('button', { name: /Search Passages/i }))

    expect(searchSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        query: 'attention layer',
        scope_type: 'COLLECTION',
        scope_id: 'col-uuid-1',
      }),
      expect.any(AbortSignal)
    )

    expect(await screen.findByText(/An attention function can be described as mapping/i)).toBeDefined()
  })

  it('displays empty results state when vector search returns 0 matching passages', async () => {
    vi.spyOn(api.search, 'query').mockResolvedValueOnce({
      query: 'rare quantum topic',
      scope_type: 'GLOBAL',
      scope_id: null,
      total_results: 0,
      result_count: 0,
      results: [],
    })

    render(<SearchView />)

    const input = screen.getByPlaceholderText(/Ask a question or enter concepts/i)
    fireEvent.change(input, { target: { value: 'rare quantum topic' } })
    fireEvent.click(screen.getByRole('button', { name: /Search Passages/i }))

    expect(await screen.findByRole('heading', { name: /No Matching Passages Found/i })).toBeDefined()
    expect(screen.getByText(/Try rephrasing your search terms/i)).toBeDefined()
  })

  it('handles server error (503 Vector Store unavailable) with retry button', async () => {
    const searchSpy = vi.spyOn(api.search, 'query')
      .mockRejectedValueOnce(new ApiError(503, 'Vector search service is currently unavailable. Please try again later.'))
      .mockResolvedValueOnce(mockSearchSuccessResponse)

    render(<SearchView />)

    const input = screen.getByPlaceholderText(/Ask a question or enter concepts/i)
    fireEvent.change(input, { target: { value: 'test query' } })
    fireEvent.click(screen.getByRole('button', { name: /Search Passages/i }))

    expect(await screen.findByText(/Vector search service is currently unavailable/i)).toBeDefined()

    // Click retry
    const retryBtn = screen.getByRole('button', { name: /Retry/i })
    fireEvent.click(retryBtn)

    expect(await screen.findByText(/Attention Is All You Need/i)).toBeDefined()
    expect(screen.getByTestId('results-count').textContent).toContain('Found 2 ranked passages')
    expect(searchSpy).toHaveBeenCalledTimes(2)
  })

  it('handles document not ready error (400) gracefully', async () => {
    vi.spyOn(api.search, 'query').mockRejectedValueOnce(
      new ApiError(400, "Target document 'paper.pdf' is currently in status 'EXTRACTING' and not ready for queries.")
    )

    render(<SearchView />)

    fireEvent.click(screen.getByLabelText(/Document Scoped/i))
    const docSelect = await screen.findByLabelText(/Select target document/i)
    fireEvent.change(docSelect, { target: { value: 'doc-uuid-1' } })

    const input = screen.getByPlaceholderText(/Ask a question or enter concepts/i)
    fireEvent.change(input, { target: { value: 'test query' } })
    fireEvent.click(screen.getByRole('button', { name: /Search Passages/i }))

    expect(await screen.findByText(/is currently in status 'EXTRACTING' and not ready for queries/i)).toBeDefined()
  })

  it('prevents stale out-of-order responses from overwriting results for a newer query', async () => {
    let resolveFirstQuery: (res: SearchResponse) => void = () => {}
    const firstPromise = new Promise<SearchResponse>((resolve) => {
      resolveFirstQuery = resolve
    })

    const secondResponse: SearchResponse = {
      query: 'second faster query',
      scope_type: 'GLOBAL',
      scope_id: null,
      total_results: 1,
      result_count: 1,
      results: [
        {
          chunk_id: 999,
          document_id: 'doc-uuid-1',
          document_title: 'Attention Is All You Need',
          page_number: 1,
          content: 'NEWEST RESPONSE CONTENT',
          similarity_score: 0.95,
          char_start: 0,
          char_end: 23,
        },
      ],
    }

    vi.spyOn(api.search, 'query')
      .mockImplementationOnce(() => firstPromise)
      .mockResolvedValueOnce(secondResponse)

    render(<SearchView />)

    const input = screen.getByPlaceholderText(/Ask a question or enter concepts/i)
    const submitBtn = screen.getByRole('button', { name: /Search Passages/i })

    // Fire first query (slow)
    fireEvent.change(input, { target: { value: 'first slow query' } })
    fireEvent.click(submitBtn)

    // Fire second query (fast)
    fireEvent.change(input, { target: { value: 'second faster query' } })
    fireEvent.click(submitBtn)

    // Second response arrives first and renders
    expect(await screen.findByText(/NEWEST RESPONSE CONTENT/i)).toBeDefined()

    // First response finally resolves late with older stale data
    resolveFirstQuery({
      query: 'first slow query',
      scope_type: 'GLOBAL',
      scope_id: null,
      total_results: 1,
      result_count: 1,
      results: [
        {
          chunk_id: 111,
          document_id: 'doc-uuid-1',
          document_title: 'Attention Is All You Need',
          page_number: 1,
          content: 'STALE OBSOLETE CONTENT THAT SHOULD BE DISCARDED',
          similarity_score: 0.1,
          char_start: 0,
          char_end: 45,
        },
      ],
    })

    // Assert that the newer content is preserved and the stale content NEVER appears
    await waitFor(() => {
      expect(screen.queryByText(/STALE OBSOLETE CONTENT/i)).toBeNull()
      expect(screen.getByText(/NEWEST RESPONSE CONTENT/i)).toBeDefined()
    })
  })

  it('renders resolved document filename or title and falls back only when metadata is missing', async () => {
    const titleTestResponse: SearchResponse = {
      query: 'deep learning representations',
      scope_type: 'GLOBAL',
      scope_id: null,
      total_results: 3,
      result_count: 3,
      results: [
        {
          chunk_id: 1,
          document_id: 'doc-uuid-1',
          document_title: 'Attention Is All You Need',
          page_number: 1,
          content: 'Passage from paper with title.',
          similarity_score: 0.9,
          char_start: 0,
          char_end: 30,
        },
        {
          chunk_id: 2,
          document_id: 'doc-uuid-2',
          document_title: 'Untitled Document',
          page_number: 2,
          content: 'Passage where title was null but filename exists in documents.',
          similarity_score: 0.8,
          char_start: 0,
          char_end: 40,
        },
        {
          chunk_id: 3,
          document_id: 'doc-unknown',
          document_title: 'Untitled Document',
          page_number: 3,
          content: 'Passage where both title and document are genuinely missing.',
          similarity_score: 0.7,
          char_start: 0,
          char_end: 45,
        },
      ],
    }

    vi.spyOn(api.search, 'query').mockResolvedValueOnce(titleTestResponse)

    render(<SearchView />)

    const input = screen.getByPlaceholderText(/Ask a question or enter concepts/i)
    fireEvent.change(input, { target: { value: 'deep learning representations' } })
    fireEvent.click(screen.getByRole('button', { name: /Search Passages/i }))

    // Doc 1: explicitly titled
    expect(await screen.findByText(/Attention Is All You Need/i)).toBeDefined()
    // Doc 2: title was null, resolved to filename 'bert_paper.pdf'
    expect(screen.getByText(/bert_paper\.pdf/i)).toBeDefined()
    // Doc 3: genuinely missing metadata, falls back to 'Untitled Document'
    expect(screen.getByText(/Untitled Document/i)).toBeDefined()
  })
})
