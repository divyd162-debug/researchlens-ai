import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { api, buildApiUrl, normalizeBaseUrl } from './api'

describe('API URL Resolution & Normalization (ENV-01)', () => {
  describe('normalizeBaseUrl', () => {
    it('defaults to /api/v1 when input is undefined, null, empty, or whitespace', () => {
      expect(normalizeBaseUrl(undefined)).toBe('/api/v1')
      expect(normalizeBaseUrl(null)).toBe('/api/v1')
      expect(normalizeBaseUrl('')).toBe('/api/v1')
      expect(normalizeBaseUrl('   ')).toBe('/api/v1')
    })

    it('appends /api/v1 when given host-only base without API prefix', () => {
      expect(normalizeBaseUrl('http://localhost:8000')).toBe('http://localhost:8000/api/v1')
      expect(normalizeBaseUrl('http://localhost:8000/')).toBe('http://localhost:8000/api/v1')
      expect(normalizeBaseUrl('https://api.example.com')).toBe('https://api.example.com/api/v1')
    })

    it('appends /v1 when given base ending in /api without version', () => {
      expect(normalizeBaseUrl('http://localhost:8000/api')).toBe('http://localhost:8000/api/v1')
      expect(normalizeBaseUrl('http://localhost:8000/api/')).toBe('http://localhost:8000/api/v1')
    })

    it('preserves existing /api/v1 suffix and strips trailing slash cleanly', () => {
      expect(normalizeBaseUrl('http://localhost:8000/api/v1')).toBe('http://localhost:8000/api/v1')
      expect(normalizeBaseUrl('http://localhost:8000/api/v1/')).toBe('http://localhost:8000/api/v1')
      expect(normalizeBaseUrl('/api/v1')).toBe('/api/v1')
      expect(normalizeBaseUrl('/api/v1/')).toBe('/api/v1')
    })
  })

  describe('buildApiUrl', () => {
    it('builds standard endpoints using default base URL (/api/v1)', () => {
      expect(buildApiUrl('/auth/login')).toBe('/api/v1/auth/login')
      expect(buildApiUrl('/search')).toBe('/api/v1/search')
      expect(buildApiUrl('/documents')).toBe('/api/v1/documents')
      expect(buildApiUrl('/chat/sessions?limit=50&offset=0')).toBe('/api/v1/chat/sessions?limit=50&offset=0')
    })

    it('builds correct URLs when explicit host base URL is configured', () => {
      expect(buildApiUrl('/auth/login', 'http://localhost:8000')).toBe('http://localhost:8000/api/v1/auth/login')
      expect(buildApiUrl('/search', 'http://localhost:8000/api/v1')).toBe('http://localhost:8000/api/v1/search')
      expect(buildApiUrl('/documents', 'http://localhost:8000/api/v1/')).toBe('http://localhost:8000/api/v1/documents')
    })

    it('safely handles endpoints without leading slash', () => {
      expect(buildApiUrl('auth/login', 'http://localhost:8000')).toBe('http://localhost:8000/api/v1/auth/login')
    })
  })

  describe('API Client Request Dispatching', () => {
    const originalFetch = window.fetch

    beforeEach(() => {
      window.fetch = vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({}),
      } as Response)
    })

    afterEach(() => {
      window.fetch = originalFetch
    })

    it('dispatches login request to /api/v1/auth/login', async () => {
      await api.auth.login({ email: 'test@example.com', password: 'password123' })
      expect(window.fetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/api\/v1\/auth\/login$/),
        expect.objectContaining({
          method: 'POST',
        })
      )
    })

    it('dispatches document listing request to /api/v1/documents with pagination query', async () => {
      await api.documents.list(20, 10)
      expect(window.fetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/api\/v1\/documents\?limit=20&offset=10$/),
        expect.objectContaining({
          method: 'GET',
        })
      )
    })

    it('dispatches search request to /api/v1/search', async () => {
      await api.search.query({ query: 'quantum computing', scope_type: 'GLOBAL' })
      expect(window.fetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/api\/v1\/search$/),
        expect.objectContaining({
          method: 'POST',
        })
      )
    })

    it('dispatches addDocuments request to /api/v1/collections/:id/documents', async () => {
      await api.collections.addDocuments('col-123', { document_ids: ['doc-1', 'doc-2'] })
      expect(window.fetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/api\/v1\/collections\/col-123\/documents$/),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ document_ids: ['doc-1', 'doc-2'] }),
        })
      )
    })

    it('dispatches removeDocument request to /api/v1/collections/:id/documents/:docId', async () => {
      await api.collections.removeDocument('col-123', 'doc-456')
      expect(window.fetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/api\/v1\/collections\/col-123\/documents\/doc-456$/),
        expect.objectContaining({
          method: 'DELETE',
        })
      )
    })

    it('dispatches listDocuments request to /api/v1/collections/:id/documents with pagination', async () => {
      await api.collections.listDocuments('col-123', 25, 5)
      expect(window.fetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/api\/v1\/collections\/col-123\/documents\?limit=25&offset=5$/),
        expect.objectContaining({
          method: 'GET',
        })
      )
    })

    it('dispatches grounded RAG query request to /api/v1/query', async () => {
      await api.query.rag({
        query: 'What are transformer self-attention dimensions?',
        scope_type: 'GLOBAL',
        top_k: 5,
        similarity_threshold: 0.6,
      })
      expect(window.fetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/api\/v1\/query$/),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            query: 'What are transformer self-attention dimensions?',
            scope_type: 'GLOBAL',
            top_k: 5,
            similarity_threshold: 0.6,
          }),
        })
      )
    })
  })
})
