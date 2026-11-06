import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import App from './App'
import { api } from './services/api'

describe('ResearchLens AI — Application Shell & Template Architecture', () => {
  beforeEach(() => {
    window.location.hash = ''
    window.scrollTo = vi.fn()
    vi.restoreAllMocks()
  })

  it('renders application shell with branding, navigation tabs, hero section, and core architecture', async () => {
    render(<App />)

    // Brand and announcement banner
    expect(screen.getAllByText(/researchlens/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/GROUNDED RAG/i)).toBeDefined()

    // Navigation items inside main navigation bar
    const nav = screen.getByRole('navigation', { name: /Main Navigation/i })
    expect(within(nav).getByRole('button', { name: 'Explore' })).toBeDefined()
    expect(within(nav).getByRole('button', { name: 'Documents' })).toBeDefined()
    expect(within(nav).getByRole('button', { name: 'Search' })).toBeDefined()
    expect(within(nav).getByRole('button', { name: /Research Chat/i })).toBeDefined()
    expect(within(nav).getByRole('button', { name: 'Collections' })).toBeDefined()
    expect(within(nav).getByRole('button', { name: /API & MCP/i })).toBeDefined()

    // Hero headline and primary CTA
    expect(screen.getByRole('heading', { level: 1 }).textContent).toMatch(/Turn scientific literature/i)
    expect(screen.getByRole('button', { name: /Upload Paper/i })).toBeDefined()

    // Capabilities Grid
    expect(screen.getByRole('heading', { name: /A complete academic retrieval ecosystem/i })).toBeDefined()
    expect(screen.getByText(/Dense Semantic Vector Search/i)).toBeDefined()
  })

  it('switches between tabs: Documents, Search, Research Chat, Collections, and API & MCP', async () => {
    render(<App />)

    const nav = screen.getByRole('navigation', { name: /Main Navigation/i })

    // Switch to Documents tab
    const docsTab = within(nav).getByRole('button', { name: 'Documents' })
    fireEvent.click(docsTab)
    expect(await screen.findByRole('heading', { name: /Research Documents/i })).toBeDefined()
    expect(window.location.hash).toBe('#documents')

    // Switch to Search tab
    const searchTab = within(nav).getByRole('button', { name: 'Search' })
    fireEvent.click(searchTab)
    expect(await screen.findByRole('heading', { name: /Semantic Passage Search/i })).toBeDefined()
    expect(window.location.hash).toBe('#search')

    // Switch to Research Chat tab
    const chatTab = within(nav).getByRole('button', { name: /Research Chat/i })
    fireEvent.click(chatTab)
    expect(await screen.findByRole('heading', { name: /Conversations/i })).toBeDefined()
    expect(window.location.hash).toBe('#chat')

    // Switch to Collections tab
    const collectionsTab = within(nav).getByRole('button', { name: 'Collections' })
    fireEvent.click(collectionsTab)
    expect(await screen.findByRole('heading', { name: /Research Collections/i })).toBeDefined()
    expect(window.location.hash).toBe('#collections')

    // Switch to API & MCP tab
    const mcpTab = within(nav).getByRole('button', { name: /API & MCP/i })
    fireEvent.click(mcpTab)
    expect(await screen.findByRole('heading', { name: /Connect ResearchLens AI to Claude Desktop & Cursor/i })).toBeDefined()
    expect(window.location.hash).toBe('#mcp')
  })

  it('opens and closes the PDF Upload modal via header button', async () => {
    render(<App />)

    // Click "+ Upload Paper" in header
    const uploadBtn = screen.getByRole('button', { name: /Upload Paper/i })
    fireEvent.click(uploadBtn)

    // Verify modal is open
    expect(screen.getByRole('heading', { name: /Upload Research PDF/i })).toBeDefined()
    expect(screen.getByText(/Select PDF Document/i)).toBeDefined()

    // Cancel / Close modal
    const cancelBtn = screen.getByRole('button', { name: /Cancel/i })
    fireEvent.click(cancelBtn)
    expect(screen.queryByRole('heading', { name: /Upload Research PDF/i })).toBeNull()
  })

  it('selects research engine and updates specifications in PromoHero', async () => {
    render(<App />)

    // Click on Meta FAISS IndexFlatIP selector card
    const faissCard = screen.getByRole('button', { name: /Meta FAISS IndexFlatIP/i })
    fireEvent.click(faissCard)

    // Active architecture updates
    expect(screen.getByText(/Active Architecture: Meta FAISS IndexFlatIP/i)).toBeDefined()
    expect(screen.getByText(/FAISS IndexFlatIP \(Dense\)/i)).toBeDefined()
  })

  it('opens Research Parameter Studio Modal, executes inquiry, and displays citation evidence', async () => {
    vi.spyOn(api.search, 'query').mockResolvedValue({
      query: 'Self-Attention vs. Recurrence Computational Scaling',
      scope_type: 'GLOBAL',
      scope_id: null,
      document_ids: null,
      total_results: 1,
      result_count: 1,
      results: [
        {
          chunk_id: 1,
          document_id: 'doc-uuid-1',
          document_title: 'Attention Is All You Need (Vaswani et al.)',
          page_number: 6,
          content: 'Based on the retrieved primary literature, computational complexity per layer scales as O(n² · d) for self-attention, whereas recurrent layers scale as O(n · d²).',
          similarity_score: 0.92,
          char_start: 100,
          char_end: 250,
        },
      ],
    })

    render(<App />)

    // Find and click a Run Inquiry button from presets
    const runInquiryButtons = screen.getAllByRole('button', { name: /Run Inquiry/i })
    expect(runInquiryButtons.length).toBeGreaterThan(0)
    fireEvent.click(runInquiryButtons[0])

    // Research Modal is visible
    expect(screen.getByRole('heading', { name: /ResearchLens AI — Grounded Parameter & Inquiry Studio/i })).toBeDefined()
    const dialog = screen.getByRole('dialog')

    // Click Execute with ResearchLens
    const executeBtn = within(dialog).getByRole('button', { name: /Execute with ResearchLens/i })
    fireEvent.click(executeBtn)

    // Progress bar updates
    expect(within(dialog).getByText(/Retrieving Dense Passages/i)).toBeDefined()

    await waitFor(
      () => {
        expect(within(dialog).getByText(/Grounded Synthesis/i)).toBeDefined()
      },
      { timeout: 3000 }
    )

    // Verify citation evidence is rendered
    expect(within(dialog).getByText(/Page 6/i)).toBeDefined()

    // Close modal
    const closeBtn = within(dialog).getByRole('button', { name: /Close modal/i })
    fireEvent.click(closeBtn)
    expect(screen.queryByRole('heading', { name: /ResearchLens AI — Grounded Parameter & Inquiry Studio/i })).toBeNull()
  })

  it('filters academic inquiry presets by category', async () => {
    render(<App />)

    const transformersFilter = screen.getAllByRole('button', { name: 'Transformers' })[0]
    fireEvent.click(transformersFilter)

    expect(screen.getByText(/Self-Attention vs. Recurrence Computational Scaling/i)).toBeDefined()
  })

  it('copies ResearchLens MCP configuration in API & MCP tab', async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined)
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextMock,
      },
    })

    render(<App />)

    // Navigate to MCP
    const nav = screen.getByRole('navigation', { name: /Main Navigation/i })
    const mcpTab = within(nav).getByRole('button', { name: /^API & MCP/i })
    fireEvent.click(mcpTab)

    // Click Copy Config
    const copyBtn = screen.getByRole('button', { name: /Copy Config/i })
    fireEvent.click(copyBtn)

    expect(writeTextMock).toHaveBeenCalledWith(expect.stringContaining('researchlens'))
    expect(screen.getAllByText(/Copied/i).length).toBeGreaterThan(0)
  })

  it('allows user to subscribe to academic intelligence newsletter in footer', async () => {
    render(<App />)

    const emailInput = screen.getByPlaceholderText(/Enter your academic email/i)
    const subscribeBtn = screen.getByRole('button', { name: /Subscribe/i })

    fireEvent.change(emailInput, { target: { value: 'researcher@stanford.edu' } })
    fireEvent.click(subscribeBtn)

    expect(screen.getByText(/Subscribed! Check your inbox for benchmark releases\./i)).toBeDefined()
  })
})
