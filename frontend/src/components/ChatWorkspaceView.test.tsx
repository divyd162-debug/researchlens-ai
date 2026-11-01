import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { ChatWorkspaceView } from './ChatWorkspaceView'
import { api, ApiError } from '../services/api'
import {
  ChatSessionResponse,
  ChatMessageListResponse,
  ChatTurnResponse,
} from '../types'

const mockSessions: ChatSessionResponse[] = [
  {
    id: 'session-uuid-1',
    title: 'Attention Mechanism Analysis',
    scope_type: 'GLOBAL',
    target_document_id: null,
    target_collection_id: null,
    is_archived: false,
    created_at: '2026-10-19T10:00:00Z',
    updated_at: '2026-10-19T10:30:00Z',
    message_count: 2,
  },
  {
    id: 'session-uuid-2',
    title: 'Deep Learning Optimization',
    scope_type: 'DOCUMENT',
    target_document_id: 'doc-uuid-1',
    target_collection_id: null,
    is_archived: false,
    created_at: '2026-10-19T11:00:00Z',
    updated_at: '2026-10-19T11:15:00Z',
    message_count: 0,
  },
  {
    id: 'session-uuid-3',
    title: 'Archived Survey',
    scope_type: 'COLLECTION',
    target_document_id: null,
    target_collection_id: 'col-uuid-1',
    is_archived: true,
    created_at: '2026-10-18T09:00:00Z',
    updated_at: '2026-10-18T09:45:00Z',
    message_count: 4,
  },
]

const mockSession1Messages: ChatMessageListResponse = {
  session_id: 'session-uuid-1',
  total: 2,
  items: [
    {
      id: 'msg-uuid-1',
      chat_session_id: 'session-uuid-1',
      sender_type: 'USER',
      content: 'What is self-attention?',
      token_count: null,
      created_at: '2026-10-19T10:00:00Z',
      citations: [],
    },
    {
      id: 'msg-uuid-2',
      chat_session_id: 'session-uuid-1',
      sender_type: 'ASSISTANT',
      content: 'Self-attention connects different positions of a single sequence to compute a representation.',
      token_count: 25,
      created_at: '2026-10-19T10:00:05Z',
      citations: [
        {
          id: 'cit-uuid-1',
          message_id: 'msg-uuid-2',
          chunk_id: 101,
          quoted_text: 'Self-attention allows the model to associate each word with other words.',
          page_number: 3,
          document_title: 'Attention Is All You Need.pdf',
          similarity_score: 0.92,
          created_at: '2026-10-19T10:00:05Z',
        },
      ],
    },
  ],
}

const mockTurnResponse: ChatTurnResponse = {
  session_id: 'session-uuid-1',
  user_message: {
    id: 'msg-uuid-user-new',
    chat_session_id: 'session-uuid-1',
    sender_type: 'USER',
    content: 'What is transformer architecture?',
    token_count: null,
    created_at: '2026-10-19T12:00:00Z',
    citations: [],
  },
  assistant_message: {
    id: 'msg-uuid-asst-new',
    chat_session_id: 'session-uuid-1',
    sender_type: 'ASSISTANT',
    content: 'The Transformer is a deep learning model based solely on attention mechanisms.',
    token_count: 35,
    created_at: '2026-10-19T12:00:03Z',
    citations: [
      {
        id: 'cit-uuid-turn-1',
        message_id: 'msg-uuid-asst-new',
        chunk_id: 201,
        quoted_text: 'The Transformer is the first transduction model relying entirely on self-attention.',
        page_number: 1,
        document_title: 'Attention Is All You Need.pdf',
        similarity_score: 0.88,
        created_at: '2026-10-19T12:00:03Z',
      },
    ],
  },
  retrieved_chunk_count: 1,
  model_name: 'gemini-1.5-pro',
  provider_name: 'google',
  prompt_tokens: 150,
  completion_tokens: 35,
  total_tokens: 185,
}

describe('ChatWorkspaceView Component', () => {
  beforeEach(() => {
    window.location.hash = '#chat'
    vi.restoreAllMocks()
  })

  it('renders loading spinner initially, then displays session list and total count', async () => {
    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    render(<ChatWorkspaceView />)

    // Spinner visible initially
    expect(screen.getByLabelText(/loading conversations/i)).toBeDefined()

    // Loaded sessions list
    expect(await screen.findByText('Attention Mechanism Analysis')).toBeDefined()
    expect(screen.getByText('Deep Learning Optimization')).toBeDefined()
    expect(screen.getByText('Archived Survey')).toBeDefined()

    // Total count badge
    expect(screen.getByTitle('Total chat sessions').textContent).toBe('3')
  })

  it('displays empty state when user has zero chat sessions', async () => {
    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: [],
      total: 0,
      limit: 100,
      offset: 0,
    })

    render(<ChatWorkspaceView />)

    expect(await screen.findByText(/No conversations found/i)).toBeDefined()
    expect(screen.getByRole('button', { name: /start first chat/i })).toBeDefined()
  })

  it('handles session list API failure with inline retry button', async () => {
    vi.spyOn(api.chat, 'listSessions').mockRejectedValueOnce(
      new ApiError(500, 'Server connection failure while listing sessions.')
    )

    render(<ChatWorkspaceView />)

    expect(
      await screen.findByText('Server connection failure while listing sessions.')
    ).toBeDefined()

    // Retry recovery
    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    fireEvent.click(screen.getByRole('button', { name: /retry/i }))
    expect(await screen.findByText('Attention Mechanism Analysis')).toBeDefined()
  })

  it('creates a new GLOBAL chat session, selects it, and updates URL hash', async () => {
    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    const newSession: ChatSessionResponse = {
      id: 'session-new-uuid',
      title: 'New Research Chat',
      scope_type: 'GLOBAL',
      target_document_id: null,
      target_collection_id: null,
      is_archived: false,
      created_at: '2026-10-20T02:00:00Z',
      updated_at: '2026-10-20T02:00:00Z',
      message_count: 0,
    }

    vi.spyOn(api.chat, 'createSession').mockResolvedValueOnce(newSession)
    vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce({
      session_id: 'session-new-uuid',
      items: [],
      total: 0,
    })

    render(<ChatWorkspaceView />)
    await screen.findByText('Attention Mechanism Analysis')

    const newChatBtn = screen.getByRole('button', { name: /create new conversation/i })
    fireEvent.click(newChatBtn)

    const createBtn = await screen.findByRole('button', { name: /create conversation/i })
    fireEvent.click(createBtn)

    // Verify creation API call payload
    expect(api.chat.createSession).toHaveBeenCalledWith({
      scope_type: 'GLOBAL',
      title: 'New Research Chat',
    })

    // Verify session appeared in the list and is selected
    expect(await screen.findByRole('heading', { level: 2, name: 'New Research Chat' })).toBeDefined()
    expect(window.location.hash).toBe('#chat?session=session-new-uuid')

    // Verify empty messages note for new session
    expect(screen.getByText(/This conversation is currently empty/i)).toBeDefined()
  })

  it('handles create session API failure without breaking session list', async () => {
    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    vi.spyOn(api.chat, 'createSession').mockRejectedValueOnce(
      new ApiError(400, 'Unable to create session due to invalid scope.')
    )

    render(<ChatWorkspaceView />)
    await screen.findByText('Attention Mechanism Analysis')

    const newChatBtn = screen.getByRole('button', { name: /create new conversation/i })
    fireEvent.click(newChatBtn)

    const createBtn = await screen.findByRole('button', { name: /create conversation/i })
    fireEvent.click(createBtn)

    expect(await screen.findByText('Unable to create session due to invalid scope.')).toBeDefined()
  })

  it('creates a DOCUMENTS multi-document session with selected documents', async () => {
    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    const mockDocs = [
      {
        id: 'doc-uuid-a',
        filename: 'paper_a.pdf',
        title: 'Attention Paper A',
        file_size_bytes: 1024,
        page_count: 10,
        processing_status: 'READY',
        created_at: '2026-10-20T00:00:00Z',
        updated_at: '2026-10-20T00:00:00Z',
      },
      {
        id: 'doc-uuid-b',
        filename: 'paper_b.pdf',
        title: 'Deep Learning Paper B',
        file_size_bytes: 2048,
        page_count: 15,
        processing_status: 'READY',
        created_at: '2026-10-20T00:00:00Z',
        updated_at: '2026-10-20T00:00:00Z',
      },
      {
        id: 'doc-uuid-c',
        filename: 'paper_c.pdf',
        title: 'Pending Paper C',
        file_size_bytes: 512,
        page_count: 5,
        processing_status: 'PENDING',
        created_at: '2026-10-20T00:00:00Z',
        updated_at: '2026-10-20T00:00:00Z',
      },
    ]

    vi.spyOn(api.documents, 'list').mockResolvedValueOnce({
      items: mockDocs,
      total: 3,
      limit: 100,
      offset: 0,
    })

    const multiDocSession: ChatSessionResponse = {
      id: 'session-multi-uuid',
      title: 'Comparative Survey',
      scope_type: 'DOCUMENTS',
      target_document_id: null,
      target_collection_id: null,
      is_archived: false,
      created_at: '2026-10-20T03:00:00Z',
      updated_at: '2026-10-20T03:00:00Z',
      message_count: 0,
      selected_documents: [
        {
          id: 'doc-uuid-a',
          filename: 'paper_a.pdf',
          title: 'Attention Paper A',
          page_count: 10,
          processing_status: 'READY',
        },
        {
          id: 'doc-uuid-b',
          filename: 'paper_b.pdf',
          title: 'Deep Learning Paper B',
          page_count: 15,
          processing_status: 'READY',
        },
      ],
    }

    vi.spyOn(api.chat, 'createSession').mockResolvedValueOnce(multiDocSession)
    vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce({
      session_id: 'session-multi-uuid',
      items: [],
      total: 0,
    })

    render(<ChatWorkspaceView />)
    await screen.findByText('Attention Mechanism Analysis')

    // Open creation modal
    fireEvent.click(screen.getByRole('button', { name: /create new conversation/i }))

    // Fill title
    const titleInput = await screen.findByLabelText(/conversation title/i)
    fireEvent.change(titleInput, { target: { value: 'Comparative Survey' } })

    // Select "Selected Documents" scope
    const multiDocOption = screen.getByText('Selected Documents')
    fireEvent.click(multiDocOption)

    // Select ready documents
    const docA = await screen.findByLabelText(/Select Attention Paper A/i)
    const docB = screen.getByLabelText(/Select Deep Learning Paper B/i)
    const docC = screen.getByLabelText(/Select Pending Paper C/i)

    expect(docC).toHaveProperty('disabled', true)

    fireEvent.click(docA)
    fireEvent.click(docB)

    // Submit
    const createBtn = screen.getByRole('button', { name: /create conversation/i })
    fireEvent.click(createBtn)

    await waitFor(() => {
      expect(api.chat.createSession).toHaveBeenCalledWith({
        title: 'Comparative Survey',
        scope_type: 'DOCUMENTS',
        document_ids: ['doc-uuid-a', 'doc-uuid-b'],
      })
    })

    // Header displays multi-doc scope
    expect(await screen.findByText(/Scope: 2 Documents ▾/i)).toBeDefined()
  })

  it('displays multi-document scope popover with selected document list', async () => {
    const multiDocSession: ChatSessionResponse = {
      id: 'session-multi-uuid-2',
      title: 'Multi-Doc Analysis',
      scope_type: 'DOCUMENTS',
      target_document_id: null,
      target_collection_id: null,
      is_archived: false,
      created_at: '2026-10-20T03:00:00Z',
      updated_at: '2026-10-20T03:00:00Z',
      message_count: 0,
      selected_documents: [
        {
          id: 'doc-uuid-1',
          filename: 'Doc1.pdf',
          title: 'Document One Title',
          page_count: 5,
          processing_status: 'READY',
        },
        {
          id: 'doc-uuid-2',
          filename: 'Doc2.pdf',
          title: 'Document Two Title',
          page_count: 12,
          processing_status: 'READY',
        },
      ],
    }

    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: [multiDocSession],
      total: 1,
      limit: 100,
      offset: 0,
    })

    vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce({
      session_id: 'session-multi-uuid-2',
      items: [],
      total: 0,
    })

    render(<ChatWorkspaceView />)
    const sessionItem = await screen.findByRole('button', {
      name: /Chat session: Multi-Doc Analysis/i,
    })
    fireEvent.click(sessionItem)

    // Scope button visible
    const popoverBtn = await screen.findByRole('button', { name: /view selected documents/i })
    expect(popoverBtn.textContent).toContain('Scope: 2 Documents ▾')

    // Click to open popover
    fireEvent.click(popoverBtn)

    // Popover content visible
    expect(await screen.findByRole('dialog', { name: /selected documents list/i })).toBeDefined()
    expect(screen.getByText('Document One Title')).toBeDefined()
    expect(screen.getByText('Document Two Title')).toBeDefined()
    expect(screen.getByText('(5 pages)')).toBeDefined()
    expect(screen.getByText('(12 pages)')).toBeDefined()
  })

  it('selects an existing session and loads its messages and citation snapshots', async () => {
    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)

    render(<ChatWorkspaceView />)
    const sessionItem = await screen.findByRole('button', {
      name: /Chat session: Attention Mechanism Analysis/i,
    })

    fireEvent.click(sessionItem)

    // Message query executed
    expect(api.chat.getSessionMessages).toHaveBeenCalledWith(
      'session-uuid-1',
      expect.any(AbortSignal)
    )

    // Header displayed
    expect(await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })).toBeDefined()
    expect(window.location.hash).toBe('#chat?session=session-uuid-1')

    // Messages displayed
    expect(screen.getByText('What is self-attention?')).toBeDefined()
    expect(screen.getByText(/Self-attention connects different positions/i)).toBeDefined()

    // Citations displayed
    expect(screen.getByText(/1 Verified Evidence Citation/i)).toBeDefined()
    expect(screen.getByText(/Attention Is All You Need\.pdf/i)).toBeDefined()
    expect(screen.getByText('Page 3')).toBeDefined()
    expect(screen.getByText('92% match')).toBeDefined()
    expect(
      screen.getByText(/Self-attention allows the model to associate each word with other words\./i)
    ).toBeDefined()
  })

  it('renames a session with inline edit form and updates the list', async () => {
    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    render(<ChatWorkspaceView />)
    await screen.findByText('Attention Mechanism Analysis')

    const renameBtn = screen.getByRole('button', {
      name: 'Rename Attention Mechanism Analysis',
    })
    fireEvent.click(renameBtn)

    const input = screen.getByLabelText(/edit conversation title/i)
    fireEvent.change(input, { target: { value: 'Transformers Deep Dive' } })

    const updatedSession: ChatSessionResponse = {
      ...mockSessions[0],
      title: 'Transformers Deep Dive',
    }
    vi.spyOn(api.chat, 'updateSession').mockResolvedValueOnce(updatedSession)

    const saveBtn = screen.getByRole('button', { name: /save/i })
    fireEvent.click(saveBtn)

    expect(api.chat.updateSession).toHaveBeenCalledWith('session-uuid-1', {
      title: 'Transformers Deep Dive',
    })

    expect(await screen.findByText('Transformers Deep Dive')).toBeDefined()
  })

  it('archives a session and toggles filter views', async () => {
    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    render(<ChatWorkspaceView />)
    await screen.findByText('Attention Mechanism Analysis')

    const archiveBtn = screen.getByRole('button', {
      name: 'Archive Attention Mechanism Analysis',
    })

    const archivedSession: ChatSessionResponse = {
      ...mockSessions[0],
      is_archived: true,
    }
    vi.spyOn(api.chat, 'updateSession').mockResolvedValueOnce(archivedSession)

    fireEvent.click(archiveBtn)

    await waitFor(() => {
      expect(api.chat.updateSession).toHaveBeenCalledWith('session-uuid-1', {
        is_archived: true,
      })
    })

    // Filter by Active: session-uuid-1 should now be excluded
    const activeTab = screen.getByRole('tab', { name: /^active$/i })
    fireEvent.click(activeTab)

    expect(screen.queryByText('Attention Mechanism Analysis')).toBeNull()
    expect(screen.getByText('Deep Learning Optimization')).toBeDefined()

    // Filter by Archived: session-uuid-1 and session-uuid-3 should be listed
    const archivedTab = screen.getByRole('tab', { name: /^archived$/i })
    fireEvent.click(archivedTab)

    expect(screen.getByText('Attention Mechanism Analysis')).toBeDefined()
    expect(screen.getByText('Archived Survey')).toBeDefined()
  })

  it('restores selected session from URL hash on mount', async () => {
    window.location.hash = '#chat?session=session-uuid-2'

    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce({
      session_id: 'session-uuid-2',
      items: [],
      total: 0,
    })

    render(<ChatWorkspaceView />)

    expect(await screen.findByRole('heading', { level: 2, name: 'Deep Learning Optimization' })).toBeDefined()
    expect(api.chat.getSessionMessages).toHaveBeenCalledWith('session-uuid-2', expect.any(AbortSignal))
  })

  it('handles invalid session ID in URL hash with clear error state and recovery button', async () => {
    window.location.hash = '#chat?session=invalid-uuid-999'

    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    vi.spyOn(api.chat, 'getSessionMessages').mockRejectedValueOnce(
      new ApiError(404, 'Chat session not found.')
    )

    render(<ChatWorkspaceView />)

    expect(await screen.findByRole('heading', { level: 3, name: 'Conversation Not Available' })).toBeDefined()
    expect(screen.getByText('Chat session not found.')).toBeDefined()

    const backBtn = screen.getByRole('button', { name: /back to conversations/i })
    fireEvent.click(backBtn)

    expect(window.location.hash).toBe('#chat')
    expect(await screen.findByRole('heading', { level: 3, name: 'Select a Conversation' })).toBeDefined()
  })

  it('clears selected session upon browser back-navigation to #chat', async () => {
    window.location.hash = '#chat?session=session-uuid-1'

    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)

    render(<ChatWorkspaceView />)
    expect(await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })).toBeDefined()

    // Simulate browser Back button navigating to #chat
    window.location.hash = '#chat'
    window.dispatchEvent(new HashChangeEvent('hashchange'))

    expect(await screen.findByRole('heading', { level: 3, name: 'Select a Conversation' })).toBeDefined()
  })

  it('prevents stale response race when rapidly switching sessions', async () => {
    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    let resolveSession1!: (value: ChatMessageListResponse) => void
    const session1Promise = new Promise<ChatMessageListResponse>((resolve) => {
      resolveSession1 = resolve
    })

    const session2Messages: ChatMessageListResponse = {
      session_id: 'session-uuid-2',
      total: 1,
      items: [
        {
          id: 'msg-s2-1',
          chat_session_id: 'session-uuid-2',
          sender_type: 'USER',
          content: 'Session 2 unique question',
          token_count: null,
          created_at: '2026-10-19T11:00:00Z',
          citations: [],
        },
      ],
    }

    vi.spyOn(api.chat, 'getSessionMessages')
      .mockReturnValueOnce(session1Promise)
      .mockResolvedValueOnce(session2Messages)

    render(<ChatWorkspaceView />)
    await screen.findByText('Attention Mechanism Analysis')

    // Click Session 1 (starts slow promise)
    const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
    fireEvent.click(session1Btn)

    // Click Session 2 immediately (resolves fast)
    const session2Btn = screen.getByRole('button', { name: /Chat session: Deep Learning Optimization/i })
    fireEvent.click(session2Btn)

    // Wait for Session 2 messages to appear
    expect(await screen.findByText('Session 2 unique question')).toBeDefined()

    // Now resolve Session 1 late
    resolveSession1(mockSession1Messages)

    // Verify Session 2 remains displayed and is NOT overwritten by Session 1
    await waitFor(() => {
      expect(screen.queryByText('What is self-attention?')).toBeNull()
      expect(screen.getByText('Session 2 unique question')).toBeDefined()
    })
  })

  it('displays action error when unarchiving fails and preserves archived state', async () => {
    vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
      items: mockSessions,
      total: 3,
      limit: 100,
      offset: 0,
    })

    vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce({
      session_id: 'session-uuid-3',
      total: 0,
      items: [],
    })

    render(<ChatWorkspaceView />)
    await screen.findByText('Archived Survey')

    // Select the archived session
    const archivedSessionItem = screen.getByRole('button', { name: /Chat session: Archived Survey/i })
    fireEvent.click(archivedSessionItem)

    await screen.findByRole('heading', { level: 2, name: 'Archived Survey' })

    // Attempt to unarchive; mock backend 400 error
    vi.spyOn(api.chat, 'updateSession').mockRejectedValueOnce(
      new ApiError(400, 'Cannot unarchive collection-scoped session whose target collection was deleted.')
    )

    const unarchiveBtn = screen.getByRole('button', { name: /^unarchive$/i })
    fireEvent.click(unarchiveBtn)

    // Assert error message is displayed
    const errorElements = await screen.findAllByText(
      /Cannot unarchive collection-scoped session whose target collection was deleted/i
    )
    expect(errorElements.length).toBeGreaterThan(0)

    // Assert session remains archived
    expect(screen.getByText('Archived (Read-Only)')).toBeDefined()
  })

  describe('Phase 2 Interactive Grounded Composer & Turn Handling', () => {
    it('enables composer for active session and disables it with lock notice for archived session', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      // Select active session 1
      const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
      fireEvent.click(session1Btn)

      // Active session has interactive textarea and send button
      const textarea = await screen.findByRole('textbox', { name: /Ask a research question/i })
      expect(textarea).toBeDefined()
      expect(screen.getByRole('button', { name: /Send research question/i })).toBeDefined()
      expect(screen.queryByText(/This conversation is archived. Unarchive to resume questions./i)).toBeNull()

      // Now switch to archived session 3
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce({
        session_id: 'session-uuid-3',
        total: 0,
        items: [],
      })
      const session3Btn = screen.getByRole('button', { name: /Chat session: Archived Survey/i })
      fireEvent.click(session3Btn)

      await screen.findByRole('heading', { level: 2, name: 'Archived Survey' })

      // Textarea is gone and lock notice is rendered
      expect(screen.queryByRole('textbox', { name: /Ask a research question/i })).toBeNull()
      expect(screen.getByText(/This conversation is archived. Unarchive to resume questions./i)).toBeDefined()
    })

    it('disables submit button when query is empty or only whitespace and ignores empty Enter keypress', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)
      const postSpy = vi.spyOn(api.chat, 'postMessage')

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
      fireEvent.click(session1Btn)

      const textarea = await screen.findByRole('textbox', { name: /Ask a research question/i })
      const sendBtn = screen.getByRole('button', { name: /Send research question/i })

      // Empty by default: disabled
      expect(sendBtn.hasAttribute('disabled')).toBe(true)

      // Enter keypress on empty textarea does not submit
      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false })
      expect(postSpy).not.toHaveBeenCalled()

      // Whitespace only: still disabled
      fireEvent.change(textarea, { target: { value: '    \n  \t  ' } })
      expect(sendBtn.hasAttribute('disabled')).toBe(true)

      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false })
      expect(postSpy).not.toHaveBeenCalled()
    })

    it('submits query via api.chat.postMessage, appends turn messages, updates message count, and clears draft', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)
      const postSpy = vi.spyOn(api.chat, 'postMessage').mockResolvedValueOnce(mockTurnResponse)

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
      fireEvent.click(session1Btn)

      const textarea = (await screen.findByRole('textbox', {
        name: /Ask a research question/i,
      })) as HTMLTextAreaElement
      const sendBtn = screen.getByRole('button', { name: /Send research question/i })

      fireEvent.change(textarea, { target: { value: '   What is transformer architecture?   ' } })
      expect(sendBtn.hasAttribute('disabled')).toBe(false)

      fireEvent.click(sendBtn)

      // Submitted with trimmed content
      expect(postSpy).toHaveBeenCalledWith('session-uuid-1', {
        content: 'What is transformer architecture?',
      })

      // Turn response messages are appended
      await screen.findByText('The Transformer is a deep learning model based solely on attention mechanisms.')
      expect(screen.getByText('What is transformer architecture?')).toBeDefined()

      // Composer draft is cleared
      expect(textarea.value).toBe('')

      // Sidebar session count bumped to 4 (Session 1 now has 4 msgs, joining Session 3)
      expect(screen.getAllByText('4 msgs').length).toBe(2)
    })

    it('submits query on Enter keypress and allows newline on Shift+Enter', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)
      const postSpy = vi.spyOn(api.chat, 'postMessage').mockResolvedValueOnce(mockTurnResponse)

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
      fireEvent.click(session1Btn)

      const textarea = await screen.findByRole('textbox', { name: /Ask a research question/i })

      fireEvent.change(textarea, { target: { value: 'What is transformer architecture?' } })

      // Shift+Enter should NOT submit
      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true })
      expect(postSpy).not.toHaveBeenCalled()

      // Plain Enter submits
      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false })
      expect(postSpy).toHaveBeenCalledTimes(1)
      expect(postSpy).toHaveBeenCalledWith('session-uuid-1', {
        content: 'What is transformer architecture?',
      })
    })

    it('renders citation details including document title, page number, quoted excerpt, and match score percentage', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)
      vi.spyOn(api.chat, 'postMessage').mockResolvedValueOnce(mockTurnResponse)

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
      fireEvent.click(session1Btn)

      const textarea = await screen.findByRole('textbox', { name: /Ask a research question/i })
      fireEvent.change(textarea, { target: { value: 'What is transformer architecture?' } })
      fireEvent.click(screen.getByRole('button', { name: /Send research question/i }))

      // Citation verification
      await screen.findByText(/1 Verified Evidence Citation:/i)
      expect(screen.getAllByText(/Attention Is All You Need\.pdf/i).length).toBeGreaterThan(0)
      expect(screen.getByText('Page 1')).toBeDefined()
      expect(screen.getByText('88% match')).toBeDefined()
      expect(screen.getByText(/The Transformer is the first transduction model relying entirely on self-attention\./i)).toBeDefined()
    })

    it('renders no-citations notice when assistant message has empty citations', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)

      const noCitationTurn: ChatTurnResponse = {
        ...mockTurnResponse,
        assistant_message: {
          ...mockTurnResponse.assistant_message,
          id: 'asst-no-cit-id',
          content: 'No indexed sources directly discuss this topic.',
          citations: [],
        },
      }
      vi.spyOn(api.chat, 'postMessage').mockResolvedValueOnce(noCitationTurn)

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
      fireEvent.click(session1Btn)

      const textarea = await screen.findByRole('textbox', { name: /Ask a research question/i })
      fireEvent.change(textarea, { target: { value: 'What about something uncited?' } })
      fireEvent.click(screen.getByRole('button', { name: /Send research question/i }))

      await screen.findByText('No indexed sources directly discuss this topic.')
      expect(screen.getByText('ℹ️ No direct document citations')).toBeDefined()
      expect(screen.getByText(/This response was generated without direct cited excerpts/i)).toBeDefined()
    })

    it('displays error alert on submission failure, preserves draft in composer, and allows dismissal', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)
      vi.spyOn(api.chat, 'postMessage').mockRejectedValueOnce(
        new ApiError(502, 'LLM provider gateway timed out.')
      )

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
      fireEvent.click(session1Btn)

      const textarea = (await screen.findByRole('textbox', {
        name: /Ask a research question/i,
      })) as HTMLTextAreaElement

      fireEvent.change(textarea, { target: { value: 'Why use multi-head attention?' } })
      fireEvent.click(screen.getByRole('button', { name: /Send research question/i }))

      // Error banner is rendered
      const errorAlert = await screen.findByText('LLM provider gateway timed out.')
      expect(errorAlert).toBeDefined()

      // The draft is preserved in the textarea!
      expect(textarea.value).toBe('Why use multi-head attention?')

      // Dismiss the error banner
      const dismissBtn = screen.getByRole('button', { name: /Dismiss error/i })
      fireEvent.click(dismissBtn)

      await waitFor(() => {
        expect(screen.queryByText('LLM provider gateway timed out.')).toBeNull()
      })
      // Draft remains intact after dismiss
      expect(textarea.value).toBe('Why use multi-head attention?')
    })

    it('prevents duplicate submissions on rapid double-clicks while request is in flight', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)

      let resolveTurn!: (val: ChatTurnResponse) => void
      const turnPromise = new Promise<ChatTurnResponse>((resolve) => {
        resolveTurn = resolve
      })
      const postSpy = vi.spyOn(api.chat, 'postMessage').mockReturnValueOnce(turnPromise)

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
      fireEvent.click(session1Btn)

      const textarea = await screen.findByRole('textbox', { name: /Ask a research question/i })
      fireEvent.change(textarea, { target: { value: 'How many heads are used?' } })

      const sendBtn = screen.getByRole('button', { name: /Send research question/i })
      fireEvent.click(sendBtn)

      // Button is disabled with 'Sending...'
      expect(sendBtn.hasAttribute('disabled')).toBe(true)
      expect(screen.getAllByText(/Sending\.\.\./i).length).toBeGreaterThan(0)

      // Second click does nothing
      fireEvent.click(sendBtn)
      expect(postSpy).toHaveBeenCalledTimes(1)

      // Resolve the turn
      resolveTurn(mockTurnResponse)
      await screen.findByText('The Transformer is a deep learning model based solely on attention mechanisms.')
    })

    it('preserves separate draft queries per session when switching back and forth', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockImplementation(async (sid: string) => {
        if (sid === 'session-uuid-1') return mockSession1Messages
        return { session_id: 'session-uuid-2', total: 0, items: [] }
      })

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      // Select Session 1
      const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
      fireEvent.click(session1Btn)

      let textarea = (await screen.findByRole('textbox', {
        name: /Ask a research question/i,
      })) as HTMLTextAreaElement

      // Type draft in Session 1
      fireEvent.change(textarea, { target: { value: 'Draft for Session 1' } })
      expect(textarea.value).toBe('Draft for Session 1')

      // Switch to Session 2
      const session2Btn = screen.getByRole('button', { name: /Chat session: Deep Learning Optimization/i })
      fireEvent.click(session2Btn)

      await screen.findByRole('heading', { level: 2, name: 'Deep Learning Optimization' })
      textarea = screen.getByRole('textbox', { name: /Ask a research question/i }) as HTMLTextAreaElement
      expect(textarea.value).toBe('')

      // Type draft in Session 2
      fireEvent.change(textarea, { target: { value: 'Draft for Session 2' } })
      expect(textarea.value).toBe('Draft for Session 2')

      // Switch back to Session 1
      fireEvent.click(session1Btn)
      await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })
      textarea = screen.getByRole('textbox', { name: /Ask a research question/i }) as HTMLTextAreaElement
      expect(textarea.value).toBe('Draft for Session 1')

      // Switch back to Session 2
      fireEvent.click(session2Btn)
      await screen.findByRole('heading', { level: 2, name: 'Deep Learning Optimization' })
      textarea = screen.getByRole('textbox', { name: /Ask a research question/i }) as HTMLTextAreaElement
      expect(textarea.value).toBe('Draft for Session 2')
    })

    it('prevents late response from in-flight query in Session 1 from polluting Session 2 after switching', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockImplementation(async (sid: string) => {
        if (sid === 'session-uuid-1') return mockSession1Messages
        return { session_id: 'session-uuid-2', total: 0, items: [] }
      })

      let resolveTurn!: (val: ChatTurnResponse) => void
      const turnPromise = new Promise<ChatTurnResponse>((resolve) => {
        resolveTurn = resolve
      })
      vi.spyOn(api.chat, 'postMessage').mockReturnValueOnce(turnPromise)

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      // Select Session 1
      const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
      fireEvent.click(session1Btn)

      const textarea = await screen.findByRole('textbox', { name: /Ask a research question/i })
      fireEvent.change(textarea, { target: { value: 'Session 1 question' } })
      fireEvent.click(screen.getByRole('button', { name: /Send research question/i }))

      // While request is pending, switch to Session 2
      const session2Btn = screen.getByRole('button', { name: /Chat session: Deep Learning Optimization/i })
      fireEvent.click(session2Btn)

      await screen.findByRole('heading', { level: 2, name: 'Deep Learning Optimization' })

      // Resolve Session 1 turn response
      resolveTurn(mockTurnResponse)

      // Wait a tick
      await new Promise((r) => setTimeout(r, 60))

      // Session 2 active message stream must NOT have Session 1's assistant message
      expect(screen.queryByText('The Transformer is a deep learning model based solely on attention mechanisms.')).toBeNull()

      // But Session 1 in the sidebar list has updated message_count (2 + 2 = 4 msgs, joining Session 3)
      expect(screen.getAllByText('4 msgs').length).toBe(2)
    })

    it('calls scrollIntoView when message list updates and scrollIntoView is supported', async () => {
      const scrollIntoViewMock = vi.fn()
      window.HTMLElement.prototype.scrollIntoView = scrollIntoViewMock

      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      const session1Btn = screen.getByRole('button', { name: /Chat session: Attention Mechanism Analysis/i })
      fireEvent.click(session1Btn)

      await screen.findByText('What is self-attention?')
      expect(scrollIntoViewMock).toHaveBeenCalled()
    })
  })

  describe('Phase 3 Safe Session Deletion', () => {
    it('opens DeleteChatModal with conversation title when clicking delete in sidebar without selecting session', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      const deleteIconBtn = screen.getByRole('button', { name: 'Delete Attention Mechanism Analysis' })
      fireEvent.click(deleteIconBtn)

      const dialog = await screen.findByRole('dialog')
      expect(dialog).toBeDefined()
      expect(screen.getByRole('heading', { level: 3, name: 'Delete Conversation' })).toBeDefined()
      expect(within(dialog).getByText('Attention Mechanism Analysis')).toBeDefined()
    })

    it('opens DeleteChatModal when clicking Delete button in conversation header', async () => {
      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)
      await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })

      const headerDeleteBtn = screen.getByRole('button', { name: /^delete$/i })
      fireEvent.click(headerDeleteBtn)

      const dialog = await screen.findByRole('dialog')
      expect(dialog).toBeDefined()
      expect(screen.getByRole('heading', { level: 3, name: 'Delete Conversation' })).toBeDefined()
      expect(within(dialog).getByText('Attention Mechanism Analysis')).toBeDefined()
    })

    it('closes DeleteChatModal without calling API when Cancel or Close is clicked', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      const deleteSpy = vi.spyOn(api.chat, 'deleteSession')

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      // Open modal
      fireEvent.click(screen.getByRole('button', { name: 'Delete Attention Mechanism Analysis' }))
      expect(await screen.findByRole('dialog')).toBeDefined()

      // Click Cancel
      fireEvent.click(screen.getByRole('button', { name: /cancel/i }))
      await waitFor(() => {
        expect(screen.queryByRole('dialog')).toBeNull()
      })
      expect(deleteSpy).not.toHaveBeenCalled()

      // Open modal again and click close (×) button
      fireEvent.click(screen.getByRole('button', { name: 'Delete Attention Mechanism Analysis' }))
      expect(await screen.findByRole('dialog')).toBeDefined()

      fireEvent.click(screen.getByRole('button', { name: /close dialog/i }))
      await waitFor(() => {
        expect(screen.queryByRole('dialog')).toBeNull()
      })
      expect(deleteSpy).not.toHaveBeenCalled()
    })

    it('shows in-flight Deleting... state and disables actions during deletion', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })

      let resolveDelete!: (val: { message: string }) => void
      const deletePromise = new Promise<{ message: string }>((resolve) => {
        resolveDelete = resolve
      })
      vi.spyOn(api.chat, 'deleteSession').mockReturnValueOnce(deletePromise)

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      fireEvent.click(screen.getByRole('button', { name: 'Delete Attention Mechanism Analysis' }))
      expect(await screen.findByRole('dialog')).toBeDefined()

      const confirmBtn = screen.getByRole('button', { name: 'Delete Conversation' })
      const cancelBtn = screen.getByRole('button', { name: 'Cancel' })
      const closeBtn = screen.getByRole('button', { name: 'Close dialog' })

      fireEvent.click(confirmBtn)

      // In-flight state assertions
      expect(confirmBtn.textContent).toBe('Deleting...')
      expect(confirmBtn).toHaveProperty('disabled', true)
      expect(cancelBtn).toHaveProperty('disabled', true)
      expect(closeBtn).toHaveProperty('disabled', true)

      // Resolve deletion
      resolveDelete({ message: 'Chat session deleted successfully.' })
      await waitFor(() => {
        expect(screen.queryByRole('dialog')).toBeNull()
      })
    })

    it('displays API error message in modal when deletion fails and preserves session in list', async () => {
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'deleteSession').mockRejectedValueOnce(
        new ApiError(500, 'Database error deleting session.')
      )

      render(<ChatWorkspaceView />)
      await screen.findByText('Attention Mechanism Analysis')

      fireEvent.click(screen.getByRole('button', { name: 'Delete Attention Mechanism Analysis' }))
      const dialog = await screen.findByRole('dialog')
      expect(dialog).toBeDefined()

      fireEvent.click(screen.getByRole('button', { name: 'Delete Conversation' }))

      expect(await screen.findByText('Database error deleting session.')).toBeDefined()
      expect(screen.getByRole('dialog')).toBeDefined()
      expect(screen.getAllByText('Attention Mechanism Analysis').length).toBeGreaterThanOrEqual(1)
    })

    it('successfully deletes active session, selects next available session, and decrements count', async () => {
      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce({
        session_id: 'session-uuid-2',
        total: 0,
        items: [],
      })
      vi.spyOn(api.chat, 'deleteSession').mockResolvedValueOnce({
        message: 'Chat session deleted successfully.',
      })

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)
      await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })
      expect(screen.getByTitle('Total chat sessions').textContent).toBe('3')

      // Open delete dialog from header
      const headerDeleteBtn = screen.getByRole('button', { name: /^delete$/i })
      fireEvent.click(headerDeleteBtn)

      // Confirm deletion
      const confirmBtn = await screen.findByRole('button', { name: 'Delete Conversation' })
      fireEvent.click(confirmBtn)

      // Modal closes
      await waitFor(() => {
        expect(screen.queryByRole('dialog')).toBeNull()
      })

      // Session 1 is removed from sidebar
      expect(screen.queryByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })).toBeNull()
      expect(screen.queryByText('Attention Mechanism Analysis')).toBeNull()

      // Total sessions badge decremented
      expect(screen.getByTitle('Total chat sessions').textContent).toBe('2')

      // Next session (session-uuid-2) is automatically selected
      expect(await screen.findByRole('heading', { level: 2, name: 'Deep Learning Optimization' })).toBeDefined()
      expect(window.location.hash).toBe('#chat?session=session-uuid-2')
    })

    it('successfully deletes sole session and resets to empty chat state', async () => {
      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: [mockSessions[0]],
        total: 1,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)
      vi.spyOn(api.chat, 'deleteSession').mockResolvedValueOnce({
        message: 'Chat session deleted successfully.',
      })

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)
      await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })
      expect(screen.getByTitle('Total chat sessions').textContent).toBe('1')

      fireEvent.click(screen.getByRole('button', { name: /^delete$/i }))
      fireEvent.click(await screen.findByRole('button', { name: 'Delete Conversation' }))

      await waitFor(() => {
        expect(screen.queryByRole('dialog')).toBeNull()
      })

      // Count is 0
      expect(screen.getByTitle('Total chat sessions').textContent).toBe('0')
      // Reset to empty chat state
      expect(await screen.findByRole('heading', { level: 3, name: 'Select a Conversation' })).toBeDefined()
      expect(window.location.hash).toBe('#chat')
    })

    it('successfully deletes a non-active session from sidebar without altering active session', async () => {
      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)
      vi.spyOn(api.chat, 'deleteSession').mockResolvedValueOnce({
        message: 'Chat session deleted successfully.',
      })

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)
      await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })

      // Delete session 2 from the sidebar
      const deleteSession2Btn = screen.getByRole('button', { name: 'Delete Deep Learning Optimization' })
      fireEvent.click(deleteSession2Btn)

      expect(await screen.findByRole('dialog')).toBeDefined()
      fireEvent.click(screen.getByRole('button', { name: 'Delete Conversation' }))

      await waitFor(() => {
        expect(screen.queryByRole('dialog')).toBeNull()
      })

      // Session 2 is removed from sidebar
      expect(screen.queryByText('Deep Learning Optimization')).toBeNull()
      // Session 1 remains active and selected
      expect(screen.getByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })).toBeDefined()
      expect(window.location.hash).toBe('#chat?session=session-uuid-1')
      expect(screen.getByTitle('Total chat sessions').textContent).toBe('2')
    })
  })

  describe('Phase 4: Safe Concurrent Chat Operations & Busy Coordination', () => {
    it('disables header and sidebar delete buttons with tooltip while generation is in flight, restoring them afterwards', async () => {
      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)

      let resolveTurn: (value: ChatTurnResponse) => void
      const pendingTurnPromise = new Promise<ChatTurnResponse>((resolve) => {
        resolveTurn = resolve
      })
      vi.spyOn(api.chat, 'postMessage').mockReturnValueOnce(pendingTurnPromise)

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)
      await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })

      const headerDeleteBtn = screen.getByRole('button', { name: /^delete$/i })
      const sidebarDeleteBtn1 = screen.getByRole('button', { name: 'Delete Attention Mechanism Analysis' })
      const sidebarDeleteBtn2 = screen.getByRole('button', { name: 'Delete Deep Learning Optimization' })

      expect(headerDeleteBtn.hasAttribute('disabled')).toBe(false)
      expect(sidebarDeleteBtn1.hasAttribute('disabled')).toBe(false)
      expect(sidebarDeleteBtn2.hasAttribute('disabled')).toBe(false)

      const textarea = screen.getByPlaceholderText(/Ask a research question about your documents/i)
      fireEvent.change(textarea, { target: { value: 'In flight question' } })
      const sendBtn = screen.getByRole('button', { name: 'Send research question' })
      fireEvent.click(sendBtn)

      // Generation is now in flight
      await screen.findByText('Searching evidence passages and synthesizing answer...')

      expect(headerDeleteBtn.hasAttribute('disabled')).toBe(true)
      expect(headerDeleteBtn.getAttribute('title')).toBe(
        'Cannot delete conversation while generation is in progress'
      )
      expect(sidebarDeleteBtn1.hasAttribute('disabled')).toBe(true)
      expect(sidebarDeleteBtn1.getAttribute('title')).toBe(
        'Cannot delete conversation while generation is in progress'
      )
      // Other session delete is NOT disabled
      expect(sidebarDeleteBtn2.hasAttribute('disabled')).toBe(false)

      // Resolve the generation
      resolveTurn!(mockTurnResponse)

      await waitFor(() => {
        expect(headerDeleteBtn.hasAttribute('disabled')).toBe(false)
        expect(sidebarDeleteBtn1.hasAttribute('disabled')).toBe(false)
      })
    })

    it('handles 409 Conflict error on submit and restores controls without losing draft', async () => {
      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)
      vi.spyOn(api.chat, 'postMessage').mockRejectedValueOnce(
        new ApiError(409, 'SESSION_BUSY: A conversation turn is currently being generated.')
      )

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)
      await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })

      const textarea = screen.getByPlaceholderText(/Ask a research question about your documents/i)
      fireEvent.change(textarea, { target: { value: 'Concurrent turn attempt' } })
      const sendBtn = screen.getByRole('button', { name: 'Send research question' })
      fireEvent.click(sendBtn)

      // Error alert displays
      expect(
        await screen.findByText('SESSION_BUSY: A conversation turn is currently being generated.')
      ).toBeDefined()

      // Draft is preserved in textarea
      expect((textarea as HTMLTextAreaElement).value).toBe('Concurrent turn attempt')

      // Controls are restored
      expect(screen.getByRole('button', { name: /^delete$/i }).hasAttribute('disabled')).toBe(false)
      expect(screen.getByRole('button', { name: 'Send research question' }).hasAttribute('disabled')).toBe(false)
    })

    it('supports switching sessions during generation without blocking the newly selected session', async () => {
      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages')
        .mockResolvedValueOnce(mockSession1Messages)
        .mockResolvedValueOnce({ session_id: 'session-uuid-2', total: 0, items: [] })
        .mockResolvedValueOnce(mockSession1Messages)

      let resolveTurn: (value: ChatTurnResponse) => void
      const pendingTurnPromise = new Promise<ChatTurnResponse>((resolve) => {
        resolveTurn = resolve
      })
      vi.spyOn(api.chat, 'postMessage').mockReturnValueOnce(pendingTurnPromise)

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)
      await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })

      // Start generation in Session 1
      const textarea1 = screen.getByPlaceholderText(/Ask a research question about your documents/i)
      fireEvent.change(textarea1, { target: { value: 'Session 1 question' } })
      fireEvent.click(screen.getByRole('button', { name: 'Send research question' }))

      await screen.findByText('Searching evidence passages and synthesizing answer...')

      // Switch to Session 2
      const session2Item = screen.getByRole('button', { name: 'Chat session: Deep Learning Optimization' })
      fireEvent.click(session2Item)

      await screen.findByRole('heading', { level: 2, name: 'Deep Learning Optimization' })

      // Session 2 is not generating: composer is enabled and delete button is enabled
      const textarea2 = screen.getByPlaceholderText(/Ask a research question about your documents/i)
      expect(textarea2.hasAttribute('disabled')).toBe(false)
      expect(screen.getByRole('button', { name: /^delete$/i }).hasAttribute('disabled')).toBe(false)

      // Switch back to Session 1
      const session1Item = screen.getByRole('button', { name: 'Chat session: Attention Mechanism Analysis' })
      fireEvent.click(session1Item)

      await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })
      // Session 1 is still generating and shows in-flight prompt
      expect(screen.getAllByText('Session 1 question').length).toBeGreaterThan(0)
      expect(screen.getByText('Searching evidence passages and synthesizing answer...')).toBeDefined()

      // Resolve turn
      resolveTurn!(mockTurnResponse)
      await waitFor(() => {
        expect(screen.queryByText('Searching evidence passages and synthesizing answer...')).toBeNull()
      })
    })

    it('guards handleOpenDeleteModal against busy sessions', async () => {
      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(mockSession1Messages)

      let resolveTurn: (value: ChatTurnResponse) => void
      const pendingTurnPromise = new Promise<ChatTurnResponse>((resolve) => {
        resolveTurn = resolve
      })
      vi.spyOn(api.chat, 'postMessage').mockReturnValueOnce(pendingTurnPromise)

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)
      await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })

      const textarea = screen.getByPlaceholderText(/Ask a research question about your documents/i)
      fireEvent.change(textarea, { target: { value: 'In flight question' } })
      fireEvent.click(screen.getByRole('button', { name: 'Send research question' }))

      await screen.findByText('Searching evidence passages and synthesizing answer...')

      // Modal should not open
      expect(screen.queryByRole('dialog')).toBeNull()

      resolveTurn!(mockTurnResponse)
      await waitFor(() => {
        expect(screen.queryByText('Searching evidence passages and synthesizing answer...')).toBeNull()
      })
    })
  })

  describe('Phase 5: Adaptive Markdown Rendering, Security & Citation Formatting', () => {
    it('renders assistant messages with rich Markdown headings, tables, lists, and code blocks while preserving verified citations', async () => {
      const markdownMessages: ChatMessageListResponse = {
        session_id: 'session-uuid-1',
        total: 2,
        items: [
          {
            id: 'msg-user-1',
            chat_session_id: 'session-uuid-1',
            sender_type: 'USER',
            content: 'Explain Naive Bayes performance without SMOTE',
            token_count: 10,
            created_at: '2026-10-19T10:00:00Z',
            citations: [],
          },
          {
            id: 'msg-asst-1',
            chat_session_id: 'session-uuid-1',
            sender_type: 'ASSISTANT',
            content:
              '## Performance Overview\n' +
              'Naïve Bayes exhibits high recall but lower precision on the CICIDS2017 dataset [CHUNK:101].\n\n' +
              '### Benchmark Metrics\n' +
              '| Metric | Value |\n' +
              '|---|---:|\n' +
              '| Precision | 0.6364 |\n' +
              '| Recall | 0.9742 |\n' +
              '| FAR | 0.1820 |\n\n' +
              '- Identified **97.42%** of true positive threats\n' +
              '- High false alarm rate of **18.20%**\n\n' +
              'Configuration parameter: `use_smote=False`\n\n' +
              '```python\n' +
              'nb = GaussianNB()\n' +
              '```',
            token_count: 50,
            created_at: '2026-10-19T10:00:05Z',
            citations: [
              {
                id: 'cit-uuid-1',
                message_id: 'msg-asst-1',
                chunk_id: 101,
                quoted_text: 'Table 6: Naive Bayes without SMOTE achieved Precision: 0.6364, Recall: 0.9742.',
                page_number: 7,
                document_title: 'NIDS Research Paper.pdf',
                similarity_score: 0.94,
                created_at: '2026-10-19T10:00:05Z',
              },
            ],
          },
        ],
      }

      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(markdownMessages)

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)

      // 1. Headings rendered with appropriate level
      expect(await screen.findByRole('heading', { level: 2, name: 'Performance Overview' })).toBeDefined()
      expect(screen.getByRole('heading', { level: 3, name: 'Benchmark Metrics' })).toBeDefined()

      // 2. Table rendered with headers, cells, and responsive wrapper
      expect(screen.getByRole('table')).toBeDefined()
      expect(screen.getByText('Precision')).toBeDefined()
      expect(screen.getByText('0.6364')).toBeDefined()
      expect(screen.getByText('0.9742')).toBeDefined()
      expect(screen.getByText('0.1820')).toBeDefined()
      expect(screen.getByRole('region', { name: 'Table' }).className).toContain('table-responsive-wrapper')

      // 3. Lists and bold formatting
      expect(screen.getByText(/Identified/)).toBeDefined()
      expect(screen.getByText('97.42%').tagName).toBe('STRONG')

      // 4. Inline code and code blocks
      expect(screen.getByText('use_smote=False').tagName).toBe('CODE')
      expect(screen.getByText(/nb = GaussianNB\(\)/)).toBeDefined()

      // 5. Existing verified evidence citation cards remain intact and rendered below
      expect(screen.getByText(/1 Verified Evidence Citation/i)).toBeDefined()
      expect(screen.getByText(/NIDS Research Paper\.pdf/i)).toBeDefined()
      expect(screen.getByText('Page 7')).toBeDefined()
      expect(screen.getByText('94% match')).toBeDefined()
      expect(screen.getByText(/Table 6: Naive Bayes without SMOTE/i)).toBeDefined()
    })

    it('sanitizes untrusted HTML in assistant messages without executing scripts or injecting arbitrary tags', async () => {
      const xssMessages: ChatMessageListResponse = {
        session_id: 'session-uuid-1',
        total: 1,
        items: [
          {
            id: 'msg-xss-1',
            chat_session_id: 'session-uuid-1',
            sender_type: 'ASSISTANT',
            content: 'Safe explanation <script>window.pwned = true</script> and <img src=x onerror=alert(1)> done.',
            token_count: 20,
            created_at: '2026-10-19T10:00:00Z',
            citations: [],
          },
        ],
      }

      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(xssMessages)

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)
      await screen.findByRole('heading', { level: 2, name: 'Attention Mechanism Analysis' })

      // Scripts and img tags must not be created in the DOM
      expect(document.querySelector('script')).toBeNull()
      expect(document.querySelector('img')).toBeNull()
      expect(screen.getByText(/Safe explanation/)).toBeDefined()
    })

    it('renders safe https links with target=_blank and neutralizes javascript: URLs', async () => {
      const linkMessages: ChatMessageListResponse = {
        session_id: 'session-uuid-1',
        total: 1,
        items: [
          {
            id: 'msg-link-1',
            chat_session_id: 'session-uuid-1',
            sender_type: 'ASSISTANT',
            content: 'Reference: [NIDS Benchmark Paper](https://arxiv.org/abs/1706.03762) and [Malicious](javascript:alert("XSS")).',
            token_count: 20,
            created_at: '2026-10-19T10:00:00Z',
            citations: [],
          },
        ],
      }

      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(linkMessages)

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)

      const safeLink = await screen.findByRole('link', { name: 'NIDS Benchmark Paper' })
      expect(safeLink.getAttribute('href')).toBe('https://arxiv.org/abs/1706.03762')
      expect(safeLink.getAttribute('target')).toBe('_blank')
      expect(safeLink.getAttribute('rel')).toContain('noopener')

      // Malicious link is neutralized to a safe non-clickable span
      expect(screen.queryByRole('link', { name: 'Malicious' })).toBeNull()
      expect(screen.getByText('Malicious').tagName).toBe('SPAN')
    })

    it('resiliently renders malformed Markdown without crashing the workspace', async () => {
      const malformedMessages: ChatMessageListResponse = {
        session_id: 'session-uuid-1',
        total: 1,
        items: [
          {
            id: 'msg-malformed-1',
            chat_session_id: 'session-uuid-1',
            sender_type: 'ASSISTANT',
            content: 'Malformed text: **unclosed bold and ```python\nunclosed code and | incomplete | table |',
            token_count: 20,
            created_at: '2026-10-19T10:00:00Z',
            citations: [],
          },
        ],
      }

      window.location.hash = '#chat?session=session-uuid-1'
      vi.spyOn(api.chat, 'listSessions').mockResolvedValueOnce({
        items: mockSessions,
        total: 3,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.chat, 'getSessionMessages').mockResolvedValueOnce(malformedMessages)

      render(<ChatWorkspaceView initialSessionId="session-uuid-1" />)
      expect(await screen.findByText(/Malformed text:/)).toBeDefined()
    })
  })
})

