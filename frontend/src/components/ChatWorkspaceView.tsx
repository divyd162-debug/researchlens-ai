import React, { useEffect, useState, useCallback, useRef } from 'react'
import { api, ApiError } from '../services/api'
import {
  ChatMessageResponse,
  ChatSessionResponse,
} from '../types'
import { DeleteChatModal } from './DeleteChatModal'
import { MarkdownContent } from './MarkdownContent'
import { NewConversationModal } from './NewConversationModal'

interface ChatWorkspaceViewProps {
  initialSessionId?: string | null
}

const formatSessionDate = (dateStr?: string | null): string => {
  if (!dateStr) return ''
  try {
    const d = new Date(dateStr)
    return isNaN(d.getTime()) ? '' : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  } catch {
    return ''
  }
}

const formatMessageTime = (dateStr?: string | null): string => {
  if (!dateStr) return ''
  try {
    const d = new Date(dateStr)
    return isNaN(d.getTime()) ? '' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

const getSessionIdFromHash = (): string | null => {
  const hash = window.location.hash
  if (hash.includes('session=')) {
    const parts = hash.split('session=')
    if (parts[1]) {
      return parts[1].split('&')[0]
    }
  }
  return null
}

export const ChatWorkspaceView: React.FC<ChatWorkspaceViewProps> = ({ initialSessionId }) => {
  // Session list states
  const [sessions, setSessions] = useState<ChatSessionResponse[]>([])
  const [totalSessions, setTotalSessions] = useState<number>(0)
  const [isLoadingSessions, setIsLoadingSessions] = useState<boolean>(true)
  const [sessionError, setSessionError] = useState<string | null>(null)
  const [activeFilter, setActiveFilter] = useState<'all' | 'active' | 'archived'>('all')

  // Selected session and message history states
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(() => {
    return initialSessionId || getSessionIdFromHash()
  })
  const [messages, setMessages] = useState<ChatMessageResponse[]>([])
  const [isLoadingMessages, setIsLoadingMessages] = useState<boolean>(false)
  const [messageError, setMessageError] = useState<string | null>(null)

  // Creation modal state
  const [isNewModalOpen, setIsNewModalOpen] = useState<boolean>(false)
  const [showSourcesPopover, setShowSourcesPopover] = useState<boolean>(false)

  // Renaming state
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState<string>('')
  const [isUpdatingTitle, setIsUpdatingTitle] = useState<boolean>(false)
  const [actionError, setActionError] = useState<string | null>(null)

  // Archiving in-flight state
  const [isArchivingId, setIsArchivingId] = useState<string | null>(null)

  // Delete modal states
  const [sessionToDelete, setSessionToDelete] = useState<ChatSessionResponse | null>(null)
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState<boolean>(false)

  // Phase 2 Composer and interactive turn states
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [submittingSessions, setSubmittingSessions] = useState<Record<string, string>>({})
  const [submitErrors, setSubmitErrors] = useState<Record<string, string | null>>({})

  // Helper to check if a specific session has a generation in flight
  const isSessionSubmitting = (sessionId?: string | null): boolean =>
    Boolean(sessionId && submittingSessions[sessionId])

  // Abort controller refs and active request tracking for concurrency safety
  const listAbortRef = useRef<AbortController | null>(null)
  const messagesAbortRef = useRef<AbortController | null>(null)
  const activeMessageSessionIdRef = useRef<string | null>(null)
  const activeListRequestIdRef = useRef<number>(0)
  const selectedSessionIdRef = useRef<string | null>(selectedSessionId)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  // Keep selectedSessionIdRef in sync
  useEffect(() => {
    selectedSessionIdRef.current = selectedSessionId
  }, [selectedSessionId])

  // Auto-scroll to bottom of conversation
  const scrollToBottom = useCallback((smooth = true) => {
    if (messagesEndRef.current && typeof messagesEndRef.current.scrollIntoView === 'function') {
      messagesEndRef.current.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto' })
    }
  }, [])

  // Load chat sessions from API
  const loadSessions = useCallback(async (selectFirst = false) => {
    if (listAbortRef.current) {
      listAbortRef.current.abort()
    }
    const controller = new AbortController()
    listAbortRef.current = controller
    const requestId = ++activeListRequestIdRef.current

    setIsLoadingSessions(true)
    setSessionError(null)

    try {
      const data = await api.chat.listSessions(100, 0, undefined, controller.signal)
      if (requestId === activeListRequestIdRef.current) {
        setSessions(data.items)
        setTotalSessions(data.total)

        // Handle auto-selection if requested
        if (selectFirst && data.items.length > 0) {
          const firstActive = data.items.find((s) => !s.is_archived) || data.items[0]
          setSelectedSessionId(firstActive.id)
          window.location.hash = `#chat?session=${firstActive.id}`
        }
      }
    } catch (err) {
      if (requestId !== activeListRequestIdRef.current) {
        return
      }
      if (err instanceof Error && err.name === 'AbortError') {
        return
      }
      if (err instanceof ApiError) {
        setSessionError(err.detail)
      } else {
        setSessionError('Failed to load conversations from server.')
      }
    } finally {
      if (requestId === activeListRequestIdRef.current) {
        setIsLoadingSessions(false)
      }
    }
  }, [])

  // Load messages for currently selected session with concurrency protection
  const loadMessages = useCallback(async (sessionId: string) => {
    if (messagesAbortRef.current) {
      messagesAbortRef.current.abort()
    }
    const controller = new AbortController()
    messagesAbortRef.current = controller
    activeMessageSessionIdRef.current = sessionId

    setIsLoadingMessages(true)
    setMessageError(null)

    try {
      const data = await api.chat.getSessionMessages(sessionId, controller.signal)
      if (activeMessageSessionIdRef.current === sessionId) {
        setMessages(data.items)
        // Keep session message count synchronized with actual loaded count
        setSessions((prev) =>
          prev.map((s) => (s.id === sessionId ? { ...s, message_count: data.total } : s))
        )
      }
    } catch (err) {
      if (activeMessageSessionIdRef.current !== sessionId) {
        return
      }
      if (err instanceof Error && err.name === 'AbortError') {
        return
      }
      if (err instanceof ApiError) {
        setMessageError(err.detail)
      } else {
        setMessageError('Failed to load messages for this conversation.')
      }
    } finally {
      if (activeMessageSessionIdRef.current === sessionId) {
        setIsLoadingMessages(false)
      }
    }
  }, [])

  // Initial load
  useEffect(() => {
    loadSessions()
    return () => {
      if (listAbortRef.current) listAbortRef.current.abort()
      if (messagesAbortRef.current) messagesAbortRef.current.abort()
    }
  }, [loadSessions])

  // React to hash change for browser back/forward/refresh
  useEffect(() => {
    const handleHashChange = () => {
      const sid = getSessionIdFromHash()
      setSelectedSessionId((prev) => (prev !== sid ? sid : prev))
    }
    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

  // Fetch messages whenever selected session changes
  useEffect(() => {
    if (selectedSessionId) {
      loadMessages(selectedSessionId)
    } else {
      setMessages([])
      setMessageError(null)
    }
  }, [selectedSessionId, loadMessages])

  // Select a session
  const handleSelectSession = (session: ChatSessionResponse) => {
    setSelectedSessionId(session.id)
    setShowSourcesPopover(false)
    window.location.hash = `#chat?session=${session.id}`
    setActionError(null)
  }

  // Handle successful creation from modal
  const handleNewConversationSuccess = (newSession: ChatSessionResponse) => {
    setSessions((prev) => [newSession, ...prev])
    setTotalSessions((prev) => prev + 1)
    setSelectedSessionId(newSession.id)
    setShowSourcesPopover(false)
    window.location.hash = `#chat?session=${newSession.id}`
  }

  // Start renaming inline
  const handleStartRename = (session: ChatSessionResponse, e: React.MouseEvent) => {
    e.stopPropagation()
    setEditingSessionId(session.id)
    setEditTitle(session.title)
    setActionError(null)
  }

  // Cancel renaming
  const handleCancelRename = (e: React.MouseEvent) => {
    e.stopPropagation()
    setEditingSessionId(null)
    setEditTitle('')
  }

  // Save renamed title
  const handleSaveRename = async (sessionId: string, e: React.FormEvent) => {
    e.preventDefault()
    e.stopPropagation()

    const trimmed = editTitle.trim()
    if (!trimmed) {
      setActionError('Title cannot be empty.')
      return
    }

    setIsUpdatingTitle(true)
    setActionError(null)

    try {
      const updated = await api.chat.updateSession(sessionId, { title: trimmed })
      setSessions((prev) => prev.map((s) => (s.id === sessionId ? updated : s)))
      setEditingSessionId(null)
      setEditTitle('')
    } catch (err) {
      if (err instanceof ApiError) {
        setActionError(err.detail)
      } else {
        setActionError('Failed to rename conversation.')
      }
    } finally {
      setIsUpdatingTitle(false)
    }
  }

  // Toggle archive state
  const handleToggleArchive = async (session: ChatSessionResponse, e: React.MouseEvent) => {
    e.stopPropagation()
    setIsArchivingId(session.id)
    setActionError(null)

    const nextState = !session.is_archived
    try {
      const updated = await api.chat.updateSession(session.id, { is_archived: nextState })
      setSessions((prev) => prev.map((s) => (s.id === session.id ? updated : s)))
    } catch (err) {
      if (err instanceof ApiError) {
        setActionError(err.detail)
      } else {
        setActionError(nextState ? 'Failed to archive conversation.' : 'Failed to unarchive conversation.')
      }
    } finally {
      setIsArchivingId(null)
    }
  }

  // Open delete confirmation modal
  const handleOpenDeleteModal = (session: ChatSessionResponse, e?: React.MouseEvent) => {
    if (e) {
      e.stopPropagation()
    }
    if (isSessionSubmitting(session.id)) {
      setActionError('Cannot delete conversation while generation is in progress.')
      return
    }
    setSessionToDelete(session)
    setIsDeleteModalOpen(true)
    setActionError(null)
  }

  // Close delete confirmation modal
  const handleCloseDeleteModal = () => {
    setIsDeleteModalOpen(false)
    setSessionToDelete(null)
  }

  // Handle confirmed deletion success
  const handleDeleteSuccess = (deletedId: string) => {
    const remaining = sessions.filter((s) => s.id !== deletedId)
    setSessions(remaining)
    setTotalSessions((prev) => Math.max(0, prev - 1))

    setDrafts((prev) => {
      const next = { ...prev }
      delete next[deletedId]
      return next
    })
    setSubmittingSessions((prev) => {
      const next = { ...prev }
      delete next[deletedId]
      return next
    })
    setSubmitErrors((prev) => {
      const next = { ...prev }
      delete next[deletedId]
      return next
    })

    if (selectedSessionId === deletedId) {
      if (remaining.length > 0) {
        const nextSession = remaining.find((s) => !s.is_archived) || remaining[0]
        setSelectedSessionId(nextSession.id)
        window.location.hash = `#chat?session=${nextSession.id}`
      } else {
        setSelectedSessionId(null)
        window.location.hash = '#chat'
        setMessages([])
        setMessageError(null)
      }
    }
  }

  // Auto-scroll when messages update or in-flight state changes
  useEffect(() => {
    if (messages.length > 0 || isSessionSubmitting(selectedSessionId)) {
      scrollToBottom(false)
    }
  }, [messages, submittingSessions, selectedSessionId, scrollToBottom])

  // Submit research query and stream/persist response
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) {
      e.preventDefault()
    }
    if (!selectedSessionId || isSessionSubmitting(selectedSessionId)) {
      return
    }

    const currentDraft = (drafts[selectedSessionId] || '').trim()
    if (!currentDraft) {
      return
    }

    const targetSessionId = selectedSessionId
    setSubmittingSessions((prev) => ({ ...prev, [targetSessionId]: currentDraft }))
    setSubmitErrors((prev) => ({ ...prev, [targetSessionId]: null }))

    try {
      const turn = await api.chat.postMessage(targetSessionId, { content: currentDraft })

      // Clear draft for this session upon success
      setDrafts((prev) => ({ ...prev, [targetSessionId]: '' }))

      // Only append to active messages if the user is still on targetSessionId
      if (selectedSessionIdRef.current === targetSessionId) {
        setMessages((prev) => {
          const existingIds = new Set(prev.map((m) => m.id))
          const next = [...prev]
          if (!existingIds.has(turn.user_message.id)) {
            next.push(turn.user_message)
          }
          if (!existingIds.has(turn.assistant_message.id)) {
            next.push(turn.assistant_message)
          }
          return next
        })
        setTimeout(() => scrollToBottom(true), 50)
      }

      // Reconcile session metadata in sidebar
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id === targetSessionId) {
            return {
              ...s,
              message_count: s.message_count + 2,
              updated_at: turn.assistant_message.created_at || new Date().toISOString(),
            }
          }
          return s
        })
      )
    } catch (err) {
      // Draft remains in drafts[targetSessionId]
      const errMsg =
        err instanceof ApiError
          ? err.detail
          : 'Failed to send message. Please check your connection and try again.'
      setSubmitErrors((prev) => ({ ...prev, [targetSessionId]: errMsg }))
    } finally {
      setSubmittingSessions((prev) => {
        const next = { ...prev }
        delete next[targetSessionId]
        return next
      })
    }
  }

  // Filtered sessions list
  const filteredSessions = sessions.filter((s) => {
    if (activeFilter === 'active') return !s.is_archived
    if (activeFilter === 'archived') return s.is_archived
    return true
  })

  const selectedSession = sessions.find((s) => s.id === selectedSessionId) || null

  return (
    <div className="chat-workspace">
      {/* Sidebar: Conversation Sessions */}
      <aside className="chat-sidebar" aria-label="Conversation Sessions Sidebar">
        <div className="chat-sidebar-header">
          <div className="chat-sidebar-title-row">
            <h3>Conversations</h3>
            <span className="badge badge-secondary" title="Total chat sessions">
              {totalSessions}
            </span>
          </div>

          <button
            type="button"
            className="btn btn-primary btn-sm btn-new-chat"
            onClick={() => setIsNewModalOpen(true)}
            aria-label="Create new conversation"
          >
            + New Chat
          </button>
        </div>

        {actionError ? (
          <div className="chat-inline-alert alert-error" role="alert">
            <span>{actionError}</span>
            <button
              type="button"
              className="alert-close-btn"
              onClick={() => setActionError(null)}
              aria-label="Dismiss error"
            >
              &times;
            </button>
          </div>
        ) : null}

        {/* Filter Pills */}
        <div className="chat-filter-tabs" role="tablist" aria-label="Session status filter">
          <button
            type="button"
            role="tab"
            aria-selected={activeFilter === 'all'}
            className={`filter-tab ${activeFilter === 'all' ? 'active' : ''}`}
            onClick={() => setActiveFilter('all')}
          >
            All
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeFilter === 'active'}
            className={`filter-tab ${activeFilter === 'active' ? 'active' : ''}`}
            onClick={() => setActiveFilter('active')}
          >
            Active
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeFilter === 'archived'}
            className={`filter-tab ${activeFilter === 'archived' ? 'active' : ''}`}
            onClick={() => setActiveFilter('archived')}
          >
            Archived
          </button>
        </div>

        {/* Sessions List Content */}
        <div className="chat-sessions-list">
          {isLoadingSessions ? (
            <div className="loading-state chat-loading-state">
              <div className="spinner" aria-label="Loading conversations" />
              <p>Loading sessions...</p>
            </div>
          ) : sessionError ? (
            <div className="chat-error-state">
              <p>{sessionError}</p>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => loadSessions()}
              >
                Retry
              </button>
            </div>
          ) : filteredSessions.length === 0 ? (
            <div className="chat-empty-state">
              <p className="empty-text">
                {activeFilter === 'archived'
                  ? 'No archived conversations.'
                  : 'No conversations found.'}
              </p>
              {activeFilter !== 'archived' && (
                <button
                  type="button"
                  className="btn btn-outline btn-sm"
                  onClick={() => setIsNewModalOpen(true)}
                >
                  Start first chat
                </button>
              )}
            </div>
          ) : (
            <ul className="sessions-ul" role="list">
              {filteredSessions.map((session) => {
                const isSelected = session.id === selectedSessionId
                const isEditing = editingSessionId === session.id
                const isArchiving = isArchivingId === session.id

                return (
                  <li
                    key={session.id}
                    className={`session-item ${isSelected ? 'selected' : ''} ${session.is_archived ? 'archived' : ''}`}
                    onClick={() => handleSelectSession(session)}
                    tabIndex={0}
                    role="button"
                    aria-label={`Chat session: ${session.title}`}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        handleSelectSession(session)
                      }
                    }}
                  >
                    {isEditing ? (
                      <form
                        className="session-rename-form"
                        onSubmit={(e) => handleSaveRename(session.id, e)}
                        onClick={(e) => e.stopPropagation()}
                      >
                        <input
                          type="text"
                          className="input-rename"
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          disabled={isUpdatingTitle}
                          autoFocus
                          maxLength={255}
                          aria-label="Edit conversation title"
                        />
                        <div className="rename-actions">
                          <button
                            type="submit"
                            className="btn btn-primary btn-xs"
                            disabled={isUpdatingTitle || !editTitle.trim()}
                          >
                            Save
                          </button>
                          <button
                            type="button"
                            className="btn btn-secondary btn-xs"
                            onClick={handleCancelRename}
                            disabled={isUpdatingTitle}
                          >
                            Cancel
                          </button>
                        </div>
                      </form>
                    ) : (
                      <>
                        <div className="session-item-header">
                          <span className="session-title" title={session.title}>
                            {session.title}
                          </span>
                          <span
                            className={`scope-pill scope-${session.scope_type.toLowerCase()}`}
                            title={`Scope: ${session.scope_type}`}
                          >
                            {session.scope_type}
                          </span>
                        </div>

                        <div className="session-item-meta">
                          <span className="session-count">
                            {session.message_count} {session.message_count === 1 ? 'msg' : 'msgs'}
                          </span>
                          <span className="session-date">
                            {formatSessionDate(session.updated_at)}
                          </span>
                          {session.is_archived && (
                            <span className="archived-tag">Archived</span>
                          )}
                        </div>

                        <div className="session-item-actions" onClick={(e) => e.stopPropagation()}>
                          <button
                            type="button"
                            className="btn-icon"
                            title="Rename conversation"
                            aria-label={`Rename ${session.title}`}
                            onClick={(e) => handleStartRename(session, e)}
                          >
                            ✏️
                          </button>
                          <button
                            type="button"
                            className="btn-icon"
                            title={session.is_archived ? 'Unarchive conversation' : 'Archive conversation'}
                            aria-label={session.is_archived ? `Unarchive ${session.title}` : `Archive ${session.title}`}
                            onClick={(e) => handleToggleArchive(session, e)}
                            disabled={isArchiving}
                          >
                            {session.is_archived ? '📂' : '📦'}
                          </button>
                          <button
                            type="button"
                            className="btn-icon btn-icon-danger"
                            title={
                              isSessionSubmitting(session.id)
                                ? 'Cannot delete conversation while generation is in progress'
                                : 'Delete conversation'
                            }
                            aria-label={`Delete ${session.title}`}
                            onClick={(e) => handleOpenDeleteModal(session, e)}
                            disabled={isSessionSubmitting(session.id)}
                          >
                            🗑️
                          </button>
                        </div>
                      </>
                    )}
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      </aside>

      {/* Main Conversation Pane */}
      <section className="chat-main" aria-label="Active Conversation Workspace">
        {selectedSession ? (
          <div className="chat-conversation-container">
            {/* Conversation Header */}
            <div className="chat-conversation-header">
              <div className="chat-header-info">
                <h2>{selectedSession.title}</h2>
                <div className="chat-header-badges">
                  {selectedSession.scope_type === 'DOCUMENTS' ? (
                    <div className="scope-documents-wrapper" style={{ position: 'relative', display: 'inline-block' }}>
                      <button
                        type="button"
                        className="scope-pill scope-documents scope-popover-trigger"
                        onClick={() => setShowSourcesPopover((prev) => !prev)}
                        aria-label="View selected documents"
                        aria-expanded={showSourcesPopover}
                      >
                        Scope: {selectedSession.selected_documents?.length || 0} Documents ▾
                      </button>
                      {showSourcesPopover && (
                        <div className="scope-sources-popover" role="dialog" aria-label="Selected Documents List">
                          <div className="popover-header">
                            <strong>Selected Documents ({selectedSession.selected_documents?.length || 0})</strong>
                            <button
                              type="button"
                              className="btn-close-sm"
                              onClick={() => setShowSourcesPopover(false)}
                              aria-label="Close popover"
                            >
                              &times;
                            </button>
                          </div>
                          <ul className="popover-doc-list">
                            {(selectedSession.selected_documents || []).length === 0 ? (
                              <li className="popover-doc-item text-muted text-xs">No documents attached.</li>
                            ) : (
                              (selectedSession.selected_documents || []).map((doc) => (
                                <li key={doc.id} className="popover-doc-item">
                                  <span className="doc-item-title" title={doc.title || doc.filename}>
                                    {doc.title || doc.filename}
                                  </span>
                                  {doc.page_count != null && (
                                    <span className="doc-item-pages text-muted text-xs">
                                      ({doc.page_count} {doc.page_count === 1 ? 'page' : 'pages'})
                                    </span>
                                  )}
                                </li>
                              ))
                            )}
                          </ul>
                        </div>
                      )}
                    </div>
                  ) : (
                    <span className={`scope-pill scope-${selectedSession.scope_type.toLowerCase()}`}>
                      Scope: {selectedSession.scope_type}
                    </span>
                  )}
                  {selectedSession.is_archived ? (
                    <span className="badge badge-warning">Archived (Read-Only)</span>
                  ) : (
                    <span className="badge badge-success">Active</span>
                  )}
                  <span className="text-muted text-sm">
                    Created {formatSessionDate(selectedSession.created_at)}
                  </span>
                </div>
              </div>

              <div className="chat-header-actions">
                <button
                  type="button"
                  className="btn btn-outline btn-sm"
                  onClick={(e) => handleStartRename(selectedSession, e)}
                >
                  Rename
                </button>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={(e) => handleToggleArchive(selectedSession, e)}
                  disabled={isArchivingId === selectedSession.id}
                >
                  {selectedSession.is_archived ? 'Unarchive' : 'Archive'}
                </button>
                <button
                  type="button"
                  className="btn btn-danger btn-sm"
                  onClick={(e) => handleOpenDeleteModal(selectedSession, e)}
                  disabled={isSessionSubmitting(selectedSession.id)}
                  title={
                    isSessionSubmitting(selectedSession.id)
                      ? 'Cannot delete conversation while generation is in progress'
                      : undefined
                  }
                >
                  Delete
                </button>
              </div>
            </div>

            {actionError ? (
              <div className="chat-inline-alert alert-error" style={{ margin: '0.75rem 1.5rem 0 1.5rem' }} role="alert">
                <span>{actionError}</span>
                <button
                  type="button"
                  className="alert-close-btn"
                  onClick={() => setActionError(null)}
                  aria-label="Dismiss error"
                >
                  &times;
                </button>
              </div>
            ) : null}

            {/* Messages Area */}
            <div className="chat-messages-area" aria-label="Conversation message history">
              {isLoadingMessages ? (
                <div className="loading-state chat-loading-state">
                  <div className="spinner" aria-label="Loading message history" />
                  <p>Loading messages...</p>
                </div>
              ) : messageError ? (
                <div className="chat-error-state">
                  <p>{messageError}</p>
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    onClick={() => loadMessages(selectedSession.id)}
                  >
                    Retry
                  </button>
                </div>
              ) : messages.length === 0 && !isSessionSubmitting(selectedSession.id) ? (
                <div className="chat-empty-messages">
                  <div className="empty-messages-icon">💬</div>
                  <h3>No Messages Yet</h3>
                  <p className="text-muted">
                    This conversation is currently empty. Ask a research question to retrieve verified
                    evidence passages from your library.
                  </p>
                  <div className="phase-note">
                    <strong>Tip:</strong> Ask a research question below to retrieve verified evidence passages and grounded answers from your documents.
                  </div>
                </div>
              ) : (
                <div className="chat-messages-stream">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`message-bubble-wrapper ${
                        msg.sender_type === 'USER' ? 'user-turn' : 'assistant-turn'
                      }`}
                    >
                      <div className="message-header-row">
                        <span className="sender-label">
                          {msg.sender_type === 'USER' ? '👤 You' : '🤖 ResearchLens Assistant'}
                        </span>
                        <span className="message-timestamp">
                          {formatMessageTime(msg.created_at)}
                        </span>
                      </div>

                      <div className="message-content">
                        {msg.sender_type === 'ASSISTANT' ? (
                          <MarkdownContent content={msg.content} />
                        ) : (
                          <p>{msg.content}</p>
                        )}
                      </div>

                      {/* Evidence Citations or No Citations Notice */}
                      {msg.sender_type === 'ASSISTANT' && (
                        msg.citations && msg.citations.length > 0 ? (
                          <div className="message-citations" aria-label="Cited source passages">
                            <span className="citations-heading">
                              📎 {msg.citations.length} Verified Evidence Citation
                              {msg.citations.length === 1 ? '' : 's'}:
                            </span>
                            <div className="citations-grid">
                              {msg.citations.map((cit) => (
                                <div key={cit.id} className="citation-card">
                                  <div className="citation-header">
                                    <span className="citation-doc" title={cit.document_title}>
                                      📄 {cit.document_title}
                                    </span>
                                    <span className="citation-page">Page {cit.page_number}</span>
                                    {cit.similarity_score !== null && (
                                      <span
                                        className="citation-score"
                                        title="Vector similarity score"
                                      >
                                        {Math.round(cit.similarity_score * 100)}% match
                                      </span>
                                    )}
                                  </div>
                                  <blockquote className="citation-quote">
                                    &ldquo;{cit.quoted_text}&rdquo;
                                  </blockquote>
                                </div>
                              ))}
                            </div>
                          </div>
                        ) : (
                          <div className="no-citations-notice" aria-label="No citations available">
                            <span className="no-citations-badge">ℹ️ No direct document citations</span>
                            <span className="no-citations-text">
                              This response was generated without direct cited excerpts from the current index.
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  ))}

                  {/* In-Flight Turn Indicator */}
                  {isSessionSubmitting(selectedSession.id) && (
                    <>
                      {submittingSessions[selectedSession.id] && (
                        <div className="message-bubble-wrapper user-turn in-flight-user-turn">
                          <div className="message-header-row">
                            <span className="sender-label">👤 You</span>
                            <span className="message-timestamp">Sending...</span>
                          </div>
                          <div className="message-content">
                            <p>{submittingSessions[selectedSession.id]}</p>
                          </div>
                        </div>
                      )}
                      <div className="message-bubble-wrapper assistant-turn in-flight-turn" aria-live="polite">
                        <div className="message-header-row">
                          <span className="sender-label">🤖 ResearchLens Assistant</span>
                          <span className="message-timestamp">Analyzing...</span>
                        </div>
                        <div className="message-content in-flight-content">
                          <div className="in-flight-spinner-row">
                            <div className="typing-dots" aria-label="Generating response">
                              <span></span>
                              <span></span>
                              <span></span>
                            </div>
                            <span className="in-flight-text">
                              Searching evidence passages and synthesizing answer...
                            </span>
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Interactive Composer / Archived Notice */}
            {selectedSession.is_archived ? (
              <div className="chat-composer-placeholder" aria-label="Message composer area">
                <div className="composer-disabled-box">
                  <span className="composer-lock-icon">🔒</span>
                  <span className="composer-placeholder-text">
                    This conversation is archived. Unarchive to resume questions.
                  </span>
                </div>
              </div>
            ) : (
              <div className="chat-composer-container" aria-label="Message composer area">
                {submitErrors[selectedSession.id] ? (
                  <div className="chat-inline-alert alert-error" style={{ marginBottom: '0.75rem' }} role="alert">
                    <span>{submitErrors[selectedSession.id]}</span>
                    <button
                      type="button"
                      className="alert-close-btn"
                      onClick={() =>
                        setSubmitErrors((prev) => ({ ...prev, [selectedSession.id]: null }))
                      }
                      aria-label="Dismiss error"
                    >
                      &times;
                    </button>
                  </div>
                ) : null}

                <form
                  className="chat-composer-form"
                  onSubmit={handleSendMessage}
                  aria-label="Ask a research question form"
                >
                  <div className="composer-input-row">
                    <textarea
                      id="chat-message-input"
                      className="chat-composer-textarea"
                      rows={2}
                      placeholder="Ask a research question about your documents... (Shift+Enter for newline)"
                      aria-label="Ask a research question"
                      value={drafts[selectedSession.id] || ''}
                      onChange={(e) => {
                        const val = e.target.value
                        setDrafts((prev) => ({ ...prev, [selectedSession.id]: val }))
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                          e.preventDefault()
                          handleSendMessage()
                        }
                      }}
                      disabled={isSessionSubmitting(selectedSession.id)}
                    />
                    <button
                      type="submit"
                      className="btn btn-primary chat-send-btn"
                      disabled={
                        isSessionSubmitting(selectedSession.id) ||
                        !(drafts[selectedSession.id] || '').trim()
                      }
                      aria-label="Send research question"
                    >
                      {isSessionSubmitting(selectedSession.id) ? (
                        <span className="send-spinner-text">
                          <span className="mini-spinner" aria-hidden="true" />
                          Sending...
                        </span>
                      ) : (
                        'Send'
                      )}
                    </button>
                  </div>
                  <div className="composer-hint-row">
                    <span className="composer-hint">
                      Press <strong>Enter</strong> to send &bull; <strong>Shift+Enter</strong> for newline
                    </span>
                  </div>
                </form>
              </div>
            )}
          </div>
        ) : messageError ? (
          <div className="chat-no-selection" role="alert">
            <div className="no-selection-illustration">⚠️</div>
            <h3>Conversation Not Available</h3>
            <p className="text-muted">{messageError}</p>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => {
                setSelectedSessionId(null)
                window.location.hash = '#chat'
              }}
            >
              Back to Conversations
            </button>
          </div>
        ) : (isLoadingMessages || isLoadingSessions) ? (
          <div className="loading-state chat-loading-state">
            <div className="spinner" aria-label="Loading conversation" />
            <p>Loading conversation...</p>
          </div>
        ) : (
          <div className="chat-no-selection">
            <div className="no-selection-illustration">💡</div>
            <h3>Select a Conversation</h3>
            <p className="text-muted">
              Choose an existing chat from the sidebar or click <strong>+ New Chat</strong> to start
              a grounded dialogue with your research documents.
            </p>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => setIsNewModalOpen(true)}
            >
              + Start New Chat
            </button>
          </div>
        )}
      </section>

      <DeleteChatModal
        session={sessionToDelete}
        isOpen={isDeleteModalOpen}
        onClose={handleCloseDeleteModal}
        onSuccess={handleDeleteSuccess}
      />

      <NewConversationModal
        isOpen={isNewModalOpen}
        onClose={() => setIsNewModalOpen(false)}
        onSuccess={handleNewConversationSuccess}
      />
    </div>
  )
}
