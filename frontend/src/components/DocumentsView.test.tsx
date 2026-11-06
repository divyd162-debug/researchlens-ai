import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { DocumentsView } from './DocumentsView'
import { api, ApiError } from '../services/api'
import { DocumentResponse } from '../types'

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
    processing_status: 'PROCESSING',
    created_at: '2026-10-18T11:00:00Z',
    updated_at: '2026-10-18T11:00:00Z',
  },
]

describe('DocumentsView Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(api.documents, 'list').mockResolvedValue({
      items: mockDocs,
      total: mockDocs.length,
      limit: 50,
      offset: 0,
    })
    vi.spyOn(api.documents, 'delete').mockResolvedValue({
      message: 'Document deleted successfully',
    })
  })

  it('renders loading state initially and then displays document table with actions', async () => {
    render(<DocumentsView />)

    expect(screen.getByText(/Loading research documents.../i)).toBeDefined()

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Research Documents/i })).toBeDefined()
    })

    expect(screen.getByText('2 Documents')).toBeDefined()
    expect(screen.getByText('Attention Is All You Need')).toBeDefined()
    expect(screen.getByText('attention_is_all_you_need.pdf')).toBeDefined()
    expect(screen.getByText('bert_paper.pdf')).toBeDefined()

    const deleteButtons = screen.getAllByRole('button', { name: /Delete/i })
    expect(deleteButtons.length).toBe(2)
    expect(screen.getByRole('button', { name: 'Delete Attention Is All You Need' })).toBeDefined()
    expect(screen.getByRole('button', { name: 'Delete bert_paper.pdf' })).toBeDefined()
  })

  it('renders empty state when no documents exist', async () => {
    vi.spyOn(api.documents, 'list').mockResolvedValue({
      items: [],
      total: 0,
      limit: 50,
      offset: 0,
    })

    render(<DocumentsView />)

    await waitFor(() => {
      expect(screen.getByText(/No Documents Uploaded/i)).toBeDefined()
    })

    expect(screen.getByText('0 Documents')).toBeDefined()
    expect(screen.getByRole('button', { name: /\+ Upload First Document/i })).toBeDefined()
  })

  it('opens DeleteDocumentModal with document title and confirmation warning when clicking Delete', async () => {
    render(<DocumentsView />)

    await waitFor(() => {
      expect(screen.getByText('Attention Is All You Need')).toBeDefined()
    })

    const deleteBtn = screen.getByRole('button', { name: 'Delete Attention Is All You Need' })
    fireEvent.click(deleteBtn)

    const modal = screen.getByRole('dialog')
    expect(modal).toBeDefined()
    expect(screen.getByRole('heading', { name: 'Delete Document' })).toBeDefined()
    expect(screen.getByText(/Are you sure you want to delete/i)).toBeDefined()
    expect(within(modal).getByText('Attention Is All You Need')).toBeDefined()
    expect(
      screen.getByText(/Historical chat conversations and citation snapshots referencing this document will be preserved/i)
    ).toBeDefined()

    expect(screen.getByRole('button', { name: /Cancel/i })).toBeDefined()
    expect(screen.getByRole('button', { name: 'Confirm delete document' })).toBeDefined()
  })

  it('closes DeleteDocumentModal without calling API when Cancel is clicked', async () => {
    render(<DocumentsView />)

    await waitFor(() => {
      expect(screen.getByText('Attention Is All You Need')).toBeDefined()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Delete Attention Is All You Need' }))
    expect(screen.getByRole('dialog')).toBeDefined()

    fireEvent.click(screen.getByRole('button', { name: /Cancel/i }))

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull()
    })

    expect(api.documents.delete).not.toHaveBeenCalled()
    expect(screen.getByText('Attention Is All You Need')).toBeDefined()
    expect(screen.getByText('2 Documents')).toBeDefined()
  })

  it('closes DeleteDocumentModal without calling API when close (×) button is clicked', async () => {
    render(<DocumentsView />)

    await waitFor(() => {
      expect(screen.getByText('Attention Is All You Need')).toBeDefined()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Delete Attention Is All You Need' }))
    expect(screen.getByRole('dialog')).toBeDefined()

    fireEvent.click(screen.getByRole('button', { name: /Close dialog/i }))

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull()
    })

    expect(api.documents.delete).not.toHaveBeenCalled()
  })

  it('successfully deletes document, updates UI list, and decrements total count badge', async () => {
    render(<DocumentsView />)

    await waitFor(() => {
      expect(screen.getByText('Attention Is All You Need')).toBeDefined()
    })

    expect(screen.getByText('2 Documents')).toBeDefined()

    // Trigger delete on doc-uuid-1
    fireEvent.click(screen.getByRole('button', { name: 'Delete Attention Is All You Need' }))

    // Confirm deletion in modal
    const confirmBtn = screen.getByRole('button', { name: 'Confirm delete document' })
    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(api.documents.delete).toHaveBeenCalledWith('doc-uuid-1')
    })

    // Modal closes
    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull()
    })

    // doc-uuid-1 is removed from table
    expect(screen.queryByText('Attention Is All You Need')).toBeNull()

    // doc-uuid-2 remains in table
    expect(screen.getByText('bert_paper.pdf')).toBeDefined()

    // Total count badge decremented to 1 Document
    expect(screen.getByText('1 Document')).toBeDefined()
  })

  it('displays API error message in modal when deletion fails and keeps document in list', async () => {
    vi.spyOn(api.documents, 'delete').mockRejectedValueOnce(
      new ApiError(500, 'Failed to unlink document vector embeddings from index.')
    )

    render(<DocumentsView />)

    await waitFor(() => {
      expect(screen.getByText('Attention Is All You Need')).toBeDefined()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Delete Attention Is All You Need' }))
    fireEvent.click(screen.getByRole('button', { name: 'Confirm delete document' }))

    await waitFor(() => {
      expect(screen.getByText('Failed to unlink document vector embeddings from index.')).toBeDefined()
    })

    // Modal stays open so the user can inspect or retry
    expect(screen.getByRole('dialog')).toBeDefined()

    // Document is NOT removed from background table
    const table = screen.getByRole('table')
    expect(within(table).getByText('Attention Is All You Need')).toBeDefined()
    expect(screen.getByText('2 Documents')).toBeDefined()
  })

  it('shows in-flight Deleting... state and disables actions during deletion', async () => {
    let resolveDeletePromise!: (value: { message: string }) => void
    const deletePromise = new Promise<{ message: string }>((resolve) => {
      resolveDeletePromise = resolve
    })

    vi.spyOn(api.documents, 'delete').mockReturnValueOnce(deletePromise)

    render(<DocumentsView />)

    await waitFor(() => {
      expect(screen.getByText('Attention Is All You Need')).toBeDefined()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Delete Attention Is All You Need' }))
    const confirmBtn = screen.getByRole('button', { name: 'Confirm delete document' })
    const cancelBtn = screen.getByRole('button', { name: /Cancel/i })
    const closeBtn = screen.getByRole('button', { name: /Close dialog/i })

    fireEvent.click(confirmBtn)

    // Button shows "Deleting..." and is disabled
    expect(confirmBtn.textContent).toBe('Deleting...')
    expect((confirmBtn as HTMLButtonElement).disabled).toBe(true)
    expect((cancelBtn as HTMLButtonElement).disabled).toBe(true)
    expect((closeBtn as HTMLButtonElement).disabled).toBe(true)

    // Resolve in-flight request
    resolveDeletePromise({ message: 'Document deleted successfully' })

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).toBeNull()
    })
  })

  it('handles load error with Retry button', async () => {
    vi.spyOn(api.documents, 'list')
      .mockRejectedValueOnce(new ApiError(500, 'Database unavailable'))
      .mockResolvedValueOnce({
        items: mockDocs,
        total: mockDocs.length,
        limit: 50,
        offset: 0,
      })

    render(<DocumentsView />)

    await waitFor(() => {
      expect(screen.getByText('Database unavailable')).toBeDefined()
    })

    fireEvent.click(screen.getByRole('button', { name: /Retry/i }))

    await waitFor(() => {
      expect(screen.getByText('Attention Is All You Need')).toBeDefined()
    })
  })

  describe('Add to Collection workflow', () => {
    it('opens AddToCollectionModal and displays collections dropdown', async () => {
      vi.spyOn(api.collections, 'list').mockResolvedValueOnce({
        items: [
          {
            id: 'col-1',
            name: 'ML Papers',
            description: 'Machine learning',
            document_count: 2,
            created_at: '2026-10-18T10:00:00Z',
            updated_at: '2026-10-18T10:00:00Z',
          },
        ],
        total: 1,
        limit: 100,
        offset: 0,
      })

      render(<DocumentsView />)

      await waitFor(() => {
        expect(screen.getByText('Attention Is All You Need')).toBeDefined()
      })

      const addBtn = screen.getByRole('button', { name: 'Add Attention Is All You Need to collection' })
      fireEvent.click(addBtn)

      await waitFor(() => {
        expect(screen.getByRole('heading', { name: 'Add to Collection' })).toBeDefined()
      })

      await waitFor(() => {
        expect(screen.getByText(/ML Papers \(2 docs\)/i)).toBeDefined()
      })

      expect(screen.getByRole('button', { name: 'Add to Collection' })).toBeDefined()
    })

    it('successfully adds document to collection and shows success state', async () => {
      vi.spyOn(api.collections, 'list').mockResolvedValueOnce({
        items: [
          {
            id: 'col-1',
            name: 'ML Papers',
            description: 'Machine learning',
            document_count: 2,
            created_at: '2026-10-18T10:00:00Z',
            updated_at: '2026-10-18T10:00:00Z',
          },
        ],
        total: 1,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.collections, 'addDocuments').mockResolvedValueOnce({
        id: 'col-1',
        name: 'ML Papers',
        description: 'Machine learning',
        document_count: 3,
        created_at: '2026-10-18T10:00:00Z',
        updated_at: '2026-10-18T10:00:00Z',
      })

      render(<DocumentsView />)

      await waitFor(() => {
        expect(screen.getByText('Attention Is All You Need')).toBeDefined()
      })

      fireEvent.click(screen.getByRole('button', { name: 'Add Attention Is All You Need to collection' }))

      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Add to Collection' })).toBeDefined()
      })

      fireEvent.click(screen.getByRole('button', { name: 'Add to Collection' }))

      await waitFor(() => {
        expect(api.collections.addDocuments).toHaveBeenCalledWith('col-1', {
          document_ids: ['doc-uuid-1'],
        })
      })

      await waitFor(() => {
        expect(screen.getByText(/Document added to collection successfully!/i)).toBeDefined()
      })
    })

    it('shows informative message when no collections exist', async () => {
      vi.spyOn(api.collections, 'list').mockResolvedValueOnce({
        items: [],
        total: 0,
        limit: 100,
        offset: 0,
      })

      render(<DocumentsView />)

      await waitFor(() => {
        expect(screen.getByText('Attention Is All You Need')).toBeDefined()
      })

      fireEvent.click(screen.getByRole('button', { name: 'Add Attention Is All You Need to collection' }))

      await waitFor(() => {
        expect(screen.getByText(/You have not created any collections yet/i)).toBeDefined()
      })

      const submitBtn = screen.getByRole('button', { name: 'Add to Collection' })
      expect((submitBtn as HTMLButtonElement).disabled).toBe(true)
    })

    it('displays error in modal when addDocuments API call fails', async () => {
      vi.spyOn(api.collections, 'list').mockResolvedValueOnce({
        items: [
          {
            id: 'col-1',
            name: 'ML Papers',
            description: null,
            document_count: 0,
            created_at: '2026-10-18T10:00:00Z',
            updated_at: '2026-10-18T10:00:00Z',
          },
        ],
        total: 1,
        limit: 100,
        offset: 0,
      })
      vi.spyOn(api.collections, 'addDocuments').mockRejectedValueOnce(
        new ApiError(404, 'Collection not found.')
      )

      render(<DocumentsView />)

      await waitFor(() => {
        expect(screen.getByText('Attention Is All You Need')).toBeDefined()
      })

      fireEvent.click(screen.getByRole('button', { name: 'Add Attention Is All You Need to collection' }))

      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Add to Collection' })).toBeDefined()
      })

      fireEvent.click(screen.getByRole('button', { name: 'Add to Collection' }))

      await waitFor(() => {
        expect(screen.getByText('Collection not found.')).toBeDefined()
      })
    })
  })
})
