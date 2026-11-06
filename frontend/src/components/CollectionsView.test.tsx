import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { CollectionsView } from './CollectionsView'
import { api, ApiError } from '../services/api'
import { CollectionResponse, DocumentResponse } from '../types'

const mockCollections: CollectionResponse[] = [
  {
    id: 'col-1',
    name: 'Quantum Optics',
    description: 'Photon entanglement and cavity QED research',
    document_count: 2,
    created_at: '2026-10-18T10:00:00Z',
    updated_at: '2026-10-18T10:00:00Z',
  },
  {
    id: 'col-2',
    name: 'Deep Learning',
    description: null,
    document_count: 0,
    created_at: '2026-10-18T11:00:00Z',
    updated_at: '2026-10-18T11:00:00Z',
  },
]

const mockMemberDocs: DocumentResponse[] = [
  {
    id: 'doc-1',
    filename: 'quantum_paper.pdf',
    title: 'Quantum Entanglement',
    file_size_bytes: 1048576,
    page_count: 14,
    processing_status: 'READY',
    created_at: '2026-10-18T10:00:00Z',
    updated_at: '2026-10-18T10:00:00Z',
  },
  {
    id: 'doc-2',
    filename: 'cavity_qed.pdf',
    title: null,
    file_size_bytes: 2097152,
    page_count: 22,
    processing_status: 'EXTRACTING',
    created_at: '2026-10-18T11:00:00Z',
    updated_at: '2026-10-18T11:00:00Z',
  },
]

describe('CollectionsView Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.spyOn(api.collections, 'list').mockResolvedValue({
      items: mockCollections,
      total: mockCollections.length,
      limit: 50,
      offset: 0,
    })
    vi.spyOn(api.collections, 'listDocuments').mockResolvedValue({
      items: mockMemberDocs,
      total: mockMemberDocs.length,
      limit: 50,
      offset: 0,
    })
    vi.spyOn(api.collections, 'removeDocument').mockResolvedValue({
      ...mockCollections[0],
      document_count: 1,
    })
  })

  it('renders loading state and displays collections grid', async () => {
    render(<CollectionsView />)

    expect(screen.getByText(/Loading research collections.../i)).toBeDefined()

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Research Collections/i })).toBeDefined()
    })

    expect(screen.getByText('2 Collections')).toBeDefined()
    expect(screen.getByText('Quantum Optics')).toBeDefined()
    expect(screen.getByText('Photon entanglement and cavity QED research')).toBeDefined()
    expect(screen.getByText('2 docs')).toBeDefined()
    expect(screen.getByText('Deep Learning')).toBeDefined()
    expect(screen.getByText('No description provided')).toBeDefined()
    expect(screen.getByText('0 docs')).toBeDefined()
  })

  it('renders empty state when user has no collections', async () => {
    vi.spyOn(api.collections, 'list').mockResolvedValueOnce({
      items: [],
      total: 0,
      limit: 50,
      offset: 0,
    })

    render(<CollectionsView />)

    await waitFor(() => {
      expect(screen.getByText(/No Collections Found/i)).toBeDefined()
    })

    expect(screen.getByRole('button', { name: /\+ Create First Collection/i })).toBeDefined()
  })

  it('opens collection detail interface when clicking a collection card', async () => {
    render(<CollectionsView />)

    await waitFor(() => {
      expect(screen.getByText('Quantum Optics')).toBeDefined()
    })

    // Click on Quantum Optics card
    fireEvent.click(screen.getByText('Quantum Optics'))

    // Detail interface appears
    await waitFor(() => {
      expect(screen.getByText('← Back to Collections')).toBeDefined()
    })

    expect(screen.getByRole('heading', { name: 'Quantum Optics' })).toBeDefined()
    expect(screen.getByText('Photon entanglement and cavity QED research')).toBeDefined()
    expect(screen.getByText('2 Documents')).toBeDefined()

    // Member documents loaded
    await waitFor(() => {
      expect(screen.getByText('Quantum Entanglement')).toBeDefined()
      expect(screen.getByText('cavity_qed.pdf')).toBeDefined()
    })

    // Check processing statuses are displayed
    expect(screen.getByText('READY')).toBeDefined()
    expect(screen.getByText('EXTRACTING')).toBeDefined()
  })

  it('displays empty state in collection detail when collection has no documents', async () => {
    vi.spyOn(api.collections, 'listDocuments').mockResolvedValueOnce({
      items: [],
      total: 0,
      limit: 50,
      offset: 0,
    })

    render(<CollectionsView />)

    await waitFor(() => {
      expect(screen.getByText('Deep Learning')).toBeDefined()
    })

    fireEvent.click(screen.getByText('Deep Learning'))

    await waitFor(() => {
      expect(screen.getByText(/No Documents in this Collection/i)).toBeDefined()
    })

    expect(screen.getByText(/Go to the Documents tab and click "\+ Add to Collection"/i)).toBeDefined()
  })

  it('removes document from collection and updates count correctly', async () => {
    render(<CollectionsView />)

    await waitFor(() => {
      expect(screen.getByText('Quantum Optics')).toBeDefined()
    })

    fireEvent.click(screen.getByText('Quantum Optics'))

    await waitFor(() => {
      expect(screen.getByText('Quantum Entanglement')).toBeDefined()
    })

    // Click remove button on Quantum Entanglement
    const removeBtn = screen.getByRole('button', { name: 'Remove Quantum Entanglement from collection' })
    fireEvent.click(removeBtn)

    // Modal opens
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Remove Document from Collection' })).toBeDefined()
    })

    expect(screen.getByText(/The original research paper in your library will/i)).toBeDefined()

    // Confirm removal
    const confirmRemoveBtn = screen.getByRole('button', { name: 'Remove from Collection' })
    fireEvent.click(confirmRemoveBtn)

    await waitFor(() => {
      expect(api.collections.removeDocument).toHaveBeenCalledWith('col-1', 'doc-1')
    })

    // Quantum Entanglement removed from table
    await waitFor(() => {
      expect(screen.queryByText('Quantum Entanglement')).toBeNull()
    })

    // cavity_qed.pdf remains
    expect(screen.getByText('cavity_qed.pdf')).toBeDefined()

    // Count synchronized to 1 Document
    expect(screen.getByText('1 Document')).toBeDefined()

    // Click back to collections
    fireEvent.click(screen.getByText('← Back to Collections'))

    // Card in list also displays updated count (1 doc)
    await waitFor(() => {
      expect(screen.getByText('Quantum Optics')).toBeDefined()
    })
    expect(screen.getByText('1 doc')).toBeDefined()
  })

  it('handles error during removal gracefully and keeps document in list', async () => {
    vi.spyOn(api.collections, 'removeDocument').mockRejectedValueOnce(
      new ApiError(404, 'Document is not a member of this collection.')
    )

    render(<CollectionsView />)

    await waitFor(() => {
      expect(screen.getByText('Quantum Optics')).toBeDefined()
    })

    fireEvent.click(screen.getByText('Quantum Optics'))

    await waitFor(() => {
      expect(screen.getByText('Quantum Entanglement')).toBeDefined()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Remove Quantum Entanglement from collection' }))

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Remove from Collection' })).toBeDefined()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Remove from Collection' }))

    await waitFor(() => {
      expect(screen.getByText('Document is not a member of this collection.')).toBeDefined()
    })

    // Cancel modal
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    // Document is still present
    expect(screen.getByText('Quantum Entanglement')).toBeDefined()
  })

  it('handles member list load error with Retry button', async () => {
    vi.spyOn(api.collections, 'listDocuments')
      .mockRejectedValueOnce(new ApiError(500, 'Server error loading members'))
      .mockResolvedValueOnce({
        items: mockMemberDocs,
        total: mockMemberDocs.length,
        limit: 50,
        offset: 0,
      })

    render(<CollectionsView />)

    await waitFor(() => {
      expect(screen.getByText('Quantum Optics')).toBeDefined()
    })

    fireEvent.click(screen.getByText('Quantum Optics'))

    await waitFor(() => {
      expect(screen.getByText('Server error loading members')).toBeDefined()
    })

    fireEvent.click(screen.getByRole('button', { name: /Retry/i }))

    await waitFor(() => {
      expect(screen.getByText('Quantum Entanglement')).toBeDefined()
    })
  })
})
