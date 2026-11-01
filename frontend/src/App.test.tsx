import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import App from './App'
import { api, ApiError, clearToken, setToken } from './services/api'

describe('ResearchLens AI Frontend Shell & Authentication', () => {
  beforeEach(() => {
    localStorage.clear()
    clearToken()
    window.location.hash = ''
    vi.restoreAllMocks()
  })

  it('renders application shell, navbar branding, and initial sign-in view', async () => {
    render(<App />)

    // Verify brand heading in navbar
    expect(screen.getAllByText(/ResearchLens AI/i).length).toBeGreaterThan(0)

    // Verify unauthenticated navbar links
    const nav = screen.getByRole('navigation', { name: /main navigation/i })
    expect(within(nav).getByRole('button', { name: /sign in/i })).toBeDefined()
    expect(within(nav).getByRole('button', { name: /register/i })).toBeDefined()

    // Verify login form is presented by default for unauthenticated users
    expect(screen.getByRole('heading', { name: /Sign In to ResearchLens AI/i })).toBeDefined()
    const form = screen.getByRole('form', { name: /sign in form/i })
    expect(within(form).getByLabelText(/Email Address/i)).toBeDefined()
    expect(within(form).getByLabelText(/Password/i)).toBeDefined()
  })

  it('toggles between Login and Register views', async () => {
    render(<App />)

    // Switch to Register via navbar link
    const nav = screen.getByRole('navigation', { name: /main navigation/i })
    fireEvent.click(within(nav).getByRole('button', { name: /register/i }))

    expect(await screen.findByRole('heading', { name: /Create an Account/i })).toBeDefined()
    const regForm = screen.getByRole('form', { name: /registration form/i })
    expect(within(regForm).getByLabelText(/Confirm Password/i)).toBeDefined()

    // Switch back to Login via navbar
    const signInBtn = within(nav).getByRole('button', { name: /sign in/i })
    fireEvent.click(signInBtn)

    expect(await screen.findByRole('heading', { name: /Sign In to ResearchLens AI/i })).toBeDefined()
  })

  it('validates required fields and email format on login', async () => {
    render(<App />)

    const form = screen.getByRole('form', { name: /sign in form/i })
    const submitBtn = within(form).getByRole('button', { name: /sign in/i })

    // Empty submission
    fireEvent.click(submitBtn)
    expect(await screen.findByText(/Please enter your email address/i)).toBeDefined()

    // Invalid email structure
    const emailInput = within(form).getByLabelText(/Email Address/i)
    fireEvent.change(emailInput, { target: { value: 'notanemail' } })
    fireEvent.click(submitBtn)
    expect(await screen.findByText(/Please enter a valid email address/i)).toBeDefined()

    // Empty password
    fireEvent.change(emailInput, { target: { value: 'test@example.com' } })
    fireEvent.click(submitBtn)
    expect(await screen.findByText(/Please enter your password/i)).toBeDefined()
  })

  it('validates password length and matching confirmation on registration', async () => {
    render(<App />)

    const nav = screen.getByRole('navigation', { name: /main navigation/i })
    fireEvent.click(within(nav).getByRole('button', { name: /register/i }))
    await screen.findByRole('heading', { name: /Create an Account/i })

    const regForm = screen.getByRole('form', { name: /registration form/i })
    const emailInput = within(regForm).getByLabelText(/Email Address/i)
    const passwordInput = within(regForm).getByLabelText(/^Password/i)
    const confirmInput = within(regForm).getByLabelText(/Confirm Password/i)
    const createBtn = within(regForm).getByRole('button', { name: /create account/i })

    // Short password (< 8 chars)
    fireEvent.change(emailInput, { target: { value: 'newuser@example.com' } })
    fireEvent.change(passwordInput, { target: { value: '123' } })
    fireEvent.change(confirmInput, { target: { value: '123' } })
    fireEvent.click(createBtn)
    expect(await screen.findByText(/Password must be at least 8 characters long/i)).toBeDefined()

    // Password mismatch
    fireEvent.change(passwordInput, { target: { value: 'securepassword123' } })
    fireEvent.change(confirmInput, { target: { value: 'differentpassword' } })
    fireEvent.click(createBtn)
    expect(await screen.findByText(/Passwords do not match/i)).toBeDefined()
  })

  it('authenticates user on login, updates navbar with navigation tabs and displays dashboard', async () => {
    vi.spyOn(api.auth, 'login').mockResolvedValueOnce({
      access_token: 'fake-jwt-token-12345',
      token_type: 'bearer',
      expires_in: 3600,
    })

    vi.spyOn(api.auth, 'getMe').mockResolvedValueOnce({
      id: '12345678-1234-1234-1234-123456789abc',
      email: 'alice@researcher.com',
      is_active: true,
      is_verified: false,
      created_at: '2026-10-17T20:00:00Z',
      updated_at: '2026-10-17T20:00:00Z',
    })

    vi.spyOn(api.collections, 'list').mockResolvedValueOnce({
      items: [],
      total: 3,
      limit: 1,
      offset: 0,
    })

    vi.spyOn(api.documents, 'list').mockResolvedValueOnce({
      items: [],
      total: 5,
      limit: 1,
      offset: 0,
    })

    render(<App />)

    const form = screen.getByRole('form', { name: /sign in form/i })
    const emailInput = within(form).getByLabelText(/Email Address/i)
    const passwordInput = within(form).getByLabelText(/Password/i)
    const submitBtn = within(form).getByRole('button', { name: /sign in/i })

    fireEvent.change(emailInput, { target: { value: 'alice@researcher.com' } })
    fireEvent.change(passwordInput, { target: { value: 'mypassword123' } })
    fireEvent.click(submitBtn)

    // Wait for authentication and dashboard view
    expect(await screen.findByRole('heading', { name: /Welcome back, alice@researcher.com/i })).toBeDefined()
    expect(screen.getByTitle('alice@researcher.com')).toBeDefined()
    expect(screen.getByRole('button', { name: /log out/i })).toBeDefined()

    // Authenticated navigation items visible
    const nav = screen.getByRole('navigation', { name: /main navigation/i })
    expect(within(nav).getByRole('button', { name: /dashboard/i })).toBeDefined()
    expect(within(nav).getByRole('button', { name: /collections/i })).toBeDefined()
    expect(within(nav).getByRole('button', { name: /documents/i })).toBeDefined()

    // Metrics counts rendered
    expect(await screen.findByText('3')).toBeDefined()
    expect(await screen.findByText('5')).toBeDefined()
  })

  it('navigates to Collections and Documents views and displays content', async () => {
    // Start with authenticated token in storage
    setToken('valid-existing-token')

    vi.spyOn(api.auth, 'getMe').mockResolvedValue({
      id: '12345678-1234-1234-1234-123456789abc',
      email: 'alice@researcher.com',
      is_active: true,
      is_verified: false,
      created_at: '2026-10-17T20:00:00Z',
      updated_at: '2026-10-17T20:00:00Z',
    })

    vi.spyOn(api.collections, 'list').mockResolvedValue({
      items: [
        {
          id: 'col-1',
          name: 'Quantum Optics',
          description: 'Papers on photon entanglement and cavity QED',
          document_count: 2,
          created_at: '2026-10-17T18:00:00Z',
          updated_at: '2026-10-17T18:00:00Z',
        },
      ],
      total: 1,
      limit: 50,
      offset: 0,
    })

    vi.spyOn(api.documents, 'list').mockResolvedValue({
      items: [
        {
          id: 'doc-1',
          filename: 'quantum_entanglement.pdf',
          title: 'Quantum Entanglement in Microcavities',
          file_size_bytes: 1048576,
          page_count: 12,
          processing_status: 'PENDING',
          error_message: null,
          created_at: '2026-10-17T18:30:00Z',
          updated_at: '2026-10-17T18:30:00Z',
        },
      ],
      total: 1,
      limit: 50,
      offset: 0,
    })

    render(<App />)

    // Wait for session verification
    expect(await screen.findByTitle('alice@researcher.com')).toBeDefined()

    const nav = screen.getByRole('navigation', { name: /main navigation/i })

    // Navigate to Collections
    fireEvent.click(within(nav).getByRole('button', { name: /^collections$/i }))
    expect(await screen.findByText('Quantum Optics')).toBeDefined()
    expect(screen.getByText(/Papers on photon entanglement and cavity QED/i)).toBeDefined()

    // Navigate to Documents
    fireEvent.click(within(nav).getByRole('button', { name: /^documents$/i }))
    expect(await screen.findByText('Quantum Entanglement in Microcavities')).toBeDefined()
    expect(screen.getByText('1.00 MB')).toBeDefined()
    expect(screen.getAllByText('PENDING').length).toBeGreaterThan(0)
  })

  it('logs out and returns to sign-in view', async () => {
    setToken('valid-existing-token')

    vi.spyOn(api.auth, 'getMe').mockResolvedValue({
      id: '12345678-1234-1234-1234-123456789abc',
      email: 'bob@researcher.com',
      is_active: true,
      is_verified: false,
      created_at: '2026-10-17T20:00:00Z',
      updated_at: '2026-10-17T20:00:00Z',
    })

    vi.spyOn(api.collections, 'list').mockResolvedValue({ items: [], total: 0, limit: 1, offset: 0 })
    vi.spyOn(api.documents, 'list').mockResolvedValue({ items: [], total: 0, limit: 1, offset: 0 })

    render(<App />)

    expect(await screen.findByRole('heading', { name: /Welcome back, bob@researcher.com/i })).toBeDefined()

    const logoutBtn = screen.getByRole('button', { name: /log out/i })
    fireEvent.click(logoutBtn)

    // Verify token was cleared from localStorage
    await waitFor(() => {
      expect(localStorage.getItem('researchlens_access_token')).toBeNull()
    })

    // Verify returned to sign in view
    expect(await screen.findByRole('heading', { name: /Sign In to ResearchLens AI/i })).toBeDefined()
    expect(screen.queryByRole('button', { name: /log out/i })).toBeNull()
  })

  it('displays API error banner when login credentials are invalid (401)', async () => {
    vi.spyOn(api.auth, 'login').mockRejectedValueOnce(
      new ApiError(401, 'Invalid email or password.')
    )

    render(<App />)

    const form = screen.getByRole('form', { name: /sign in form/i })
    fireEvent.change(within(form).getByLabelText(/Email Address/i), { target: { value: 'wrong@example.com' } })
    fireEvent.change(within(form).getByLabelText(/Password/i), { target: { value: 'wrongpass' } })
    fireEvent.click(within(form).getByRole('button', { name: /sign in/i }))

    expect(await screen.findByRole('alert')).toBeDefined()
    expect(screen.getByText(/Invalid email or password/i)).toBeDefined()
  })

  it('displays API error banner when registration email is already taken (409)', async () => {
    render(<App />)
    const nav = screen.getByRole('navigation', { name: /main navigation/i })

    fireEvent.click(within(nav).getByRole('button', { name: /register/i }))
    await screen.findByRole('heading', { name: /Create an Account/i })

    vi.spyOn(api.auth, 'register').mockRejectedValueOnce(
      new ApiError(409, 'An account with this email already exists.')
    )

    const regForm = screen.getByRole('form', { name: /registration form/i })
    fireEvent.change(within(regForm).getByLabelText(/Email Address/i), { target: { value: 'existing@example.com' } })
    fireEvent.change(within(regForm).getByLabelText(/^Password/i), { target: { value: 'validpassword123' } })
    fireEvent.change(within(regForm).getByLabelText(/Confirm Password/i), { target: { value: 'validpassword123' } })
    fireEvent.click(within(regForm).getByRole('button', { name: /create account/i }))

    expect(await screen.findByRole('alert')).toBeDefined()
    expect(screen.getByText(/An account with this email already exists/i)).toBeDefined()
  })

  it('handles expired session by resetting state and clearing stored token', async () => {
    setToken('expired-jwt-token')

    // Simulate getMe failing with 401
    vi.spyOn(api.auth, 'getMe').mockRejectedValueOnce(
      new Error('Token expired')
    )

    render(<App />)

    // Verify token was cleared
    await waitFor(() => {
      expect(localStorage.getItem('researchlens_access_token')).toBeNull()
    })

    // User should see login view
    expect(await screen.findByRole('heading', { name: /Sign In to ResearchLens AI/i })).toBeDefined()
  })

  it('guards protected views when accessed directly via hash by unauthenticated user', async () => {
    window.location.hash = '#collections'

    render(<App />)

    // Should redirect to login screen
    expect(await screen.findByRole('heading', { name: /Sign In to ResearchLens AI/i })).toBeDefined()
    expect(screen.queryByText(/Quantum Optics/i)).toBeNull()
  })

  it('creates a new collection successfully and refreshes the collections list', async () => {
    setToken('valid-existing-token')

    vi.spyOn(api.auth, 'getMe').mockResolvedValue({
      id: '12345678-1234-1234-1234-123456789abc',
      email: 'alice@researcher.com',
      is_active: true,
      is_verified: false,
      created_at: '2026-10-17T20:00:00Z',
      updated_at: '2026-10-17T20:00:00Z',
    })

    vi.spyOn(api.collections, 'list')
      .mockResolvedValueOnce({ items: [], total: 0, limit: 50, offset: 0 })
      .mockResolvedValueOnce({
        items: [
          {
            id: 'col-new-1',
            name: 'Deep Learning Optimization',
            description: 'Papers on AdamW, gradient clipping, and schedules',
            document_count: 0,
            created_at: '2026-10-17T21:00:00Z',
            updated_at: '2026-10-17T21:00:00Z',
          },
        ],
        total: 1,
        limit: 50,
        offset: 0,
      })

    const createSpy = vi.spyOn(api.collections, 'create').mockResolvedValueOnce({
      id: 'col-new-1',
      name: 'Deep Learning Optimization',
      description: 'Papers on AdamW, gradient clipping, and schedules',
      document_count: 0,
      created_at: '2026-10-17T21:00:00Z',
      updated_at: '2026-10-17T21:00:00Z',
    })

    render(<App />)

    // Wait for session verification
    expect(await screen.findByTitle('alice@researcher.com')).toBeDefined()

    const nav = screen.getByRole('navigation', { name: /main navigation/i })
    fireEvent.click(within(nav).getByRole('button', { name: /^collections$/i }))

    const newColBtn = await screen.findByRole('button', { name: /\+ (new|create) collection/i })
    fireEvent.click(newColBtn)

    const modal = await screen.findByRole('dialog', { name: /create new collection/i })
    const nameInput = within(modal).getByLabelText(/Collection Name/i)
    const descInput = within(modal).getByLabelText(/Description/i)
    const submitBtn = within(modal).getByRole('button', { name: /^create collection$/i })

    fireEvent.change(nameInput, { target: { value: 'Deep Learning Optimization' } })
    fireEvent.change(descInput, { target: { value: 'Papers on AdamW, gradient clipping, and schedules' } })
    fireEvent.click(submitBtn)

    await waitFor(() => {
      expect(createSpy).toHaveBeenCalledWith({
        name: 'Deep Learning Optimization',
        description: 'Papers on AdamW, gradient clipping, and schedules',
      })
    })

    expect(await screen.findByText('Deep Learning Optimization')).toBeDefined()
    expect(screen.getByText(/Papers on AdamW, gradient clipping, and schedules/i)).toBeDefined()
  })

  it('handles duplicate collection name conflict (409) with inline error without closing the modal', async () => {
    setToken('valid-existing-token')

    vi.spyOn(api.auth, 'getMe').mockResolvedValue({
      id: '12345678-1234-1234-1234-123456789abc',
      email: 'alice@researcher.com',
      is_active: true,
      is_verified: false,
      created_at: '2026-10-17T20:00:00Z',
      updated_at: '2026-10-17T20:00:00Z',
    })

    vi.spyOn(api.collections, 'list').mockResolvedValue({
      items: [
        {
          id: 'col-1',
          name: 'Quantum Optics',
          description: null,
          document_count: 0,
          created_at: '2026-10-17T20:00:00Z',
          updated_at: '2026-10-17T20:00:00Z',
        },
      ],
      total: 1,
      limit: 50,
      offset: 0,
    })

    vi.spyOn(api.collections, 'create').mockRejectedValueOnce(
      new ApiError(409, "A collection with the name 'Quantum Optics' already exists.")
    )

    render(<App />)

    // Wait for session verification
    expect(await screen.findByTitle('alice@researcher.com')).toBeDefined()

    const nav = screen.getByRole('navigation', { name: /main navigation/i })
    fireEvent.click(within(nav).getByRole('button', { name: /^collections$/i }))

    const newColBtn = await screen.findByRole('button', { name: /\+ (new|create) collection/i })
    fireEvent.click(newColBtn)

    const modal = await screen.findByRole('dialog', { name: /create new collection/i })
    const nameInput = within(modal).getByLabelText(/Collection Name/i)
    const submitBtn = within(modal).getByRole('button', { name: /^create collection$/i })

    fireEvent.change(nameInput, { target: { value: 'Quantum Optics' } })
    fireEvent.click(submitBtn)

    const alert = await within(modal).findByRole('alert')
    expect(within(alert).getByText(/A collection with the name 'Quantum Optics' already exists/i)).toBeDefined()

    expect(screen.getByRole('dialog', { name: /create new collection/i })).toBeDefined()
    expect((nameInput as HTMLInputElement).value).toBe('Quantum Optics')
  })

  it('uploads a valid PDF document successfully and displays it in the table with PENDING status', async () => {
    setToken('valid-existing-token')

    vi.spyOn(api.auth, 'getMe').mockResolvedValue({
      id: '12345678-1234-1234-1234-123456789abc',
      email: 'alice@researcher.com',
      is_active: true,
      is_verified: false,
      created_at: '2026-10-17T20:00:00Z',
      updated_at: '2026-10-17T20:00:00Z',
    })

    vi.spyOn(api.documents, 'list')
      .mockResolvedValueOnce({ items: [], total: 0, limit: 50, offset: 0 })
      .mockResolvedValueOnce({
        items: [
          {
            id: 'doc-uploaded-1',
            filename: 'attention_is_all_you_need.pdf',
            title: 'Attention Is All You Need',
            file_size_bytes: 2097152,
            page_count: 15,
            processing_status: 'PENDING',
            error_message: null,
            created_at: '2026-10-17T21:10:00Z',
            updated_at: '2026-10-17T21:10:00Z',
          },
        ],
        total: 1,
        limit: 50,
        offset: 0,
      })

    const uploadSpy = vi.spyOn(api.documents, 'upload').mockResolvedValueOnce({
      id: 'doc-uploaded-1',
      filename: 'attention_is_all_you_need.pdf',
      title: 'Attention Is All You Need',
      file_size_bytes: 2097152,
      page_count: 15,
      processing_status: 'PENDING',
      error_message: null,
      created_at: '2026-10-17T21:10:00Z',
      updated_at: '2026-10-17T21:10:00Z',
    })

    render(<App />)

    // Wait for session verification
    expect(await screen.findByTitle('alice@researcher.com')).toBeDefined()

    const nav = screen.getByRole('navigation', { name: /main navigation/i })
    fireEvent.click(within(nav).getByRole('button', { name: /^documents$/i }))

    const uploadBtn = await screen.findByRole('button', { name: /\+ upload (first )?document/i })
    fireEvent.click(uploadBtn)

    const modal = await screen.findByRole('dialog', { name: /upload research pdf/i })
    const fileInput = within(modal).getByLabelText(/Select PDF Document/i)
    const titleInput = within(modal).getByLabelText(/Document Title/i)

    const fakePdf = new File([new Uint8Array([0x25, 0x50, 0x44, 0x46])], 'attention_is_all_you_need.pdf', {
      type: 'application/pdf',
    })

    fireEvent.change(fileInput, { target: { files: [fakePdf] } })
    fireEvent.change(titleInput, { target: { value: 'Attention Is All You Need' } })

    const submitBtn = within(modal).getByRole('button', { name: /^upload pdf$/i })
    fireEvent.click(submitBtn)

    await waitFor(() => {
      expect(uploadSpy).toHaveBeenCalledWith(fakePdf, 'Attention Is All You Need')
    })

    expect(await screen.findByText('Attention Is All You Need')).toBeDefined()
    expect(screen.getByText('2.00 MB')).toBeDefined()
    expect(screen.getAllByText('PENDING').length).toBeGreaterThan(0)
  })

  it('handles upload errors (oversized file or API failure) with clear feedback without crashing', async () => {
    setToken('valid-existing-token')

    vi.spyOn(api.auth, 'getMe').mockResolvedValue({
      id: '12345678-1234-1234-1234-123456789abc',
      email: 'alice@researcher.com',
      is_active: true,
      is_verified: false,
      created_at: '2026-10-17T20:00:00Z',
      updated_at: '2026-10-17T20:00:00Z',
    })

    vi.spyOn(api.documents, 'list').mockResolvedValue({ items: [], total: 0, limit: 50, offset: 0 })

    vi.spyOn(api.documents, 'upload').mockRejectedValueOnce(
      new ApiError(413, 'File size exceeds maximum allowed limit of 26214400 bytes.')
    )

    render(<App />)

    // Wait for session verification
    expect(await screen.findByTitle('alice@researcher.com')).toBeDefined()

    const nav = screen.getByRole('navigation', { name: /main navigation/i })
    fireEvent.click(within(nav).getByRole('button', { name: /^documents$/i }))

    const uploadBtn = await screen.findByRole('button', { name: /\+ upload (first )?document/i })
    fireEvent.click(uploadBtn)

    const modal = await screen.findByRole('dialog', { name: /upload research pdf/i })
    const fileInput = within(modal).getByLabelText(/Select PDF Document/i)

    const fakePdf = new File([new Uint8Array([0x25, 0x50, 0x44, 0x46])], 'large_document.pdf', {
      type: 'application/pdf',
    })

    fireEvent.change(fileInput, { target: { files: [fakePdf] } })

    const submitBtn = within(modal).getByRole('button', { name: /^upload pdf$/i })
    fireEvent.click(submitBtn)

    const alert = await within(modal).findByRole('alert')
    expect(within(alert).getByText(/File size exceeds maximum allowed limit/i)).toBeDefined()

    expect(screen.getByRole('dialog', { name: /upload research pdf/i })).toBeDefined()
  })

  it('navigates to Semantic Search via navbar and displays search interface', async () => {
    setToken('valid-existing-token')

    vi.spyOn(api.auth, 'getMe').mockResolvedValue({
      id: '12345678-1234-1234-1234-123456789abc',
      email: 'alice@researcher.com',
      is_active: true,
      is_verified: false,
      created_at: '2026-10-17T20:00:00Z',
      updated_at: '2026-10-17T20:00:00Z',
    })

    vi.spyOn(api.collections, 'list').mockResolvedValue({ items: [], total: 0, limit: 100, offset: 0 })
    vi.spyOn(api.documents, 'list').mockResolvedValue({ items: [], total: 0, limit: 100, offset: 0 })

    render(<App />)

    expect(await screen.findByTitle('alice@researcher.com')).toBeDefined()

    const nav = screen.getByRole('navigation', { name: /main navigation/i })
    const searchNavBtn = within(nav).getByRole('button', { name: /^search$/i })
    fireEvent.click(searchNavBtn)

    expect(await screen.findByRole('heading', { name: /Semantic Passage Search/i })).toBeDefined()
    expect(window.location.hash).toBe('#search')
  })

  it('navigates to Semantic Search from Dashboard metric card', async () => {
    setToken('valid-existing-token')

    vi.spyOn(api.auth, 'getMe').mockResolvedValue({
      id: '12345678-1234-1234-1234-123456789abc',
      email: 'alice@researcher.com',
      is_active: true,
      is_verified: false,
      created_at: '2026-10-17T20:00:00Z',
      updated_at: '2026-10-17T20:00:00Z',
    })

    vi.spyOn(api.collections, 'list').mockResolvedValue({ items: [], total: 0, limit: 100, offset: 0 })
    vi.spyOn(api.documents, 'list').mockResolvedValue({ items: [], total: 0, limit: 100, offset: 0 })

    render(<App />)

    expect(await screen.findByRole('heading', { name: /Welcome back, alice@researcher.com/i })).toBeDefined()

    const searchMetricCard = screen.getByRole('button', { name: /Semantic Search/i })
    fireEvent.click(searchMetricCard)

    expect(await screen.findByRole('heading', { name: /Semantic Passage Search/i })).toBeDefined()
    expect(window.location.hash).toBe('#search')
  })

  it('navigates to Chat workspace via navbar and loads chat shell', async () => {
    setToken('valid-existing-token')

    vi.spyOn(api.auth, 'getMe').mockResolvedValue({
      id: '12345678-1234-1234-1234-123456789abc',
      email: 'alice@researcher.com',
      is_active: true,
      is_verified: false,
      created_at: '2026-10-17T20:00:00Z',
      updated_at: '2026-10-17T20:00:00Z',
    })

    vi.spyOn(api.collections, 'list').mockResolvedValue({ items: [], total: 0, limit: 100, offset: 0 })
    vi.spyOn(api.documents, 'list').mockResolvedValue({ items: [], total: 0, limit: 100, offset: 0 })
    vi.spyOn(api.chat, 'listSessions').mockResolvedValue({ items: [], total: 0, limit: 100, offset: 0 })

    render(<App />)

    expect(await screen.findByTitle('alice@researcher.com')).toBeDefined()

    const nav = screen.getByRole('navigation', { name: /main navigation/i })
    const chatNavBtn = within(nav).getByRole('button', { name: /^chat$/i })
    fireEvent.click(chatNavBtn)

    expect(await screen.findByRole('heading', { level: 3, name: 'Conversations' })).toBeDefined()
    expect(window.location.hash).toBe('#chat')
  })

  it('navigates to Chat workspace from Dashboard metric card', async () => {
    setToken('valid-existing-token')

    vi.spyOn(api.auth, 'getMe').mockResolvedValue({
      id: '12345678-1234-1234-1234-123456789abc',
      email: 'alice@researcher.com',
      is_active: true,
      is_verified: false,
      created_at: '2026-10-17T20:00:00Z',
      updated_at: '2026-10-17T20:00:00Z',
    })

    vi.spyOn(api.collections, 'list').mockResolvedValue({ items: [], total: 0, limit: 100, offset: 0 })
    vi.spyOn(api.documents, 'list').mockResolvedValue({ items: [], total: 0, limit: 100, offset: 0 })
    vi.spyOn(api.chat, 'listSessions').mockResolvedValue({ items: [], total: 0, limit: 100, offset: 0 })

    render(<App />)

    expect(await screen.findByRole('heading', { name: /Welcome back, alice@researcher.com/i })).toBeDefined()

    const chatCard = screen.getByRole('button', { name: /Research Chat/i })
    fireEvent.click(chatCard)

    expect(await screen.findByRole('heading', { level: 3, name: 'Conversations' })).toBeDefined()
    expect(window.location.hash).toBe('#chat')
  })
})

