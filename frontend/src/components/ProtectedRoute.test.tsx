import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ProtectedRoute } from './ProtectedRoute'
import * as AuthContextModule from '../context/AuthContext'

describe('ProtectedRoute Component', () => {
  it('renders loading state when session is being verified', () => {
    vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: true,
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    })

    render(
      <ProtectedRoute onNavigate={vi.fn()} targetView="documents">
        <div>Protected Content</div>
      </ProtectedRoute>
    )

    expect(screen.getByText(/Verifying session.../i)).toBeDefined()
    expect(screen.queryByText('Protected Content')).toBeNull()
  })

  it('renders sign-in requirement and LoginView when unauthenticated', () => {
    vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    })

    const onNavigate = vi.fn()

    render(
      <ProtectedRoute onNavigate={onNavigate} targetView="documents">
        <div>Protected Content</div>
      </ProtectedRoute>
    )

    expect(screen.getByText(/Authentication Required:/i)).toBeDefined()
    expect(screen.getByRole('heading', { name: /Research Documents/i })).toBeDefined()
    expect(screen.getByRole('form', { name: /sign in form/i })).toBeDefined()
    expect(screen.queryByText('Protected Content')).toBeNull()
  })

  it('renders protected children when authenticated', () => {
    vi.spyOn(AuthContextModule, 'useAuth').mockReturnValue({
      user: {
        id: 'user-1',
        email: 'researcher@example.com',
        is_active: true,
        is_verified: true,
        created_at: '2026-01-01T00:00:00Z',
        updated_at: '2026-01-01T00:00:00Z',
      },
      token: 'jwt-valid-token',
      isAuthenticated: true,
      isLoading: false,
      login: vi.fn(),
      register: vi.fn(),
      logout: vi.fn(),
    })

    render(
      <ProtectedRoute onNavigate={vi.fn()} targetView="documents">
        <div>Protected Content</div>
      </ProtectedRoute>
    )

    expect(screen.getByText('Protected Content')).toBeDefined()
    expect(screen.queryByText(/Authentication Required:/i)).toBeNull()
  })
})
