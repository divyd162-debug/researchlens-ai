/**
 * API client and authentication token management service.
 */
import {
  ChatMessageCreateRequest,
  ChatMessageListResponse,
  ChatSessionCreateRequest,
  ChatSessionListResponse,
  ChatSessionResponse,
  ChatSessionUpdateRequest,
  ChatTurnResponse,
  CollectionCreateRequest,
  CollectionListResponse,
  CollectionResponse,
  DocumentListResponse,
  DocumentResponse,
  MessageResponse,
  QueryRequest,
  RAGResponse,
  SearchRequest,
  SearchResponse,
  TokenResponse,
  UserLoginRequest,
  UserRegisterRequest,
  UserResponse,
} from '../types'

const TOKEN_STORAGE_KEY = 'researchlens_access_token'

export const normalizeBaseUrl = (rawBase?: string | null): string => {
  if (!rawBase || !rawBase.trim()) {
    return '/api/v1'
  }
  let base = rawBase.trim().replace(/\/+$/, '')
  if (!base.endsWith('/api/v1')) {
    if (base.endsWith('/api')) {
      base = `${base}/v1`
    } else {
      base = `${base}/api/v1`
    }
  }
  return base
}

export const buildApiUrl = (endpoint: string, baseUrl?: string): string => {
  const base = normalizeBaseUrl(baseUrl ?? (import.meta.env.VITE_API_BASE_URL as string | undefined))
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`
  return `${base}${cleanEndpoint}`
}

let unauthorizedCallback: (() => void) | null = null

export const setUnauthorizedCallback = (callback: (() => void) | null): void => {
  unauthorizedCallback = callback
}

export const getToken = (): string | null => {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY)
  } catch {
    return null
  }
}

export const setToken = (token: string): void => {
  try {
    localStorage.setItem(TOKEN_STORAGE_KEY, token)
  } catch {
    // Graceful fallback if storage is restricted
  }
}

export const clearToken = (): void => {
  try {
    localStorage.removeItem(TOKEN_STORAGE_KEY)
  } catch {
    // Graceful fallback
  }
}

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = buildApiUrl(endpoint)
  const headers = new Headers(options.headers || {})

  if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json')
  }

  const token = getToken()
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  const response = await fetch(url, {
    ...options,
    headers,
  })

  if (response.status === 401) {
    clearToken()
    if (unauthorizedCallback) {
      unauthorizedCallback()
    }
  }

  if (!response.ok) {
    let errorMessage = `Request failed with status ${response.status}`
    try {
      const errorData = await response.json()
      if (typeof errorData?.detail === 'string') {
        errorMessage = errorData.detail
      } else if (Array.isArray(errorData?.detail)) {
        errorMessage = errorData.detail.map((err: { msg?: string }) => err.msg || JSON.stringify(err)).join(', ')
      }
    } catch {
      // Use fallback status message if body is not JSON
    }
    throw new ApiError(response.status, errorMessage)
  }

  if (response.status === 204) {
    return {} as T
  }

  return response.json() as Promise<T>
}

export const api = {
  auth: {
    login: async (payload: UserLoginRequest): Promise<TokenResponse> => {
      const result = await request<TokenResponse>('/auth/login', {
        method: 'POST',
        body: JSON.stringify(payload),
      })
      if (result.access_token) {
        setToken(result.access_token)
      }
      return result
    },

    register: async (payload: UserRegisterRequest): Promise<UserResponse> => {
      return request<UserResponse>('/auth/register', {
        method: 'POST',
        body: JSON.stringify(payload),
      })
    },

    getMe: async (): Promise<UserResponse> => {
      return request<UserResponse>('/auth/me', {
        method: 'GET',
      })
    },

    logout: (): void => {
      clearToken()
    },
  },

  collections: {
    list: async (limit = 50, offset = 0): Promise<CollectionListResponse> => {
      return request<CollectionListResponse>(`/collections?limit=${limit}&offset=${offset}`, {
        method: 'GET',
      })
    },

    create: async (payload: CollectionCreateRequest): Promise<CollectionResponse> => {
      return request<CollectionResponse>('/collections', {
        method: 'POST',
        body: JSON.stringify(payload),
      })
    },

    addDocuments: async (
      collectionId: string,
      payload: { document_ids?: string[]; document_id?: string }
    ): Promise<CollectionResponse> => {
      return request<CollectionResponse>(`/collections/${collectionId}/documents`, {
        method: 'POST',
        body: JSON.stringify(payload),
      })
    },

    removeDocument: async (
      collectionId: string,
      documentId: string
    ): Promise<CollectionResponse> => {
      return request<CollectionResponse>(`/collections/${collectionId}/documents/${documentId}`, {
        method: 'DELETE',
      })
    },

    listDocuments: async (
      collectionId: string,
      limit = 50,
      offset = 0
    ): Promise<DocumentListResponse> => {
      return request<DocumentListResponse>(
        `/collections/${collectionId}/documents?limit=${limit}&offset=${offset}`,
        {
          method: 'GET',
        }
      )
    },
  },

  documents: {
    list: async (limit = 50, offset = 0): Promise<DocumentListResponse> => {
      return request<DocumentListResponse>(`/documents?limit=${limit}&offset=${offset}`, {
        method: 'GET',
      })
    },

    upload: async (file: File, title?: string): Promise<DocumentResponse> => {
      const formData = new FormData()
      formData.append('file', file)
      if (title && title.trim()) {
        formData.append('title', title.trim())
      }
      return request<DocumentResponse>('/documents', {
        method: 'POST',
        body: formData,
      })
    },

    delete: async (id: string): Promise<MessageResponse> => {
      return request<MessageResponse>(`/documents/${id}`, {
        method: 'DELETE',
      })
    },

    retry: async (id: string): Promise<DocumentResponse> => {
      return request<DocumentResponse>(`/documents/${id}/retry`, {
        method: 'POST',
      })
    },
  },

  search: {
    query: async (payload: SearchRequest, signal?: AbortSignal): Promise<SearchResponse> => {
      return request<SearchResponse>('/search', {
        method: 'POST',
        body: JSON.stringify(payload),
        signal,
      })
    },
  },

  query: {
    rag: async (payload: QueryRequest, signal?: AbortSignal): Promise<RAGResponse> => {
      return request<RAGResponse>('/query', {
        method: 'POST',
        body: JSON.stringify(payload),
        signal,
      })
    },
  },

  chat: {
    listSessions: async (
      limit = 50,
      offset = 0,
      isArchived?: boolean,
      signal?: AbortSignal
    ): Promise<ChatSessionListResponse> => {
      let queryStr = `limit=${limit}&offset=${offset}`
      if (typeof isArchived === 'boolean') {
        queryStr += `&is_archived=${isArchived}`
      }
      return request<ChatSessionListResponse>(`/chat/sessions?${queryStr}`, {
        method: 'GET',
        signal,
      })
    },

    getSession: async (sessionId: string, signal?: AbortSignal): Promise<ChatSessionResponse> => {
      return request<ChatSessionResponse>(`/chat/sessions/${sessionId}`, {
        method: 'GET',
        signal,
      })
    },

    createSession: async (payload: ChatSessionCreateRequest): Promise<ChatSessionResponse> => {
      return request<ChatSessionResponse>('/chat/sessions', {
        method: 'POST',
        body: JSON.stringify(payload),
      })
    },

    updateSession: async (
      sessionId: string,
      payload: ChatSessionUpdateRequest
    ): Promise<ChatSessionResponse> => {
      return request<ChatSessionResponse>(`/chat/sessions/${sessionId}`, {
        method: 'PATCH',
        body: JSON.stringify(payload),
      })
    },

    deleteSession: async (sessionId: string): Promise<MessageResponse> => {
      return request<MessageResponse>(`/chat/sessions/${sessionId}`, {
        method: 'DELETE',
      })
    },

    getSessionMessages: async (
      sessionId: string,
      signal?: AbortSignal
    ): Promise<ChatMessageListResponse> => {
      return request<ChatMessageListResponse>(`/chat/sessions/${sessionId}/messages`, {
        method: 'GET',
        signal,
      })
    },

    postMessage: async (
      sessionId: string,
      payload: ChatMessageCreateRequest,
      signal?: AbortSignal
    ): Promise<ChatTurnResponse> => {
      return request<ChatTurnResponse>(`/chat/sessions/${sessionId}/messages`, {
        method: 'POST',
        body: JSON.stringify(payload),
        signal,
      })
    },
  },
}
