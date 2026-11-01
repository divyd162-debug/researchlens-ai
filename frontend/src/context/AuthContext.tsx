import React, { createContext, useContext, useEffect, useState } from 'react'
import { api, clearToken, getToken, setUnauthorizedCallback } from '../services/api'
import { UserLoginRequest, UserRegisterRequest, UserResponse } from '../types'

interface AuthContextType {
  user: UserResponse | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (credentials: UserLoginRequest) => Promise<void>
  register: (credentials: UserRegisterRequest) => Promise<UserResponse>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserResponse | null>(null)
  const [token, setTokenState] = useState<string | null>(() => getToken())
  const [isLoading, setIsLoading] = useState<boolean>(true)

  const logout = (): void => {
    api.auth.logout()
    setTokenState(null)
    setUser(null)
  }

  useEffect(() => {
    setUnauthorizedCallback(() => {
      logout()
    })
    return () => {
      setUnauthorizedCallback(null)
    }
  }, [])

  useEffect(() => {
    let isMounted = true

    const verifySession = async (): Promise<void> => {
      const storedToken = getToken()
      if (!storedToken) {
        if (isMounted) {
          setIsLoading(false)
        }
        return
      }

      try {
        const profile = await api.auth.getMe()
        if (isMounted) {
          setUser(profile)
          setTokenState(storedToken)
        }
      } catch {
        if (isMounted) {
          clearToken()
          setTokenState(null)
          setUser(null)
        }
      } finally {
        if (isMounted) {
          setIsLoading(false)
        }
      }
    }

    verifySession()

    return () => {
      isMounted = false
    }
  }, [])

  const login = async (credentials: UserLoginRequest): Promise<void> => {
    setIsLoading(true)
    try {
      const tokenResponse = await api.auth.login(credentials)
      setTokenState(tokenResponse.access_token)
      const profile = await api.auth.getMe()
      setUser(profile)
    } finally {
      setIsLoading(false)
    }
  }

  const register = async (credentials: UserRegisterRequest): Promise<UserResponse> => {
    setIsLoading(true)
    try {
      return await api.auth.register(credentials)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user && !!token,
        isLoading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
