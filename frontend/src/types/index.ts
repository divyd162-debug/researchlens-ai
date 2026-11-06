/**
 * TypeScript core interfaces for ResearchLens AI frontend.
 */

export interface UserResponse {
  id: string
  email: string
  is_active: boolean
  is_verified: boolean
  created_at: string
  updated_at: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
  expires_in: number
}

export interface UserLoginRequest {
  email: string
  password: string
}

export interface UserRegisterRequest {
  email: string
  password: string
}

export interface CollectionCreateRequest {
  name: string
  description?: string | null
}

export interface AddDocumentsToCollectionRequest {
  document_ids?: string[]
  document_id?: string
}

export interface CollectionResponse {
  id: string
  name: string
  description?: string | null
  document_count: number
  created_at: string
  updated_at: string
}

export interface CollectionListResponse {
  items: CollectionResponse[]
  total: number
  limit: number
  offset: number
}

export interface DocumentResponse {
  id: string
  filename: string
  title?: string | null
  file_size_bytes: number
  page_count?: number | null
  processing_status: string
  error_message?: string | null
  created_at: string
  updated_at: string
}

export interface DocumentListResponse {
  items: DocumentResponse[]
  total: number
  limit: number
  offset: number
}

export interface MessageResponse {
  message: string
}

export type SearchScopeType = 'GLOBAL' | 'DOCUMENT' | 'COLLECTION' | 'DOCUMENTS' | 'global' | 'document' | 'collection' | 'documents'

export type DocumentStatus = 'pending' | 'extracting' | 'chunking' | 'embedding' | 'ready' | 'failed'

export interface DocumentItem {
  id: string
  title: string
  filename: string
  pageCount: number
  fileSizeBytes: number
  status: DocumentStatus
  statusMessage?: string
  uploadedAt: string
  authors: string[]
  doi?: string
  year: number
  abstract: string
  collectionIds: string[]
  chunkCount?: number
  topics: string[]
}

export interface CollectionItem {
  id: string
  name: string
  description: string
  documentCount: number
  createdAt: string
  updatedAt: string
  documentIds: string[]
  colorTheme?: string
}

export interface SearchScope {
  type: SearchScopeType
  targetId?: string
  targetName?: string
  selectedDocIds?: string[]
}

export interface SearchResult {
  id: string
  documentId: string
  documentTitle: string
  page: number
  passage: string
  matchedKeywords: string[]
  sectionTitle?: string
  doi?: string
  year: number
  authors: string[]
}

export interface Citation {
  id: string
  documentId: string
  documentTitle: string
  page: number
  quotedPassage: string
  sectionTitle?: string
  doi?: string
  authors?: string[]
  year?: number
  charStart?: number
  charEnd?: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  citations?: Citation[]
  scope?: SearchScope
}

export interface ChatSession {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  scope: SearchScope
  messages: ChatMessage[]
  isArchived?: boolean
}

export interface ResearchActivity {
  id: string
  type: 'search' | 'chat' | 'upload'
  title: string
  timestamp: string
  scopeDescription: string
  linkTab: 'search' | 'chat' | 'documents'
  linkId?: string
}

export interface OverviewMetrics {
  totalDocuments: number
  totalCollections: number
  totalConversations: number
  processingDocuments: number
  storageUsedBytes: number
  storageMaxBytes: number
}

export interface SearchRequest {
  query: string
  scope_type?: SearchScopeType
  scope_id?: string | null
  document_ids?: string[] | null
  top_k?: number
  similarity_threshold?: number
}

export interface SearchPassageResponse {
  chunk_id: number
  document_id: string
  document_title: string
  page_number: number
  content: string
  similarity_score: number
  char_start: number
  char_end: number
}

export interface SearchResponse {
  query: string
  scope_type: SearchScopeType
  scope_id?: string | null
  document_ids?: string[] | null
  total_results: number
  result_count: number
  results: SearchPassageResponse[]
}

export interface CitationSnapshot {
  chunk_id: number
  document_id: string
  quoted_text: string
  page_number: number
  document_title: string
  similarity_score: number
  citation_frequency: number
}

export interface QueryRequest {
  query: string
  scope_type?: ChatScopeType
  scope_id?: string | null
  document_ids?: string[] | null
  top_k?: number
  similarity_threshold?: number
}

export interface RAGResponse {
  query: string
  answer: string
  citations: CitationSnapshot[]
  unknown_chunk_ids: number[]
  malformed_markers: string[]
  unsupported_numbers: string[]
  status: 'SUCCESS' | 'NO_RELEVANT_CONTEXT' | string
  model_name: string
  provider_name: string
  retrieved_chunk_count: number
  prompt_tokens?: number | null
  completion_tokens?: number | null
  total_tokens?: number | null
}

export type ActiveView = 'dashboard' | 'collections' | 'documents' | 'search' | 'chat' | 'login' | 'register' | 'explore' | 'studio' | 'settings' | 'mcp'

export type ChatScopeType = 'GLOBAL' | 'DOCUMENT' | 'COLLECTION' | 'DOCUMENTS'
export type SenderType = 'USER' | 'ASSISTANT' | 'SYSTEM'

export interface DocumentScopeSummary {
  id: string
  filename: string
  title?: string | null
  page_count?: number | null
  processing_status: string
}

export interface MessageCitationResponse {
  id: string
  message_id: string
  chunk_id: number | null
  quoted_text: string
  page_number: number
  document_title: string
  similarity_score: number | null
  created_at: string
}

export interface ChatMessageResponse {
  id: string
  chat_session_id: string
  sender_type: SenderType
  content: string
  token_count: number | null
  created_at: string
  citations: MessageCitationResponse[]
}

export interface ChatSessionResponse {
  id: string
  title: string
  scope_type: ChatScopeType
  target_document_id: string | null
  target_collection_id: string | null
  is_archived: boolean
  created_at: string
  updated_at: string
  message_count: number
  selected_documents?: DocumentScopeSummary[]
}

export interface ChatSessionListResponse {
  items: ChatSessionResponse[]
  total: number
  limit: number
  offset: number
}

export interface ChatMessageListResponse {
  session_id: string
  items: ChatMessageResponse[]
  total: number
}

export interface ChatSessionCreateRequest {
  title?: string | null
  scope_type?: ChatScopeType
  target_document_id?: string | null
  target_collection_id?: string | null
  document_ids?: string[] | null
}

export interface ChatSessionUpdateRequest {
  title?: string | null
  is_archived?: boolean | null
}

export interface ChatMessageCreateRequest {
  content: string
  top_k?: number
  similarity_threshold?: number
}

export interface ChatTurnResponse {
  session_id: string
  user_message: ChatMessageResponse
  assistant_message: ChatMessageResponse
  retrieved_chunk_count: number
  model_name: string
  provider_name: string
  prompt_tokens?: number | null
  completion_tokens?: number | null
  total_tokens?: number | null
}

