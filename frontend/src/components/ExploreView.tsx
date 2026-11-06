import { useState } from 'react'
import { Sparkles } from 'lucide-react'
import { PromoHero } from './PromoHero'
import { CapabilitiesGrid } from './CapabilitiesGrid'
import { MetricsBanner } from './MetricsBanner'
import { ResearchPresets } from './ResearchPresets'
import { CorpusShowcase } from './CorpusShowcase'
import { ResearchModal, ResearchModalData } from './ResearchModal'
import {
  ResearchEngineItem,
  ResearchPresetWorkflow,
  CorpusShowcaseItem,
} from '../data/researchLensData'
import { ActiveView } from '../types'

interface ExploreViewProps {
  onNavigate: (view: ActiveView) => void
  onPrepareChatQuery?: (query: string, scope?: string) => void
  onOpenUpload?: () => void
}

export const ExploreView = ({
  onNavigate,
  onPrepareChatQuery,
  onOpenUpload,
}: ExploreViewProps) => {
  const [modalOpen, setModalOpen] = useState(false)
  const [modalData, setModalData] = useState<ResearchModalData | null>(null)
  const [toastMessage, setToastMessage] = useState<string | null>(null)

  const showToast = (msg: string) => {
    setToastMessage(msg)
    setTimeout(() => setToastMessage(null), 3000)
  }

  const handleSelectEngine = (engine: ResearchEngineItem) => {
    showToast(`Loaded ${engine.title} specifications`)
  }

  const handleOpenStudioModal = (engine: ResearchEngineItem) => {
    setModalData({
      title: `${engine.title} Studio`,
      prompt: `Analyze methodological rigor and benchmark claims using ${engine.title}`,
      scope: 'GLOBAL',
      model: engine.id,
      topK: 5,
      threshold: 0.65,
    })
    setModalOpen(true)
  }

  const handleQuickQuery = (
    query: string,
    scope: 'GLOBAL' | 'COLLECTION' | 'DOCUMENT'
  ) => {
    setModalData({
      title: 'Execute Grounded Inquiry',
      prompt: query,
      scope: scope,
      topK: 5,
      threshold: 0.65,
    })
    setModalOpen(true)
  }

  const handleExecutePreset = (preset: ResearchPresetWorkflow) => {
    setModalData({
      title: preset.title,
      prompt: preset.prompt,
      scope: preset.scope,
      topK: preset.topK,
      threshold: preset.threshold,
    })
    setModalOpen(true)
  }

  const handleInspectPaper = (item: CorpusShowcaseItem) => {
    setModalData({
      title: item.title,
      prompt: item.sampleQuery,
      scope: 'GLOBAL',
      isDemonstration: true,
      demonstrationNotice: `Demonstration Reference: "${item.title}" is an illustrative showcase paper. Inquiries will retrieve and synthesize evidence from your indexed research library.`,
      sourceTitle: `${item.title} (${item.authors.split(',')[0]} et al., ${item.publishedYear})`,
      topK: 4,
      threshold: 0.65,
    })
    setModalOpen(true)
  }

  return (
    <div className="explore-view-container">
      {/* Toast Notification */}
      {toastMessage && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-6 right-6 z-50 bg-neutral-950 text-white px-4 py-2.5 rounded-xl shadow-2xl text-xs font-semibold flex items-center gap-2 border border-neutral-800 animate-in slide-in-from-bottom-2 duration-200"
        >
          <Sparkles className="w-3.5 h-3.5 text-[#D1FE17]" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* 1. Hero Showcase */}
      <PromoHero
        onSelectEngine={handleSelectEngine}
        onQuickQuery={handleQuickQuery}
        onOpenStudioModal={handleOpenStudioModal}
      />

      {/* 2. Flagship Capabilities Architecture */}
      <CapabilitiesGrid
        onSelectCapability={(id) => {
          if (id === 'semantic-search-card') {
            onNavigate('search')
          } else if (id === 'grounded-rag-card' || id === 'citation-card') {
            onNavigate('chat')
          } else if (id === 'ingestion-card') {
            if (onOpenUpload) onOpenUpload()
            else onNavigate('documents')
          } else if (id === 'collections-card') {
            onNavigate('collections')
          } else if (id === 'mcp-api-card') {
            onNavigate('mcp')
          } else {
            handleQuickQuery('Summarize empirical results with verifiable citations', 'GLOBAL')
          }
        }}
      />

      {/* 3. Citation Architecture & Benchmark Banner */}
      <MetricsBanner
        onLearnMore={() => {
          showToast('Citation Architecture: Exact page-bounded chunks verified against canonical PostgreSQL records.')
        }}
        onUploadPaper={() => {
          if (onOpenUpload) onOpenUpload()
          else onNavigate('documents')
        }}
      />

      {/* 4. Trending Academic Inquiry Presets */}
      <ResearchPresets onExecutePreset={handleExecutePreset} />

      {/* 5. Pre-Indexed Literature Corpus Showcase */}
      <CorpusShowcase onInspectPaper={handleInspectPaper} />

      {/* Interactive Studio Modal */}
      <ResearchModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        initialData={modalData}
        onOpenChatWithQuery={(q, sc) => {
          if (onPrepareChatQuery) {
            onPrepareChatQuery(q, sc)
          }
          onNavigate('chat')
        }}
      />
    </div>
  )
}
export default ExploreView
