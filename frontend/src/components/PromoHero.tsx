import { useState } from 'react'
import {
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Cpu,
  Layers,
  FileCheck2,
  Database,
} from 'lucide-react'
import { RESEARCH_ENGINES, ResearchEngineItem } from '../data/researchLensData'

interface PromoHeroProps {
  onSelectEngine: (engine: ResearchEngineItem) => void
  onQuickQuery: (query: string, scope: 'GLOBAL' | 'COLLECTION' | 'DOCUMENT') => void
  onOpenStudioModal: (engine: ResearchEngineItem) => void
}

export const PromoHero = ({
  onSelectEngine,
  onQuickQuery,
  onOpenStudioModal,
}: PromoHeroProps) => {
  const [selectedEngine, setSelectedEngine] = useState<ResearchEngineItem>(RESEARCH_ENGINES[2]) // Default Grounded RAG
  const [queryInput, setQueryInput] = useState('')

  const handleQuerySubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!queryInput.trim()) return
    onQuickQuery(queryInput, 'GLOBAL')
  }

  const sampleQueries = [
    'How does self-attention complexity scale with sequence length compared to recurrence?',
    'What are the exact parameter bounds and learning rates used in the ablation baseline?',
    'Summarize the empirical results on GLUE and SQuAD benchmarks with cited percentages',
    'Extract the loss function formulation and convergence proof assumptions from Section 3',
  ]

  const getEngineIcon = (id: string) => {
    switch (id) {
      case 'minilm-embeddings':
        return <Cpu className="w-4 h-4 text-neutral-900" />
      case 'faiss-vector-store':
        return <Database className="w-4 h-4 text-neutral-900" />
      case 'grounded-rag':
        return <ShieldCheck className="w-4 h-4 text-neutral-900" />
      case 'corpus-knowledge-graph':
        return <Layers className="w-4 h-4 text-neutral-900" />
      default:
        return <FileCheck2 className="w-4 h-4 text-neutral-900" />
    }
  }

  return (
    <section className="relative overflow-hidden pt-8 pb-14 bg-gradient-to-b from-white via-[#FAFAFB] to-[#F4F4F6] border-b border-neutral-200/70">
      {/* Decorative ambient subtle background glow (Template Style) */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[350px] bg-gradient-to-tr from-lime-200/35 via-emerald-100/20 to-sky-100/30 blur-3xl -z-10 pointer-events-none rounded-full" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Top Header & Tagline */}
        <div className="text-center max-w-3xl mx-auto mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-neutral-200 shadow-xs mb-4">
            <span className="flex h-2 w-2 rounded-full bg-lime-500 animate-pulse" />
            <span className="text-xs font-semibold text-neutral-800 tracking-wide uppercase">
              Next-Gen Research Platform
            </span>
            <span className="text-xs text-neutral-400">|</span>
            <span className="text-xs font-medium text-neutral-600">
              Grounded Dense Vector Retrieval &amp; Verifiable RAG
            </span>
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-neutral-950 tracking-tight leading-[1.1] mb-5">
            Turn scientific literature <br className="hidden sm:block" />
            into <span className="bg-gradient-to-r from-neutral-950 via-neutral-800 to-neutral-700 bg-clip-text text-transparent">verifiable knowledge</span>
          </h1>

          <p className="text-base sm:text-lg text-neutral-600 leading-relaxed max-w-2xl mx-auto">
            Harness ResearchLens AI&apos;s dense vector models. From paragraph-level semantic FAISS retrieval to strictly grounded conversational RAG with exact PDF page citations.
          </p>
        </div>

        {/* Featured Engine Interactive Stage */}
        <div className="bg-white rounded-2xl border border-neutral-200/90 shadow-sm overflow-hidden mb-10">
          <div className="grid grid-cols-1 lg:grid-cols-12">

            {/* Visual Preview Screen */}
            <div className="lg:col-span-7 relative min-h-[320px] sm:min-h-[420px] bg-neutral-950 flex items-center justify-center overflow-hidden group">
              <img
                src={selectedEngine.posterUrl}
                alt={selectedEngine.title}
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105 opacity-80"
              />

              {/* Overlay Gradient for readability */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/40 to-black/20 pointer-events-none" />

              {/* Top status chips */}
              <div className="absolute top-4 left-4 flex items-center gap-2">
                <span className="bg-[#D1FE17] text-neutral-950 text-xs font-bold px-2.5 py-1 rounded-md shadow-xs">
                  {selectedEngine.badge || selectedEngine.tag}
                </span>
                <span className="bg-black/60 backdrop-blur-md text-white text-xs px-2.5 py-1 rounded-md border border-white/10 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                  Live Retrieval Engine
                </span>
              </div>

              {/* Bottom Card Title & Quick sample prompt fill */}
              <div className="absolute bottom-4 left-4 right-4 text-white">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-6 h-6 rounded-md bg-[#D1FE17] text-neutral-950 flex items-center justify-center font-bold text-xs">
                    {getEngineIcon(selectedEngine.id)}
                  </div>
                  <p className="text-xs uppercase tracking-wider text-[#D1FE17] font-semibold">
                    Active Architecture: {selectedEngine.title}
                  </p>
                </div>
                <p className="text-sm font-medium text-neutral-200 line-clamp-2 drop-shadow-sm">
                  &ldquo;{selectedEngine.subtitle}&rdquo;
                </p>
                <div className="mt-3 flex items-center gap-2 flex-wrap">
                  <button
                    type="button"
                    onClick={() => {
                      const sample = sampleQueries[Math.floor(Math.random() * sampleQueries.length)]
                      setQueryInput(sample)
                    }}
                    className="text-[11px] inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/10 backdrop-blur-md text-white hover:bg-white/20 transition-all border border-white/15 cursor-pointer"
                  >
                    <Sparkles className="w-3 h-3 text-[#D1FE17]" />
                    Insert sample academic query
                  </button>
                  <button
                    type="button"
                    onClick={() => onOpenStudioModal(selectedEngine)}
                    className="text-[11px] inline-flex items-center gap-1 text-neutral-300 hover:text-white underline cursor-pointer bg-transparent border-0 p-0"
                  >
                    Inspect model parameters &rarr;
                  </button>
                </div>
              </div>
            </div>

            {/* Engine Info & Interactive Prompt Launcher */}
            <div className="lg:col-span-5 p-6 sm:p-8 flex flex-col justify-between bg-white">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-bold tracking-wider text-neutral-500 uppercase">
                    {selectedEngine.tag}
                  </span>
                  <span className="text-xs text-neutral-400 font-mono">
                    ENGINE: {selectedEngine.id}
                  </span>
                </div>

                <h3 className="text-2xl font-bold text-neutral-950 mb-2">
                  {selectedEngine.title}
                </h3>

                <p className="text-sm text-neutral-600 mb-6 leading-relaxed">
                  {selectedEngine.description}
                </p>

                {/* Technical Specifications specs pills */}
                <div className="grid grid-cols-2 gap-3 mb-6">
                  {selectedEngine.specs.map((spec) => (
                    <div
                      key={spec.label}
                      className="bg-neutral-50 p-3 rounded-xl border border-neutral-200/80"
                    >
                      <span className="text-[11px] text-neutral-500 block uppercase font-medium">
                        {spec.label}
                      </span>
                      <span className="text-xs sm:text-sm font-bold text-neutral-900 flex items-center gap-1 mt-0.5">
                        {spec.value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Interactive Inquiry Box */}
              <form onSubmit={handleQuerySubmit} className="space-y-3 pt-2 border-t border-neutral-100">
                <label
                  htmlFor="hero-query-input"
                  className="text-xs font-semibold text-neutral-700 flex items-center gap-1.5"
                >
                  <Sparkles className="w-3.5 h-3.5 text-neutral-950" />
                  Try inquiry with {selectedEngine.title}
                </label>
                <div className="relative">
                  <textarea
                    id="hero-query-input"
                    rows={2}
                    value={queryInput}
                    onChange={(e) => setQueryInput(e.target.value)}
                    placeholder="e.g. How does self-attention scale computationally with sequence length?"
                    className="w-full text-xs p-3 bg-neutral-50 border border-neutral-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-neutral-950/10 focus:border-neutral-400 focus:bg-white text-neutral-900 resize-none transition-all"
                  />
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-[11px] text-neutral-400">
                    Grounded with exact passage citations
                  </span>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-neutral-950 text-white hover:bg-neutral-800 rounded-xl text-xs font-semibold inline-flex items-center gap-2 transition-transform active:scale-95 shadow-sm cursor-pointer"
                  >
                    <span>Execute Inquiry</span>
                    <ArrowRight className="w-3.5 h-3.5 text-[#D1FE17]" />
                  </button>
                </div>
              </form>

            </div>

          </div>
        </div>

        {/* Engine Carousel Selector Cards */}
        <div>
          <div className="flex items-center justify-between mb-3 px-1">
            <h4 className="text-xs font-bold uppercase tracking-wider text-neutral-500">
              Select Retrieval Engine to Inspect
            </h4>
            <span className="text-xs text-neutral-400">
              Click to view architectural specifications
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {RESEARCH_ENGINES.map((item) => {
              const isSelected = selectedEngine.id === item.id
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => {
                    setSelectedEngine(item)
                    onSelectEngine(item)
                  }}
                  className={`text-left p-3 rounded-xl border transition-all relative overflow-hidden group cursor-pointer ${
                    isSelected
                      ? 'bg-neutral-900 text-white border-neutral-900 shadow-md ring-2 ring-neutral-950/20'
                      : 'bg-white hover:bg-neutral-50 border-neutral-200/90 text-neutral-900 hover:border-neutral-300'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span
                      className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                        isSelected
                          ? 'bg-[#D1FE17] text-neutral-950'
                          : 'bg-neutral-100 text-neutral-700'
                      }`}
                    >
                      {item.tag}
                    </span>
                    {isSelected && (
                      <span className="w-2 h-2 rounded-full bg-[#D1FE17]" />
                    )}
                  </div>
                  <h5 className="font-bold text-sm tracking-tight truncate mb-1">
                    {item.title}
                  </h5>
                  <p
                    className={`text-[11px] line-clamp-2 leading-snug ${
                      isSelected ? 'text-neutral-300' : 'text-neutral-500'
                    }`}
                  >
                    {item.subtitle}
                  </p>
                </button>
              )
            })}
          </div>
        </div>

      </div>
    </section>
  )
}
export default PromoHero
