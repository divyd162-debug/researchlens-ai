import { useState } from 'react'
import { Eye, Heart, Sparkles, Copy, Check, BookOpen } from 'lucide-react'
import {
  PRESET_RESEARCH_WORKFLOWS,
  ResearchPresetWorkflow,
} from '../data/researchLensData'

interface ResearchPresetsProps {
  onExecutePreset: (preset: ResearchPresetWorkflow) => void
}

export const ResearchPresets = ({
  onExecutePreset,
}: ResearchPresetsProps) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('All')
  const [presets, setPresets] = useState<ResearchPresetWorkflow[]>(
    PRESET_RESEARCH_WORKFLOWS
  )
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const categories = [
    'All',
    'Transformers',
    'Retrieval & RAG',
    'Vision & Multimodal',
    'Optimization',
    'Bioinformatics',
  ]

  const filteredPresets =
    selectedCategory === 'All'
      ? presets
      : presets.filter((p) => p.category === selectedCategory)

  const handleToggleLike = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setPresets((prev) =>
      prev.map((item) => {
        if (item.id === id) {
          const isLiked = !item.isLiked
          return {
            ...item,
            isLiked,
            likes: isLiked ? item.likes + 1 : item.likes - 1,
          }
        }
        return item
      })
    )
  }

  const handleCopyPrompt = (
    preset: ResearchPresetWorkflow,
    e: React.MouseEvent
  ) => {
    e.stopPropagation()
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(preset.prompt)
      }
    } catch {
      // fallback
    }
    setCopiedId(preset.id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  return (
    <section id="presets" className="py-16 bg-[#FAFAFB] border-b border-neutral-200/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4">
          <div>
            <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-neutral-500 mb-2">
              <span className="w-2 h-2 rounded-full bg-lime-500" />
              Academic Inquiry Presets
            </div>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-neutral-950 tracking-tight">
              Trending Scientific Investigations
            </h2>
            <p className="text-neutral-600 text-sm sm:text-base mt-2 max-w-xl">
              Execute battle-tested inquiries across foundational architectures. Click <strong>Run Inquiry</strong> to synthesize grounded answers with citations.
            </p>
          </div>

          {/* Stat Counter */}
          <div className="flex items-center gap-2 bg-white px-3.5 py-2 rounded-xl border border-neutral-200 shadow-xs">
            <Sparkles className="w-4 h-4 text-neutral-950" />
            <span className="text-xs font-semibold text-neutral-800">
              Deterministic passage citations verified against PostgreSQL
            </span>
          </div>
        </div>

        {/* Category Pills */}
        <div className="flex items-center gap-2 overflow-x-auto pb-4 mb-8 hide-scrollbar">
          {categories.map((cat) => {
            const isActive = selectedCategory === cat
            return (
              <button
                key={cat}
                type="button"
                onClick={() => setSelectedCategory(cat)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
                  isActive
                    ? 'bg-neutral-950 text-white shadow-xs'
                    : 'bg-white text-neutral-600 hover:text-neutral-950 hover:bg-neutral-100 border border-neutral-200'
                }`}
              >
                {cat}
              </button>
            )
          })}
        </div>

        {/* Vertical Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-5">
          {filteredPresets.map((preset) => {
            return (
              <div
                key={preset.id}
                className="group relative bg-white rounded-2xl border border-neutral-200/90 overflow-hidden shadow-xs hover:shadow-xl transition-all duration-300 flex flex-col cursor-pointer"
                onClick={() => onExecutePreset(preset)}
              >
                {/* 9:16 Poster Container */}
                <div className="relative aspect-[9/16] w-full bg-neutral-950 overflow-hidden">
                  <img
                    src={preset.posterUrl}
                    alt={preset.title}
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105 opacity-80"
                  />

                  {/* Top Stats Bar */}
                  <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between text-white text-[11px] font-medium z-10">
                    <span className="bg-black/60 backdrop-blur-md px-2 py-0.5 rounded-md flex items-center gap-1 border border-white/10">
                      <Eye className="w-3 h-3 text-[#D1FE17]" />
                      {preset.views}
                    </span>

                    <button
                      type="button"
                      onClick={(e) => handleToggleLike(preset.id, e)}
                      className={`px-2 py-0.5 rounded-md flex items-center gap-1 transition-all cursor-pointer ${
                        preset.isLiked
                          ? 'bg-red-500 text-white'
                          : 'bg-black/60 backdrop-blur-md text-white hover:bg-black/80 border border-white/10'
                      }`}
                      aria-label="Like inquiry"
                    >
                      <Heart
                        className={`w-3 h-3 ${preset.isLiked ? 'fill-current' : ''}`}
                      />
                      <span>{preset.likes.toLocaleString()}</span>
                    </button>
                  </div>

                  {/* Play/Inspect Icon on Hover */}
                  <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/35">
                    <div className="w-12 h-12 rounded-full bg-white/95 text-neutral-950 flex items-center justify-center shadow-lg transform group-hover:scale-110 transition-transform">
                      <BookOpen className="w-5 h-5" />
                    </div>
                  </div>

                  {/* Bottom Vignette */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/30 to-transparent pointer-events-none" />

                  {/* Prompt preview overlaid on bottom of card */}
                  <div className="absolute bottom-3 left-3 right-3 text-white z-10">
                    <span className="text-[10px] uppercase font-bold tracking-wider text-[#D1FE17] block mb-0.5">
                      {preset.category}
                    </span>
                    <h4 className="text-xs font-bold leading-snug line-clamp-2 drop-shadow-sm mb-1.5">
                      {preset.title}
                    </h4>
                    <p className="text-[10px] text-neutral-300 line-clamp-2 leading-relaxed opacity-90">
                      {preset.prompt}
                    </p>
                  </div>
                </div>

                {/* Card Bottom Meta & Actions */}
                <div className="p-3 bg-white flex flex-col justify-between flex-grow gap-2.5">
                  {/* Creator Bar */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <img
                        src={preset.creator.avatar}
                        alt={preset.creator.name}
                        className="w-5 h-5 rounded-full object-cover border border-neutral-200"
                      />
                      <span className="text-[11px] font-semibold text-neutral-800 truncate max-w-[90px]">
                        {preset.creator.name}
                      </span>
                    </div>

                    <button
                      type="button"
                      onClick={(e) => handleCopyPrompt(preset, e)}
                      title="Copy inquiry query text"
                      className="p-1 rounded-md text-neutral-400 hover:text-neutral-700 hover:bg-neutral-100 cursor-pointer bg-transparent border-0"
                    >
                      {copiedId === preset.id ? (
                        <Check className="w-3.5 h-3.5 text-emerald-600" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>

                  {/* Run Inquiry Button */}
                  <button
                    type="button"
                    onClick={() => onExecutePreset(preset)}
                    className="w-full py-1.5 px-2 bg-neutral-950 hover:bg-neutral-800 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors shadow-xs cursor-pointer"
                  >
                    <Sparkles className="w-3 h-3 text-[#D1FE17]" />
                    <span>Run Inquiry</span>
                  </button>
                </div>
              </div>
            )
          })}
        </div>

      </div>
    </section>
  )
}
export default ResearchPresets
