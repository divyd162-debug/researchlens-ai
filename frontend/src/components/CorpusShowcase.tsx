import { useState } from 'react'
import { Heart, ArrowUpRight, Cpu, BookOpen } from 'lucide-react'
import { CORPUS_SHOWCASE, CorpusShowcaseItem } from '../data/researchLensData'

interface CorpusShowcaseProps {
  onInspectPaper: (item: CorpusShowcaseItem) => void
}

export const CorpusShowcase = ({ onInspectPaper }: CorpusShowcaseProps) => {
  const [selectedFilter, setSelectedFilter] = useState<string>('All')
  const [items, setItems] = useState<CorpusShowcaseItem[]>(CORPUS_SHOWCASE)

  const filters = [
    'All',
    'Transformers',
    'Retrieval',
    'Vision & Multimodal',
    'Optimization',
    'Bioinformatics',
  ]

  const filteredItems =
    selectedFilter === 'All'
      ? items
      : items.filter((i) => i.category === selectedFilter)

  const handleLike = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setItems((prev) =>
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

  return (
    <section id="corpus" className="py-16 bg-white border-b border-neutral-200/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Top Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4">
          <div>
            <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-neutral-500 mb-2">
              <span className="w-2 h-2 rounded-full bg-blue-500" />
              Reference Paper Architectures (Demonstration Examples)
            </div>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-neutral-950 tracking-tight">
              Foundational Research Showcase
            </h2>
            <p className="text-neutral-600 text-sm sm:text-base mt-2 max-w-xl">
              Inspect sample chunk boundaries and inquiry patterns across seminal academic works. Upload your own scientific preprints in the <strong>Documents</strong> tab to build your live searchable library.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-neutral-500 bg-neutral-100 px-2.5 py-1 rounded-md border border-neutral-200">
              Curated Reference Examples
            </span>
          </div>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-2 overflow-x-auto pb-4 mb-8 hide-scrollbar">
          {filters.map((filter) => {
            const isActive = selectedFilter === filter
            return (
              <button
                key={filter}
                type="button"
                onClick={() => setSelectedFilter(filter)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
                  isActive
                    ? 'bg-neutral-950 text-white shadow-xs'
                    : 'bg-neutral-50 text-neutral-600 hover:text-neutral-950 hover:bg-neutral-100 border border-neutral-200/80'
                }`}
              >
                {filter}
              </button>
            )
          })}
        </div>

        {/* Grid Display */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredItems.map((item) => {
            return (
              <div
                key={item.id}
                onClick={() => onInspectPaper(item)}
                className="group relative bg-[#FAFAFB] rounded-2xl border border-neutral-200/90 overflow-hidden shadow-xs hover:shadow-xl transition-all duration-300 flex flex-col cursor-pointer"
              >
                {/* Visual Image */}
                <div className="relative aspect-video w-full bg-neutral-900 overflow-hidden">
                  <img
                    src={item.imageUrl}
                    alt={item.title}
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105 opacity-80"
                  />

                  {/* Top Badges */}
                  <div className="absolute top-3 left-3 right-3 flex items-center justify-between text-white text-[11px] font-medium z-10">
                    <span className="bg-black/60 backdrop-blur-md px-2.5 py-0.5 rounded-md border border-white/10 flex items-center gap-1.5 text-[#D1FE17] font-semibold">
                      <Cpu className="w-3 h-3" />
                      {item.model}
                    </span>

                    <button
                      type="button"
                      onClick={(e) => handleLike(item.id, e)}
                      className={`px-2 py-0.5 rounded-md flex items-center gap-1 transition-all cursor-pointer ${
                        item.isLiked
                          ? 'bg-red-500 text-white'
                          : 'bg-black/60 backdrop-blur-md text-white hover:bg-black/80 border border-white/10'
                      }`}
                      aria-label="Like paper"
                    >
                      <Heart
                        className={`w-3 h-3 ${item.isLiked ? 'fill-current' : ''}`}
                      />
                      <span>{item.likes}</span>
                    </button>
                  </div>

                  {/* Gradient Overlay */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/85 via-black/20 to-transparent pointer-events-none" />

                  {/* Bottom Image Info */}
                  <div className="absolute bottom-3 left-3 right-3 text-white z-10 flex items-center justify-between text-xs">
                    <span className="bg-white/20 backdrop-blur-md px-2 py-0.5 rounded border border-white/15">
                      {item.publishedYear} &bull; {item.pages} Pages
                    </span>
                    <span className="text-white/90 flex items-center gap-1">
                      <BookOpen className="w-3.5 h-3.5 text-[#D1FE17]" />
                      {item.chunkCount} Chunks
                    </span>
                  </div>
                </div>

                {/* Content Details */}
                <div className="p-5 flex flex-col justify-between flex-grow bg-white">
                  <div>
                    <h3 className="font-bold text-base text-neutral-950 mb-1 group-hover:text-black line-clamp-1">
                      {item.title}
                    </h3>
                    <p className="text-[11px] text-neutral-400 font-medium mb-2.5 truncate">
                      {item.authors}
                    </p>
                    <p className="text-xs text-neutral-600 line-clamp-2 leading-relaxed mb-4">
                      {item.summary}
                    </p>
                  </div>

                  {/* Footer & Inspect Trigger */}
                  <div className="pt-3 border-t border-neutral-100 flex items-center justify-between">
                    <div className="flex flex-col">
                      <span className="text-[10px] text-neutral-400 uppercase font-semibold">
                        Relevance Score
                      </span>
                      <span className="text-xs font-bold text-emerald-600">
                        {(item.relevanceScore * 100).toFixed(0)}% Match
                      </span>
                    </div>

                    <span className="inline-flex items-center gap-1 text-xs font-bold text-neutral-900 group-hover:text-black group-hover:translate-x-0.5 transition-transform">
                      <span>Inspect Evidence</span>
                      <ArrowUpRight className="w-3.5 h-3.5" />
                    </span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>

      </div>
    </section>
  )
}
export default CorpusShowcase
