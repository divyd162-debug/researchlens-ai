import { Trophy, Award, BookOpen, Clock, ArrowRight, ExternalLink } from 'lucide-react'

interface MetricsBannerProps {
  onLearnMore: () => void
  onUploadPaper?: () => void
}

export const MetricsBanner = ({
  onLearnMore,
  onUploadPaper,
}: MetricsBannerProps) => {
  return (
    <section className="py-12 bg-gradient-to-r from-neutral-900 via-neutral-950 to-neutral-900 text-white relative overflow-hidden">
      {/* Decorative subtle lighting (Template Style) */}
      <div className="absolute -top-24 right-1/4 w-96 h-96 bg-lime-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-24 left-1/4 w-96 h-96 bg-sky-500/10 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="bg-neutral-900/80 border border-neutral-700/60 rounded-3xl p-6 sm:p-10 backdrop-blur-xl">
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-8">

            {/* Left Content */}
            <div className="space-y-4 max-w-2xl">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-lime-400/10 border border-lime-400/20 text-[#D1FE17] text-xs font-semibold">
                <Trophy className="w-3.5 h-3.5" />
                <span>Deterministic Source Grounding — Anti-Hallucination Pipeline</span>
              </div>

              <h3 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white leading-tight">
                Verifiable Passage Citations on Academic Literature
              </h3>

              <p className="text-neutral-300 text-sm sm:text-base leading-relaxed">
                Indexed with sentence-transformers and PyMuPDF physical page boundaries. Every synthesized answer anchors directly to its source passage with validated chunk identifiers and page numbers.
              </p>

              {/* Badges / Metrics */}
              <div className="flex flex-wrap items-center gap-4 pt-2 text-xs text-neutral-300">
                <div className="flex items-center gap-1.5">
                  <Clock className="w-4 h-4 text-amber-400" />
                  <span>Real-Time Dense Vector Search via Meta FAISS</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Award className="w-4 h-4 text-[#D1FE17]" />
                  <span>Exact Physical Page Pinning &amp; Validation</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <BookOpen className="w-4 h-4 text-cyan-400" />
                  <span>Transactional PostgreSQL &amp; Cascading Chunk Store</span>
                </div>
              </div>
            </div>

            {/* Right Action buttons */}
            <div className="flex flex-col sm:flex-row lg:flex-col gap-3 w-full lg:w-auto shrink-0">
              <button
                type="button"
                onClick={onUploadPaper || onLearnMore}
                className="px-6 py-3 rounded-xl bg-[#D1FE17] text-neutral-950 hover:bg-[#c2ed14] font-bold text-sm inline-flex items-center justify-center gap-2 transition-transform active:scale-95 shadow-md cursor-pointer"
              >
                <span>Upload Research Paper</span>
                <ArrowRight className="w-4 h-4" />
              </button>

              <button
                type="button"
                onClick={onLearnMore}
                className="px-6 py-3 rounded-xl bg-neutral-800/80 hover:bg-neutral-800 text-white font-semibold text-sm border border-neutral-700/80 inline-flex items-center justify-center gap-2 transition-colors cursor-pointer"
              >
                <span>View Grounding Architecture</span>
                <ExternalLink className="w-4 h-4 text-neutral-400" />
              </button>
            </div>

          </div>
        </div>
      </div>
    </section>
  )
}
export default MetricsBanner
