import {
  Search,
  ShieldCheck,
  FileText,
  Layers,
  Terminal,
  FolderGit2,
  Sparkles,
  ArrowRight,
} from 'lucide-react'
import { CORE_CAPABILITIES } from '../data/researchLensData'

interface CapabilitiesGridProps {
  onSelectCapability: (id: string) => void
}

export const CapabilitiesGrid = ({
  onSelectCapability,
}: CapabilitiesGridProps) => {
  const getIcon = (name: string) => {
    switch (name) {
      case 'Search':
        return <Search className="w-5 h-5 text-neutral-900" />
      case 'ShieldCheck':
        return <ShieldCheck className="w-5 h-5 text-neutral-900" />
      case 'FileText':
        return <FileText className="w-5 h-5 text-neutral-900" />
      case 'Layers':
        return <Layers className="w-5 h-5 text-neutral-900" />
      case 'Terminal':
        return <Terminal className="w-5 h-5 text-neutral-900" />
      case 'FolderGit2':
        return <FolderGit2 className="w-5 h-5 text-neutral-900" />
      default:
        return <Sparkles className="w-5 h-5 text-neutral-900" />
    }
  }

  return (
    <section id="capabilities" className="py-16 bg-white border-b border-neutral-200/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-4">
          <div>
            <div className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-neutral-500 mb-2">
              <span className="w-2 h-2 rounded-full bg-neutral-900" />
              Core Architecture &amp; Capabilities
            </div>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-neutral-950 tracking-tight">
              A complete academic retrieval ecosystem
            </h2>
            <p className="text-neutral-600 text-sm sm:text-base mt-2 max-w-xl">
              From paragraph-level dense embeddings to multi-turn conversational synthesis and Anthropic Claude MCP agent integrations.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-neutral-500 hidden sm:inline">6 Core modules operational</span>
            <button
              type="button"
              onClick={() => onSelectCapability('all')}
              className="px-4 py-2 text-xs font-semibold rounded-lg bg-neutral-100 hover:bg-neutral-200 text-neutral-800 transition-colors cursor-pointer"
            >
              Compare Specifications
            </button>
          </div>
        </div>

        {/* 6 Capability Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {CORE_CAPABILITIES.map((capability) => {
            return (
              <div
                key={capability.id}
                className="group relative bg-[#FAFAFB] hover:bg-white rounded-2xl p-6 sm:p-7 border border-neutral-200/90 hover:border-neutral-300 hover:shadow-lg transition-all duration-300 flex flex-col justify-between"
              >
                <div>
                  {/* Top Bar with Icon & Tag */}
                  <div className="flex items-center justify-between mb-5">
                    <div className="w-10 h-10 rounded-xl bg-white border border-neutral-200 flex items-center justify-center shadow-xs group-hover:scale-105 transition-transform">
                      {getIcon(capability.iconName)}
                    </div>
                    <span className="text-[11px] font-bold tracking-wide uppercase px-2.5 py-1 rounded-full bg-white border border-neutral-200 text-neutral-800">
                      {capability.tag}
                    </span>
                  </div>

                  {/* Category */}
                  <span className="text-[11px] font-bold text-neutral-400 uppercase tracking-wider">
                    {capability.category}
                  </span>

                  {/* Title */}
                  <h3 className="text-xl font-bold text-neutral-950 mt-1 mb-3 group-hover:text-black">
                    {capability.title}
                  </h3>

                  {/* Description */}
                  <p className="text-xs text-neutral-600 leading-relaxed mb-6">
                    {capability.description}
                  </p>
                </div>

                {/* Bottom Footer Details & Trigger */}
                <div className="pt-4 border-t border-neutral-200/70 flex items-center justify-between">
                  <span className="text-xs font-semibold text-neutral-700">
                    {capability.stats}
                  </span>

                  <button
                    type="button"
                    onClick={() => onSelectCapability(capability.id)}
                    className="inline-flex items-center gap-1.5 text-xs font-bold text-neutral-900 group-hover:text-black hover:underline cursor-pointer bg-transparent border-0 p-0"
                  >
                    <span>Launch Studio</span>
                    <ArrowRight className="w-3.5 h-3.5 text-neutral-500 group-hover:translate-x-1 group-hover:text-black transition-all" />
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
export default CapabilitiesGrid
