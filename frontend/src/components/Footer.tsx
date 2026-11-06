import { useState } from 'react'
import { Check } from 'lucide-react'
import { FOOTER_RESEARCH_SECTIONS } from '../data/researchLensData'
import { ActiveView } from '../types'

interface FooterProps {
  onNavigateTab?: (tab: ActiveView) => void
  onSubscribe?: (email: string) => void
}

export const Footer = ({ onNavigateTab, onSubscribe }: FooterProps) => {
  const [email, setEmail] = useState('')
  const [subscribed, setSubscribed] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!email.trim() || !email.includes('@')) return
    if (onSubscribe) onSubscribe(email)
    setSubscribed(true)
    setEmail('')
    setTimeout(() => setSubscribed(false), 4000)
  }

  const handleLinkClick = (href: string, e: React.MouseEvent) => {
    if (href.startsWith('#')) {
      const target = href.replace('#', '')
      if (['documents', 'search', 'chat', 'collections', 'mcp', 'explore'].includes(target)) {
        e.preventDefault()
        if (onNavigateTab) {
          onNavigateTab(target as ActiveView)
          window.scrollTo({ top: 0, behavior: 'smooth' })
        }
      }
    }
  }

  return (
    <footer className="bg-neutral-950 text-neutral-400 text-xs border-t border-neutral-800">

      {/* Newsletter / Early Access Banner (Template Style) */}
      <div className="border-b border-neutral-800/80 bg-neutral-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 flex flex-col md:flex-row items-center justify-between gap-6">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-widest text-[#D1FE17] block mb-1">
              Academic Intelligence Dispatch
            </span>
            <h4 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              Get empirical retrieval benchmarks and architecture breakdowns
            </h4>
            <p className="text-xs text-neutral-400 mt-1">
              Join 42,000+ machine learning researchers, AI engineers, and scientists.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="flex items-center gap-2 w-full md:w-auto">
            {subscribed ? (
              <div className="flex items-center gap-2 px-4 py-2.5 bg-emerald-950 border border-emerald-800 text-emerald-300 rounded-xl text-xs font-semibold">
                <Check className="w-4 h-4 text-emerald-400" />
                <span>Subscribed! Check your inbox for benchmark releases.</span>
              </div>
            ) : (
              <>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Enter your academic email..."
                  required
                  className="bg-neutral-800/80 border border-neutral-700 text-white placeholder:text-neutral-500 text-xs px-3.5 py-2.5 rounded-xl w-full sm:w-72 focus:outline-none focus:border-neutral-500"
                />
                <button
                  type="submit"
                  className="px-4 py-2.5 bg-[#D1FE17] text-neutral-950 font-bold rounded-xl text-xs hover:bg-[#c2ed14] transition-colors shrink-0 shadow-sm cursor-pointer"
                >
                  Subscribe
                </button>
              </>
            )}
          </form>
        </div>
      </div>

      {/* Main Multi-Column Links Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8 mb-12">
          {FOOTER_RESEARCH_SECTIONS.map((section) => (
            <div key={section.title}>
              <h5 className="font-bold text-white text-xs uppercase tracking-wider mb-4">
                {section.title}
              </h5>
              <ul className="space-y-2.5">
                {section.links.map((link) => (
                  <li key={link.name}>
                    <a
                      href={link.href}
                      onClick={(e) => handleLinkClick(link.href, e)}
                      className="text-neutral-400 hover:text-white transition-colors flex items-center justify-between group"
                    >
                      <span className="truncate">{link.name}</span>
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom Attribution and Socials */}
        <div className="pt-8 border-t border-neutral-800/80 flex flex-col md:flex-row items-center justify-between gap-4">

          {/* Logo & Copyright */}
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded-md bg-white text-neutral-950 flex items-center justify-center font-black text-sm">
              <svg className="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">
                <path d="M12 2L3 7v10l9 5 9-5V7l-9-5zm0 3.3L18 8.7v6.6L12 18.7 6 15.3V8.7L12 5.3z" />
              </svg>
            </div>
            <span className="font-heading font-bold text-white text-sm">researchlens.ai</span>
            <span className="text-neutral-500">&copy; 2026 ResearchLens AI. Grounded research assistant with verifiable passage citations.</span>
          </div>

          {/* Social Icons & Status */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5 text-[11px] text-neutral-400 mr-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>Meta FAISS &amp; PostgreSQL: Operational</span>
            </div>

            {/* X / Twitter */}
            <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="text-neutral-400 hover:text-white p-1" aria-label="Twitter">
              <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
              </svg>
            </a>

            {/* Discord */}
            <a href="https://discord.com" target="_blank" rel="noopener noreferrer" className="text-neutral-400 hover:text-white p-1" aria-label="Discord">
              <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994.021-.041.001-.09-.041-.106a13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.929 1.793 8.18 1.793 12.061 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.894.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.028zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" />
              </svg>
            </a>

            {/* YouTube */}
            <a href="https://youtube.com" target="_blank" rel="noopener noreferrer" className="text-neutral-400 hover:text-white p-1" aria-label="YouTube">
              <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
              </svg>
            </a>

            {/* GitHub */}
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-neutral-400 hover:text-white p-1" aria-label="GitHub">
              <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
              </svg>
            </a>
          </div>

        </div>

      </div>
    </footer>
  )
}
export default Footer
