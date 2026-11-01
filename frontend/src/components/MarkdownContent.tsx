import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface MarkdownContentProps {
  content: string
  className?: string
}

/**
 * Security URL sanitizer: blocks dangerous link schemes (javascript:, data:, vbscript:, etc.).
 * Strictly allows only https://, http://, and mailto: protocols.
 */
export const safeUrlTransform = (url: string): string => {
  const sanitized = url.trim()
  const lower = sanitized.toLowerCase()
  if (
    lower.startsWith('https://') ||
    lower.startsWith('http://') ||
    lower.startsWith('mailto:')
  ) {
    return sanitized
  }
  return ''
}

/**
 * MarkdownContent renders LLM and system messages using GitHub Flavored Markdown (GFM).
 *
 * Security & UX features:
 * - HTML tags are escaped and NEVER rendered as arbitrary DOM (no dangerouslySetInnerHTML, no rehype-raw).
 * - Links are checked against safe protocols and rendered with target="_blank" rel="noopener noreferrer".
 * - Tables are automatically wrapped in a horizontally scrollable container with overflow protection.
 * - Fenced code blocks are horizontally scrollable without breaking parent chat layout.
 */
export const MarkdownContent: React.FC<MarkdownContentProps> = ({ content, className = '' }) => {
  return (
    <div className={`markdown-body ${className}`.trim()}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        urlTransform={safeUrlTransform}
        components={{
          table: ({ children, ...props }) => (
            <div className="table-responsive-wrapper" tabIndex={0} role="region" aria-label="Table">
              <table {...props}>{children}</table>
            </div>
          ),
          a: ({ href, children, ...props }) => {
            const safeHref = href && safeUrlTransform(href) ? href : undefined
            if (!safeHref) {
              return <span>{children}</span>
            }
            return (
              <a href={safeHref} target="_blank" rel="noopener noreferrer" {...props}>
                {children}
              </a>
            )
          },
          pre: ({ children, ...props }) => (
            <pre tabIndex={0} {...props}>
              {children}
            </pre>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
