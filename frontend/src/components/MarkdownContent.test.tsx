import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MarkdownContent, safeUrlTransform } from './MarkdownContent'

describe('MarkdownContent Component', () => {
  it('renders headings with proper hierarchy', () => {
    const content = '# Heading 1\n## Heading 2\n### Heading 3'
    render(<MarkdownContent content={content} />)

    expect(screen.getByRole('heading', { level: 1, name: 'Heading 1' })).toBeDefined()
    expect(screen.getByRole('heading', { level: 2, name: 'Heading 2' })).toBeDefined()
    expect(screen.getByRole('heading', { level: 3, name: 'Heading 3' })).toBeDefined()
  })

  it('renders paragraphs, bold text, and italic text', () => {
    const content = 'This is a **bold claim** and an *italic note*.\n\nSecond paragraph.'
    render(<MarkdownContent content={content} />)

    expect(screen.getByText(/This is a/)).toBeDefined()
    expect(screen.getByText('bold claim').tagName).toBe('STRONG')
    expect(screen.getByText('italic note').tagName).toBe('EM')
    expect(screen.getByText('Second paragraph.')).toBeDefined()
  })

  it('renders unordered and ordered lists', () => {
    const content = '- Item A\n- Item B\n\n1. Step One\n2. Step Two'
    render(<MarkdownContent content={content} />)

    expect(screen.getByText('Item A')).toBeDefined()
    expect(screen.getByText('Item B')).toBeDefined()
    expect(screen.getByText('Step One')).toBeDefined()
    expect(screen.getByText('Step Two')).toBeDefined()
  })

  it('renders GFM markdown tables with header and cells', () => {
    const content = `
| Metric | Value |
|---|---:|
| Precision | 0.6364 |
| Recall | 0.9742 |
| FAR | 0.1820 |
`
    render(<MarkdownContent content={content} />)

    expect(screen.getByRole('table')).toBeDefined()
    expect(screen.getByText('Metric')).toBeDefined()
    expect(screen.getByText('0.6364')).toBeDefined()
    expect(screen.getByText('0.9742')).toBeDefined()
    expect(screen.getByText('0.1820')).toBeDefined()

    // Responsive table wrapper present
    const tableWrapper = screen.getByRole('region', { name: 'Table' })
    expect(tableWrapper.className).toContain('table-responsive-wrapper')
  })

  it('renders code blocks and inline code', () => {
    const content = 'Inline `codeToken` and block:\n\n```python\ndef solve():\n    return 42\n```'
    render(<MarkdownContent content={content} />)

    expect(screen.getByText('codeToken').tagName).toBe('CODE')
    expect(screen.getByText(/def solve\(\):/)).toBeDefined()
  })

  it('renders blockquotes and horizontal rules', () => {
    const content = '> Grounded research excerpt.\n\n---'
    render(<MarkdownContent content={content} />)

    expect(screen.getByText('Grounded research excerpt.').closest('blockquote')).toBeDefined()
    expect(screen.getByRole('separator')).toBeDefined()
  })

  it('blocks dangerous links (javascript:, data:, vbscript:) and renders safe links with target=_blank', () => {
    const content = '[Safe Link](https://arxiv.org/abs/1706.03762) and [Unsafe Link](javascript:alert("XSS"))'
    render(<MarkdownContent content={content} />)

    const safeLink = screen.getByRole('link', { name: 'Safe Link' })
    expect(safeLink.getAttribute('href')).toBe('https://arxiv.org/abs/1706.03762')
    expect(safeLink.getAttribute('target')).toBe('_blank')
    expect(safeLink.getAttribute('rel')).toContain('noopener')

    // Unsafe link is stripped of href and rendered as non-clickable span
    expect(screen.queryByRole('link', { name: 'Unsafe Link' })).toBeNull()
    expect(screen.getByText('Unsafe Link').tagName).toBe('SPAN')
  })

  it('does NOT render raw HTML tags from model output as arbitrary DOM elements', () => {
    const content = 'Text with <script>window.pwned = true</script> and <img src=x onerror=alert(1)>.'
    render(<MarkdownContent content={content} />)

    expect(document.querySelector('script')).toBeNull()
    expect(document.querySelector('img')).toBeNull()
    // Content is rendered safely as escaped text
    expect(screen.getByText(/Text with/)).toBeDefined()
  })

  it('resiliently handles malformed markdown without crashing', () => {
    const malformed = 'Unclosed **bold and unclosed `code and broken | table | without end'
    render(<MarkdownContent content={malformed} />)

    expect(screen.getByText(/Unclosed/)).toBeDefined()
  })
})

describe('safeUrlTransform unit validation', () => {
  it('allows http, https, and mailto', () => {
    expect(safeUrlTransform('https://example.com')).toBe('https://example.com')
    expect(safeUrlTransform('http://example.com')).toBe('http://example.com')
    expect(safeUrlTransform('mailto:user@example.com')).toBe('mailto:user@example.com')
  })

  it('rejects dangerous protocols', () => {
    expect(safeUrlTransform('javascript:alert(1)')).toBe('')
    expect(safeUrlTransform('JAVASCRIPT:alert(1)')).toBe('')
    expect(safeUrlTransform('data:text/html,<script>alert(1)</script>')).toBe('')
    expect(safeUrlTransform('vbscript:msgbox(1)')).toBe('')
    expect(safeUrlTransform('file:///etc/passwd')).toBe('')
  })
})
