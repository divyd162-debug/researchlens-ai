export interface ResearchEngineItem {
  id: string
  title: string
  tag: string
  tagColor?: string
  subtitle: string
  posterUrl: string
  badge?: string
  description: string
  linkText: string
  specs: { label: string; value: string }[]
}

export interface ResearchPresetWorkflow {
  id: string
  title: string
  category: string
  views: string
  citationsCount: number
  isLiked?: boolean
  likes: number
  posterUrl: string
  creator: {
    name: string
    avatar: string
    handle: string
  }
  prompt: string
  scope: 'GLOBAL' | 'COLLECTION' | 'DOCUMENT'
  topK: number
  threshold: number
  latency: string
}

export interface CorpusShowcaseItem {
  id: string
  title: string
  category: 'All' | 'Transformers' | 'Retrieval' | 'Vision & Multimodal' | 'Optimization' | 'Bioinformatics'
  imageUrl: string
  authors: string
  publishedYear: number
  relevanceScore: number
  pages: number
  chunkCount: number
  model: string
  summary: string
  keyFinding: string
  sampleQuery: string
  isLiked?: boolean
  likes: number
}

export const RESEARCH_ENGINES: ResearchEngineItem[] = [
  {
    id: 'minilm-embeddings',
    title: 'all-MiniLM-L6-v2',
    tag: 'BI-ENCODER',
    subtitle: '384-dimensional dense semantic embedding space with high cosine sensitivity',
    posterUrl: 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=1200&q=80',
    badge: '384-D VECTORS',
    description: 'Calculates normalized passage representations capturing nuanced academic phrasing, algorithmic complexity claims, and domain jargon.',
    linkText: 'Inspect Vectors',
    specs: [
      { label: 'Target Latency', value: '~18ms / chunk (CPU target)' },
      { label: 'Vector Dimension', value: '384 Dimensions' },
      { label: 'Context Window', value: '256 Word Tokens' },
      { label: 'Similarity Metric', value: 'Inner Product / Cosine' },
    ],
  },
  {
    id: 'faiss-vector-store',
    title: 'Meta FAISS IndexFlatIP',
    tag: 'VECTOR INDEX',
    subtitle: 'Deterministic in-memory vector index with transactional PostgreSQL persistence',
    posterUrl: 'https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1200&q=80',
    badge: 'DENSE VECTOR INDEX',
    description: 'Exact inner product neighbor searches over hundreds of thousands of academic passages with zero approximate recall degradation.',
    linkText: 'Explore Index',
    specs: [
      { label: 'Vector Architecture', value: 'FAISS IndexFlatIP (Dense)' },
      { label: 'Recall Accuracy', value: '100% Exact Nearest' },
      { label: 'Thread Safety', value: 'Reentrant Read-Write' },
      { label: 'Recovery Strategy', value: 'Atomic Cold-Boot Sync' },
    ],
  },
  {
    id: 'grounded-rag',
    title: 'Grounded Synthesis Pipeline',
    tag: 'VERIFIABLE RAG',
    subtitle: 'Strict citation-enforced generation with fallback detection & source verification',
    posterUrl: 'https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?auto=format&fit=crop&w=1200&q=80',
    badge: 'CITATION VERIFIED',
    description: 'Extracts exact source passages, enforces citation validation, and flags ungrounded claims before delivering answers to researchers.',
    linkText: 'Launch RAG Studio',
    specs: [
      { label: 'Citation Fidelity', value: '100% Verifiable Page/Chunk' },
      { label: 'Synthesis LLM', value: 'OmniRoute / Anthropic Claude' },
      { label: 'Grounding Verification', value: 'PostgreSQL Chunk Match' },
      { label: 'Fallback Logic', value: 'Graceful Abstention' },
    ],
  },
  {
    id: 'pdf-pipeline',
    title: 'Layout-Aware PDF Extractor',
    tag: 'DOCUMENT INGESTION',
    subtitle: 'Page-preserving text extraction with table, formula, and boundary detection',
    posterUrl: 'https://images.unsplash.com/photo-1534447677768-be436bb09401?auto=format&fit=crop&w=1200&q=80',
    badge: 'MULTI-COLUMN PARSER',
    description: 'Deconstructs complex academic two-column formats, preserving section headings, authors, bibliographies, and physical page numbers.',
    linkText: 'Upload Document',
    specs: [
      { label: 'Estimated Throughput', value: 'Up to 12 Pages / Sec' },
      { label: 'Chunking Strategy', value: 'Recursive Character Split' },
      { label: 'Page Preservation', value: '100% Hard Bounded' },
      { label: 'Sanitization', value: 'Strict PDF Traversal Defense' },
    ],
  },
  {
    id: 'citation-verifier',
    title: 'Semantic Evidence Verifier',
    tag: 'INTEGRITY GUARANTEE',
    subtitle: 'Post-generation token overlap and fuzzy n-gram citation grounding validator',
    posterUrl: 'https://images.unsplash.com/photo-1563089145-599997674d42?auto=format&fit=crop&w=1200&q=80',
    badge: 'PASSAGE VERIFIED',
    description: 'Inspects every citation tag `[Page X]` against retrieved database chunks to ensure claims are directly rooted in primary literature.',
    linkText: 'Verify Citations',
    specs: [
      { label: 'Verification Rate', value: 'Real-time Inline Filter' },
      { label: 'Citation Mode', value: 'Numbered [Page X]' },
      { label: 'Validation Engine', value: 'Strict Token N-Gram Match' },
      { label: 'Evidence Drawer', value: 'Direct Source Highlighting' },
    ],
  },
  {
    id: 'corpus-knowledge-graph',
    title: 'Curated Research Collections',
    tag: 'ORGANIZATION',
    subtitle: 'Domain-specific collections for targeted reading lists and scoped inquiries',
    posterUrl: 'https://images.unsplash.com/photo-1511512578047-dfb367046420?auto=format&fit=crop&w=1200&q=80',
    badge: 'SCOPED SEARCH',
    description: 'Organize research papers into themed collections such as LLM Alignment, Cyber Security, or Neuroscience to restrict vector search boundaries.',
    linkText: 'Manage Collections',
    specs: [
      { label: 'Scope Control', value: 'Global, Collection, Doc' },
      { label: 'Relational Model', value: 'PostgreSQL Foreign Keys' },
      { label: 'Document Re-use', value: 'Many-to-Many Association' },
      { label: 'Access Control', value: 'Private Isolated Workspaces' },
    ],
  },
]

export const CORE_CAPABILITIES = [
  {
    id: 'semantic-search-card',
    title: 'Dense Semantic Vector Search',
    category: 'RETRIEVAL ENGINE',
    description: 'Query literature by conceptual intent rather than rigid keywords. Meta FAISS retrieves paragraph-level evidence in under 5 milliseconds.',
    stats: 'Dense In-Memory Index',
    tag: 'Top Feature',
    accent: 'bg-lime-400',
    iconName: 'Search',
  },
  {
    id: 'grounded-rag-card',
    title: 'Grounded Question Answering',
    category: 'SYNTHESIS ENGINE',
    description: 'Ask complex scientific questions across single papers or vast reading collections. Answers cite exact pages and supporting passage text.',
    stats: 'Verifiable Passage Quotes',
    tag: 'Verifiable RAG',
    accent: 'bg-emerald-400',
    iconName: 'ShieldCheck',
  },
  {
    id: 'ingestion-card',
    title: 'High-Throughput PDF Ingestion',
    category: 'INGESTION PIPELINE',
    description: 'Robust two-column document parsing, chunking with configurable overlap, and SHA-256 deduplication protecting your library.',
    stats: 'Preserves Hard Page Numbers',
    tag: 'Robust Pipeline',
    accent: 'bg-cyan-400',
    iconName: 'FileText',
  },
  {
    id: 'citation-card',
    title: 'Passage Evidence Drawer',
    category: 'INTERPRETABILITY',
    description: 'Click on any in-text citation badge to inspect the original author text, page context, and similarity score in a dedicated inspection panel.',
    stats: 'Full Provenance Tracking',
    tag: 'Audit Ready',
    accent: 'bg-amber-400',
    iconName: 'Layers',
  },
  {
    id: 'mcp-api-card',
    title: 'Claude MCP & REST API',
    category: 'DEVELOPER & AGENTS',
    description: 'Connect ResearchLens AI to Claude Desktop, Cursor IDE, or automated scientific workflows via our Anthropic Model Context Protocol server.',
    stats: 'Standardized Tool Interface',
    tag: 'Developer Ready',
    accent: 'bg-violet-400',
    iconName: 'Terminal',
  },
  {
    id: 'collections-card',
    title: 'Curated Literature Collections',
    category: 'KNOWLEDGE BASE',
    description: 'Group literature into specialized topical clusters to perform scoped searches and comparative inquiries without interference.',
    stats: 'Multi-scope Filtering',
    tag: 'Workflow Centric',
    accent: 'bg-pink-400',
    iconName: 'FolderGit2',
  },
]

export const PRESET_RESEARCH_WORKFLOWS: ResearchPresetWorkflow[] = [
  {
    id: 'preset-1',
    title: 'Self-Attention vs. Recurrence Computational Scaling',
    category: 'Transformers',
    views: '14.2k',
    citationsCount: 18,
    likes: 1240,
    posterUrl: 'https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=720&q=80',
    creator: {
      name: 'Dr. Vaswani et al.',
      avatar: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=120&q=80',
      handle: '@nlp_architectures',
    },
    prompt: 'How does self-attention computational complexity per layer scale with sequence length compared to recurrent and convolutional layers?',
    scope: 'GLOBAL',
    topK: 5,
    threshold: 0.65,
    latency: '34ms',
  },
  {
    id: 'preset-2',
    title: 'Dense Passage Retrieval (DPR) Dual-Encoder Loss',
    category: 'Retrieval & RAG',
    views: '9.8k',
    citationsCount: 24,
    likes: 890,
    posterUrl: 'https://images.unsplash.com/photo-1592945403244-b3fbafd7f539?auto=format&fit=crop&w=720&q=80',
    creator: {
      name: 'Karpukhin et al.',
      avatar: 'https://images.unsplash.com/photo-1517841905240-472988babdf9?auto=format&fit=crop&w=120&q=80',
      handle: '@dense_retrieval',
    },
    prompt: 'What objective function and in-batch negative sampling strategies are employed during bi-encoder training for open-domain QA retrieval?',
    scope: 'GLOBAL',
    topK: 6,
    threshold: 0.70,
    latency: '41ms',
  },
  {
    id: 'preset-3',
    title: 'Residual Networks Gradient Flow & Degradation Proof',
    category: 'Vision & Multimodal',
    views: '12.5k',
    citationsCount: 32,
    likes: 1450,
    posterUrl: 'https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=720&q=80',
    creator: {
      name: 'He, Zhang, Ren & Sun',
      avatar: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=120&q=80',
      handle: '@deep_residual',
    },
    prompt: 'How do identity shortcut mappings alleviate the degradation problem in extremely deep neural networks without introducing extra parameters?',
    scope: 'GLOBAL',
    topK: 5,
    threshold: 0.68,
    latency: '39ms',
  },
  {
    id: 'preset-4',
    title: 'FlashAttention IO-Aware Memory Hierarchy Bounds',
    category: 'Optimization',
    views: '18.1k',
    citationsCount: 29,
    likes: 2120,
    posterUrl: 'https://images.unsplash.com/photo-1569718212165-3a8278d5f624?auto=format&fit=crop&w=720&q=80',
    creator: {
      name: 'Tri Dao et al.',
      avatar: 'https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?auto=format&fit=crop&w=120&q=80',
      handle: '@hardware_aware',
    },
    prompt: 'Explain the tiling technique and online softmax recomputation used to avoid materializing the N-by-N attention matrix in HBM.',
    scope: 'GLOBAL',
    topK: 6,
    threshold: 0.72,
    latency: '28ms',
  },
  {
    id: 'preset-5',
    title: 'Constitutional AI Self-Critique & Preference RL',
    category: 'Transformers',
    views: '11.4k',
    citationsCount: 15,
    likes: 970,
    posterUrl: 'https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=720&q=80',
    creator: {
      name: 'Bai, Kadavath et al.',
      avatar: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?auto=format&fit=crop&w=120&q=80',
      handle: '@alignment_lab',
    },
    prompt: 'How does constitutional RL use iterative self-critique based on natural language principles to remove human supervision during alignment?',
    scope: 'GLOBAL',
    topK: 5,
    threshold: 0.64,
    latency: '45ms',
  },
  {
    id: 'preset-6',
    title: 'AlphaFold 2 Evoformer Pair Representation',
    category: 'Bioinformatics',
    views: '22.7k',
    citationsCount: 45,
    likes: 2840,
    posterUrl: 'https://images.unsplash.com/photo-1544551763-46a013bb70d5?auto=format&fit=crop&w=720&q=80',
    creator: {
      name: 'Jumper, Evans et al.',
      avatar: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=120&q=80',
      handle: '@structural_bio',
    },
    prompt: 'Explain the triangular multiplicative updates and axial attention on evolutionary and pair representations in the Evoformer module.',
    scope: 'GLOBAL',
    topK: 8,
    threshold: 0.70,
    latency: '52ms',
  },
]

export const CORPUS_SHOWCASE: CorpusShowcaseItem[] = [
  {
    id: 'doc-showcase-1',
    title: 'Attention Is All You Need',
    category: 'Transformers',
    imageUrl: 'https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?auto=format&fit=crop&w=1000&q=80',
    authors: 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al.',
    publishedYear: 2017,
    relevanceScore: 0.96,
    pages: 15,
    chunkCount: 42,
    model: 'all-MiniLM-L6-v2',
    summary: 'The seminal paper introducing the Transformer architecture based entirely on self-attention mechanisms, dispensing with recurrence and convolutions.',
    keyFinding: 'Achieved 28.4 BLEU on English-to-German WMT 2014, outperforming existing state-of-the-art models including ensembles with significantly faster training.',
    sampleQuery: 'What are the dimensions of Query, Key, and Value vectors in Multi-Head Attention?',
    likes: 3120,
  },
  {
    id: 'doc-showcase-2',
    title: 'BERT: Pre-training of Deep Bidirectional Transformers',
    category: 'Transformers',
    imageUrl: 'https://images.unsplash.com/photo-1578749556568-bc2c40e68b61?auto=format&fit=crop&w=1000&q=80',
    authors: 'Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova',
    publishedYear: 2018,
    relevanceScore: 0.94,
    pages: 16,
    chunkCount: 38,
    model: 'all-MiniLM-L6-v2',
    summary: 'Introduces bidirectional Transformer pre-training using masked language modeling (MLM) and next sentence prediction (NSP) to learn contextualized representations.',
    keyFinding: 'Advanced the state of the art on eleven natural language processing tasks, pushing GLUE benchmark score to 80.5% and SQuAD v1.1 F1 score to 93.2%.',
    sampleQuery: 'How is the Masked Language Model objective formulated during pre-training?',
    likes: 2450,
  },
  {
    id: 'doc-showcase-3',
    title: 'Retrieval-Augmented Generation for Knowledge-Intensive Tasks',
    category: 'Retrieval',
    imageUrl: 'https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?auto=format&fit=crop&w=1000&q=80',
    authors: 'Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, et al.',
    publishedYear: 2020,
    relevanceScore: 0.98,
    pages: 19,
    chunkCount: 54,
    model: 'Meta FAISS + RAG',
    summary: 'Explores general-purpose fine-tuning recipes for models that combine pre-trained parametric memory with non-parametric vector index memory.',
    keyFinding: 'RAG models generate more specific, diverse, and factual language than parametric-only seq2seq baselines, setting new records on Open-Domain QA.',
    sampleQuery: 'How does RAG-Sequence differ from RAG-Token marginalization across retrieved passages?',
    likes: 2890,
  },
  {
    id: 'doc-showcase-4',
    title: 'Deep Residual Learning for Image Recognition',
    category: 'Vision & Multimodal',
    imageUrl: 'https://images.unsplash.com/photo-1585320806297-9794b3e4eeae?auto=format&fit=crop&w=1000&q=80',
    authors: 'Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun',
    publishedYear: 2015,
    relevanceScore: 0.91,
    pages: 12,
    chunkCount: 31,
    model: 'all-MiniLM-L6-v2',
    summary: 'Proposes a residual learning framework to ease the training of networks that are substantially deeper than those previously used.',
    keyFinding: 'Enabled training of networks up to 152 layers—8x deeper than VGG nets—winning 1st place on the ILSVRC 2015 classification task with 3.57% top-5 error.',
    sampleQuery: 'What happens mathematically when residual mapping F(x) equals zero in an identity block?',
    likes: 1980,
  },
  {
    id: 'doc-showcase-5',
    title: 'FlashAttention: Fast and Memory-Efficient Exact Attention',
    category: 'Optimization',
    imageUrl: 'https://images.unsplash.com/photo-1518709268805-4e9042af9f23?auto=format&fit=crop&w=1000&q=80',
    authors: 'Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré',
    publishedYear: 2022,
    relevanceScore: 0.95,
    pages: 14,
    chunkCount: 36,
    model: 'all-MiniLM-L6-v2',
    summary: 'Introduces an IO-aware exact attention algorithm that uses tiling to reduce memory reads/writes between GPU high bandwidth memory and on-chip SRAM.',
    keyFinding: 'Yields 2-4x wall-clock speedup on GPT-2 and enables training on sequences up to 64k tokens without approximation.',
    sampleQuery: 'What is the IO complexity of standard attention compared to FlashAttention with SRAM size M?',
    likes: 3410,
  },
  {
    id: 'doc-showcase-6',
    title: 'Highly Accurate Protein Structure Prediction with AlphaFold',
    category: 'Bioinformatics',
    imageUrl: 'https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?auto=format&fit=crop&w=1000&q=80',
    authors: 'John Jumper, Richard Evans, Alexander Pritzel, Tim Green, et al.',
    publishedYear: 2021,
    relevanceScore: 0.97,
    pages: 22,
    chunkCount: 65,
    model: 'Meta FAISS + RAG',
    summary: 'Presents AlphaFold 2, a computational method that can regularly predict protein 3D structures with atomic accuracy even where no homologous structure is known.',
    keyFinding: 'Demonstrated median backbone accuracy of 0.96Å r.m.s.d.95 in CASP14, competitive with experimental x-ray crystallography structures.',
    sampleQuery: 'How does the invariant point attention (IPA) module enforce SE(3) equivariance in the 3D structure module?',
    likes: 4200,
  },
]

export const FOOTER_RESEARCH_SECTIONS = [
  {
    title: 'Research Tools',
    links: [
      { name: 'Semantic Search Engine', href: '#search' },
      { name: 'Research Q&A Assistant', href: '#chat' },
      { name: 'PDF Ingestion & OCR', href: '#documents' },
      { name: 'Curated Collections', href: '#collections' },
      { name: 'Citation Evidence Drawer', href: '#explore' },
      { name: 'Interactive Studio Modal', href: '#explore' },
    ],
  },
  {
    title: 'Vector Models',
    links: [
      { name: 'all-MiniLM-L6-v2 (384-d)', href: '#explore' },
      { name: 'Meta FAISS IndexFlatIP', href: '#explore' },
      { name: 'BGE-Large Bi-Encoder', href: '#explore' },
      { name: 'Specter-2 Academic Embeddings', href: '#explore' },
      { name: 'Dense Passage Retrieval (DPR)', href: '#explore' },
      { name: 'Cosine & Dot Product Metrics', href: '#explore' },
    ],
  },
  {
    title: 'RAG Pipeline',
    links: [
      { name: 'Multi-turn Dialog History', href: '#chat' },
      { name: 'Grounded Passage Validation', href: '#chat' },
      { name: 'Numbered [Page X] Citations', href: '#chat' },
      { name: 'Dynamic Token Overlap Check', href: '#explore' },
      { name: 'Deterministic Abstract Filtering', href: '#search' },
      { name: 'Anti-Hallucination Fencing', href: '#explore' },
    ],
  },
  {
    title: 'Literature Corpus',
    links: [
      { name: 'Attention Is All You Need', href: '#explore' },
      { name: 'BERT Pre-training Deep Bi-Dir', href: '#explore' },
      { name: 'Retrieval-Augmented Generation', href: '#explore' },
      { name: 'Deep Residual Learning (ResNet)', href: '#explore' },
      { name: 'FlashAttention IO-Aware', href: '#explore' },
      { name: 'AlphaFold 2 Protein Structure', href: '#explore' },
    ],
  },
  {
    title: 'Developers & MCP',
    links: [
      { name: 'FastAPI OpenAPI v1 Schema', href: '/docs' },
      { name: 'Anthropic Claude Desktop MCP', href: '#mcp' },
      { name: 'PostgreSQL Relational Storage', href: '#mcp' },
      { name: 'FAISS Heartbeat Advisory Lease', href: '#mcp' },
      { name: 'Single-Worker Mutex Locking', href: '#mcp' },
      { name: 'Cluster Readiness Health Check', href: '/health/ready' },
    ],
  },
  {
    title: 'About ResearchLens',
    links: [
      { name: 'Mission & Scientific Rigor', href: '#explore' },
      { name: 'Pre-Publication Review Audit', href: '#explore' },
      { name: 'Zero-Hallucination Benchmark', href: '#explore' },
      { name: 'Documentation & Architecture', href: '#explore' },
      { name: 'Privacy & Terms of Service', href: '#explore' },
      { name: 'Open-Source Academic License', href: '#explore' },
    ],
  },
]
