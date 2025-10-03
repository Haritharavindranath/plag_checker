# Workaround for Streamlit + torch file watcher error
import os
os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import streamlit as st
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional
import requests
from dataclasses import dataclass
import ast
import hashlib
import re
from collections import defaultdict
import json
import google.generativeai as genai

# LangChain imports
from langchain_community.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class PlagiarismResult:
    original_code: str
    matched_code: str
    similarity_score: float
    ast_similarity: float
    structural_similarity: float
    source_url: str
    repository_name: str
    file_path: str
    match_type: str  # 'exact', 'structural', 'semantic'
    confidence_level: str  # 'high', 'medium', 'low'

class ASTAnalyzer:
    """Advanced AST-based code structure analyzer"""
    
    @staticmethod
    def extract_ast_features(code: str, language: str = 'python') -> Dict:
        """Extract structural features from code using AST"""
        try:
            if language == 'python':
                tree = ast.parse(code)
                return ASTAnalyzer._analyze_python_ast(tree)
            else:
                # For other languages, use regex-based pattern extraction
                return ASTAnalyzer._analyze_generic_structure(code)
        except:
            return {}
    
    @staticmethod
    def _analyze_python_ast(tree: ast.AST) -> Dict:
        """Deep Python AST analysis"""
        features = {
            'functions': [],
            'classes': [],
            'imports': [],
            'control_flow': [],
            'complexity_score': 0,
            'structure_hash': ''
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                    'body_structure': ASTAnalyzer._get_body_structure(node.body)
                }
                features['functions'].append(func_info)
                features['complexity_score'] += len(node.body)
            
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'bases': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                    'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                }
                features['classes'].append(class_info)
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    features['imports'].extend([alias.name for alias in node.names])
                else:
                    features['imports'].append(node.module)
            
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                features['control_flow'].append(type(node).__name__)
        
        # Create structural hash
        structure_str = json.dumps(features, sort_keys=True)
        features['structure_hash'] = hashlib.md5(structure_str.encode()).hexdigest()
        
        return features
    
    @staticmethod
    def _get_body_structure(body: List) -> List[str]:
        """Extract structure of function body"""
        return [type(stmt).__name__ for stmt in body]
    
    @staticmethod
    def _analyze_generic_structure(code: str) -> Dict:
        """Analyze code structure for non-Python languages"""
        features = {
            'functions': re.findall(r'function\s+(\w+)|def\s+(\w+)|\w+\s+\w+\s*\([^)]*\)\s*{', code),
            'classes': re.findall(r'class\s+(\w+)', code),
            'imports': re.findall(r'import\s+[\w.]+|from\s+[\w.]+\s+import|#include\s*<[\w.]+>', code),
            'control_flow': len(re.findall(r'\b(if|for|while|switch|try|catch)\b', code)),
            'complexity_score': len(code.split('\n'))
        }
        return features
    
    @staticmethod
    def calculate_ast_similarity(features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two AST feature sets"""
        if not features1 or not features2:
            return 0.0
        
        score = 0.0
        weights = {
            'structure_hash': 0.3,
            'functions': 0.25,
            'classes': 0.20,
            'imports': 0.15,
            'control_flow': 0.10
        }
        
        # Exact structure match
        if features1.get('structure_hash') == features2.get('structure_hash'):
            score += weights['structure_hash']
        
        # Function similarity
        func_sim = ASTAnalyzer._calculate_list_similarity(
            [f['name'] for f in features1.get('functions', [])],
            [f['name'] for f in features2.get('functions', [])]
        )
        score += func_sim * weights['functions']
        
        # Class similarity
        class_sim = ASTAnalyzer._calculate_list_similarity(
            [c['name'] for c in features1.get('classes', [])],
            [c['name'] for c in features2.get('classes', [])]
        )
        score += class_sim * weights['classes']
        
        # Import similarity
        import_sim = ASTAnalyzer._calculate_list_similarity(
            features1.get('imports', []),
            features2.get('imports', [])
        )
        score += import_sim * weights['imports']
        
        # Control flow similarity
        cf1 = features1.get('control_flow', [])
        cf2 = features2.get('control_flow', [])
        if cf1 and cf2:
            cf_sim = 1.0 - abs(len(cf1) - len(cf2)) / max(len(cf1), len(cf2))
            score += cf_sim * weights['control_flow']
        
        return min(score, 1.0)
    
    @staticmethod
    def _calculate_list_similarity(list1: List, list2: List) -> float:
        """Calculate Jaccard similarity between two lists"""
        if not list1 or not list2:
            return 0.0
        set1, set2 = set(list1), set(list2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

class CodeNormalizer:
    """Normalize code to detect plagiarism with variable renaming"""
    
    @staticmethod
    def normalize_python(code: str) -> str:
        """Normalize Python code by removing comments and standardizing naming"""
        try:
            tree = ast.parse(code)
            
            # Remove docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    if (ast.get_docstring(node)):
                        node.body = node.body[1:] if len(node.body) > 1 else node.body
            
            # Variable name standardization would go here
            normalized = ast.unparse(tree)
            
            # Remove comments and extra whitespace
            lines = [line.split('#')[0].strip() for line in normalized.split('\n')]
            return '\n'.join(line for line in lines if line)
        except:
            return code
    
    @staticmethod
    def extract_code_patterns(code: str) -> List[str]:
        """Extract common code patterns for fingerprinting"""
        patterns = []
        
        # Extract all function/method calls
        patterns.extend(re.findall(r'\b\w+\s*\([^)]*\)', code))
        
        # Extract control structures
        patterns.extend(re.findall(r'\b(if|for|while|try)\b[^:]+:', code))
        
        # Extract assignment patterns
        patterns.extend(re.findall(r'\w+\s*=\s*[^;]+', code))
        
        return patterns

class GitHubSearchAgent:
    """Advanced GitHub search with intelligent query generation"""
    
    def __init__(self, github_token: str = None):
        self.github_token = github_token
        self.headers = {"Authorization": f"token {github_token}"} if github_token else {}
        self.base_url = "https://api.github.com/search/code"
    
    def generate_search_queries(self, code: str, ast_features: Dict) -> List[str]:
        """Generate multiple targeted search queries"""
        queries = []
        
        # Query 1: Function signatures
        functions = ast_features.get('functions', [])
        if functions:
            func_names = ' '.join([f['name'] for f in functions[:3]])
            queries.append(func_names)
        
        # Query 2: Class names
        classes = ast_features.get('classes', [])
        if classes:
            class_names = ' '.join([c['name'] for c in classes[:2]])
            queries.append(class_names)
        
        # Query 3: Unique string literals
        strings = re.findall(r'["\']([^"\']{10,50})["\']', code)
        if strings:
            queries.append(strings[0])
        
        # Query 4: Import patterns
        imports = ast_features.get('imports', [])
        if len(imports) > 2:
            queries.append(' '.join(imports[:3]))
        
        # Query 5: Code fingerprint (unique patterns)
        patterns = CodeNormalizer.extract_code_patterns(code)
        if patterns:
            queries.append(' '.join(patterns[:2]))
        
        return [q for q in queries if len(q) > 15]
    
    def search_code(self, queries: List[str], language: str = "python", max_results: int = 10) -> List[Dict]:
        """Multi-query GitHub search"""
        all_results = []
        seen_repos = set()
        
        for query in queries[:3]:  # Limit to top 3 queries
            params = {
                "q": f"{query} language:{language}",
                "per_page": max_results,
                "sort": "indexed"
            }
            
            try:
                response = requests.get(self.base_url, headers=self.headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    items = response.json().get("items", [])
                    
                    for item in items:
                        repo_name = item.get("repository", {}).get("full_name", "")
                        
                        # Avoid duplicates from same repo
                        if repo_name not in seen_repos:
                            seen_repos.add(repo_name)
                            all_results.append(item)
                
                elif response.status_code == 403:
                    st.warning("GitHub API rate limit reached. Consider adding a GitHub token.")
                    break
            
            except Exception as e:
                st.warning(f"GitHub search error: {e}")
                continue
        
        return all_results[:max_results]
    
    def fetch_file_content(self, url: str) -> Optional[str]:
        """Fetch actual file content from GitHub"""
        try:
            # Convert to raw URL
            raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            response = requests.get(raw_url, timeout=10)
            
            if response.status_code == 200:
                return response.text
        except:
            pass
        return None

class CodeSimilarityMLModel:
    """ML-based code similarity using TF-IDF and advanced embeddings"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=5000
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="microsoft/codebert-base",
            model_kwargs={'device': 'cpu'}
        )
    
    def calculate_similarity(self, code1: str, code2: str) -> Dict[str, float]:
        """Multi-metric similarity calculation"""
        similarities = {}
        
        # 1. TF-IDF similarity (character n-grams)
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([code1, code2])
            similarities['tfidf'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            similarities['tfidf'] = 0.0
        
        # 2. CodeBERT embeddings
        try:
            emb1 = self.embeddings.embed_query(code1[:512])  # Limit for performance
            emb2 = self.embeddings.embed_query(code2[:512])
            similarities['semantic'] = float(cosine_similarity([emb1], [emb2])[0][0])
        except:
            similarities['semantic'] = 0.0
        
        # 3. Normalized edit distance
        similarities['structural'] = self._normalized_levenshtein(
            CodeNormalizer.normalize_python(code1),
            CodeNormalizer.normalize_python(code2)
        )
        
        # Weighted average
        weights = {'tfidf': 0.3, 'semantic': 0.4, 'structural': 0.3}
        similarities['overall'] = sum(similarities[k] * weights[k] for k in weights)
        
        return similarities
    
    @staticmethod
    def _normalized_levenshtein(s1: str, s2: str) -> float:
        """Normalized Levenshtein distance"""
        if not s1 or not s2:
            return 0.0
        
        len1, len2 = len(s1), len(s2)
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1
        
        current_row = range(len1 + 1)
        for i in range(1, len2 + 1):
            previous_row, current_row = current_row, [i] + [0] * len1
            for j in range(1, len1 + 1):
                add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                if s1[j - 1] != s2[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)
        
        distance = current_row[len1]
        return 1.0 - (distance / max(len1, len2))

class PlagiarismDetectionPipeline:
    """Multi-agent LangChain pipeline for plagiarism detection"""
    
    def __init__(self, gemini_api_key: str, github_token: str = None):
        # Initialize Gemini directly (no langchain wrapper needed)
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-pro')
        
        # Initialize components
        self.github_agent = GitHubSearchAgent(github_token)
        self.ast_analyzer = ASTAnalyzer()
        self.ml_model = CodeSimilarityMLModel()
        
        # Initialize memory for context
        self.conversation_history = []
    
    def _call_llm(self, prompt: str) -> str:
        """Helper method to call Gemini LLM"""
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _setup_agents(self):
        """Setup agent prompts (using native Gemini instead of LangChain wrappers)"""
        
        # We'll use direct prompting instead of LangChain agents
        # This avoids the langchain_google_genai dependency
        pass
    
    def _analyze_code_with_llm(self, code: str, features: Dict) -> str:
        """Agent 1: Code Analysis"""
        prompt = f"""You are an expert code analyst. Analyze this code and its structural features.

Code Sample:
{code[:1000]}

AST Features:
{json.dumps(features, indent=2)}

Provide:
1. Code complexity assessment
2. Key identifying features
3. Potential search strategies
4. Risk factors for plagiarism

Be concise and technical."""
        
        return self._call_llm(prompt)
    
    def _evaluate_match_with_llm(self, original: str, matched: str, similarity_scores: Dict) -> str:
        """Agent 2: Match Evaluation"""
        prompt = f"""You are a plagiarism detection expert. Evaluate if this is genuine plagiarism.

Original Code:
{original[:800]}

Potentially Matched Code:
{matched[:800]}

Similarity Metrics:
{json.dumps(similarity_scores, indent=2)}

Determine:
1. Is this likely plagiarism? (YES/NO/UNCERTAIN)
2. Match type: (EXACT_COPY/MODIFIED_COPY/STRUCTURAL_CLONE/FALSE_POSITIVE)
3. Confidence level: (HIGH/MEDIUM/LOW)
4. Key evidence

Be strict and evidence-based."""
        
        return self._call_llm(prompt)
    
    def _generate_report_with_llm(self, findings: str, statistics: Dict) -> str:
        """Agent 3: Report Generation"""
        prompt = f"""You are a technical report writer. Create a comprehensive plagiarism detection report.

Findings:
{findings}

Statistics:
{json.dumps(statistics, indent=2)}

Create a professional report with:
1. Executive Summary
2. Severity Assessment (CRITICAL/HIGH/MEDIUM/LOW/NONE)
3. Detailed Findings (numbered list with evidence)
4. Repository Links and Sources
5. Recommendations
6. Confidence Assessment

Use clear formatting with headers and bullet points."""
        
        return self._call_llm(prompt)
    
    def analyze_repository(self, repo_url: str, file_extensions: List[str], threshold: float = 0.75, 
                          progress_callback=None) -> Tuple[List[PlagiarismResult], str]:
        """Main analysis pipeline"""
        
        # Step 1: Load repository
        if progress_callback:
            progress_callback("Loading repository...")
        
        documents = self._load_repository(repo_url, file_extensions)
        
        if not documents:
            return [], "No code files found in repository."
        
        # Step 2: Analyze each file
        if progress_callback:
            progress_callback(f"Analyzing {len(documents)} files...")
        
        all_results = []
        
        for i, doc in enumerate(documents[:20]):  # Limit to 20 files for performance
            if progress_callback:
                progress_callback(f"Processing file {i+1}/{min(len(documents), 20)}: {doc.metadata.get('source', 'Unknown')}")
            
            code = doc.page_content
            
            # Skip small files
            if len(code) < 200:
                continue
            
            # Extract AST features
            ast_features = self.ast_analyzer.extract_ast_features(code)
            
            # Agent 1: Analyze code
            analysis = self._analyze_code_with_llm(code, ast_features)
            
            # Generate search queries
            search_queries = self.github_agent.generate_search_queries(code, ast_features)
            
            if not search_queries:
                continue
            
            # Search GitHub
            if progress_callback:
                progress_callback(f"Searching GitHub for matches...")
            
            github_results = self.github_agent.search_code(search_queries)
            
            # Analyze each match
            for gh_result in github_results:
                matched_url = gh_result.get("html_url", "")
                repo_name = gh_result.get("repository", {}).get("full_name", "Unknown")
                
                # Skip if same repository
                if repo_url.lower() in matched_url.lower():
                    continue
                
                # Fetch content
                matched_content = self.github_agent.fetch_file_content(matched_url)
                
                if not matched_content or len(matched_content) < 200:
                    continue
                
                # Calculate similarities
                ml_similarities = self.ml_model.calculate_similarity(code, matched_content)
                
                # AST comparison
                matched_ast = self.ast_analyzer.extract_ast_features(matched_content)
                ast_sim = self.ast_analyzer.calculate_ast_similarity(ast_features, matched_ast)
                
                overall_score = (ml_similarities['overall'] + ast_sim) / 2
                
                # Only process high-similarity matches
                if overall_score < threshold:
                    continue
                
                # Agent 2: Evaluate match
                evaluation = self._evaluate_match_with_llm(
                    original=code[:800],
                    matched=matched_content[:800],
                    similarity_scores={
                        'ml_overall': ml_similarities['overall'],
                        'ast_similarity': ast_sim,
                        'tfidf': ml_similarities['tfidf'],
                        'semantic': ml_similarities['semantic']
                    }
                )
                
                # Parse evaluation
                is_plagiarism = "YES" in evaluation.upper() or "LIKELY" in evaluation.upper()
                
                if is_plagiarism:
                    match_type = "exact" if overall_score > 0.95 else "structural" if ast_sim > 0.8 else "semantic"
                    confidence = "high" if overall_score > 0.85 else "medium" if overall_score > 0.75 else "low"
                    
                    result = PlagiarismResult(
                        original_code=code[:1000],
                        matched_code=matched_content[:1000],
                        similarity_score=overall_score,
                        ast_similarity=ast_sim,
                        structural_similarity=ml_similarities['structural'],
                        source_url=matched_url,
                        repository_name=repo_name,
                        file_path=doc.metadata.get("source", "Unknown"),
                        match_type=match_type,
                        confidence_level=confidence
                    )
                    
                    all_results.append(result)
        
        # Step 3: Generate report
        if progress_callback:
            progress_callback("Generating final report...")
        
        report = self._generate_comprehensive_report(all_results)
        
        return all_results, report
    
    def _load_repository(self, repo_url: str, file_extensions: List[str]) -> List[Document]:
        """Load repository with cleanup"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            loader = GitLoader(
                clone_url=repo_url,
                repo_path=temp_dir,
                file_filter=lambda path: any(path.endswith(ext) for ext in file_extensions)
            )
            return loader.load()
        finally:
            if os.path.exists(temp_dir):
                try:
                    def handle_remove_readonly(func, path, exc):
                        import stat
                        if not os.access(path, os.W_OK):
                            os.chmod(path, stat.S_IWUSR)
                            func(path)
                    shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
                except:
                    pass
    
    def _generate_comprehensive_report(self, results: List[PlagiarismResult]) -> str:
        """Generate final report using Agent 3"""
        if not results:
            return "✅ **No Plagiarism Detected**\n\nThe repository appears to be original work."
        
        # Compile statistics
        stats = {
            'total_matches': len(results),
            'high_confidence': len([r for r in results if r.confidence_level == 'high']),
            'repositories_found': len(set(r.repository_name for r in results)),
            'avg_similarity': sum(r.similarity_score for r in results) / len(results),
            'match_types': {
                'exact': len([r for r in results if r.match_type == 'exact']),
                'structural': len([r for r in results if r.match_type == 'structural']),
                'semantic': len([r for r in results if r.match_type == 'semantic'])
            }
        }
        
        # Prepare findings
        findings_text = []
        for i, result in enumerate(results[:10], 1):
            findings_text.append(f"""
Finding #{i}:
- File: {result.file_path}
- Match Type: {result.match_type.upper()}
- Confidence: {result.confidence_level.upper()}
- Similarity: {result.similarity_score:.1%} (AST: {result.ast_similarity:.1%})
- Source Repository: {result.repository_name}
- URL: {result.source_url}
""")
        
        report = self._generate_report_with_llm(
            findings='\n'.join(findings_text),
            statistics=stats
        )
        
        return report

def main():
    st.set_page_config(
        page_title="Advanced Code Plagiarism Detector",
        page_icon="🔬",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Advanced Code Plagiarism Detection System</h1>
        <p>Multi-Agent AI Pipeline with AST Analysis, ML Models & Semantic Search</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("### 🔑 API Keys")
        gemini_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Get free key at https://makersuite.google.com/app/apikey"
        )
        
        if st.button("🔗 Get Free Gemini Key"):
            st.markdown("[Get API Key](https://makersuite.google.com/app/apikey)")
        
        github_token = st.text_input(
            "GitHub Token (Optional)",
            type="password",
            help="Increases rate limits (60 → 5000 req/hour)"
        )
        
        st.divider()
        
        st.markdown("### 🎯 Detection Settings")
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.65,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help="Higher = stricter detection"
        )
        
        languages = st.multiselect(
            "File Extensions",
            [".py", ".js", ".java", ".cpp", ".c", ".ts", ".go", ".rb", ".php"],
            default=[".py"],
            help="Select programming languages to analyze"
        )
        
        st.divider()
        
        st.markdown("### ✨ Features")
        st.caption("✅ AST-based structural analysis")
        st.caption("✅ ML similarity models (CodeBERT)")
        st.caption("✅ TF-IDF character n-grams")
        st.caption("✅ Multi-agent LangChain pipeline")
        st.caption("✅ Intelligent GitHub search")
        st.caption("✅ Code normalization")
        
        st.divider()
        
        st.caption("🆓 100% Free Technologies")
        st.caption("• Google Gemini API")
        st.caption("• HuggingFace Models")
        st.caption("• GitHub API")
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        repo_url = st.text_input(
            "🔗 GitHub Repository URL",
            placeholder="https://github.com/username/repository",
            help="Enter the full repository URL to analyze"
        )
    
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    
    # Info cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><b>🧬 AST Analysis</b><br>Structural detection</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><b>🤖 ML Models</b><br>Semantic similarity</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><b>🔍 Smart Search</b><br>Multi-query GitHub</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><b>👥 Multi-Agent</b><br>LangChain pipeline</div>', unsafe_allow_html=True)
    
    if analyze_btn:
        if not repo_url:
            st.error("❌ Please enter a GitHub repository URL")
            return
        
        if not gemini_key:
            st.error("❌ Please provide Google Gemini API key")
            st.info("Get your FREE API key: https://makersuite.google.com/app/apikey")
            return
        
        if not languages:
            st.error("❌ Please select at least one file extension")
            return
        
        # Progress tracking
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            # Initialize pipeline
            with st.spinner("🔧 Initializing AI agents and ML models..."):
                pipeline = PlagiarismDetectionPipeline(gemini_key, github_token)
            
            status_placeholder.success("✅ Pipeline initialized with 3 specialized agents")
            
            # Progress callback
            def update_progress(message):
                progress_placeholder.info(f"🔄 {message}")
            
            # Run analysis
            results, report = pipeline.analyze_repository(
                repo_url=repo_url,
                file_extensions=languages,
                threshold=threshold,
                progress_callback=update_progress
            )
            
            progress_placeholder.empty()
            
            # Display results
            st.divider()
            
            if results:
                # Statistics
                st.header("📊 Detection Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Matches", len(results))
                
                with col2:
                    high_conf = len([r for r in results if r.confidence_level == 'high'])
                    st.metric("High Confidence", high_conf, delta="Critical" if high_conf > 0 else None)
                
                with col3:
                    unique_repos = len(set(r.repository_name for r in results))
                    st.metric("Source Repositories", unique_repos)
                
                with col4:
                    avg_sim = sum(r.similarity_score for r in results) / len(results)
                    st.metric("Avg Similarity", f"{avg_sim:.1%}")
                
                # Severity indicator
                if high_conf > 5:
                    st.error("🚨 **CRITICAL**: Multiple high-confidence plagiarism instances detected!")
                elif high_conf > 2:
                    st.warning("⚠️ **HIGH SEVERITY**: Significant plagiarism detected")
                elif len(results) > 5:
                    st.warning("⚠️ **MEDIUM SEVERITY**: Several potential matches found")
                else:
                    st.info("ℹ️ **LOW SEVERITY**: Minor similarities detected")
                
                st.divider()
                
                # AI-Generated Report
                st.header("📝 AI Analysis Report")
                st.markdown(report)
                
                st.divider()
                
                # Detailed findings
                st.header("🔍 Detailed Findings")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    confidence_filter = st.multiselect(
                        "Filter by Confidence",
                        ["high", "medium", "low"],
                        default=["high", "medium", "low"]
                    )
                
                with col2:
                    match_type_filter = st.multiselect(
                        "Filter by Match Type",
                        ["exact", "structural", "semantic"],
                        default=["exact", "structural", "semantic"]
                    )
                
                # Filter results
                filtered_results = [
                    r for r in results 
                    if r.confidence_level in confidence_filter 
                    and r.match_type in match_type_filter
                ]
                
                # Display each finding
                for i, result in enumerate(filtered_results, 1):
                    # Color code by confidence
                    if result.confidence_level == 'high':
                        color = "🔴"
                        badge_color = "red"
                    elif result.confidence_level == 'medium':
                        color = "🟡"
                        badge_color = "orange"
                    else:
                        color = "🟢"
                        badge_color = "green"
                    
                    with st.expander(
                        f"{color} Finding #{i} - {result.match_type.upper()} - "
                        f"{result.similarity_score:.1%} similarity - "
                        f"**{result.repository_name}**",
                        expanded=(i <= 3 and result.confidence_level == 'high')
                    ):
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Overall Similarity", f"{result.similarity_score:.1%}")
                        
                        with col2:
                            st.metric("AST Similarity", f"{result.ast_similarity:.1%}")
                        
                        with col3:
                            st.metric("Structural Match", f"{result.structural_similarity:.1%}")
                        
                        with col4:
                            st.markdown(f"**Confidence:** <span style='color:{badge_color}'>{result.confidence_level.upper()}</span>", unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # Source information
                        st.markdown("### 📍 Source Information")
                        st.markdown(f"**Repository:** [{result.repository_name}](https://github.com/{result.repository_name})")
                        st.markdown(f"**File:** `{result.file_path}`")
                        st.markdown(f"**Match URL:** [{result.source_url}]({result.source_url})")
                        st.markdown(f"**Match Type:** `{result.match_type.upper()}`")
                        
                        st.divider()
                        
                        # Code comparison
                        st.markdown("### 🔬 Code Comparison")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Your Code:**")
                            st.code(result.original_code, language="python")
                        
                        with col2:
                            st.markdown(f"**Matched Code from [{result.repository_name}]({result.source_url}):**")
                            st.code(result.matched_code, language="python")
                        
                        # Action buttons
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.link_button(
                                "🔗 View Full Source",
                                result.source_url,
                                use_container_width=True
                            )
                        
                        with col2:
                            st.link_button(
                                "📂 View Repository",
                                f"https://github.com/{result.repository_name}",
                                use_container_width=True
                            )
                
                # Export option
                st.divider()
                
                if st.button("📥 Export Report as JSON"):
                    export_data = {
                        'repository': repo_url,
                        'analysis_date': st.session_state.get('analysis_date', 'N/A'),
                        'total_findings': len(results),
                        'threshold': threshold,
                        'findings': [
                            {
                                'file': r.file_path,
                                'similarity': r.similarity_score,
                                'ast_similarity': r.ast_similarity,
                                'match_type': r.match_type,
                                'confidence': r.confidence_level,
                                'source_repo': r.repository_name,
                                'source_url': r.source_url
                            }
                            for r in results
                        ]
                    }
                    
                    st.download_button(
                        "Download JSON Report",
                        data=json.dumps(export_data, indent=2),
                        file_name="plagiarism_report.json",
                        mime="application/json"
                    )
            
            else:
                # No plagiarism detected
                st.success("🎉 **No Plagiarism Detected!**")
                st.balloons()
                
                st.markdown("""
                ### ✅ Analysis Complete
                
                The repository has been thoroughly analyzed using:
                - 🧬 **AST structural analysis**
                - 🤖 **ML-based semantic similarity**
                - 🔍 **Multi-query GitHub search**
                - 👥 **Multi-agent verification**
                
                **No significant code plagiarism was found.**
                
                The code appears to be original work or uses properly attributed open-source components.
                """)
                
                st.info("""
                **What was checked:**
                - External repositories on GitHub
                - Structural code patterns (AST)
                - Semantic code similarity (CodeBERT)
                - Character-level n-gram analysis (TF-IDF)
                """)
        
        except Exception as e:
            st.error(f"❌ An error occurred during analysis")
            st.exception(e)
            
            st.markdown("""
            ### 🔧 Troubleshooting Tips:
            - Verify the repository URL is correct and public
            - Check your Gemini API key is valid
            - Ensure you have internet connectivity
            - Try with fewer file extensions
            - Check GitHub API rate limits (add token if needed)
            """)
    
    # Footer
    st.divider()
    
    with st.expander("ℹ️ How This System Works"):
        st.markdown("""
        ### 🔬 Advanced Multi-Layer Detection
        
        This system uses a sophisticated **4-agent LangChain pipeline** with multiple detection methods:
        
        #### 🤖 **Agent 1: Code Analysis Agent**
        - Extracts AST (Abstract Syntax Tree) features
        - Identifies key code patterns and structures
        - Generates intelligent search queries
        - Assesses code complexity
        
        #### 🔍 **Agent 2: GitHub Search Agent**
        - Multi-query search strategy
        - Searches by function signatures, class names, and patterns
        - Fetches and analyzes actual source code
        - Filters duplicate results
        
        #### ⚖️ **Agent 3: Match Evaluation Agent**
        - Combines multiple similarity metrics:
          - **TF-IDF**: Character n-gram analysis (3-5 grams)
          - **CodeBERT**: Deep semantic understanding
          - **AST Comparison**: Structural similarity
          - **Normalized Edit Distance**: Code normalization
        - LLM-powered validation to reduce false positives
        - Assigns confidence levels and match types
        
        #### 📝 **Agent 4: Report Generation Agent**
        - Synthesizes all findings
        - Generates comprehensive reports
        - Provides actionable recommendations
        - Risk assessment and severity scoring
        
        ### 🎯 Why This Approach is Superior:
        
        ✅ **AST Analysis** - Detects structural plagiarism (renamed variables, reordered code)  
        ✅ **ML Models** - Understands semantic similarity beyond exact matching  
        ✅ **Code Normalization** - Strips formatting to find hidden copies  
        ✅ **Multi-Agent Validation** - Reduces false positives with AI reasoning  
        ✅ **External-Only Search** - Focuses on real plagiarism from other repos  
        
        ### 🆓 Free & Open Source
        - Google Gemini API (60 req/min free tier)
        - HuggingFace models (runs locally)
        - GitHub API (60 req/hour without token, 5000 with)
        """)
    
    with st.expander("🚀 Performance Tips"):
        st.markdown("""
        ### Optimize Your Analysis:
        
        1. **Add GitHub Token** - Increases API limits from 60 to 5,000 requests/hour
        2. **Adjust Threshold** - Higher threshold (0.85+) = stricter, fewer false positives
        3. **Select Specific Languages** - Analyze only relevant file types
        4. **Smaller Repos First** - Test with smaller repositories initially
        
        ### Current Limitations:
        - Analyzes up to 20 files per run (configurable in code)
        - GitHub API rate limits apply
        - CodeBERT embeddings limited to 512 tokens
        
        ### Best Practices:
        - Review high-confidence matches manually
        - Check if matched code is properly attributed/licensed
        - Consider common libraries and frameworks (may show similarities)
        """)

if __name__ == "__main__":
    main()