import os
import torch
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI

# Local model imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import logging

logger = logging.getLogger("")

class KCCService:
    """
    KCC (Krishi Crop Companion) Service - Robust Smart Integration System
    Handles JSON failures gracefully - follows robust_smart_integration.py
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, '..', '..', 'ml-models', 'KCC', 'varieties_weights')
        self.base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Will be set during initialization
        self.search_llm = None
        self.processing_llm = None
        self.extraction_chain = None
        self.synthesis_chain = None
        self.fallback_chain = None
        self.search_prompt = None
    
    async def load_model(self, google_api_key: str = None, project_id: str = "krishi-saarthi-main", location: str = "us-central1"):
        """
        Load the KCC model with Gemini integration
        """
        if self.is_loaded:
            logger.info("KCC model already loaded, skipping initialization")
            return
        
        try:
            import time
            total_start = time.time()
            
            logger.info("Loading KCC service...")
            logger.info(f"Base model: {self.base_model_id}")
            
            api_key = google_api_key or os.getenv("GEMINI_API_KEY")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                logger.info("Google API key configured")
            else:
                logger.warning("No Google API key found - will use fallback mode")
            
            service_account_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'krishi-saarthi-main-163bc6406584.json')
            if os.path.exists(service_account_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
                logger.info("Google Cloud service account configured")
            else:
                logger.warning("Service account file not found - Vertex AI will not be available")
            
            await self._setup_tinyllama()
            self._setup_gemini_models(project_id, location)
            self._setup_prompts()
            
            self.is_loaded = True
            total_time = time.time() - total_start
            
            logger.info(f"KCC service loaded successfully in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load KCC service: {e}")
            raise
    
    async def _setup_tinyllama(self):
        """Setup TinyLlama prompt-tuned model"""
        logger.info("Loading TinyLlama model...")
        
        try:
            model_path = os.path.abspath(self.model_path)
            logger.info(f"Model path: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory not found: {model_path}")
            
            config_file = os.path.join(model_path, "adapter_config.json")
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"adapter_config.json not found at: {config_file}")
            
            peft_config = PeftConfig.from_pretrained(model_path)
            
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.model = PeftModel.from_pretrained(base_model, model_path)
            
            self.tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.eval()
            logger.info("TinyLlama loaded successfully")
            
            if torch.cuda.is_available():
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Using CPU")
            
        except Exception as e:
            logger.error(f"Error loading TinyLlama: {e}")
            raise
    
    def _setup_gemini_models(self, project_id: str, location: str):
        """Setup Gemini models for search and processing"""
        logger.info("Setting up Gemini models...")
        
        try:
            # Gemini with Google Search (for verification)
            self.search_llm = ChatVertexAI(
                model="gemini-2.5-flash",
                temperature=0,
                max_tokens=None,
                max_retries=3,
                project=project_id,
                location=location,
            ).bind_tools([{"google_search": {}}])
            
            logger.info("Vertex AI Gemini with search ready")
        except Exception as e:
            logger.warning(f"Vertex AI failed ({e}), using regular Gemini...")
            self.search_llm = None
        
        self.processing_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            convert_system_message_to_human=True
        )
        
        logger.info("Gemini models ready")
    
    def _setup_prompts(self):
        """Setup simplified prompts that avoid JSON parsing"""
        logger.info("Setting up robust prompts...")
        
        # Simple extraction prompt (no JSON required)
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting agricultural information from text.

Analyze the response and extract:
1. Soybean variety names mentioned
2. Maturity period information (days)
3. Yield information (quintals per hectare)
4. Physical characteristics

Format your response as:
VARIETIES: [list variety names separated by commas]
MATURITY: [maturity information found]
YIELD: [yield information found]
CHARACTERISTICS: [physical traits mentioned]"""),
            ("human", "Extract information from this agricultural response:\n\n{tinyllama_output}")
        ])
        
        # Simple search prompt
        self.search_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an agricultural researcher with access to search capabilities.

Search for reliable information about the soybean variety mentioned in the query. Focus on:
- Official government agricultural databases
- Research institution publications
- Verified seed company information

Provide accurate information about yield, maturity, and characteristics."""),
            ("human", "Search for reliable information about: {search_query}")
        ])
        
        # Simple synthesis prompt
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert agricultural advisor.

Based on the original domain knowledge and verified search results, provide helpful agricultural advice.

Give practical information about:
- Variety characteristics
- Maturity periods
- Yield expectations
- Growing recommendations

Be helpful and accurate. Don't mention verification processes."""),
            ("human", """Original Query: {original_query}

Domain Response: {tinyllama_output}

Search Results: {search_results}

Provide helpful agricultural advice.  

Give the final output in the following form: 
- Heading 1: One liner summary 
- Heading 2: One liner summary.

At most there should be four points, make sure to cover all the points but in a compact format.
""")
        ])
        
        # Fallback prompt
        self.fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an honest agricultural advisor.

The specific variety information couldn't be verified. Provide general guidance about:
1. Where to find reliable soybean variety information
2. General soybean variety characteristics
3. How to choose appropriate varieties

Be helpful and direct farmers to reliable sources."""),
            ("human", """The user asked: {original_query}

Provide helpful guidance about soybean varieties and where to find reliable information.""")
        ])
        
        self.extraction_chain = self.extraction_prompt | self.processing_llm
        self.synthesis_chain = self.synthesis_prompt | self.processing_llm
        self.fallback_chain = self.fallback_prompt | self.processing_llm
        
        logger.info("All robust prompts ready")
    
    def get_tinyllama_response(self, query: str) -> str:
        """Stage 1: Get response from TinyLlama"""
        logger.info("====== Stage 1: TinyLlama Response ======")
        
        # TinyLlama chat format
        prompt = f"<|user|>\n{query}\n<|assistant|>\n"
        
        # Tokenize with attention mask
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def extract_claims_simple(self, tinyllama_output: str) -> Dict:
        """Stage 2: Extract claims using simple text parsing"""
        logger.info("====== Stage 2: Claims Extraction ======")
        
        try:
            result = self.extraction_chain.invoke({"tinyllama_output": tinyllama_output})
            extraction_text = result.content
            
            varieties = []
            maturity = ""
            yield_info = ""
            characteristics = ""
            
            for line in extraction_text.split('\n'):
                if line.startswith('VARIETIES:'):
                    varieties_text = line.replace('VARIETIES:', '').strip()
                    varieties = [v.strip() for v in varieties_text.split(',') if v.strip()]
                elif line.startswith('MATURITY:'):
                    maturity = line.replace('MATURITY:', '').strip()
                elif line.startswith('YIELD:'):
                    yield_info = line.replace('YIELD:', '').strip()
                elif line.startswith('CHARACTERISTICS:'):
                    characteristics = line.replace('CHARACTERISTICS:', '').strip()
            
            return {
                "varieties": varieties,
                "maturity": maturity,
                "yield": yield_info,
                "characteristics": characteristics,
                "has_varieties": len(varieties) > 0
            }
            
        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
            # Fallback: try to extract variety names from the original text
            varieties = self._extract_varieties_fallback(tinyllama_output)
            return {
                "varieties": varieties,
                "maturity": "",
                "yield": "",
                "characteristics": "",
                "has_varieties": len(varieties) > 0
            }
    
    def _extract_varieties_fallback(self, text: str) -> List[str]:
        """Fallback variety extraction using regex patterns"""
        # Common soybean variety patterns
        patterns = [
            r'JS[-\s]?\d+',  # JS-335, JS 335, etc.
            r'NRC[-\s]?\d+',  # NRC-150, NRC 150, etc.
            r'Pusa[-\s]?\d+',  # Pusa-24, etc.
            r'RVSM[-\s]?\d+',  # RVSM-201, etc.
            r'[A-Z]{2,4}[-\s]?\d+',  # General pattern
        ]
        
        varieties = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            varieties.extend(matches)
        
        varieties = list(set([v.strip() for v in varieties if v.strip()]))
        return varieties[:3]  # Limit to 3 varieties
    
    def perform_search_simple(self, search_query: str) -> str:
        """Perform simple search"""
        logger.info(f"====== Stage 3: Google Search (Via Gemini Vertex AI) ======")
        logger.info(f"Searching for: {search_query}")
        
        try:
            if self.search_llm:
                result = self.search_llm.invoke(f"Search for reliable agricultural information about: {search_query}")
                return result.content
            else:
                search_chain = self.search_prompt | self.processing_llm
                result = search_chain.invoke({"search_query": search_query})
                return result.content
                
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            return f"Search failed for {search_query}. Using general knowledge."
    
    def synthesize_response_simple(self, original_query: str, tinyllama_output: str, search_results: str) -> str:
        """Stage 4: Synthesize final response"""
        logger.info("====== Stage 4: Response Synthesis ======")
        
        try:
            result = self.synthesis_chain.invoke({
                "original_query": original_query,
                "tinyllama_output": tinyllama_output,
                "search_results": search_results
            })
            return result.content
            
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")
            return self.execute_fallback_simple(original_query)
    
    def execute_fallback_simple(self, original_query: str) -> str:
        """Execute simple fallback"""
        logger.info("====== FALLBACK: General Guidance ======")
        
        try:
            result = self.fallback_chain.invoke({"original_query": original_query})
            return result.content
        except Exception as e:
            logger.warning(f"Even fallback failed: {e}")
            return f"""I apologize, but I'm having technical difficulties verifying the specific soybean variety information you requested.

For reliable information about soybean varieties, I recommend:

1. **Contact your local Agricultural Extension Office**
2. **Visit ICAR (Indian Council of Agricultural Research) websites**
3. **Consult certified seed dealers in your area**
4. **Check with your State Agricultural University**

These sources will provide accurate, region-specific information about soybean varieties, their yields, and maturity periods.

Your query was: {original_query}"""
    
    async def generate_response(self, query: str) -> str:
        """Main processing pipeline - Robust version (follows robust_smart_integration.py)"""
        if not self.is_loaded:
            logger.warning("Model not loaded, cannot process query without initialization")
            raise RuntimeError("KCC service not properly initialized with API keys")
        
        try:
            import time
            inference_start = time.time()
            
            logger.info("====== KCC Processing Pipeline ======")
            logger.info(f"Processing query: {query}")
            
            tinyllama_output = self.get_tinyllama_response(query)
            logger.info(f"====== TinyLlama Output ======")
            logger.info(f"Response: {tinyllama_output}")
            
            extracted_claims = self.extract_claims_simple(tinyllama_output)
            logger.info(f"====== Extraction Output ======")
            logger.info(f"Varieties found: {extracted_claims.get('varieties', [])}")
            
            search_results = ""
            if extracted_claims.get("has_varieties", False):
                for variety in extracted_claims["varieties"][:2]:  # Limit searches
                    variety_search = self.perform_search_simple(f"{variety} soybean variety yield maturity")
                    search_results += f"\n--- {variety} ---\n{variety_search}\n"
            else:
                general_search = self.perform_search_simple(f"soybean varieties {query}")
                search_results = general_search
            
            if search_results.strip():
                final_response = self.synthesize_response_simple(query, tinyllama_output, search_results)
            else:
                final_response = self.execute_fallback_simple(query)
            
            total_time = time.time() - inference_start
            logger.info("====== Final Output ======")
            logger.info(f"Inference completed in {total_time:.3f}s")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in processing: {e}")
            return self.execute_fallback_simple(query)
    
    def get_status(self) -> dict:
        """
        Get service status
        """
        return {
            "service": "KCC",
            "loaded": self.is_loaded,
            "model_path": self.model_path,
            "base_model": self.base_model_id,
            "has_gemini": self.processing_llm is not None,
            "has_search": self.search_llm is not None
        }

# Global service instance
kcc_varieties_service = KCCService()