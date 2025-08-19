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

class KCCCulturalService:
    """
    KCC (Krishi Crop Companion) Cultural Practices Service - Robust Smart Integration System
    Handles agricultural cultural practices queries with Gemini integration
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, '..', '..', 'ml-models', 'KCC', 'cultural_practices_weights')
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
        Load the KCC Cultural Practices model with Gemini integration
        """
        if self.is_loaded:
            logger.info("KCC Cultural Practices model already loaded, skipping initialization")
            return
        
        try:
            import time
            total_start = time.time()
            
            logger.info("Loading KCC Cultural Practices service...")
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
            
            logger.info(f"KCC Cultural Practices service loaded successfully in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load KCC Cultural Practices service: {e}")
            raise
    
    async def _setup_tinyllama(self):
        """Setup TinyLlama prompt-tuned model for cultural practices"""
        logger.info("Loading TinyLlama Cultural Practices model...")
        
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
            logger.info("TinyLlama Cultural Practices loaded successfully")
            
            if torch.cuda.is_available():
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Using CPU")
            
        except Exception as e:
            logger.error(f"Error loading TinyLlama Cultural Practices: {e}")
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
        """Setup cultural practices specific prompts"""
        logger.info("Setting up cultural practices prompts...")
        
        # Cultural practices extraction prompt
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting comprehensive agricultural cultural practices information from text.

Analyze the response and extract:
1. Sowing/planting details: seed rates, spacing, depth, timing, methods
2. Weather considerations: monsoon timing, seasonal recommendations
3. Fertilizer applications: types (urea, NPK), rates, timing, methods
4. Regional recommendations: block-wise, district-wise advice
5. Crop management: germination timelines, post-planting care

Format your response as:
SOWING: [seed rates, spacing, depth, timing, methods found]
WEATHER: [weather considerations, seasonal timing mentioned]
FERTILIZATION: [fertilizer types, rates, application methods, timing mentioned]
REGIONAL: [location-specific recommendations found]
MANAGEMENT: [crop care, germination, post-planting advice mentioned]"""),
            ("human", "Extract comprehensive cultural practices information from this agricultural response:\n\n{tinyllama_output}")
        ])
        
        # Cultural practices search prompt
        self.search_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an agricultural researcher with access to search capabilities.

Search for reliable information about cultural practices mentioned in the query. Focus on:
- Seed rates and spacing recommendations for different crops
- Weather-based sowing timing and monsoon considerations
- Fertilizer application methods and rates (urea, NPK, organic)
- Regional farming practices for specific blocks/districts
- Crop management timelines and best practices
- Agricultural extension services recommendations

Provide accurate information about farming methods, timing, rates, and regional specificity."""),
            ("human", "Search for reliable cultural practices information about: {search_query}")
        ])
        
        # Cultural practices synthesis prompt
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert agricultural advisor specializing in cultural practices for Indian farming conditions.

Based on the original domain knowledge and verified search results, provide practical farming guidance.

Give detailed, actionable information about:
- Seed rates and sowing methods (per acre/hectare)
- Optimal spacing between rows and plants
- Sowing depth and timing considerations
- Weather-based recommendations and monsoon timing
- Fertilizer applications (types, rates, timing)
- Regional specific practices for different blocks/districts
- Post-sowing care and management practices

Be helpful, practical, and provide specific numbers/rates where possible. Don't mention verification processes."""),
            ("human", """Original Query: {original_query}

Domain Response: {tinyllama_output}

Search Results: {search_results}

Provide comprehensive agricultural cultural practices guidance.

Give the final output in the following form: 
- Heading 1: One liner summary with specific details
- Heading 2: One liner summary with specific recommendations
- Heading 3: One liner summary with timing/rates
- Heading 4: One liner summary with care instructions

Make sure to include specific numbers, rates, and actionable advice in a compact format.
""")
        ])
        
        # Cultural practices fallback prompt
        self.fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an honest agricultural advisor specializing in cultural practices.

The specific cultural practices information couldn't be verified. Provide general guidance about:
1. Where to find reliable seed rate and spacing information
2. How to access weather-based sowing recommendations
3. Sources for regional farming practices and extension services
4. General fertilizer application guidelines

Be helpful and direct farmers to reliable sources with actionable advice."""),
            ("human", """The user asked: {original_query}

Provide helpful guidance about cultural practices and where to find reliable farming information with specific focus on their query.""")
        ])
        
        self.extraction_chain = self.extraction_prompt | self.processing_llm
        self.synthesis_chain = self.synthesis_prompt | self.processing_llm
        self.fallback_chain = self.fallback_prompt | self.processing_llm
        
        logger.info("All cultural practices prompts ready")
    
    def get_tinyllama_response(self, query: str) -> str:
        """Stage 1: Get response from TinyLlama Cultural Practices model"""
        logger.info("====== Stage 1: TinyLlama Cultural Practices Response ======")
        
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
    
    def extract_practices_simple(self, tinyllama_output: str) -> Dict:
        """Stage 2: Extract cultural practices using simple text parsing"""
        logger.info("====== Stage 2: Cultural Practices Extraction ======")
        
        try:
            result = self.extraction_chain.invoke({"tinyllama_output": tinyllama_output})
            extraction_text = result.content
            
            sowing = ""
            weather = ""
            fertilization = ""
            regional = ""
            management = ""
            
            for line in extraction_text.split('\n'):
                if line.startswith('SOWING:'):
                    sowing = line.replace('SOWING:', '').strip()
                elif line.startswith('WEATHER:'):
                    weather = line.replace('WEATHER:', '').strip()
                elif line.startswith('FERTILIZATION:'):
                    fertilization = line.replace('FERTILIZATION:', '').strip()
                elif line.startswith('REGIONAL:'):
                    regional = line.replace('REGIONAL:', '').strip()
                elif line.startswith('MANAGEMENT:'):
                    management = line.replace('MANAGEMENT:', '').strip()
            
            return {
                "sowing": sowing,
                "weather": weather,
                "fertilization": fertilization,
                "regional": regional,
                "management": management,
                "has_practices": any([sowing, weather, fertilization, regional, management])
            }
            
        except Exception as e:
            logger.warning(f"Cultural practices extraction failed: {e}")
            # Fallback: try to extract key farming terms from the original text
            practices = self._extract_practices_fallback(tinyllama_output)
            return {
                "sowing": practices.get("sowing", ""),
                "weather": practices.get("weather", ""),
                "fertilization": practices.get("fertilization", ""),
                "regional": practices.get("regional", ""),
                "management": practices.get("management", ""),
                "has_practices": any(practices.values())
            }
    
    def _extract_practices_fallback(self, text: str) -> Dict:
        """Fallback practices extraction using keyword detection"""
        practices = {
            "sowing": "",
            "weather": "",
            "fertilization": "",
            "regional": "",
            "management": ""
        }
        
        sowing_keywords = ['sowing', 'planting', 'seed rate', 'spacing', 'depth', 'per acre', 'hectare']
        if any(keyword in text.lower() for keyword in sowing_keywords):
            practices["sowing"] = "Sowing information mentioned"
        
        weather_keywords = ['weather', 'monsoon', 'rain', 'temperature', 'forecast', 'seasonal']
        if any(keyword in text.lower() for keyword in weather_keywords):
            practices["weather"] = "Weather information mentioned"
        
        fertilization_keywords = ['fertilizer', 'urea', 'NPK', 'manure', 'compost', 'nutrient']
        if any(keyword in text.lower() for keyword in fertilization_keywords):
            practices["fertilization"] = "Fertilization information mentioned"
        
        regional_keywords = ['block', 'district', 'vidisha', 'guna', 'development block']
        if any(keyword in text.lower() for keyword in regional_keywords):
            practices["regional"] = "Regional information mentioned"
            
        management_keywords = ['germination', 'care', 'management', 'post-planting', 'crop management']
        if any(keyword in text.lower() for keyword in management_keywords):
            practices["management"] = "Management information mentioned"
        
        return practices
    
    def perform_search_simple(self, search_query: str) -> str:
        """Perform simple search for cultural practices"""
        logger.info(f"====== Stage 3: Google Search (Cultural Practices) ======")
        logger.info(f"Searching for: {search_query}")
        
        try:
            if self.search_llm:
                result = self.search_llm.invoke(f"Search for reliable agricultural cultural practices information about: {search_query}")
                return result.content
            else:
                search_chain = self.search_prompt | self.processing_llm
                result = search_chain.invoke({"search_query": search_query})
                return result.content
                
        except Exception as e:
            logger.warning(f"Cultural practices search failed: {e}")
            return f"Search failed for {search_query}. Using general knowledge."
    
    def synthesize_response_simple(self, original_query: str, tinyllama_output: str, search_results: str) -> str:
        """Stage 4: Synthesize final cultural practices response"""
        logger.info("====== Stage 4: Cultural Practices Response Synthesis ======")
        
        try:
            result = self.synthesis_chain.invoke({
                "original_query": original_query,
                "tinyllama_output": tinyllama_output,
                "search_results": search_results
            })
            return result.content
            
        except Exception as e:
            logger.warning(f"Cultural practices synthesis failed: {e}")
            return self.execute_fallback_simple(original_query)
    
    def execute_fallback_simple(self, original_query: str) -> str:
        """Execute cultural practices fallback"""
        logger.info("====== FALLBACK: General Cultural Practices Guidance ======")
        
        try:
            result = self.fallback_chain.invoke({"original_query": original_query})
            return result.content
        except Exception as e:
            logger.warning(f"Even cultural practices fallback failed: {e}")
            return f"""I apologize, but I'm having technical difficulties providing specific cultural practices information you requested.

For reliable information about agricultural cultural practices, I recommend:

1. **Contact your local Agricultural Extension Office**
2. **Visit ICAR (Indian Council of Agricultural Research) websites**
3. **Consult with State Agricultural Universities**
4. **Check with Krishi Vigyan Kendras (KVKs) in your area**

These sources will provide accurate, region-specific information about farming practices, timing, and techniques.

Your query was: {original_query}"""
    
    async def generate_response(self, query: str) -> str:
        """Main cultural practices processing pipeline"""
        if not self.is_loaded:
            logger.warning("Cultural practices model not loaded, cannot process query without initialization")
            raise RuntimeError("KCC Cultural Practices service not properly initialized")
        
        try:
            import time
            inference_start = time.time()
            
            logger.info("====== KCC Cultural Practices Processing Pipeline ======")
            logger.info(f"Processing cultural practices query: {query}")
            
            tinyllama_output = self.get_tinyllama_response(query)
            logger.info(f"====== TinyLlama Cultural Practices Output ======")
            logger.info(f"Response: {tinyllama_output}")
            
            extracted_practices = self.extract_practices_simple(tinyllama_output)
            logger.info(f"====== Cultural Practices Extraction Output ======")
            logger.info(f"Practices found: {extracted_practices.get('has_practices', False)}")
            
            search_results = ""
            if extracted_practices.get("has_practices", False):
                search_query = f"agricultural cultural practices {query}"
                search_results = self.perform_search_simple(search_query)
            else:
                general_search = self.perform_search_simple(f"farming practices {query}")
                search_results = general_search
            
            if search_results.strip():
                final_response = self.synthesize_response_simple(query, tinyllama_output, search_results)
            else:
                final_response = self.execute_fallback_simple(query)
            
            total_time = time.time() - inference_start
            logger.info("====== Cultural Practices Final Output ======")
            logger.info(f"Cultural practices inference completed in {total_time:.3f}s")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in cultural practices processing: {e}")
            return self.execute_fallback_simple(query)
    
    def get_status(self) -> dict:
        """
        Get cultural practices service status
        """
        return {
            "service": "KCC_Cultural_Practices",
            "loaded": self.is_loaded,
            "model_path": self.model_path,
            "base_model": self.base_model_id,
            "has_gemini": self.processing_llm is not None,
            "has_search": self.search_llm is not None
        }

# Global service instance
kcc_cultural_service = KCCCulturalService()