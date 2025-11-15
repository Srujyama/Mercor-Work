#!/usr/bin/env python3
"""
Multi-turn GPT-5 Autograder with Vision
Grades multi-turn task responses using GPT-5 with image support
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from openai import AsyncOpenAI
from pyairtable import Api

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from configs.api_keys import AIRTABLE_API_KEY, OPENAI_API_KEYS, GOOGLE_CREDENTIALS_PATH
from utils.file_processor import FileProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
SOURCE_BASE_ID = "appgeueGlH9mCUTvu"
SOURCE_TABLE_ID = "tblfy3EPxl1PHvKV7"
VIEW_ID = "viwxB0A2krCqfWueg"

class MultiTurnGPT5AutograderVision:
    def __init__(self):
        self.api = Api(AIRTABLE_API_KEY)
        self.source_table = self.api.table(SOURCE_BASE_ID, SOURCE_TABLE_ID)
        self.file_processor = FileProcessor(GOOGLE_CREDENTIALS_PATH)
        self.stats = {'processed': 0, 'graded': 0, 'failed': 0}
        
        # Use all 3 working API keys with cycling
        self.api_keys = OPENAI_API_KEYS
        self.clients = [AsyncOpenAI(api_key=key) for key in self.api_keys]
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.api_semaphore = asyncio.Semaphore(47)  # Same high concurrency
        
        # EXACT same system prompt as airtable_autograder.py
        self.system_prompt = """You are an expert grader evaluating solutions against specific criteria.
Your task is to determine if the given solution meets the specified criterion exactly as described.

The solution you are evaluating may have a variety of repeating formats or patterns, be very certain that the context of the section of the solution is the same as the criterion.

To 'meet the criterion' the solution must include the content style requested by the criterion description.

Make sure to analyze the entire solution, not just a portion of it, be certain and strategic in your determination.

If the solution has code, you should not give points for calculating a value unless that resulting value is clearly shown in the solution.

IMPORTANT: Provide your response in the following format:
1. First, provide exactly 10 sentences of reasoning explaining your evaluation
2. Then, provide your final decision in the format "FINAL_DECISION: true" or "FINAL_DECISION: false"

If the criterion has any ambiguity, analyze it from every angle. Be strategic and certain in your determination."""
    
    def get_records_needing_vision_grading(self) -> list:
        """Get records that have response files and need vision grading"""
        try:
            records = self.source_table.all(view=VIEW_ID)
            logger.info(f"📊 Fetched {len(records)} records from view")
            
            records_needing_grading = []
            for record in records:
                fields = record['fields']
                
                needs_grading = False
                
                # Check Gemini response with files
                gemini_files = fields.get('Gemini 2.5 Pro Response (Files)', '')
                if gemini_files and (isinstance(gemini_files, list) or str(gemini_files).strip()):
                    # Check if GPT-5 grading is missing
                    if not fields.get('[GPT5 graded] Gemini Response Score') or not fields.get('[GPT5 graded] Gemini Response Scoring Summary'):
                        needs_grading = True
                
                # Check GPT5 response with files
                gpt5_files = fields.get('GPT5 Response (Files)', '')
                if gpt5_files and (isinstance(gpt5_files, list) or str(gpt5_files).strip()):
                    # Check if GPT-5 grading is missing
                    if not fields.get('[GPT5 graded] GPT5 Response Score') or not fields.get('[GPT5 graded] GPT5 Response Scoring Summary'):
                        needs_grading = True
                
                if needs_grading:
                    records_needing_grading.append(record)
            
            logger.info(f"🎯 Found {len(records_needing_grading)} records needing vision grading")
            return records_needing_grading
            
        except Exception as e:
            logger.error(f"Error fetching records: {e}")
            return []
    
    async def process_response_files(self, file_urls) -> list:
        """Process response file URLs and convert to images"""
        if not file_urls:
            return []
        
        # Handle both list and string formats
        if isinstance(file_urls, list):
            # Airtable file attachment format
            urls = []
            for file_obj in file_urls:
                if isinstance(file_obj, dict) and 'url' in file_obj:
                    urls.append(file_obj['url'])
                elif isinstance(file_obj, str):
                    urls.append(file_obj)
        else:
            # String format - split multiple URLs
            urls = [url.strip() for url in str(file_urls).replace('\n', ',').split(',') if url.strip()]
        
        processed_images = []
        for url in urls:
            try:
                logger.info(f"🔄 Processing response file: {url}")
                file_result = await self.file_processor.process_file(url, prefer_vision=True)
                
                if file_result['type'] == 'image':
                    processed_images.append(file_result['image_data'])
                    logger.info(f"✅ Processed response file as image")
                else:
                    logger.warning(f"⚠️ Could not process response file: {url}")
                    
            except Exception as e:
                logger.error(f"Error processing response file {url}: {e}")
        
        logger.info(f"📎 Processed {len(processed_images)} response file images")
        return processed_images
    
    async def grade_single_criterion_with_vision(self, solution: str, criterion: dict, prompt: str, context: str, response_images: list) -> dict:
        """Grade solution against a single criterion with vision support"""
        async with self.api_semaphore:
            criterion_key = list(criterion.keys())[0]
            criterion_data = criterion[criterion_key]
            
            criterion_description = criterion_data.get("description", "")
            
            grading_prompt = f"""CRITERION TO EVALUATE:
{criterion_description}

FULL ORIGINAL TASK CONTEXT:
{context}

ORIGINAL PROMPT:
{prompt}

SOLUTION TO GRADE:
{solution}

SOLUTION INCLUDES GENERATED FILES/IMAGES - analyze these as part of the solution.

Does the solution meet this criterion? Provide 10 sentences of reasoning followed by your final decision."""
            
            full_prompt = f"{self.system_prompt}\n\n{grading_prompt}"
            
            try:
                # Get next client in rotation
                async with self.key_lock:
                    client = self.clients[self.key_index]
                    self.key_index = (self.key_index + 1) % len(self.clients)
                
                # Build messages with vision
                if response_images:
                    content = [{"type": "text", "text": full_prompt}]
                    for image_data in response_images:
                        content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        })
                    messages = [{"role": "user", "content": content}]
                    logger.info(f"👁️ Using vision with {len(response_images)} response images")
                else:
                    messages = [{"role": "user", "content": full_prompt}]
                
                response = await client.chat.completions.create(
                    model="gpt-5",
                    messages=messages,
                    reasoning_effort="medium"
                )
                
                # Safely extract content with validation
                if hasattr(response, 'choices') and response.choices:
                    try:
                        content = response.choices[0].message.content or ""
                    except (AttributeError, IndexError, TypeError) as parse_error:
                        logger.error(f"Error parsing GPT-5 response: {parse_error}")
                        return {"autorating": False, "error": f"Response parse error: {parse_error}"}
                else:
                    logger.error(f"Unexpected GPT-5 response format: {type(response)}")
                    return {"autorating": False, "error": "Unexpected response format"}
                
                # Parse the response for true/false decision
                decision = "false"
                if "FINAL_DECISION: true" in content:
                    decision = "true"
                elif "FINAL_DECISION: false" in content:
                    decision = "false"
                
                return {
                    "autorating": decision == "true",
                    "description": criterion_data.get("description", ""),
                    "weight": criterion_data.get("weight", ""),
                    "criterion_type": criterion_data.get("criterion_type", []),
                    "dependent_criteria": criterion_data.get("dependent_criteria", []),
                    "justification": criterion_data.get("justification", ""),
                    "sources": criterion_data.get("sources", ""),
                    "human_rating": criterion_data.get("human_rating", "")
                }
                
            except Exception as e:
                logger.error(f"Error grading criterion with vision: {e}")
                return {"autorating": False, "error": str(e)}
    
    async def grade_solution_with_vision(self, solution: str, rubric: list, prompt: str, context: str, response_images: list) -> dict:
        """Grade a complete solution with vision support"""
        try:
            # Grade all criteria in parallel with vision
            criterion_tasks = [
                self.grade_single_criterion_with_vision(solution, criterion, prompt, context, response_images)
                for criterion in rubric
            ]
            
            criterion_results = await asyncio.gather(*criterion_tasks, return_exceptions=True)
            
            # Process results
            graded_criteria = []
            total_points = 0
            
            for i, result in enumerate(criterion_results):
                criterion = rubric[i]
                criterion_key = list(criterion.keys())[0]
                
                if isinstance(result, Exception):
                    logger.error(f"Criterion {i} failed: {result}")
                    graded_criteria.append({criterion_key: {"autorating": False, "error": str(result)}})
                else:
                    graded_criteria.append({criterion_key: result})
                    if result.get('autorating'):
                        total_points += 1
            
            # Calculate percentage
            percentage = (total_points / len(rubric) * 100) if rubric else 0
            
            return {
                'percentage': percentage,
                'summary': json.dumps(graded_criteria, indent=2)
            }
            
        except Exception as e:
            logger.error(f"Error grading solution with vision: {e}")
            return {
                'percentage': 0.0,
                'summary': json.dumps([{"criterion 1": {"error": str(e), "autorating": False}}])
            }
    
    async def process_record(self, record: dict) -> bool:
        """Process a single record for vision grading"""
        try:
            fields = record['fields']
            record_id = record['id']
            
            # Get rubric, prompt, and context
            rubric_json = fields.get('Rubric JSON', '')
            prompt = fields.get('Prompt', '')
            context = fields.get('Context', '')
            
            if not rubric_json:
                logger.warning(f"No rubric found for record {record_id}")
                return False
            
            try:
                rubric = json.loads(rubric_json)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid rubric JSON in record {record_id}: {e}")
                return False
            
            updates = {}
            
            # Grade Gemini response with files if available
            gemini_response = fields.get('Gemini 2.5 Pro Response', '')
            gemini_files = fields.get('Gemini 2.5 Pro Response (Files)', '')
            if gemini_response and gemini_response.strip() and gemini_files and (isinstance(gemini_files, list) or str(gemini_files).strip()):
                if not fields.get('[GPT5 graded] Gemini Response Score') or not fields.get('[GPT5 graded] Gemini Response Scoring Summary'):
                    logger.info(f"Vision grading Gemini response for record {record_id}")
                    
                    # Process response images
                    response_images = await self.process_response_files(gemini_files)
                    
                    result = await self.grade_solution_with_vision(gemini_response, rubric, prompt, context, response_images)
                    updates['[GPT5 graded] Gemini Response Score'] = result['percentage']
                    updates['[GPT5 graded] Gemini Response Scoring Summary'] = result['summary']
            
            # Grade GPT5 response with files if available
            gpt5_response = fields.get('GPT5 Response', '')
            gpt5_files = fields.get('GPT5 Response (Files)', '')
            if gpt5_response and gpt5_response.strip() and gpt5_files and (isinstance(gpt5_files, list) or str(gpt5_files).strip()):
                if not fields.get('[GPT5 graded] GPT5 Response Score') or not fields.get('[GPT5 graded] GPT5 Response Scoring Summary'):
                    logger.info(f"Vision grading GPT5 response for record {record_id}")
                    
                    # Process response images
                    response_images = await self.process_response_files(gpt5_files)
                    
                    result = await self.grade_solution_with_vision(gpt5_response, rubric, prompt, context, response_images)
                    updates['[GPT5 graded] GPT5 Response Score'] = result['percentage']
                    updates['[GPT5 graded] GPT5 Response Scoring Summary'] = result['summary']
            
            # Update Airtable if we have updates
            if updates:
                self.source_table.update(record_id, updates)
                logger.info(f"✅ Updated record {record_id} with {len(updates)} vision grading fields")
                return True
            else:
                logger.info(f"⏭️ Record {record_id} - no vision grading needed")
                return False
                
        except Exception as e:
            logger.error(f"Error processing record: {e}")
            return False
    
    async def run(self):
        """Main execution method"""
        logger.info("🤖 Starting Multi-turn GPT-5 Vision Autograder with medium reasoning effort")
        logger.info("👁️ Grades responses that include generated images/files")
        logger.info(f"📍 Processing view: {VIEW_ID}")
        
        # Get records that need vision grading
        records = self.get_records_needing_vision_grading()
        if not records:
            logger.info("✅ No records need vision grading")
            return
        
        # Process all records
        semaphore = asyncio.Semaphore(3)  # Limit concurrency for vision processing
        
        async def process_with_semaphore(record):
            async with semaphore:
                success = await self.process_record(record)
                if success:
                    self.stats['graded'] += 1
                else:
                    self.stats['failed'] += 1
                self.stats['processed'] += 1
                await asyncio.sleep(1)  # Rate limiting
        
        # Process all records
        tasks = [process_with_semaphore(record) for record in records]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"🎉 Complete! Processed: {self.stats['processed']}, Graded: {self.stats['graded']}, Failed: {self.stats['failed']}")

async def main():
    autograder = MultiTurnGPT5AutograderVision()
    await autograder.run()

if __name__ == "__main__":
    asyncio.run(main())
