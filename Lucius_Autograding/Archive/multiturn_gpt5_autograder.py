#!/usr/bin/env python3
"""
Multi-turn GPT-5 Autograder  
Grades multi-turn task responses using GPT-5
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from openai import AsyncOpenAI
from pyairtable import Api


sys.path.append(str(Path(__file__).parent.parent))
from configs.api_keys import AIRTABLE_API_KEY, OPENAI_API_KEYS, GOOGLE_CREDENTIALS_PATH
from utils.file_processor import FileProcessor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Configuration - use environment variables if available, otherwise defaults
import os
SOURCE_BASE_ID = os.environ.get('EVAL_BASE_ID', "appgeueGlH9mCUTvu")
SOURCE_TABLE_ID = os.environ.get('EVAL_TABLE_ID', "tblfy3EPxl1PHvKV7")
VIEW_ID = os.environ.get('EVAL_VIEW_ID', "viwxB0A2krCqfWueg")

class MultiTurnGPT5Autograder:
    def __init__(self):
        self.api = Api(AIRTABLE_API_KEY)
        self.source_table = self.api.table(SOURCE_BASE_ID, SOURCE_TABLE_ID)
        self.file_processor = FileProcessor(GOOGLE_CREDENTIALS_PATH)
        self.stats = {'processed': 0, 'graded': 0, 'failed': 0}
        
        # Use all 3 working API keys with cycling (4 calls per key = 12 concurrent)
        self.api_keys = OPENAI_API_KEYS
        self.clients = [AsyncOpenAI(api_key=key) for key in self.api_keys]
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.api_semaphore = asyncio.Semaphore(47)  # 30% more than 36 (36 × 1.3 = 46.8 → 47)
        
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
    
    def get_records_needing_grading(self) -> list:
        """Get records that need GPT-5 grading"""
        try:
            records = self.source_table.all(view=VIEW_ID)
            logger.info(f"📊 Fetched {len(records)} total records from Airtable")
            
            records_needing_grading = []
            batch_count = {'Single-turn': 0, 'Multi-turn': 0, 'Other': 0}
            skip_reasons = {'no_response': 0, 'already_graded': 0, 'need_grading': 0}
            
            for record in records:
                fields = record['fields']
                
                # Only process Single-turn or Multi-turn records
                batch = fields.get('Interaction Type', '')
                if batch in ['Single-turn', 'Multi-turn', 'HLE']:
                    batch_count.setdefault(batch, 0)
                    batch_count[batch] = batch_count.get(batch, 0) + 1
                else:
                    batch_count['Other'] += 1
                    continue
                
                needs_grading = False
                has_response = False
                
                # Check if GPT-5 grading is needed (both score AND summary must be missing)
                gemini_response = fields.get('Gemini Model Response [MT]', '')
                # Handle lookup field format
                if isinstance(gemini_response, list) and gemini_response:
                    gemini_response = gemini_response[0] if gemini_response[0] else ''
                gemini_response = str(gemini_response) if gemini_response else ''
                
                gpt5_response = fields.get('GPT5 Response', '')
                gpt5_file_output = fields.get('GPT5 Response (File Output)', '')
                combined_gpt5_response = gpt5_response
                if gpt5_file_output and str(gpt5_file_output).strip():
                    if gpt5_response and gpt5_response.strip():
                        combined_gpt5_response = f"{gpt5_response}\n\nFile Output:\n{gpt5_file_output}"
                    else:
                        combined_gpt5_response = gpt5_file_output
                
                # Check if Gemini has image outputs
                gemini_file_output = fields.get('Gemini Final Output Files', '')
                has_gemini_output = (gemini_response and gemini_response.strip()) or (gemini_file_output and isinstance(gemini_file_output, list) and len(gemini_file_output) > 0)
                
                # Check Gemini response grading
                if has_gemini_output:
                    has_response = True
                    if not fields.get('GPT5 Autorater - Gemini Response Score') and not fields.get('[GPT5 graded] Gemini Response Scoring Summary'):
                        needs_grading = True
                
                # Check GPT5 response grading  
                if combined_gpt5_response and combined_gpt5_response.strip():
                    has_response = True
                    if not fields.get('GPT5 Autorater - GPT5 Response Score') and not fields.get('[GPT5 graded] GPT5 Response Scoring Summary'):
                        needs_grading = True
                
                if needs_grading:
                    records_needing_grading.append(record)
                    skip_reasons['need_grading'] += 1
                elif has_response:
                    skip_reasons['already_graded'] += 1
                else:
                    skip_reasons['no_response'] += 1
            
            logger.info(f"📊 Batch breakdown: Single-turn: {batch_count.get('Single-turn', 0)}, Multi-turn: {batch_count.get('Multi-turn', 0)}, HLE: {batch_count.get('HLE', 0)}, Other: {batch_count['Other']}")
            logger.info(f"📊 Grading status: Need grading: {skip_reasons['need_grading']}, Already graded: {skip_reasons['already_graded']}, No responses yet: {skip_reasons['no_response']}")
            logger.info(f"🎯 Found {len(records_needing_grading)} records needing GPT-5 grading")
            return records_needing_grading
            
        except Exception as e:
            logger.error(f"Error fetching records: {e}")
            return []
    
    async def grade_single_criterion(self, solution: str, criterion: dict, prompt: str, response_files: list = None) -> dict:
        """Grade solution against a single criterion using GPT-5"""
        async with self.api_semaphore:
            criterion_key = list(criterion.keys())[0]
            criterion_data = criterion[criterion_key]
            
            criterion_description = criterion_data.get("description", "")
            
            grading_prompt = f"""CRITERION TO EVALUATE:
{criterion_description}

ORIGINAL PROMPT:
{prompt}

SOLUTION TO GRADE:
{solution}

Does the solution meet this criterion? Provide 10 sentences of reasoning followed by your final decision."""
            
            full_prompt = f"{self.system_prompt}\n\n{grading_prompt}"
            
            try:
                # Get next client in rotation
                async with self.key_lock:
                    client = self.clients[self.key_index]
                    self.key_index = (self.key_index + 1) % len(self.clients)
                
                # Build messages with vision support if files exist
                if response_files and len(response_files) > 0:
                    # Vision mode with files
                    content_parts = [{"type": "text", "text": full_prompt}]
                    for image_data in response_files:
                        content_parts.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        })
                    messages = [{"role": "user", "content": content_parts}]
                    logger.info(f"🔍 GPT-5 grading with {len(response_files)} response images")
                else:
                    # Text-only mode
                    messages = [{"role": "user", "content": full_prompt}]
                    logger.info(f"🔍 GPT-5 grading text-only")
                
                response = await client.chat.completions.create(
                    model="gpt-5",
                    messages=messages,
                    reasoning_effort="medium"
                )
                
                # Handle different response formats with validation
                if hasattr(response, 'choices') and response.choices:
                    try:
                        content = response.choices[0].message.content or ""
                    except (AttributeError, IndexError, TypeError) as parse_error:
                        logger.error(f"Error parsing GPT-5 response: {parse_error}")
                        return {"autorating": False, "error": f"Response parse error: {parse_error}"}
                elif hasattr(response, 'output_text'):
                    content = response.output_text or ""
                else:
                    content = str(response)
                
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
                logger.error(f"Error grading criterion: {e}")
                return {"autorating": False, "error": str(e)}
    
    async def grade_solution(self, solution: str, rubric: list, prompt: str) -> dict:
        """Grade a complete solution against rubric"""
        try:
            # Grade all criteria IN PARALLEL (same as original airtable_autograder.py)
            criterion_tasks = [
                self.grade_single_criterion(solution, criterion, prompt)
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
            logger.error(f"Error grading solution: {e}")
            return {
                'percentage': 0.0,
                'summary': json.dumps([{"criterion 1": {"error": str(e), "autorating": False}}])
            }
    
    async def process_record(self, record: dict) -> bool:
        """Process a single record for grading"""
        try:
            fields = record['fields']
            record_id = record['id']
            
            # Get rubric and prompt
            rubric_json = fields.get('Rubric JSON', '')
            prompt = fields.get('MT Prompt', '')
            # Handle lookup field format (could be list or string)
            if isinstance(prompt, list) and prompt:
                prompt = prompt[0] if prompt[0] else ''
            prompt = str(prompt) if prompt else ''
            
            if not rubric_json:
                logger.warning(f"No rubric found for record {record_id}")
                return False
            
            try:
                rubric = json.loads(rubric_json)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid rubric JSON in record {record_id}: {e}")
                return False
            
            updates = {}
            
            # Grade Gemini 2.5 Pro Response ONLY if response exists AND grading is missing
            gemini_response = fields.get('Gemini Model Response [MT]', '')
            # Handle lookup field format (could be list or string)
            if isinstance(gemini_response, list) and gemini_response:
                gemini_response = gemini_response[0] if gemini_response[0] else ''
            gemini_response = str(gemini_response) if gemini_response else ''
            
            # Get Gemini file outputs (lookup field)
            gemini_file_output = fields.get('Gemini Final Output Files', '')
            logger.info(f"🔍 GPT-5 grading Gemini - File field: {type(gemini_file_output)} - {gemini_file_output if isinstance(gemini_file_output, list) else 'Has data' if gemini_file_output else 'Empty'}")
            
            # Check if Gemini has EITHER text response OR image outputs
            has_gemini_output = (gemini_response and gemini_response.strip()) or (gemini_file_output and isinstance(gemini_file_output, list) and len(gemini_file_output) > 0)
            
            if has_gemini_output:
                if not fields.get('GPT5 Autorater - Gemini Response Score') and not fields.get('[GPT5 graded] Gemini Response Scoring Summary'):
                    logger.info(f"GPT-5 grading Gemini response for record {record_id}")
                    
                    # If Gemini has file output, use vision grading
                    if gemini_file_output and isinstance(gemini_file_output, list) and len(gemini_file_output) > 0:
                        logger.info(f"👁️ VISION MODE: GPT-5 grading Gemini with {len(gemini_file_output)} file outputs")
                        processed_images = await self.process_response_files(gemini_file_output)
                        if processed_images and len(processed_images) > 0:
                            logger.info(f"✅ Processed {len(processed_images)} Gemini output images for GPT-5 grading")
                            result = await self.grade_solution_with_vision(gemini_response, rubric, prompt, processed_images)
                        else:
                            logger.warning(f"⚠️ No images processed from Gemini files, using text-only grading")
                            result = await self.grade_solution(gemini_response, rubric, prompt)
                    else:
                        logger.info(f"📝 Text-only grading for Gemini (no file outputs)")
                        result = await self.grade_solution(gemini_response, rubric, prompt)
                    
                    updates['GPT5 Autorater - Gemini Response Score'] = result['percentage']
                    updates['[GPT5 graded] Gemini Response Scoring Summary'] = result['summary']
            
            # Grade GPT5 Response ONLY if response exists AND grading is missing
            gpt5_response = fields.get('GPT5 Response', '')
            gpt5_file_output = fields.get('GPT5 Response (File Output)', '')
            
            # Combine regular response with file output if both exist
            combined_gpt5_response = gpt5_response
            if gpt5_file_output and str(gpt5_file_output).strip():
                if gpt5_response and gpt5_response.strip():
                    combined_gpt5_response = f"{gpt5_response}\n\nFile Output:\n{gpt5_file_output}"
                else:
                    combined_gpt5_response = gpt5_file_output
            
            if combined_gpt5_response and combined_gpt5_response.strip():
                if not fields.get('GPT5 Autorater - GPT5 Response Score') and not fields.get('[GPT5 graded] GPT5 Response Scoring Summary'):
                    logger.info(f"Grading GPT5 response for record {record_id}")
                    
                    # If there's file output, use vision grading methodology
                    if gpt5_file_output and str(gpt5_file_output).strip():
                        # Get response files from the File Output field itself
                        response_files = fields.get('GPT5 Response (File Output)', '')
                        if response_files and (isinstance(response_files, list) or str(response_files).strip()):
                            # Process the response files like vision autograder does
                            processed_images = await self.process_response_files(response_files)
                            result = await self.grade_solution_with_vision(combined_gpt5_response, rubric, prompt, processed_images)
                        else:
                            result = await self.grade_solution(combined_gpt5_response, rubric, prompt)
                    else:
                        result = await self.grade_solution(combined_gpt5_response, rubric, prompt)
                        
                    updates['GPT5 Autorater - GPT5 Response Score'] = result['percentage']
                    updates['[GPT5 graded] GPT5 Response Scoring Summary'] = result['summary']
            
            # Update Airtable if we have updates
            if updates:
                self.source_table.update(record_id, updates)
                logger.info(f"✅ Updated record {record_id} with {len(updates)} grading fields")
                return True
            else:
                logger.info(f"⏭️ Record {record_id} - no grading needed")
                return False
                
        except Exception as e:
            logger.error(f"Error processing record: {e}")
            return False
    
    async def run(self):
        """Main execution method"""
        logger.info("🤖 Starting Multi-turn GPT-5 Autograder with LOW reasoning effort")
        logger.info("📋 Using EXACT same system prompt and config as airtable_autograder.py")
        
        # Get records that need grading
        records = self.get_records_needing_grading()
        if not records:
            logger.info("✅ No records need GPT-5 grading")
            return
        
        # Process all records
        semaphore = asyncio.Semaphore(3)  # Limit concurrency
        
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

    async def process_response_files(self, file_urls) -> list:
        """Process response file URLs and convert to images"""
        if not file_urls:
            return []
        
        processed_images = []
        if isinstance(file_urls, list):
            for file_obj in file_urls:
                if isinstance(file_obj, dict) and 'url' in file_obj:
                    try:
                        file_result = await self.file_processor.process_file(file_obj['url'], prefer_vision=True)
                        if file_result['type'] == 'image':
                            processed_images.append(file_result['image_data'])
                        elif file_result['type'] == 'images':
                            processed_images.extend(file_result['image_data'])
                    except Exception as e:
                        logger.error(f"Error processing response file: {e}")
        
        return processed_images

    async def grade_solution_with_vision(self, solution: str, rubric: list, prompt: str, response_images: list) -> dict:
        """Grade solution with vision support"""
        try:
            criterion_tasks = [
                self.grade_single_criterion(solution, criterion, prompt, response_images)
                for criterion in rubric
            ]
            
            criterion_results = await asyncio.gather(*criterion_tasks, return_exceptions=True)
            
            graded_criteria = []
            total_points = 0
            
            for i, result in enumerate(criterion_results):
                criterion = rubric[i]
                criterion_key = list(criterion.keys())[0]
                
                if isinstance(result, Exception):
                    graded_criteria.append({criterion_key: {"autorating": False, "error": str(result)}})
                else:
                    graded_criteria.append({criterion_key: result})
                    if result.get('autorating'):
                        total_points += 1
            
            percentage = (total_points / len(rubric) * 100) if rubric else 0
            
            return {
                'percentage': percentage,
                'summary': json.dumps(graded_criteria, indent=2)
            }
            
        except Exception as e:
            return {
                'percentage': 0,
                'summary': json.dumps([{"criterion 1": {"error": str(e), "autorating": False}}])
            }

async def main():
    autograder = MultiTurnGPT5Autograder()
    await autograder.run()

if __name__ == "__main__":
    asyncio.run(main())
