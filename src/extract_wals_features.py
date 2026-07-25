#!/usr/bin/env python3
"""
Extract WALS-style typological features for a generated language.

Uses an LLM to analyze a language's phonology, grammar, and lexicon and
produce a JSON typological profile (word order, morphology, phoneme
inventory size, etc.) comparable to WALS (World Atlas of Language
Structures) features. Results are saved to the language's evaluation
directory as diversity_analysis.json.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dotenv import load_dotenv

load_dotenv()

from llm_client import LLMClientGemini, LLMClientDeepseek, LLMClientOpenAI, PromptManager
from utils import clean_response, load_required_files, save_memory

logger = logging.getLogger(__name__)


def build_llm_client(args):
    if args.model.startswith('gemini'):
        return LLMClientGemini(
            model_checkpoint=args.model,
            temperature=args.temperature,
            debug=args.debug,
        )
    elif args.model.startswith('deepseek'):
        return LLMClientDeepseek(
            model_checkpoint=args.model,
            temperature=args.temperature,
            debug=args.debug,
        )
    elif args.model.startswith('o') or args.model.startswith('gpt-'):
        return LLMClientOpenAI(
            model_checkpoint=args.model,
            temperature=args.temperature,
            debug=args.debug,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")


def extract_wals_features(args, llm_client) -> bool:
    """Extract WALS typological features for a single language."""
    memory_dir = os.path.join(args.language_dir, 'memory')
    required_files = {
        'phonology': 'phonology.txt',
        'grammar': 'grammar.txt',
        'lexicon': 'lexicon.csv',
    }

    language_data = load_required_files(memory_dir, required_files)
    if language_data is None:
        logger.error("Could not load required language data")
        return False

    prompt_file = os.path.join(args.prompt_dir, 'evaluation', 'diversity_medium.txt')
    if not os.path.exists(prompt_file):
        logger.error(f"Prompt file not found: {prompt_file}")
        return False

    prompt_template = PromptManager.load_prompt(prompt_file)

    filled_prompt = prompt_template
    filled_prompt = filled_prompt.replace('{phonology}', language_data['phonology'])
    filled_prompt = filled_prompt.replace('{grammar}', language_data['grammar'])
    filled_prompt = filled_prompt.replace('{lexicon}', language_data['lexicon'])

    logger.info("Requesting WALS feature extraction from LLM")

    _, response = llm_client.generate_and_extract(filled_prompt, do_sleep=False)
    if not response:
        logger.error("No response received from LLM")
        return False

    cleaned_response = clean_response(response, response_type="json")

    try:
        analysis_data = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Raw response: {response[:500]}...")
        return False

    evaluation_metadata = {
        'evaluation_type': 'wals_features',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model': args.model,
        'prompt_version': 'diversity_medium.txt',
    }

    evaluation_dir = os.path.join(args.language_dir, 'evaluation')
    save_memory(cleaned_response, evaluation_dir, 'diversity_analysis.json', evaluation_metadata)

    logger.info(f"WALS feature extraction completed. Results saved to: {evaluation_dir}")

    if 'typology' in analysis_data:
        print(f"\nWALS typological features for {args.language_id}:")
        for feature, data in analysis_data['typology'].items():
            if isinstance(data, dict) and 'value' in data and data['value'] != 'null':
                confidence = data.get('confidence', 'Unknown')
                print(f"  {feature}: {data['value']} (confidence: {confidence})")

    return True


def get_args():
    parser = ArgumentParser(
        description='Extract WALS-style typological features for a generated language',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--language-id', required=True,
                        help='Language ID to analyze (must exist under --output-dir/languages/)')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory containing generated languages')
    parser.add_argument('--prompt-dir', default='prompts',
                        help='Directory containing prompt templates')
    parser.add_argument('--model', default='gemini-2.5-pro',
                        help='Model identifier to use for feature extraction')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Temperature for sampling (None = model API default)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with dummy responses')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    args = get_args()
    args.language_dir = os.path.join(args.output_dir, 'languages', args.language_id)

    if not os.path.isdir(args.language_dir):
        print(f"Error: Language directory not found: {args.language_dir}")
        sys.exit(1)

    llm_client = build_llm_client(args)

    success = extract_wals_features(args, llm_client)
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
