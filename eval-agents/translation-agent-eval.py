import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, ItemHelpers, Runner, trace
from langfuse import Langfuse
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

"""
Enhanced version that includes BLEU score calculation for translation evaluation.
"""

def load_env():
    """Load environment variables from .env file"""
    load_dotenv()

# Set Langfuse credentials
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")

# Define translation agents
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish with perfect grammar and natural phrasing.",
)

translation_picker = Agent(
    name="translation_picker",
    instructions="""You are an expert Spanish translator. Pick the best Spanish translation from the given options.
    
    Evaluate each translation on:
    1. Accuracy - How well it preserves the original meaning
    2. Grammar - Correctness of grammar 
    3. Naturalness - How native and idiomatic it sounds
    
    First briefly explain which translation is best and why, then provide ONLY the best translation.""",
)

async def run_translation(msg):
    """Run the parallel translation process on a single message"""
    # Ensure the entire workflow is a single trace
    with trace("Parallel translation"):
        # Run translations in parallel
        res_1, res_2, res_3 = await asyncio.gather(
            Runner.run(spanish_agent, msg),
            Runner.run(spanish_agent, msg),
            Runner.run(spanish_agent, msg),
        )

        outputs = [
            ItemHelpers.text_message_outputs(res_1.new_items),
            ItemHelpers.text_message_outputs(res_2.new_items),
            ItemHelpers.text_message_outputs(res_3.new_items),
        ]

        translations = "\n\n".join(outputs)
        
        # Get the best translation
        best_translation = await Runner.run(
            translation_picker,
            f"Input: {msg}\n\nTranslations:\n{translations}",
        )

    return {
        "input": msg,
        "translations": outputs,
        "best": best_translation.final_output
    }

def calculate_bleu_score(reference, candidate):
    """
    Calculate BLEU score for a translation compared to a reference.
    
    Args:
        reference (str): The reference/expected translation
        candidate (str): The candidate translation to evaluate
        
    Returns:
        float: BLEU score between 0 and 100
    """
    # Tokenize the sentences into words
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    
    # Use smoothing to avoid 0 score when there are no n-gram matches
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU score with weights [0.25, 0.25, 0.25, 0.25] for 1-gram to 4-gram
    # and handle cases where reference might be too short for higher n-grams
    try:
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens, 
                                   weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=smoothing)
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        # Fall back to unigram BLEU if there's an issue
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens, 
                                   weights=(1, 0, 0, 0),
                                   smoothing_function=smoothing)
    
    # Convert from 0-1 scale to 0-100 scale
    return bleu_score * 100

async def evaluate_on_dataset(dataset_name, num_examples):
    """Evaluate the agent on examples from the dataset"""
    langfuse = Langfuse()
    
    print(f"Evaluating on dataset '{dataset_name}'...")
    
    # Get the dataset from Langfuse
    try:
        dataset = langfuse.get_dataset(dataset_name)
        print(f"Found dataset with {len(dataset.items)} items")
    except Exception as e:
        print(f"Error retrieving dataset: {e}")
        return []
    
    results = []
    
    # Limit the number of examples to process
    items_to_process = min(num_examples, len(dataset.items))
    print(f"Processing {items_to_process} examples...")
    
    for i in range(items_to_process):
        item = dataset.items[i]
        print(f"\n===== Processing example {i+1}/{items_to_process} =====")
        
        # Extract the input from the dataset item
        try:
            # Print the entire item to debug
            print(f"Item data: {item}")
            
            # Try to get the input field
            if hasattr(item, 'input'):
                english_text = item.input.get('text', f"Default example {i+1}")
            else:
                # If direct access fails, try using the to_dict method if available
                item_dict = item.to_dict() if hasattr(item, 'to_dict') else {}
                english_text = item_dict.get('input', {}).get('text', f"Default example {i+1}")
            
            # Run the agent on this example
            print(f"English text: {english_text}")
            result = await run_translation(english_text)
            
            # print("\nTranslation candidates:")
            # for j, translation in enumerate(result["translations"]):
            #     print(f"{j+1}: {translation}")
                
            print(f"\nBest translation: {result['best']}")
            
            # Get the expected translation
            expected_translation = None
            if hasattr(item, 'expected_output'):
                print("---item.expected_output", item.expected_output)
                expected_translation = item.expected_output.get('text')
            else:
                # If direct access fails, try using the to_dict method if available
                item_dict = item.to_dict() if hasattr(item, 'to_dict') else {}
                expected_translation = item_dict.get('expected_output', {}).get('text')

            
            # Compare the best translation with the expected translation
            if expected_translation:
                print(f"\nExpected translation: {expected_translation}")
                
                # Add comparison result to the result dictionary
                result['expected'] = expected_translation
                
                # Calculate BLEU score
                bleu = calculate_bleu_score(expected_translation, result['best'])
                print(f"BLEU score: {bleu:.2f}/100")
                result['bleu_score'] = bleu
                
                # Simple match check (could be enhanced with more sophisticated metrics)
                exact_match = result['best'] == expected_translation
                print(f"Exact match: {exact_match}")
                result['exact_match'] = exact_match
                
            else:
                print("\nNo expected translation found for comparison")
                result['expected'] = None
                result['exact_match'] = None
                result['bleu_score'] = None
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing item {i+1}: {e}")
            continue
    
    # Calculate and print overall performance metrics
    successful_comparisons = [r for r in results if r['exact_match'] is not None]
    if successful_comparisons:
        exact_matches = sum(1 for r in successful_comparisons if r['exact_match'])
        match_rate = exact_matches / len(successful_comparisons)
        
        # Calculate average BLEU score
        bleu_scores = [r['bleu_score'] for r in successful_comparisons if r['bleu_score'] is not None]
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        
        print(f"\nOverall performance:")
        print(f"Exact matches: {exact_matches}/{len(successful_comparisons)} ({match_rate:.2%})")
        print(f"Average BLEU score: {avg_bleu:.2f}/100")
    
    return results

async def main():
    # Load environment variables
    load_env()
    
    # Ensure NLTK packages needed for BLEU calculation are downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    
    # Define the Langfuse dataset name
    dataset_name = "en-es-translation-benchmark"
    
    # Choose mode
    print("1. Interactive mode")
    print("2. Dataset evaluation mode")
    choice = input("Enter choice (1/2): ")
    
    if choice == "2":
        # Dataset evaluation mode
        num_examples = int(input("How many examples to evaluate? (1-20): "))
        num_examples = min(20, max(1, num_examples))
        
        results = await evaluate_on_dataset(dataset_name, num_examples)
        
        print("\n===== EVALUATION SUMMARY =====")
        for i, result in enumerate(results):
            print(f"Example {i+1}:")
            print(f"Input: {result['input']}")
            print(f"Best translation: {result['best']}")
            if result['expected']:
                print(f"Expected translation: {result['expected']}")
                print(f"Exact match: {result['exact_match']}")
                print(f"BLEU score: {result['bleu_score']:.2f}/100")
            print("-" * 40)
    else:
        # Interactive mode
        msg = input("Enter a message to translate to Spanish: ")
        result = await run_translation(msg)
        
        print("\nTranslation candidates:")
        for i, translation in enumerate(result["translations"]):
            print(f"{i+1}: {translation}")
            
        print(f"\nBest translation: {result['best']}")

if __name__ == "__main__":
    asyncio.run(main())