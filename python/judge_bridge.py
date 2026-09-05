"""JSON-lines-free subprocess bridge to pinned, unmodified LoCoMo refined judge."""
import asyncio
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'upstream'))
from llm_judge import evaluate_llm_judge, ashutdown_llm_judge

async def main():
    request = json.load(sys.stdin)
    try:
        result = await asyncio.wait_for(evaluate_llm_judge(
            question=request['question'], reference_answer=request['gold'],
            predicted_answer=request['answer'], name='refined', model=os.environ.get('EVALUATOR_MODEL'), base_url=os.environ.get('EVALUATOR_API_BASE'), api_key=os.environ.get('EVALUATOR_API_KEY')), timeout=180)
        print(json.dumps(result))
    finally:
        await ashutdown_llm_judge()

if __name__ == '__main__':
    asyncio.run(main())
