import asyncio
import json
import logging
import os
import sys
from pprint import pprint
import time

# Configure path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from app.core.config import get_settings
from app.pipeline.orchestrator import run_pipeline
from app.api.job_store import get_job_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_runner")

async def test_docs():
    settings = get_settings()
    store = get_job_store()
    
    docs = [
        {"name": "sample_esg_report.pdf", "path": "sample_esg_report.pdf", "algo": "heuristic"},
        {"name": "sample_board_minutes.pdf", "path": "sample_board_minutes_esg.pdf", "algo": "bert_mean"},
        {"name": "sample_supplier_audit.pdf", "path": "sample_supplier_audit_esg.pdf", "algo": "heuristic"},
        {"name": "sample_data_compilation.pdf", "path": "sample_data_compilation_esg.pdf", "algo": "heuristic"},
    ]

    for doc in docs:
        filepath = os.path.join("/Volumes/ReserveDisk/codeBase/AxiomESG", doc["path"])
        if not os.path.exists(filepath):
            logger.error(f"Missing {filepath}")
            continue

        with open(filepath, "rb") as f:
            data = f.read()

        files = [(doc["name"], data, "application/pdf")]
        logger.info(f"==== Testing {doc['name']} with algorithm {doc['algo']} ====")
        
        start_time = time.time()
        try:
            output, raw_text, usage = run_pipeline(
                files=files,
                settings=settings,
                job_id=f"test-{doc['name']}",
                algorithm=doc['algo']
            )
            duration = time.time() - start_time
            
            logger.info(f"SUCCESS in {duration:.2f}s!")
            logger.info("Extraction Usage: " + str(usage))
            logger.info("Metadata Algorithm Used: " + output.metadata.algorithm_used)
            logger.info("Env metrics: " + str(len(output.environmental.metrics)))
            logger.info("Soc metrics: " + str(len(output.social.metrics)))
            logger.info("Gov metrics: " + str(len(output.governance.metrics)))
            
            if doc['name'] == "sample_board_minutes.pdf":
                # Print the narrative for sanity check
                logger.info("Governance Narrative preview:")
                logger.info(output.governance.narrative[:200] + "...")

        except Exception as e:
            logger.error(f"FAILED on {doc['name']}: {e}", exc_info=True)
            
        logger.info("=========================================================\n")

if __name__ == "__main__":
    asyncio.run(test_docs())
