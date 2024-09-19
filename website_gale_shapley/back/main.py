from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List
from fastapi import FastAPI, status
import io
import logging

import fairpyx
from .schema import Instance

logger = logging.getLogger("uvicorn")

# On startup, create database tables
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, None]:
    # before app startup
    yield
    # After app shutdown


app = FastAPI(lifespan=lifespan)


@app.post("/api/divide", status_code=status.HTTP_200_OK)
async def divide(instance: Instance) -> Dict[str, Any]:
    fairpyx_instance = fairpyx.Instance(agent_capacities=instance.agentCapabilities, 
                                item_capacities=instance.courseCapabilities, 
                                valuations=instance.bids)
    
    # Create a StringIO object to capture logs
    log_capture = io.StringIO()
    # Create a handler that writes to the StringIO object
    handler = logging.StreamHandler(log_capture)
    # Set the format for the handler
    handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Add the handler to the fairpyx logger
    fairpyx_logger = logging.getLogger('fairpyx')
    fairpyx_logger.addHandler(handler)
    
    # Set the level to capture all logs
    original_level = fairpyx_logger.level
    fairpyx_logger.setLevel(logging.DEBUG)
    
    try:
        allocation = fairpyx.divide(fairpyx.algorithms.gale_shapley, 
                                instance=fairpyx_instance, 
                                course_order_per_student=instance.courseOrderPerStudent, 
                                tie_braking_lottery=instance.tieBrakingLottery)
    finally:
        # Reset the logger level and remove the handler
        fairpyx_logger.setLevel(original_level)
        fairpyx_logger.removeHandler(handler)
    
    # Get the captured logs
    log_capture.seek(0)
    captured_logs = log_capture.getvalue().splitlines()
    
    # Log the captured logs using the uvicorn logger
    for log_line in captured_logs:
        logger.info(f"fairpyx log: {log_line}")
    
    logger.info(f"input: {instance}")
    logger.info(f"output: {allocation}")
    
    # Return both the allocation and the logs
    return {
        "allocation": allocation,
        "logs": captured_logs
    }

