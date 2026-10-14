from fastapi import APIRouter, HTTPException, Body
import time
import numpy as np
from typing import List, Any

from .response import success_response, error_response
from simulator.execution_engine import get_engine, ExecutionEngine

router = APIRouter(prefix="/api/execute", tags=["execute"])


@router.post("/build")
def execute_build(architecture: List[Any] = Body(...)):
    """Build a neural network graph from architecture definition."""
    start_time = time.time()
    engine = get_engine()
    
    try:
        result = engine.build_graph(architecture)
        return success_response(
            data=result,
            execution_time=time.time() - start_time,
            device=engine.backend,
        )
    except Exception as e:
        return error_response(
            code="BUILD_FAILED",
            message=str(e),
        )


@router.post("/forward")
def execute_forward(input_data: List[Any] = Body(...)):
    """Execute forward pass through the network."""
    start_time = time.time()
    engine = get_engine()
    
    if not hasattr(engine, '_graph') or engine._graph is None:
        return error_response(
            code="NO_GRAPH",
            message="No model built. Call /execute/build first.",
        )
    
    try:
        result = engine.forward(input_data)
        
        # Convert to JSON-serializable format
        layers_data = engine.get_layer_data_json(result)
        
        return success_response(
            data={
                "output": result.final_output.tolist() if hasattr(result.final_output, 'tolist') else result.final_output,
                "layers": layers_data,
                "total_params": result.total_params,
                "execution_time_ms": result.execution_time_ms,
            },
            execution_time=time.time() - start_time,
            device=result.device_used,
        )
    except Exception as e:
        return error_response(
            code="FORWARD_FAILED",
            message=str(e),
        )


@router.post("/backward")
def execute_backward(target: List[Any] = Body(...), loss_function: str = "mse"):
    """Execute backward pass (backpropagation)."""
    start_time = time.time()
    engine = get_engine()
    
    if not hasattr(engine, '_graph') or engine._graph is None:
        return error_response(
            code="NO_GRAPH",
            message="No model built. Call /execute/build first.",
        )
    
    try:
        result = engine.backward(target, loss_function=loss_function)
        
        # Convert gradients to JSON
        gradients_data = []
        for layer in result.layers:
            layer_dict = {
                "layer_index": layer.layer_index,
                "type": layer.layer_type,
            }
            if layer.gradient_weights is not None:
                arr = engine._to_numpy(layer.gradient_weights)
                layer_dict["gradient_weights"] = {
                    "shape": list(arr.shape),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "norm": float(np.linalg.norm(arr)),
                }
            gradients_data.append(layer_dict)
        
        return success_response(
            data={
                "loss_value": result.loss_value,
                "total_gradient_norm": result.total_gradient_norm,
                "layers": gradients_data,
                "execution_time_ms": result.execution_time_ms,
            },
            execution_time=time.time() - start_time,
            device=engine.backend,
        )
    except Exception as e:
        return error_response(
            code="BACKWARD_FAILED",
            message=str(e),
        )


@router.post("/train")
def execute_train(dataset: dict, epochs: int = 10, learning_rate: float = 0.01, loss_function: str = "mse"):
    """Execute training loop."""
    start_time = time.time()
    engine = get_engine()
    
    if not hasattr(engine, '_graph') or engine._graph is None:
        return error_response(
            code="NO_GRAPH",
            message="No model built. Call /execute/build first.",
        )
    
    try:
        result = engine.train(dataset, epochs, learning_rate=learning_rate, loss_function=loss_function)
        
        return success_response(
            data={
                "loss_history": result.loss_history,
                "metrics_history": result.metrics_history,
                "final_metrics": result.final_metrics,
                "total_epochs": result.total_epochs,
                "execution_time_ms": result.execution_time_ms,
            },
            execution_time=time.time() - start_time,
            device=engine.backend,
        )
    except Exception as e:
        return error_response(
            code="TRAIN_FAILED",
            message=str(e),
        )


@router.get("/device")
def execute_device_info():
    """Get execution engine device information."""
    engine = get_engine()
    return success_response(
        data=engine.get_device_info(),
    )


@router.post("/device")
def execute_device_set(device: str):
    """Set execution device (auto/gpu/cpu)."""
    engine = get_engine()
    engine.set_device(device)
    return success_response(
        data={
            "device": device,
            "backend": engine.backend,
            "gpu_available": engine.is_gpu_available,
        },
    )