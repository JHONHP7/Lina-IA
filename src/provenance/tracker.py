# src/provenance/tracker.py
import json
import os
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# LlamaIndex Callbacks
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks import CBEventType

# noWorkflow API interna para marcação e captura explícita
# Baseado na estrutura do repositório gems-uff/noworkflow
try:
    from noworkflow.now.collection.prov_execution import collector
    from noworkflow.now.persistence import persistence_config
    NOWORKFLOW_AVAILABLE = True
except ImportError:
    NOWORKFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class LINAProvenanceHandler(BaseCallbackHandler):
    """
    Handler de Callbacks nativo do LlamaIndex adaptado para e-Science.
    Captura proveniência retrospectiva de dados (Retrieval e LLM Generation).
    """
    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.reset_trace()

    def reset_trace(self) -> None:
        self.trace_data: Dict[str, Any] = {
            "retrieval_events": [],
            "llm_events": [],
            "timestamps": {}
        }

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> str:
        if event_type == CBEventType.QUERY:
            self.trace_data["timestamps"]["query_start"] = time.time()
            self.trace_data["user_query"] = payload.get("query_str") if payload else None
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any
    ) -> None:
        if not payload:
            return

        if event_type == CBEventType.RETRIEVE:
            nodes = payload.get("nodes", [])
            retrieval_info = {
                "event_id": event_id,
                "timestamp": datetime.now().isoformat(),
                "nodes_count": len(nodes),
                "nodes": [
                    {
                        "node_id": node.node.node_id,
                        "score": node.score if hasattr(node, "score") else None,
                        "file_name": node.node.metadata.get("file_name"),
                        "source": node.node.metadata.get("source")
                    }
                    for node in nodes
                ]
            }
            self.trace_data["retrieval_events"].append(retrieval_info)
            
            # Se o noWorkflow estiver rodando ativamente, injeta uma tag de arquivo de acesso
            if NOWORKFLOW_AVAILABLE and collector.Collector.definitions:
                for node in nodes:
                    fname = node.node.metadata.get("file_name", "unknown_chunk")
                    # Registra de forma artificial no fluxo do noWorkflow que este nó de documento foi lido
                    logger.debug(f"[noWorkflow Tag] Document chunk read: {fname}")

        elif event_type == CBEventType.LLM:
            response = payload.get("response")
            llm_info = {
                "event_id": event_id,
                "timestamp": datetime.now().isoformat(),
                "output_text": str(response),
                "token_usage": getattr(response, "raw", {}).get("usage", {}) if response else {}
            }
            self.trace_data["llm_events"].append(llm_info)

        elif event_type == CBEventType.QUERY:
            self.trace_data["timestamps"]["query_end"] = time.time()

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self.reset_trace()

    def end_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def get_provenance_data(self) -> Dict[str, Any]:
        ts = self.trace_data["timestamps"]
        if "query_start" in ts and "query_end" in ts:
            self.trace_data["latency_seconds"] = ts["query_end"] - ts["query_start"]
        return self.trace_data


def track_provenance_experiment(experiment_name: str, prospective_params: Dict[str, Any]):
    """
    Decorador para mapeamento das chamadas de funções locais.
    Une metadados Prospectivos (parâmetros) e Retrospectivos (LlamaIndex + noWorkflow).
    """
    def decorator(func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from llama_index.core import Settings
            handler = LINAProvenanceHandler()
            Settings.callback_manager.add_handler(handler)
            
            # Captura Prospectiva Inicial
            run_log = {
                "experiment": experiment_name,
                "execution_timestamp": datetime.now().isoformat(),
                "noworkflow_trial_id": None,
                "prospect_parameters": prospective_params,
                "retrospect_execution": {}
            }
            
            # Vincula ID do Trial atual do noWorkflow se executado via CLI
            if NOWORKFLOW_AVAILABLE and hasattr(collector, "trial_id"):
                run_log["noworkflow_trial_id"] = collector.trial_id
            
            start_clock = time.time()
            try:
                # Execução da função RAG propriamente dita (ex: retrieve_queryengine.query)
                result = func(*args, **kwargs)
                return result
            finally:
                total_overhead_clock = time.time() - start_clock
                
                # Coleta retrospectiva estruturada
                retrospect_data = handler.get_provenance_data()
                retrospect_data["total_wrapper_time_seconds"] = total_overhead_clock
                retrospect_data["overhead_provenance_seconds"] = total_overhead_clock - retrospect_data.get("latency_seconds", 0)
                
                run_log["retrospect_execution"] = retrospect_data
                
                # Salvamento do artefato estruturado para tabulação e posterior conversão (mlflow2prov)
                os.makedirs("data/provenance_logs", exist_ok=True)
                log_filename = f"data/provenance_logs/trial_{run_log['noworkflow_trial_id'] or int(time.time())}.json"
                with open(log_filename, "w", encoding="utf-8") as f:
                    json.dump(run_log, f, indent=4, ensure_ascii=False)
                    
                Settings.callback_manager.remove_handler(handler)
        return wrapper
    return decorator

def track_execution(func):
    """
    Legacy decorator for tracking function executions, inputs, and outputs.
    """
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"[Provenance] Executing: {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"[Provenance] Completed: {func.__name__}")
        return result
    return wrapper