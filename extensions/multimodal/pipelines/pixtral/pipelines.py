from typing import Optional

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline

available_pipelines = ['pixtral-12b']

def get_pipeline(name: str, params: dict) -> Optional[AbstractMultimodalPipeline]:
    if name == 'pixtral-12b':
        from .pixtral import Pixtral_12B_Pipeline
        return Pixtral_12B_Pipeline(params)
    return None

def get_pipeline_from_model_name(model_name: str, params: dict) -> Optional[AbstractMultimodalPipeline]:
    if 'pixtral' not in model_name.lower():
        return None
    if '12b' in model_name.lower():
        from .pixtral import Pixtral_12B_Pipeline
        return Pixtral_12B_Pipeline(params)
    return None

def get_pipelines():
    from .pixtral import Pixtral_12B_Pipeline
    return {
        Pixtral_12B_Pipeline.name(): Pixtral_12B_Pipeline
    }
