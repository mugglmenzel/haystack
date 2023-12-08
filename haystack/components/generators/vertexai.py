import dataclasses
import logging
from typing import Optional, List, Dict, Any

import vertexai
from vertexai.preview.language_models import TextGenerationModel, TextGenerationResponse

from haystack import component, default_from_dict, default_to_dict

logger = logging.getLogger(__name__)


@component
class VertexAIGenerator:
    """
    Enables text generation using large language models (LLMs) from Google Cloud Vertex AI. It supports the PaLM LLMs and Gemini LMMs.

     Input and Output Format:
         - **String Format**: This component uses strings for both input and output.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        model_name: str = "text-bison@latest",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of VertexAIGenerator.

        :param project_id: The GCP project id.
        :param location: The GCP location (e.g. region 'europe-west4').
        :param model_name: The name of the model to use.
        :param generation_kwargs: Other parameters to use for the model.
            Some of the supported parameters:
            - `max_output_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So, 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `top_k`: The number of highest probability tokens to keep for top-k-filtering.
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs or {}

        vertexai.init(project=project_id, location=location)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        :return: The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            project_id=self.project_id,
            location=self.location,
            model_name=self.model_name,
            generation_kwargs=self.generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexAIGenerator":
        """
        Deserialize this component from a dictionary.
        :param data: The dictionary representation of this component.
        :return: The deserialized component instance.
        """
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str], metadata=List[TextGenerationResponse])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param prompt: The string prompt to use for text generation.
        :param generation_kwargs: Additional keyword arguments for text generation. These parameters will
        potentially override the parameters passed in the __init__ method.
        :return: The generated response and TextGenerationResponse object with all metadata incl. safety classifications.
        """

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        response = TextGenerationModel.from_pretrained(self.model_name).predict(prompt, **generation_kwargs)

        # return the response and metadata
        return {"replies": [response.text], "metadata": [response]}
