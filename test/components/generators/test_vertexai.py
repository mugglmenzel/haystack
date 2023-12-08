from unittest.mock import patch, MagicMock, Mock

import pytest
import vertexai
from vertexai.preview.language_models import TextGenerationModel, TextGenerationResponse

from haystack.components.generators import VertexAIGenerator


@pytest.fixture
def mock_text_generation():
    with patch("vertexai.preview.language_models.TextGenerationModel.predict", autospec=True) as mock_text_generation:
        mock_response = Mock()
        mock_response.text = "I'm fine, thanks."
        mock_text_generation.return_value = mock_response
        yield mock_text_generation


class TestVertexAIGenerator:
    def test_initialize_with_valid_model_and_generation_parameters(self):
        model_name = "text-bison@latest"
        generation_kwargs = {"temperature": 0.1}

        generator = VertexAIGenerator(
            project_id="project_id", location="us-central1", model_name=model_name, generation_kwargs=generation_kwargs
        )

        assert generator.model_name == model_name
        assert generator.project_id == "project_id"
        assert generator.location == "us-central1"
        assert generator.generation_kwargs == {**generation_kwargs}

    def test_to_dict(self):
        # Initialize the VertexAIGenerator object with valid parameters
        generator = VertexAIGenerator(
            project_id="project_id",
            location="us-central1",
            model_name="text-bison@latest",
            generation_kwargs={"temperature": 0.5},
        )

        # Call the to_dict method
        result = generator.to_dict()
        init_params = result["init_parameters"]

        # Assert that the init_params dictionary contains the expected keys and values
        assert init_params["model_name"] == "text-bison@latest"
        assert init_params["generation_kwargs"] == {"temperature": 0.5}
        assert init_params["project_id"] == "project_id"
        assert init_params["location"] == "us-central1"

    def test_from_dict(self):
        generator = VertexAIGenerator(
            project_id="project_id",
            location="us-central1",
            model_name="text-bison@latest",
            generation_kwargs={"temperature": 0.5},
        )
        # Call the to_dict method
        result = generator.to_dict()

        # now deserialize, call from_dict
        generator_2 = VertexAIGenerator.from_dict(result)
        assert generator_2.model_name == "text-bison@latest"
        assert generator_2.generation_kwargs == {"temperature": 0.5}
        assert generator_2.project_id == "project_id"
        assert generator_2.location == "us-central1"

    def test_generate_text_response_with_valid_prompt_and_generation_parameters(self, mock_text_generation):
        model = "text-bison@latest"
        generation_kwargs = {"temperature": 0.1}

        generator = VertexAIGenerator(
            project_id="project_id", location="us-central1", model_name=model, generation_kwargs=generation_kwargs
        )

        prompt = "Hello, how are you?"
        response = generator.run(prompt)

        # check kwargs passed to text_generation
        # note how n was not passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == generation_kwargs

        assert isinstance(response, dict)
        assert "replies" in response
        assert "metadata" in response
        assert isinstance(response["replies"], list)
        assert isinstance(response["metadata"], list)
        assert len(response["replies"]) == 1
        assert len(response["metadata"]) == 1
        assert [isinstance(reply, str) for reply in response["replies"]]

    def test_generate_text_with_custom_generation_parameters(self, mock_text_generation):
        generator = VertexAIGenerator(project_id="project_id", location="us-central1")

        generation_kwargs = {"temperature": 0.8, "max_output_tokens": 256}
        response = generator.run("How are you?", generation_kwargs=generation_kwargs)

        # check kwargs passed to text_generation
        _, kwargs = mock_text_generation.call_args
        assert kwargs == {"max_output_tokens": 256, "temperature": 0.8}

        # Assert that the response contains the generated replies and the right response
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        assert [isinstance(reply, str) for reply in response["replies"]]
        assert response["replies"][0] == "I'm fine, thanks."

        # Assert that the response contains the metadata
        assert "metadata" in response
        assert isinstance(response["metadata"], list)
        assert len(response["metadata"]) > 0
        assert [isinstance(reply, str) for reply in response["replies"]]
