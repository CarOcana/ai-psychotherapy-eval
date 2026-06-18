import argparse
import importlib.util
import os
import sys
import types
import unittest


def stub_optional_dependencies():
    pandas_mod = types.ModuleType("pandas")
    pandas_mod.read_csv = None
    sys.modules.setdefault("pandas", pandas_mod)

    tqdm_mod = types.ModuleType("tqdm")
    class DummyTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable or []
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix_str(self, *args, **kwargs):
            return None
        @staticmethod
        def write(*args, **kwargs):
            return None
    tqdm_mod.tqdm = DummyTqdm
    sys.modules.setdefault("tqdm", tqdm_mod)

    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    genai_types_mod = types.ModuleType("google.genai.types")
    genai_mod.Client = object
    genai_mod.types = genai_types_mod
    google_mod.genai = genai_mod
    sys.modules.setdefault("google", google_mod)
    sys.modules.setdefault("google.genai", genai_mod)
    sys.modules.setdefault("google.genai.types", genai_types_mod)

    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = object
    sys.modules.setdefault("openai", openai_mod)

    characterai_mod = types.ModuleType("PyCharacterAI")
    characterai_mod.get_client = None
    sys.modules.setdefault("PyCharacterAI", characterai_mod)


def load_runner():
    stub_optional_dependencies()
    runner_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "simulation_runner.py"
    )
    spec = importlib.util.spec_from_file_location("simulation_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["simulation_runner"] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


class LocalizationMappingTests(unittest.TestCase):
    def test_field_mapping_renames_nested_keys(self):
        response = {"raiz": {"valor": 7}}
        mapping = {"field_maps": {"raiz.valor": "root.value"}}

        self.assertEqual(
            runner.apply_schema_mapping(response, mapping),
            {"root": {"value": 7}}
        )

    def test_field_and_enum_mapping_support_arrays(self):
        response = {
            "respuestas": [
                {"id": "pregunta1", "severidad": "No aplica"},
                {"id": "pregunta2", "severidad": "Moderadamente"}
            ]
        }
        mapping = {
            "field_maps": {
                "respuestas[]": "question_responses[]",
                "question_responses[].id": "question_responses[].question_id",
                "question_responses[].severidad": "question_responses[].severity"
            },
            "enum_maps": {
                "question_responses[].severity": {
                    "No aplica": "N/A",
                    "Moderadamente": "Moderately"
                }
            }
        }

        self.assertEqual(
            runner.apply_schema_mapping(response, mapping),
            {
                "question_responses": [
                    {"question_id": "pregunta1", "severity": "N/A"},
                    {"question_id": "pregunta2", "severity": "Moderately"}
                ]
            }
        )

    def test_localized_schema_canonicalizes_response(self):
        model_schema = {
            "type": "object",
            "properties": {
                "respuesta": {
                    "type": "string",
                    "enum": ["si"]
                }
            },
            "required": ["respuesta"]
        }
        canonical_schema = {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "enum": ["yes"]
                }
            },
            "required": ["response"]
        }
        schema = runner.LocalizedSchema(
            name="example",
            model_schema=model_schema,
            canonical_schema=canonical_schema,
            mapping={
                "field_maps": {"respuesta": "response"},
                "enum_maps": {"response": {"si": "yes"}}
            }
        )

        self.assertEqual(
            runner.canonicalize_response_for_schema({"respuesta": "si"}, schema),
            {"response": "yes"}
        )


class LanguageConfigTests(unittest.TestCase):
    def test_spanish_language_resolves_locale_paths(self):
        args = argparse.Namespace(
            preset=None,
            config="run_simulation/configs/dummy_1s_8t.json",
            run_name=None,
            num_sessions=None,
            num_turns=None,
            pairings_file=None,
            personas_file=None,
            language="es"
        )

        config = runner.build_runtime_config(args)

        self.assertEqual(config["language"], "es")
        self.assertEqual(config["paths"]["prompt_dir"], os.path.join("locales", "es", "prompts"))
        self.assertEqual(config["paths"]["json_schema_dir"], os.path.join("locales", "es", "json_schemas"))
        self.assertEqual(config["paths"]["personas_file"], os.path.join("locales", "es", "patient_personas.csv"))
        self.assertEqual(config["locale"]["manifest"], os.path.join(runner.SCRIPT_DIR, "locales", "es", "manifest.json"))


if __name__ == "__main__":
    unittest.main()
