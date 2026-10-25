import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

CONTRACTS = Path(__file__).parent.parent / "contracts"
SCHEMAS = {p.name: json.loads(p.read_text()) for p in CONTRACTS.glob("*.schema.json")}
REGISTRY = Registry().with_resources(
    (name, Resource.from_contents(schema)) for name, schema in SCHEMAS.items()
)


def validator(file, definition=None):
    schema = SCHEMAS[file]
    if definition is not None:
        schema = {"$ref": f"{file}#/$defs/{definition}"}
    return Draft202012Validator(schema, registry=REGISTRY, format_checker=Draft202012Validator.FORMAT_CHECKER)


TARGETS = [
    ("chunk.schema.json", None),
    ("retrieve.schema.json", "request"),
    ("retrieve.schema.json", "response"),
    ("generate.schema.json", "request"),
    ("generate.schema.json", "response"),
]


def examples(file, definition):
    schema = SCHEMAS[file]
    return (schema["$defs"][definition] if definition else schema)["examples"]


@pytest.mark.parametrize("schema", SCHEMAS.values(), ids=SCHEMAS.keys())
def test_schemas_are_valid(schema):
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(("file", "definition"), TARGETS)
def test_examples_validate(file, definition):
    for example in examples(file, definition):
        validator(file, definition).validate(example)


def test_chunk_rejects_bad_records():
    v = validator("chunk.schema.json")
    good = examples("chunk.schema.json", None)[0]
    for field, bad in [("commit_sha", "main"), ("kind", "image"), ("start_line", 0), ("extra", 1)]:
        record = copy.deepcopy(good)
        record[field] = bad
        assert not v.is_valid(record), field
    missing = copy.deepcopy(good)
    del missing["source_url"]
    assert not v.is_valid(missing)
