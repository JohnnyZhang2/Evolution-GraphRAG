import os
from pathlib import Path

from scripts.check_env_sync import classify_env, OPTIONAL_ENV

# Set up a minimal temporary .env content for test (monkeypatch-like without pytest fixtures)

def write_env(lines):
    Path('.env').write_text('\n'.join(lines), encoding='utf-8')


def test_required_and_optional_classification():
    # Build a pseudo .env with only a subset
    write_env([
        'NEO4J_URI=bolt://localhost:7687',
        'NEO4J_USER=neo4j',
        'NEO4J_PASSWORD=test',
        'LLM_MODEL=openai/gpt-oss-120b',
        'EMBEDDING_MODEL=text-embedding-qwen3-embedding-0.6b',
    ])
    info = classify_env()
    # Optional variables should appear in missing_optional set
    assert all(opt in info['missing_optional'] or opt in info['commented_optional'] for opt in OPTIONAL_ENV)
    # 由于 classify_env 改为仅少量核心必需项（此测试已全部提供），允许 missing_required 为空
    assert len(info['missing_required']) == 0


def test_no_failure_when_optional_missing():
    # Provide all required (simplified assumption) - we relax to only core ones for demonstration
    write_env([
        'NEO4J_URI=bolt://localhost:7687',
        'NEO4J_USER=neo4j',
        'NEO4J_PASSWORD=test',
        'LLM_MODEL=openai/gpt-oss-120b',
        'EMBEDDING_MODEL=text-embedding-qwen3-embedding-0.6b',
        'TOP_K=8',
        'EXPAND_HOPS=1',
        'CHUNK_SIZE=800',
        'CHUNK_OVERLAP=120',
        'RELATION_EXTRACTION=true',
        'RELATION_WINDOW=2',
        'RELATION_CHUNK_TRUNC=400',
        'RELATION_LLM_TEMPERATURE=0.0',
        'RELATION_DEBUG=false',
        'REL_WEIGHT_STEP_NEXT=0.12',
        'REL_WEIGHT_REFERENCES=0.18',
        'REL_WEIGHT_FOLLOWS=0.16',
        'REL_WEIGHT_CAUSES=0.22',
        'REL_WEIGHT_SUPPORTS=0.2',
        'REL_WEIGHT_PART_OF=0.15',
        'REL_WEIGHT_SUBSTEP_OF=0.17',
        'REL_WEIGHT_CONTRASTS=0.14',
        'REL_WEIGHT_DEFAULT=0.15',
        'REL_WEIGHT_RELATES=0.15',
        'REL_WEIGHT_COOCCUR=0.10',
    ])
    info = classify_env()
    # All missing should now be optional only (or none)
    assert not info['missing_required']


if __name__ == '__main__':
    # Allow running test file directly
    test_required_and_optional_classification()
    test_no_failure_when_optional_missing()
    print('env_sync tests passed')
